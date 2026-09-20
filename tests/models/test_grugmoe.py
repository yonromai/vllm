# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import os
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig

from tests.utils import ensure_current_vllm_config
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.grugmoe import (
    _ROUTER_COMBINE_WEIGHT_EPS,
    _ROUTER_COMBINE_WEIGHT_SUM,
    GrugMoeAttention,
    GrugMoeDecoderLayer,
    GrugMoeGatedNorm,
    GrugMoeMLP,
    GrugMoeModel,
    GrugMoeRouter,
    GrugMoeRuntimeConfig,
    GrugMoeShortConv,
    _raise_for_unsupported_modes,
    _try_load_grug_expert_weight,
    get_grug_moe_runtime_info,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager


@pytest.fixture(autouse=True)
def single_rank_model_parallel():
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def _tiny_config() -> GrugMoeRuntimeConfig:
    return GrugMoeRuntimeConfig(
        vocab_size=32,
        hidden_dim=4,
        intermediate_dim=3,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        max_seq_len=8,
        sliding_window=8,
        initializer_std=0.02,
        disable_pko=True,
    ).validate()


def _tiny_hero_config() -> GrugMoeRuntimeConfig:
    return GrugMoeRuntimeConfig(
        vocab_size=32,
        artifact_schema_version=2,
        hidden_dim=16,
        intermediate_dim=6,
        shared_expert_intermediate_dim=5,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=8,
        num_layers=5,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=1,
        global_kv_heads=2,
        head_dim=4,
        max_seq_len=16,
        sliding_window=8,
        global_every=3,
        initializer_std=0.02,
        disable_pko=True,
        rope_fused=True,
        sconv=True,
        sconv_kernel=4,
        sconv_sites=("k", "attn", "mlp"),
    ).validate()


def _reference_rms_norm(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    x_float = x.float()
    out = x_float * torch.rsqrt(
        torch.mean(torch.square(x_float), dim=-1, keepdim=True) + norm.variance_epsilon
    )
    return (out * norm.weight.float()).to(x.dtype)


def _reference_gated_norm(
    x: torch.Tensor,
    module: GrugMoeGatedNorm,
) -> torch.Tensor:
    gate_hidden = F.silu(F.linear(x, module.down_proj.weight))
    gate = F.linear(gate_hidden, module.up_proj.weight)
    return x * torch.sigmoid(gate).to(x.dtype)


def _reference_dense_mlp(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    gate = F.linear(x, module.gate_proj.weight)
    up = F.linear(x, module.up_proj.weight)
    return F.linear(F.silu(gate) * up, module.down_proj.weight)


def _reference_moe(x: torch.Tensor, module: GrugMoeMLP) -> torch.Tensor:
    cfg = module.cfg
    x_flat = x.reshape(-1, x.shape[-1])
    routed_input = x_flat
    if module.latent_down_proj is not None:
        assert module.latent_norm is not None
        routed_input = _reference_rms_norm(
            F.linear(x_flat, module.latent_down_proj.weight),
            module.latent_norm,
        )
    router_logits = F.linear(x_flat.float(), module.router.weight.float())
    _, top_indices = torch.topk(
        router_logits + module.router.bias.float(),
        k=cfg.num_experts_per_token + 1,
        dim=-1,
    )
    selected = top_indices[:, : cfg.num_experts_per_token]
    combine_weights = torch.sigmoid(torch.gather(router_logits, dim=-1, index=selected))
    denom = combine_weights.sum(dim=-1, keepdim=True) + _ROUTER_COMBINE_WEIGHT_EPS
    combine_weights = combine_weights * (_ROUTER_COMBINE_WEIGHT_SUM / denom)

    routed_experts = module.experts.routed_experts
    w13_weight = routed_experts.w13_weight
    gate_proj_weight = w13_weight[:, : cfg.intermediate_dim, :]
    up_proj_weight = w13_weight[:, cfg.intermediate_dim :, :]
    down_proj_weight = routed_experts.w2_weight
    gate_weight = gate_proj_weight[selected]
    up_weight = up_proj_weight[selected]
    down_weight = down_proj_weight[selected]
    gate = torch.einsum(
        "td,tkid->tki",
        routed_input,
        gate_weight.to(routed_input.dtype),
    )
    up = torch.einsum(
        "td,tkid->tki",
        routed_input,
        up_weight.to(routed_input.dtype),
    )
    expert_out = torch.einsum(
        "tki,tkdi->tkd",
        F.silu(gate) * up,
        down_weight.to(x_flat.dtype),
    )
    out = torch.sum(
        expert_out * combine_weights.to(expert_out.dtype).unsqueeze(-1),
        dim=1,
    )
    if module.latent_up_proj is not None:
        out = F.linear(out, module.latent_up_proj.weight)
    return out.to(x.dtype).reshape_as(x)


def _fill_parameter(param: torch.Tensor, start: float, end: float) -> None:
    param.copy_(
        torch.linspace(
            start,
            end,
            steps=param.numel(),
            dtype=param.dtype,
            device=param.device,
        ).view_as(param)
    )


def _process_moe_weights(module: GrugMoeMLP) -> None:
    routed_experts = module.experts.routed_experts
    routed_experts.quant_method.process_weights_after_loading(routed_experts)


class _FakeAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        return F.linear(hidden_states, self.weight)


class _FakeDenseMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _reference_dense_mlp(x, self)


class _FixedProjection(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.value[: x.shape[0]], None


class _CaptureAttention(nn.Module):
    def __init__(self, q_size: int) -> None:
        super().__init__()
        self.q_size = q_size
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        self.k = k
        self.v = v
        return torch.zeros(q.shape[0], self.q_size, dtype=q.dtype)


class _AddLayer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        _positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states + self.value


def test_grug_model_returns_requested_eagle3_auxiliary_states():
    model = GrugMoeModel.__new__(GrugMoeModel)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.embed_tokens = nn.Embedding.from_pretrained(
        torch.arange(16).reshape(4, 4).float()
    )
    model.embed_norm = nn.Identity()
    model.embed_gated_norm = nn.Identity()
    model.layers = nn.ModuleList([_AddLayer(1), _AddLayer(2), _AddLayer(4)])
    model.norm = nn.Identity()
    model.final_gated_norm = nn.Identity()
    model._set_aux_hidden_state_layers((0, 2))
    input_ids = torch.tensor([1, 3])

    final_hidden_state, auxiliary_states = model.forward(input_ids, torch.arange(2))

    embedded = model.embed_tokens(input_ids)
    assert torch.equal(final_hidden_state, embedded + 7)
    assert len(auxiliary_states) == 2
    assert torch.equal(auxiliary_states[0], embedded)
    assert torch.equal(auxiliary_states[1], embedded + 3)


def test_grug_moe_config_parses_hf_aliases_and_rope_theta():
    hf_config = SimpleNamespace(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        shared_expert_intermediate_size=24,
        num_local_experts=6,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=4,
        max_position_embeddings=256,
        sliding_window=64,
        rms_norm_eps=1e-6,
        initializer_range=0.01,
        qk_mult=0.5,
        qk_mult_long_scale=1.25,
        disable_pko=True,
        disable_long_rope=True,
        rope={"theta": 30000.0},
        rope_parameters={"rope_theta": 50000.0},
        rope_theta=60000.0,
    )

    cfg = GrugMoeRuntimeConfig.from_hf_config(hf_config)

    assert cfg.vocab_size == 128
    assert cfg.hidden_dim == 16
    assert cfg.intermediate_dim == 32
    assert cfg.shared_expert_intermediate_dim == 24
    assert cfg.num_experts == 6
    assert cfg.num_experts_per_token == 2
    assert cfg.num_layers == 3
    assert cfg.num_heads == 4
    assert cfg.num_kv_heads == 2
    assert cfg.inferred_head_dim == 4
    assert cfg.max_seq_len == 256
    assert cfg.sliding_window == 64
    assert cfg.layer_norm_eps == 1e-6
    assert cfg.initializer_std == 0.01
    assert cfg.qk_mult == 0.5
    assert cfg.qk_mult_long_scale == 1.25
    assert cfg.disable_pko is True
    assert cfg.disable_long_rope is True
    assert cfg.rope_theta == 60000.0


def test_grug_moe_hf_config_loads_exported_artifact_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["GrugMoeForCausalLM"],
                "model_type": "grug_moe",
                "vocab_size": 128,
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "shared_expert_intermediate_dim": 24,
                "num_experts": 6,
                "num_experts_per_token": 2,
                "num_layers": 26,
                "num_heads": 4,
                "num_kv_heads": 2,
                "head_dim": 4,
                "max_seq_len": 32768,
                "sliding_window": 2048,
                "layer_norm_eps": 1e-6,
                "initializer_std": 0.01,
                "qk_mult": 0.5,
                "qk_mult_long_scale": 1.25,
                "disable_pko": True,
                "disable_long_rope": True,
                "rope": {"theta": 50000.0},
            }
        )
    )

    hf_config = get_config(tmp_path, trust_remote_code=False)
    auto_config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)

    for cfg in (hf_config, auto_config):
        assert cfg.__class__.__name__ == "GrugMoeConfig"
        assert cfg.hidden_size == 16
        assert cfg.hidden_dim == 16
        assert cfg.intermediate_size == 32
        assert cfg.intermediate_dim == 32
        assert cfg.shared_expert_intermediate_size == 24
        assert cfg.num_local_experts == 6
        assert cfg.num_experts_per_tok == 2
        assert cfg.num_hidden_layers == 26
        assert cfg.num_attention_heads == 4
        assert cfg.num_key_value_heads == 2
        assert cfg.attention_head_dim == 4
        assert cfg.max_position_embeddings == 32768
        assert cfg.sliding_window == 2048
        assert len(cfg.layer_types) == 26
        assert cfg.layer_types[0] == "sliding_attention"
        assert cfg.layer_types[3] == "full_attention"
        assert cfg.layer_types[24] == "sliding_attention"
        assert cfg.layer_types[25] == "full_attention"
        assert set(cfg.layer_types) == {"full_attention", "sliding_attention"}
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.initializer_range == 0.01
        assert cfg.qk_mult == 0.5
        assert cfg.qk_mult_long_scale == 1.25
        assert cfg.disable_pko is True
        assert cfg.disable_long_rope is True
        assert cfg.rope_theta == 50000.0
    assert cfg.rope_parameters["rope_theta"] == 50000.0

    vllm_config = EngineArgs(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
    ).create_engine_config()
    assert vllm_config.model_config.is_hybrid is False
    assert vllm_config.model_config.has_inner_state is False
    assert vllm_config.model_config.get_sliding_window() == 2048
    assert vllm_config.cache_config.sliding_window is None

    runtime_config = GrugMoeRuntimeConfig.from_hf_config(hf_config)
    final_full_layer = GrugMoeDecoderLayer(
        runtime_config,
        cache_config=vllm_config.cache_config,
        params_dtype=torch.float32,
        layer_index=25,
        prefix="layers.25",
    )

    assert final_full_layer.self_attn.attn.sliding_window is None


def test_grug_moe_hf_config_loads_hero_fields(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["GrugMoeForCausalLM"],
                "model_type": "grug_moe",
                "vocab_size": 32,
                "hidden_size": 16,
                "moe_intermediate_size": 6,
                "shared_expert_intermediate_size": 5,
                "num_shared_experts": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "latent_dim": 8,
                "num_hidden_layers": 5,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "local_kv_heads": 1,
                "global_kv_heads": 2,
                "head_dim": 4,
                "max_position_embeddings": 16,
                "sliding_window": 8,
                "global_every": 3,
                "rope_fused": True,
                "sconv": True,
                "sconv_kernel": 4,
                "sconv_sites": ["k", "attn", "mlp"],
                "grugmoe_artifact_schema_version": 2,
            }
        )
    )

    hf_config = get_config(tmp_path, trust_remote_code=False)
    runtime_config = GrugMoeRuntimeConfig.from_hf_config(hf_config)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float32",
    ).create_engine_config()

    assert hf_config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert runtime_config == _tiny_hero_config()
    assert vllm_config.model_config.is_hybrid is True
    assert vllm_config.model_config.has_inner_state is True
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_grug_moe_config_rejects_noncanonical_layer_types():
    with pytest.raises(
        ValueError,
        match="must match the GrugMoE attention architecture",
    ):
        GrugMoeConfig(
            num_layers=3,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )


@pytest.mark.parametrize(
    ("hero_field", "value"),
    [
        ("latent_dim", 8),
        ("local_kv_heads", 1),
        ("num_shared_experts", 2),
        ("global_every", 3),
        ("rope_fused", True),
        ("sconv", True),
    ],
)
def test_snowball_grug_moe_rejects_hero_behavior(
    hero_field,
    value,
):
    kwargs = {hero_field: value}
    if hero_field == "local_kv_heads":
        kwargs["global_kv_heads"] = 2
        kwargs["num_kv_heads"] = 2
        kwargs["num_heads"] = 4

    with pytest.raises(
        ValueError,
        match="require grugmoe_artifact_schema_version=2",
    ):
        GrugMoeConfig(**kwargs)


def test_grug_moe_sconv_rejects_unsupported_kernel_width():
    with pytest.raises(ValueError, match="sconv_kernel must be between 2 and 5"):
        GrugMoeConfig(
            grugmoe_artifact_schema_version=2,
            sconv=True,
            sconv_kernel=6,
        )


def _minimal_model_config_for_parallel_check(
    architectures: list[str],
    model_type: str,
) -> ModelConfig:
    model_config = object.__new__(ModelConfig)
    object.__setattr__(
        model_config,
        "model_arch_config",
        SimpleNamespace(
            total_num_attention_heads=4,
            total_num_kv_heads=4,
            is_deepseek_mla=False,
            architectures=architectures,
        ),
    )
    object.__setattr__(
        model_config, "hf_config", SimpleNamespace(model_type=model_type)
    )
    object.__setattr__(model_config, "multimodal_config", None)
    return model_config


def test_parallel_config_allows_tpu_heads_but_preserves_gpu_checks(
    monkeypatch,
):
    # ParallelConfig now eagerly rejects world_size > device_count on CUDA at
    # construction, so pretend the node has enough GPUs to build the 8-way TP
    # config this test uses only as a fixture for verify_with_parallel_config.
    monkeypatch.setattr("vllm.config.parallel.current_platform.device_count", lambda: 8)
    parallel_config = ParallelConfig(tensor_parallel_size=8)

    grug_model_config = _minimal_model_config_for_parallel_check(
        ["GrugMoeForCausalLM"],
        "grug_moe",
    )
    monkeypatch.setattr("vllm.config.model.current_platform.is_tpu", lambda: True)
    ModelConfig.verify_with_parallel_config(grug_model_config, parallel_config)

    generic_model_config = _minimal_model_config_for_parallel_check(
        ["LlamaForCausalLM"],
        "llama",
    )
    with pytest.raises(ValueError, match="Total number of attention heads"):
        ModelConfig.verify_with_parallel_config(generic_model_config, parallel_config)

    monkeypatch.setattr("vllm.config.model.current_platform.is_tpu", lambda: False)
    with pytest.raises(ValueError, match="Total number of attention heads"):
        ModelConfig.verify_with_parallel_config(grug_model_config, parallel_config)


def _minimal_vllm_config_for_mode_check(
    *,
    tensor_parallel_size=1,
    quantization_method=None,
):
    quant_config = (
        None
        if quantization_method is None
        else SimpleNamespace(get_name=lambda: quantization_method)
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tensor_parallel_size),
        lora_config=None,
        quant_config=quant_config,
    )


def _patch_grug_parallel_runtime(monkeypatch, fake_config, *, is_tpu):
    monkeypatch.setattr(
        "vllm.model_executor.models.grugmoe.get_tensor_model_parallel_world_size",
        lambda: fake_config.parallel_config.tensor_parallel_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.grugmoe.current_platform.is_tpu",
        lambda: is_tpu,
    )


def test_grug_moe_preserves_gpu_tensor_parallel_restriction(monkeypatch):
    fake_config = _minimal_vllm_config_for_mode_check(tensor_parallel_size=8)
    _patch_grug_parallel_runtime(monkeypatch, fake_config, is_tpu=False)

    with pytest.raises(NotImplementedError, match="tensor_parallel_size=1"):
        _raise_for_unsupported_modes(fake_config)


@pytest.mark.parametrize(
    ("is_tpu", "quantization_method", "supported"),
    [
        (True, "unquantized", True),
        (True, "fp8", False),
        (False, "unquantized", False),
    ],
)
def test_grug_moe_tpu_accepts_only_unquantized_backend(
    monkeypatch,
    is_tpu,
    quantization_method,
    supported,
):
    fake_config = _minimal_vllm_config_for_mode_check(
        tensor_parallel_size=8 if is_tpu else 1,
        quantization_method=quantization_method,
    )
    _patch_grug_parallel_runtime(monkeypatch, fake_config, is_tpu=is_tpu)

    if supported:
        _raise_for_unsupported_modes(fake_config)
    else:
        with pytest.raises(NotImplementedError, match="does not support quantization"):
            _raise_for_unsupported_modes(fake_config)


def test_grug_gated_norm_matches_reference_math():
    module = GrugMoeGatedNorm(
        hidden_dim=2,
        params_dtype=torch.float32,
    )
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float32)
    down_weight = torch.zeros_like(module.down_proj.weight)
    up_weight = torch.zeros_like(module.up_proj.weight)
    down_weight[0] = torch.tensor([1.0, -1.5])
    down_weight[1] = torch.tensor([-0.25, 0.5])
    up_weight[0, 0] = 0.75
    up_weight[1, 1] = -0.5
    with torch.no_grad():
        module.down_proj.weight.copy_(down_weight)
        module.up_proj.weight.copy_(up_weight)

    actual = module(x)
    gate_down = F.linear(x.float(), down_weight.float())
    gate_hidden = F.silu(gate_down)
    gate_up = F.linear(gate_hidden, up_weight.float())
    gate = torch.sigmoid(gate_up)
    expected = x * gate

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    indices = torch.tensor([1])
    captured: dict[str, torch.Tensor] = {}
    traced = module(x, capture=(indices, captured))
    torch.testing.assert_close(traced, actual, atol=0, rtol=0)
    for name, stage in (
        ("gate_down", gate_down),
        ("gate_silu", gate_hidden),
        ("gate_up", gate_up),
        ("gate_sigmoid", gate),
    ):
        torch.testing.assert_close(
            captured[name], stage.index_select(0, indices), atol=0, rtol=0
        )


def test_grug_gated_norm_preserves_projection_dispatch():
    module = GrugMoeGatedNorm(
        hidden_dim=2,
        params_dtype=torch.float32,
    )
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7]], dtype=torch.float32)
    fixed_gate = torch.tensor([[0.4, -0.8], [1.2, 0.6]], dtype=torch.float32)
    module.down_proj = _FixedProjection(
        torch.zeros(x.shape[0], module.down_proj.weight.shape[0])
    )
    module.up_proj = _FixedProjection(fixed_gate)

    actual = module(x)
    expected = x * torch.sigmoid(fixed_gate)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_grug_moe_uses_qb_bias_for_selection_and_normalized_unbiased_weights():
    cfg = _tiny_config()
    mlp = GrugMoeMLP(cfg, params_dtype=torch.float32)
    x = torch.tensor(
        [
            [0.3, -0.4, 0.2, 0.1],
            [-0.5, 0.7, -0.2, 0.4],
        ],
        dtype=torch.float32,
    )
    router_weight = torch.tensor(
        [
            [0.7, -0.3, 0.2, 0.1],
            [-0.2, 0.4, -0.6, 0.5],
            [0.1, -0.5, 0.3, -0.4],
            [-0.1, 0.2, 0.8, -0.3],
        ],
        dtype=torch.float32,
    )
    router_bias = torch.tensor([0.0, 0.0, 3.0, -2.0], dtype=torch.float32)
    generator = torch.Generator().manual_seed(1234)
    gate_weight = torch.randn(
        cfg.num_experts,
        cfg.intermediate_dim,
        cfg.hidden_dim,
        generator=generator,
    )
    up_weight = torch.randn(
        cfg.num_experts,
        cfg.intermediate_dim,
        cfg.hidden_dim,
        generator=generator,
    )
    down_weight = torch.randn(
        cfg.num_experts,
        cfg.hidden_dim,
        cfg.intermediate_dim,
        generator=generator,
    )
    with torch.no_grad():
        mlp.router.weight.copy_(router_weight)
        # Match TPU functional_call, which substitutes the registered bias
        # after the custom router has captured its initial Parameter.
        mlp.router.bias = torch.nn.Parameter(router_bias.clone())
        routed_experts = mlp.experts.routed_experts
        routed_experts.w13_weight[:, : cfg.intermediate_dim, :].copy_(gate_weight)
        routed_experts.w13_weight[:, cfg.intermediate_dim :, :].copy_(up_weight)
        routed_experts.w2_weight.copy_(down_weight)
        _process_moe_weights(mlp)

    with set_forward_context(None, get_current_vllm_config(), num_tokens=x.shape[0]):
        actual = mlp(x)
    router_logits = F.linear(x.float(), router_weight.float())
    _, top_indices = torch.topk(
        router_logits + router_bias,
        k=cfg.num_experts_per_token + 1,
        dim=-1,
    )
    selected = top_indices[:, : cfg.num_experts_per_token]
    combine_weights = torch.sigmoid(torch.gather(router_logits, -1, selected))
    denom = combine_weights.sum(dim=-1, keepdim=True) + _ROUTER_COMBINE_WEIGHT_EPS
    combine_weights = combine_weights * (_ROUTER_COMBINE_WEIGHT_SUM / denom)
    expected_tokens = []
    for token_index, token in enumerate(x):
        token_out = torch.zeros_like(token)
        for slot_index, expert_index in enumerate(selected[token_index]):
            expert_id = int(expert_index.item())
            gate = F.linear(token, gate_weight[expert_id])
            up = F.linear(token, up_weight[expert_id])
            expert_out = F.linear(F.silu(gate) * up, down_weight[expert_id])
            token_out = (
                token_out + combine_weights[token_index, slot_index] * expert_out
            )
        expected_tokens.append(token_out)
    expected = torch.stack(expected_tokens)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_hero_grug_moe_latent_experts_match_reference_math():
    cfg = _tiny_hero_config()
    mlp = GrugMoeMLP(cfg, params_dtype=torch.float32)
    x = torch.linspace(-0.7, 0.8, steps=3 * cfg.hidden_dim).view(
        3, cfg.hidden_dim
    )

    with torch.no_grad():
        _fill_parameter(mlp.router.weight, -0.3, 0.4)
        _fill_parameter(mlp.router.bias, 0.2, -0.1)
        assert mlp.latent_down_proj is not None
        assert mlp.latent_norm is not None
        assert mlp.latent_up_proj is not None
        _fill_parameter(mlp.latent_down_proj.weight, -0.2, 0.25)
        _fill_parameter(mlp.latent_norm.weight, 0.7, 1.2)
        _fill_parameter(mlp.latent_up_proj.weight, 0.3, -0.2)
        routed_experts = mlp.experts.routed_experts
        _fill_parameter(
            routed_experts.w13_weight[:, : cfg.intermediate_dim, :],
            -0.4,
            0.2,
        )
        _fill_parameter(
            routed_experts.w13_weight[:, cfg.intermediate_dim :, :],
            0.25,
            -0.35,
        )
        _fill_parameter(routed_experts.w2_weight, -0.15, 0.3)
        _process_moe_weights(mlp)

    expected = _reference_moe(x, mlp)
    with set_forward_context(None, get_current_vllm_config(), num_tokens=x.shape[0]):
        actual = mlp(x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_grug_moe_router_selects_biased_topk_and_normalized_unbiased_weights():
    router = GrugMoeRouter(
        top_k=2,
        global_num_experts=4,
        bias=torch.tensor([0.0, 0.0, 3.0, -2.0], dtype=torch.float32),
    )
    router_logits = torch.tensor(
        [
            [0.4, 0.3, -0.1, 0.2],
            [-0.2, 0.6, 0.1, 0.5],
        ],
        dtype=torch.float32,
    )

    weights, ids = router.select_experts(
        hidden_states=torch.empty(2, 4),
        router_logits=router_logits,
    )

    expected_ids = torch.topk(router_logits + router.bias, k=3, dim=-1).indices[:, :2]
    expected_weights = torch.sigmoid(
        torch.gather(router_logits, dim=-1, index=expected_ids)
    )
    denom = expected_weights.sum(dim=-1, keepdim=True) + _ROUTER_COMBINE_WEIGHT_EPS
    expected_weights = expected_weights * (_ROUTER_COMBINE_WEIGHT_SUM / denom)
    torch.testing.assert_close(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.full((2,), _ROUTER_COMBINE_WEIGHT_SUM),
    )


def test_grug_moe_router_keeps_zero_weight_underflow_finite():
    router = GrugMoeRouter(
        top_k=2,
        global_num_experts=4,
        bias=torch.zeros(4, dtype=torch.float32),
    )
    router_logits = torch.full((1, 4), -torch.inf, dtype=torch.float32)

    weights, _ = router.select_experts(
        hidden_states=torch.empty(1, 4),
        router_logits=router_logits,
    )

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, torch.zeros_like(weights))


def test_grug_moe_runtime_info_reports_worker_local_ep_contract():
    cfg = _tiny_config()
    layer = GrugMoeDecoderLayer(
        cfg,
        cache_config=None,
        params_dtype=torch.float32,
        layer_index=0,
    )
    model = SimpleNamespace(
        layers=[layer],
        start_layer=0,
        end_layer=1,
    )
    vllm_config = SimpleNamespace()

    info = get_grug_moe_runtime_info(vllm_config, model)

    assert info["tp_size"] == 1
    assert info["pp_size"] == 1
    assert info["pp_rank"] == 0
    assert info["start_layer"] == 0
    assert info["end_layer"] == 1
    assert info["dp_size"] == 1
    assert info["ep_size"] == 1
    assert info["use_ep"] is False
    assert info["num_experts"] == cfg.num_experts
    assert info["num_local_experts"] == cfg.num_experts
    assert info["top_k"] == cfg.num_experts_per_token
    assert info["local_expert_ids"] == list(range(cfg.num_experts))
    assert info["local_expert_ownership"] == "[0..3]"
    assert info["expert_placement_strategy"] == "linear"
    assert info["all2all_backend"] == "allgather_reducescatter"
    assert info["attention_backend"]


def test_grug_moe_routed_experts_keep_model_dtype():
    cfg = GrugMoeRuntimeConfig(
        vocab_size=32,
        hidden_dim=4,
        intermediate_dim=3,
        shared_expert_intermediate_dim=5,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        max_seq_len=8,
        sliding_window=8,
        initializer_std=0.02,
        disable_pko=True,
    ).validate()
    layer = GrugMoeDecoderLayer(
        cfg,
        cache_config=None,
        params_dtype=torch.bfloat16,
        layer_index=0,
    )

    assert layer.attn_gated_norm.down_proj.weight.dtype == torch.bfloat16
    assert layer.self_attn.q_proj.weight.dtype == torch.bfloat16
    assert layer.self_attn.attn_gate.weight.dtype == torch.bfloat16
    assert layer.mlp.router.weight.dtype == torch.float32
    assert layer.mlp.experts.routed_experts.w13_weight.dtype == torch.bfloat16
    assert layer.mlp.experts.routed_experts.w2_weight.dtype == torch.bfloat16
    assert layer.shared_expert is not None
    assert layer.shared_expert.gate_proj.weight.dtype == torch.bfloat16


def test_hero_grug_moe_selects_logical_kv_heads_by_layer_type():
    cfg = replace(_tiny_hero_config(), sconv=False)
    q = torch.arange(16, dtype=torch.float32).view(1, 16) + 1
    k = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]]]
    ).reshape(1, 8)
    v = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [1.1, 1.2, 1.3, 1.4]]]
    ).reshape(1, 8)
    captures = []

    for is_global in (False, True):
        module = GrugMoeAttention(
            cfg,
            cache_config=None,
            params_dtype=torch.float32,
            sliding_window=None if is_global else cfg.sliding_window,
            use_rope=False,
            qk_mult_scale=1.0,
            is_global=is_global,
            prefix=f"kv.{is_global}",
        )
        module.q_proj = _FixedProjection(q)
        module.k_proj = _FixedProjection(k)
        module.v_proj = _FixedProjection(v)
        capture = _CaptureAttention(module.q_size)
        module.attn = capture
        module(torch.zeros(1, dtype=torch.long), torch.zeros(1, cfg.hidden_dim))
        captures.append(capture)

    local_capture, global_capture = captures
    assert local_capture.k is not None
    assert local_capture.v is not None
    assert global_capture.k is not None
    assert global_capture.v is not None
    expected_local_k_head = _reference_rms_norm(
        k.view(1, 2, 4)[:, :1],
        SimpleNamespace(variance_epsilon=1e-6, weight=torch.ones(4)),
    )
    torch.testing.assert_close(
        local_capture.k.view(1, 2, 4),
        expected_local_k_head.expand(1, 2, 4),
    )
    torch.testing.assert_close(
        local_capture.v.view(1, 2, 4),
        v.view(1, 2, 4)[:, :1].expand(1, 2, 4),
    )
    assert local_capture.v.is_contiguous()
    assert not torch.equal(
        global_capture.k.view(1, 2, 4)[:, 0],
        global_capture.k.view(1, 2, 4)[:, 1],
    )
    torch.testing.assert_close(global_capture.v, v)


def test_grug_moe_attention_schedule_matches_training_architecture():
    cfg = GrugMoeRuntimeConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=3,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=5,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=16,
        sliding_window=8,
        initializer_std=0.02,
        qk_mult_long_scale=1.25,
        disable_pko=True,
        disable_long_rope=True,
    ).validate()

    short_layer = GrugMoeDecoderLayer(
        cfg,
        cache_config=None,
        params_dtype=torch.float32,
        layer_index=0,
        prefix="layers.0",
    )
    periodic_long_layer = GrugMoeDecoderLayer(
        cfg,
        cache_config=None,
        params_dtype=torch.float32,
        layer_index=3,
        prefix="layers.3",
    )
    final_long_layer = GrugMoeDecoderLayer(
        cfg,
        cache_config=None,
        params_dtype=torch.float32,
        layer_index=4,
        prefix="layers.4",
    )

    assert short_layer.self_attn.attn.sliding_window == cfg.sliding_window
    assert short_layer.self_attn.use_rope is True
    assert short_layer.self_attn.qk_mult_scale == 1.0
    assert short_layer.self_attn.rotary_emb.rotary_dim == cfg.inferred_head_dim // 2
    for layer in (periodic_long_layer, final_long_layer):
        assert layer.self_attn.attn.sliding_window is None
        assert layer.self_attn.use_rope is False
        assert layer.self_attn.qk_mult_scale == cfg.qk_mult_long_scale


def _causal_depthwise_reference(
    sequence: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(sequence)
    for token_index in range(sequence.shape[0]):
        for lag in range(weight.shape[0]):
            if token_index >= lag:
                output[token_index] += weight[lag] * sequence[token_index - lag]
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hero_grug_moe_sconv_preserves_request_history_and_isolation():
    prefix = "hero.sconv.history"
    module = GrugMoeShortConv(
        dim=2,
        kernel_size=4,
        params_dtype=torch.float32,
        cache_config=None,
        prefix=prefix,
    ).cuda()
    weight = torch.tensor(
        [[1.0, 0.75], [0.5, -0.25], [-0.25, 0.125], [0.125, 0.5]],
        device="cuda",
    )
    module.weight.data.copy_(weight)
    # State slot zero is vLLM's reserved null block.
    module.kv_cache = (torch.zeros(3, 3, 2, device="cuda"),)

    request_a = torch.tensor(
        [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], device="cuda"
    )
    request_b = torch.tensor([[10.0, 4.0], [20.0, 8.0]], device="cuda")
    prefill = torch.vstack((request_a, request_b))
    prefill_metadata = ShortConvAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=5,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=2,
        query_start_loc_p=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
        has_initial_states_p=torch.tensor([False, False], device="cuda"),
        state_indices_tensor_p=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.empty(0, device="cuda", dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([3, 2], device="cuda"),
    )
    with set_forward_context(
        {prefix: prefill_metadata},
        get_current_vllm_config(),
        num_tokens=prefill.shape[0],
    ):
        prefill_output = module(prefill)

    expected_a = _causal_depthwise_reference(request_a, weight)
    expected_b = _causal_depthwise_reference(request_b, weight)
    torch.testing.assert_close(
        prefill_output,
        torch.vstack((expected_a, expected_b)),
    )

    decode = torch.tensor([[4.0, -4.0], [30.0, 12.0]], device="cuda")
    decode_metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_reqs=2,
        query_start_loc_p=None,
        has_initial_states_p=None,
        state_indices_tensor_p=torch.empty(0, device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([4, 3], device="cuda"),
    )
    with set_forward_context(
        {prefix: decode_metadata},
        get_current_vllm_config(),
        num_tokens=decode.shape[0],
    ):
        decode_output = module(decode)

    expected_decode = torch.vstack(
        (
            _causal_depthwise_reference(torch.vstack((request_a, decode[:1])), weight)[
                -1:
            ],
            _causal_depthwise_reference(torch.vstack((request_b, decode[1:])), weight)[
                -1:
            ],
        )
    )
    torch.testing.assert_close(decode_output, expected_decode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grug_moe_cuda_accepts_bf16_expert_weights():
    cfg = GrugMoeRuntimeConfig(
        vocab_size=64,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        sliding_window=8,
        initializer_std=0.02,
        disable_pko=True,
    ).validate()
    mlp = GrugMoeMLP(
        cfg,
        params_dtype=torch.bfloat16,
    ).cuda()
    x = torch.linspace(
        -0.6,
        0.7,
        steps=3 * cfg.hidden_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).view(3, cfg.hidden_dim)

    with torch.no_grad():
        _fill_parameter(mlp.router.weight, -0.35, 0.4)
        _fill_parameter(mlp.router.bias, 0.25, -0.15)
        routed_experts = mlp.experts.routed_experts
        _fill_parameter(
            routed_experts.w13_weight[:, : cfg.intermediate_dim, :],
            -0.2,
            0.3,
        )
        _fill_parameter(
            routed_experts.w13_weight[:, cfg.intermediate_dim :, :],
            0.35,
            -0.25,
        )
        _fill_parameter(routed_experts.w2_weight, -0.1, 0.2)
        _process_moe_weights(mlp)

    expected = _reference_moe(x, mlp)
    # The fused-MoE kernels allocate through the workspace manager, which a real run
    # installs in GPUModelRunner.__init__. This test drives the layer directly, so it
    # has to do the same setup itself.
    init_workspace_manager(torch.device("cuda"))
    with set_forward_context(None, get_current_vllm_config(), num_tokens=x.shape[0]):
        actual = mlp(x)

    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_grug_moe_router_selects_logical_expert_ids():
    router = GrugMoeRouter(
        top_k=2,
        global_num_experts=8,
        bias=torch.zeros(8, dtype=torch.float32),
    )
    router_logits = torch.full((4, 8), -100.0, dtype=torch.float32)
    router_logits[0, [0, 1]] = torch.tensor([8.0, 7.0])
    router_logits[1, [2, 3]] = torch.tensor([8.0, 7.0])
    router_logits[2, [4, 5]] = torch.tensor([8.0, 7.0])
    router_logits[3, [6, 7]] = torch.tensor([8.0, 7.0])

    _, ids = router.select_experts(
        hidden_states=torch.empty(4, 4),
        router_logits=router_logits,
    )

    selected_expert_ids = sorted(int(expert_id) for expert_id in ids.flatten())
    assert selected_expert_ids == list(range(8))


def test_grug_moe_3d_expert_weights_load_into_fused_moe_layout():
    cfg = _tiny_config()
    mlp = GrugMoeMLP(cfg, params_dtype=torch.float32)
    params_dict = dict(mlp.named_parameters())
    gate_weight = torch.arange(
        cfg.num_experts * cfg.intermediate_dim * cfg.hidden_dim,
        dtype=torch.float32,
    ).view(cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim)
    up_weight = gate_weight + 1000
    down_weight = torch.arange(
        cfg.num_experts * cfg.hidden_dim * cfg.intermediate_dim,
        dtype=torch.float32,
    ).view(cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim)

    assert (
        _try_load_grug_expert_weight(
            "experts.gate_proj.weight",
            gate_weight,
            params_dict,
        )
        == "experts.routed_experts.w13_weight"
    )
    assert (
        _try_load_grug_expert_weight(
            "experts.up_proj.weight",
            up_weight,
            params_dict,
        )
        == "experts.routed_experts.w13_weight"
    )
    assert (
        _try_load_grug_expert_weight(
            "experts.down_proj.weight",
            down_weight,
            params_dict,
        )
        == "experts.routed_experts.w2_weight"
    )

    routed_experts = mlp.experts.routed_experts
    torch.testing.assert_close(
        routed_experts.w13_weight[:, : cfg.intermediate_dim, :],
        gate_weight,
    )
    torch.testing.assert_close(
        routed_experts.w13_weight[:, cfg.intermediate_dim :, :],
        up_weight,
    )
    torch.testing.assert_close(routed_experts.w2_weight, down_weight)


def test_grug_moe_split_expert_weights_load_into_fused_moe_layout():
    cfg = _tiny_config()
    mlp = GrugMoeMLP(cfg, params_dtype=torch.float32)
    params_dict = dict(mlp.named_parameters())
    gate_weight = torch.arange(
        cfg.num_experts * cfg.intermediate_dim * cfg.hidden_dim,
        dtype=torch.float32,
    ).view(cfg.num_experts, cfg.intermediate_dim, cfg.hidden_dim)
    up_weight = gate_weight + 1000
    down_weight = torch.arange(
        cfg.num_experts * cfg.hidden_dim * cfg.intermediate_dim,
        dtype=torch.float32,
    ).view(cfg.num_experts, cfg.hidden_dim, cfg.intermediate_dim)

    for expert_id in reversed(range(cfg.num_experts)):
        assert (
            _try_load_grug_expert_weight(
                f"experts.{expert_id}.gate_proj.weight",
                gate_weight[expert_id],
                params_dict,
            )
            == "experts.routed_experts.w13_weight"
        )
        assert (
            _try_load_grug_expert_weight(
                f"experts.{expert_id}.up_proj.weight",
                up_weight[expert_id],
                params_dict,
            )
            == "experts.routed_experts.w13_weight"
        )
        assert (
            _try_load_grug_expert_weight(
                f"experts.{expert_id}.down_proj.weight",
                down_weight[expert_id],
                params_dict,
            )
            == "experts.routed_experts.w2_weight"
        )

    routed_experts = mlp.experts.routed_experts
    torch.testing.assert_close(
        routed_experts.w13_weight[:, : cfg.intermediate_dim, :],
        gate_weight,
    )
    torch.testing.assert_close(
        routed_experts.w13_weight[:, cfg.intermediate_dim :, :],
        up_weight,
    )
    torch.testing.assert_close(routed_experts.w2_weight, down_weight)
    assert (
        _try_load_grug_expert_weight(
            "shared_experts.0.gate_proj.weight",
            gate_weight[0],
            params_dict,
        )
        is None
    )


def test_grug_moe_tiny_decoder_layer_matches_reference_math():
    cfg = GrugMoeRuntimeConfig(
        vocab_size=32,
        hidden_dim=4,
        intermediate_dim=3,
        shared_expert_intermediate_dim=5,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        max_seq_len=8,
        sliding_window=8,
        initializer_std=0.02,
        disable_pko=True,
    ).validate()
    torch.manual_seed(20260702)
    layer = GrugMoeDecoderLayer(
        cfg,
        cache_config=None,
        params_dtype=torch.float32,
        layer_index=0,
    )
    with torch.no_grad():
        _fill_parameter(layer.attn_gated_norm.down_proj.weight, -0.03, 0.04)
        _fill_parameter(layer.attn_gated_norm.up_proj.weight, 0.05, -0.02)
        _fill_parameter(layer.mlp_gated_norm.down_proj.weight, -0.04, 0.03)
        _fill_parameter(layer.mlp_gated_norm.up_proj.weight, 0.02, -0.05)
        _fill_parameter(layer.mlp.router.weight, -0.5, 0.6)
        _fill_parameter(layer.mlp.router.bias, 0.3, -0.2)
        routed_experts = layer.mlp.experts.routed_experts
        _fill_parameter(
            routed_experts.w13_weight[:, : cfg.intermediate_dim, :],
            -0.2,
            0.4,
        )
        _fill_parameter(
            routed_experts.w13_weight[:, cfg.intermediate_dim :, :],
            0.3,
            -0.3,
        )
        _fill_parameter(routed_experts.w2_weight, -0.1, 0.5)
        _process_moe_weights(layer.mlp)
    fake_attn = _FakeAttention(cfg.hidden_dim)
    with torch.no_grad():
        fake_attn.weight.copy_(
            torch.tensor(
                [
                    [0.2, -0.1, 0.4, 0.3],
                    [-0.5, 0.7, 0.2, -0.2],
                    [0.1, 0.3, -0.6, 0.5],
                    [0.4, -0.3, 0.2, 0.1],
                ],
                dtype=torch.float32,
            )
        )
    layer.self_attn = fake_attn
    fake_shared_expert = _FakeDenseMLP(
        cfg.hidden_dim,
        cfg.shared_expert_intermediate_dim,
    )
    with torch.no_grad():
        fake_shared_expert.gate_proj.weight.copy_(
            torch.linspace(-0.3, 0.4, steps=20, dtype=torch.float32).view(5, 4)
        )
        fake_shared_expert.up_proj.weight.copy_(
            torch.linspace(0.2, -0.5, steps=20, dtype=torch.float32).view(5, 4)
        )
        fake_shared_expert.down_proj.weight.copy_(
            torch.linspace(-0.1, 0.6, steps=20, dtype=torch.float32).view(4, 5)
        )
    layer.shared_expert = fake_shared_expert
    x = torch.tensor(
        [
            [0.2, -0.3, 0.5, 0.7],
            [-0.4, 0.6, -0.1, 0.3],
            [0.9, -0.2, 0.4, -0.5],
        ],
        dtype=torch.float32,
    )
    positions = torch.arange(x.shape[0], dtype=torch.long)

    with set_forward_context(None, get_current_vllm_config(), num_tokens=x.shape[0]):
        actual = layer(positions, x)

    attn_norm = _reference_rms_norm(x, layer.input_layernorm)
    attn_in = _reference_gated_norm(attn_norm, layer.attn_gated_norm)
    hidden_states = x + fake_attn(positions, attn_in)
    mlp_norm = _reference_rms_norm(hidden_states, layer.post_attention_layernorm)
    mlp_in = _reference_gated_norm(mlp_norm, layer.mlp_gated_norm)
    expected_mlp = _reference_moe(mlp_in, layer.mlp)
    assert layer.shared_expert is not None
    expected_mlp = expected_mlp + _reference_dense_mlp(mlp_in, layer.shared_expert)
    expected = hidden_states + expected_mlp

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_grug_xsa_correction_and_gate_match_reference_math():
    cfg = _tiny_config()
    attn = GrugMoeAttention(
        cfg,
        cache_config=None,
        params_dtype=torch.float32,
        sliding_window=cfg.sliding_window,
        use_rope=True,
        qk_mult_scale=1.0,
    )
    hidden_states = torch.tensor(
        [[0.3, -0.1, 0.2, 0.7], [-0.4, 0.6, 0.5, -0.2]],
        dtype=torch.float32,
    )
    attn_output = torch.tensor(
        [[0.8, -0.3, 0.5, 0.1], [0.2, 0.4, -0.6, 0.9]],
        dtype=torch.float32,
    )
    value_states = torch.tensor([[0.5, -0.25], [-0.4, 0.3]], dtype=torch.float32)
    gate_weight = torch.tensor(
        [[0.2, -0.3, 0.5, 0.1], [-0.4, 0.6, 0.2, -0.5]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        attn.attn_gate.weight.copy_(gate_weight)

    actual = attn.apply_xsa(hidden_states, attn_output, value_states)

    aligned_value = value_states.view(2, 1, 2).expand(2, 2, 2)
    attn_heads = attn_output.view(2, 2, 2)
    dot = torch.sum(attn_heads * aligned_value, dim=-1, keepdim=True)
    value_norm_sq = torch.sum(aligned_value * aligned_value, dim=-1, keepdim=True)
    corrected = attn_heads - (dot / (value_norm_sq + 1e-6)) * aligned_value
    gate = 2 * torch.sigmoid(F.linear(hidden_states, gate_weight))
    expected = (gate[..., None] * corrected).reshape(2, 4)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_grug_xsa_bf16_projection_matches_rounded_fp64():
    cfg = _tiny_hero_config()
    attn = GrugMoeAttention(
        cfg,
        cache_config=None,
        params_dtype=torch.bfloat16,
        sliding_window=cfg.sliding_window,
        use_rope=True,
        qk_mult_scale=1.0,
    )
    with torch.no_grad():
        attn.attn_gate.weight.zero_()
    generator = torch.Generator().manual_seed(117)
    attn_output = torch.randn(32, cfg.hidden_dim, generator=generator).bfloat16()
    value_states = torch.randn(
        32, cfg.num_kv_heads * cfg.head_dim, generator=generator
    ).bfloat16()
    actual = attn.apply_xsa(torch.zeros_like(attn_output), attn_output, value_states)

    heads = attn_output.double().view(32, cfg.num_heads, cfg.head_dim)
    values = value_states.double().view(32, cfg.num_kv_heads, cfg.head_dim)
    values = values.repeat_interleave(cfg.num_heads // cfg.num_kv_heads, dim=1)
    dot = (heads * values).sum(dim=-1, keepdim=True)
    norm_sq = (values * values).sum(dim=-1, keepdim=True)
    reference = (heads - dot / (norm_sq + 1e-6) * values).bfloat16().reshape_as(actual)

    assert torch.count_nonzero(actual == reference) >= actual.numel() - 2


def test_grug_moe_registry_imports():
    model_cls = ModelRegistry._try_load_model_cls("GrugMoeForCausalLM")

    assert model_cls is not None
    assert model_cls.__name__ == "GrugMoeForCausalLM"
