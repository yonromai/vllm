# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import math
import os
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch.utils.deterministic
from torch import nn
from transformers import AutoConfig

from tests.utils import ensure_current_vllm_config
from vllm import LLM, SamplingParams
from vllm.config import (
    CUDAGraphMode,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.config.kernel import IrOpPriorityConfig, KernelConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.grugmoe import (
    _ROUTER_COMBINE_WEIGHT_EPS,
    _ROUTER_COMBINE_WEIGHT_SUM,
    GrugMoeAttention,
    GrugMoeDecoderLayer,
    GrugMoeDenseMLP,
    GrugMoeGatedNorm,
    GrugMoeMLP,
    GrugMoeModel,
    GrugMoeRouter,
    GrugMoeRuntimeConfig,
    GrugMoeShortConv,
    _log_scaling_factor,
    _raise_for_unsupported_modes,
    _try_load_grug_expert_weight,
    get_grug_moe_runtime_info,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig
from vllm.utils.torch_utils import current_stream, set_default_torch_dtype
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
    if cfg.is_hero:
        router_logits = F.linear(
            x_flat, module.router.weight.to(x_flat.dtype)
        ).float()
    else:
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
        self.q: torch.Tensor | None = None
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        self.q = q
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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("schema", "dtype", "quantization", "checkpoint_fp8", "priority", "single_round"),
    [
        (1, "bfloat16", None, False, None, True),
        (1, "bfloat16", None, False, ["native"], False),
        (2, "bfloat16", None, False, None, False),
        (2, "bfloat16", None, False, ["vllm_c"], True),
        (1, "bfloat16", "fp8", False, None, False),
        (1, "bfloat16", "fp8", False, ["vllm_c"], True),
        (1, "bfloat16", None, True, None, False),
        (1, "float16", None, False, None, False),
    ],
)
def test_grug_moe_norm_selection_preserves_rounding(
    tmp_path, schema, dtype, quantization, checkpoint_fp8, priority, single_round
):
    GrugMoeConfig(
        architectures=["GrugMoeForCausalLM"],
        vocab_size=32,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        sliding_window=128,
        grugmoe_artifact_schema_version=schema,
        quantization_config=(
            {"quant_method": "fp8", "activation_scheme": "dynamic"}
            if checkpoint_fp8
            else None
        ),
    ).save_pretrained(tmp_path)
    config = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype=dtype,
        quantization=quantization,
        compilation_config={"mode": 3},
        kernel_config=KernelConfig(
            ir_op_priority=IrOpPriorityConfig(rms_norm=priority or [])
        ),
    ).create_engine_config()
    dtype = config.model_config.dtype
    inputs = torch.tensor(
        [[-0.75, 0.25, 0.5, 1.0], [0.125, -0.375, 0.875, -1.25]],
        dtype=dtype,
        device="cuda",
    )
    eps = 1e-5
    with (
        set_current_vllm_config(config),
        config.kernel_config.ir_op_priority.set_priority(),
    ):
        norm = RMSNorm(4, eps=eps, dtype=dtype).cuda()
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.75, 1.25, 0.5, 1.5]))
            actual = norm(inputs)
        normalized = inputs.double() * torch.rsqrt(
            inputs.double().square().mean(dim=-1, keepdim=True) + eps
        )
        if not single_round:
            normalized = normalized.to(dtype).double()
        expected = (normalized * norm.weight.double()).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
    assert runtime_config == replace(
        _tiny_hero_config(),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )
    assert vllm_config.model_config.is_hybrid is True
    assert vllm_config.model_config.has_inner_state is True
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    ("requested", "split_shortconv", "expected"),
    [
        ("NONE", False, CUDAGraphMode.NONE),
        ("PIECEWISE", False, CUDAGraphMode.NONE),
        ("PIECEWISE", True, CUDAGraphMode.PIECEWISE),
        ("FULL", True, CUDAGraphMode.NONE),
        ("FULL_AND_PIECEWISE", False, CUDAGraphMode.NONE),
        ("FULL_AND_PIECEWISE", True, CUDAGraphMode.FULL_AND_PIECEWISE),
    ],
)
def test_hero_shortconv_graph_modes(tmp_path, requested, split_shortconv, expected):
    GrugMoeConfig(
        grugmoe_artifact_schema_version=2,
        architectures=["GrugMoeForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=5,
        num_experts=4,
        num_experts_per_tok=2,
        latent_dim=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        local_kv_heads=1,
        global_kv_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=128,
        global_every=1,
        sconv=True,
    ).save_pretrained(tmp_path)
    config = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        compilation_config={
            "mode": 3,
            "cudagraph_mode": requested,
            "splitting_ops": ["vllm::grug_moe_short_conv"] if split_shortconv else [],
        },
    ).create_engine_config()
    assert config.compilation_config.cudagraph_mode == expected
    # Config normalization and the model's load-time guard must agree.
    _raise_for_unsupported_modes(config)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(NotImplementedError):
        _raise_for_unsupported_modes(config)


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
    is_checkpoint_fp8_serialized=False,
):
    quant_config = (
        None
        if quantization_method is None
        else (
            OnlineQuantizationConfig(
                QuantizationConfigArgs(moe=quantization_method.removeprefix("online:"))
            )
            if quantization_method.startswith("online:")
            else (
                Fp8Config(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized)
                if quantization_method == "fp8"
                else SimpleNamespace(get_name=lambda: quantization_method)
            )
        )
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tensor_parallel_size),
        model_config=None,
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
    ("is_tpu", "quantization_method", "serialized", "supported"),
    [
        (True, "unquantized", False, True),
        (True, "fp8", False, False),
        (False, "unquantized", False, False),
        (False, "fp8", False, True),
        (False, "fp8", True, False),
        (False, "online:fp8_per_tensor", False, True),
        (False, "online:fp8_per_block", False, False),
        (False, "online:fp8_per_channel", False, False),
        (True, "online:fp8_per_tensor", False, False),
    ],
)
def test_grug_moe_quantization_platform_support(
    monkeypatch,
    is_tpu,
    quantization_method,
    serialized,
    supported,
):
    fake_config = _minimal_vllm_config_for_mode_check(
        tensor_parallel_size=8 if is_tpu else 1,
        quantization_method=quantization_method,
        is_checkpoint_fp8_serialized=serialized,
    )
    _patch_grug_parallel_runtime(monkeypatch, fake_config, is_tpu=is_tpu)

    if supported:
        _raise_for_unsupported_modes(fake_config)
    else:
        with pytest.raises(NotImplementedError, match="supports only online FP8"):
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
    gate_hidden = F.silu(F.linear(x.float(), down_weight.float()))
    gate = torch.sigmoid(F.linear(gate_hidden, up_weight.float()))
    expected = x * gate

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


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


def test_grug_moe_router_padding_ids_ignore_stale_logits(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    router = GrugMoeRouter(
        top_k=2,
        global_num_experts=4,
        bias=torch.tensor([0.4, 0.8, 0.2, -0.1]),
    )
    first = torch.tensor([[3.0, -1.0, 2.0, 0.0], [8.0, 7.0, -8.0, -8.0]])
    second = torch.tensor([[3.0, -1.0, 2.0, 0.0], [-8.0, -8.0, 7.0, 8.0]])
    originals = first.clone(), second.clone()
    with set_forward_context(
        None,
        get_current_vllm_config(),
        num_tokens=2,
        is_padding=torch.tensor([False, True]),
    ):
        first_weights, first_ids = router.select_experts(
            hidden_states=torch.ones(2, 4), router_logits=first
        )
        second_weights, second_ids = router.select_experts(
            hidden_states=torch.ones(2, 4), router_logits=second
        )

    torch.testing.assert_close(first_ids[0], torch.tensor([0, 2], dtype=torch.int32))
    assert ((first_ids[1] >= 0) & (first_ids[1] < 4)).all()
    torch.testing.assert_close(second_ids, first_ids, rtol=0, atol=0)
    expected = torch.sigmoid(torch.tensor([3.0, 2.0]))
    expected *= _ROUTER_COMBINE_WEIGHT_SUM / (
        expected.sum() + _ROUTER_COMBINE_WEIGHT_EPS
    )
    torch.testing.assert_close(first_weights[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(second_weights, first_weights, rtol=0, atol=0)
    torch.testing.assert_close(first_weights[1], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(first, originals[0], rtol=0, atol=0)
    torch.testing.assert_close(second, originals[1], rtol=0, atol=0)


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


@pytest.mark.parametrize("use_yarn", [False, True])
def test_grug_moe_attention_schedule_matches_training_architecture(use_yarn):
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

    if use_yarn:
        cfg = replace(
            cfg,
            max_seq_len=65536,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": cfg.rope_theta,
                "partial_rotary_factor": 0.5,
                "factor": 4.0,
                "original_max_position_embeddings": 16384,
                "attention_factor": 1.0 + 0.1 * math.log(4.0),
            },
        )

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
    for profiling_metadata in (None, {}):
        # V1 omits all metadata; V2's profiling pass omits Mamba groups from
        # its dictionary. Neither pass may consume or mutate request history.
        with set_forward_context(
            profiling_metadata,
            get_current_vllm_config(),
            num_tokens=prefill.shape[0],
        ):
            profiling_output = module(prefill)
        torch.testing.assert_close(
            profiling_output, prefill * weight[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            module.kv_cache[0], torch.zeros_like(module.kv_cache[0]), rtol=0, atol=0
        )
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

    decode = torch.tensor([[4.0, -4.0], [30.0, 12.0], [5.0, -5.0]], device="cuda")
    decode_metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=3,
        num_decode_tokens=3,
        num_reqs=3,
        query_start_loc_p=None,
        has_initial_states_p=None,
        state_indices_tensor_p=torch.empty(0, device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor(
            [1, 2, 0], device="cuda", dtype=torch.int32
        ),
        num_accepted_tokens=None,
        query_start_loc_d=torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32),
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([4, 3, 1], device="cuda"),
    )
    original_decode = decode.clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill_uninitialized = torch.utils.deterministic.fill_uninitialized_memory
    try:
        # Make an unwritten null row fail reliably instead of reading allocator data.
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        with set_forward_context(
            {prefix: decode_metadata},
            get_current_vllm_config(),
            num_tokens=decode.shape[0],
        ):
            decode_output = module(decode)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill_uninitialized

    expected_decode = torch.vstack(
        (
            _causal_depthwise_reference(torch.vstack((request_a, decode[:1])), weight)[
                -1:
            ],
            _causal_depthwise_reference(torch.vstack((request_b, decode[1:2])), weight)[
                -1:
            ],
            decode[2:],
        )
    )
    torch.testing.assert_close(decode_output, expected_decode)
    torch.testing.assert_close(decode, original_decode, rtol=0, atol=0)
    torch.testing.assert_close(
        module.kv_cache[0][1], torch.vstack((request_a[1:], decode[:1])), rtol=0, atol=0
    )
    torch.testing.assert_close(
        module.kv_cache[0][2], torch.vstack((request_b, decode[1:2])), rtol=0, atol=0
    )
    torch.testing.assert_close(
        module.kv_cache[0][0], torch.zeros_like(module.kv_cache[0][0]), rtol=0, atol=0
    )

    # Decode A while B resumes prefill from its existing state. The output
    # keeps decode rows first even though the prefill kernel executes first.
    next_a = torch.tensor([[5.0, -5.0]], device="cuda")
    next_b = torch.tensor([[40.0, 16.0], [50.0, 20.0]], device="cuda")
    mixed = torch.vstack((next_a, next_b))
    mixed_metadata = replace(
        prefill_metadata,
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=1,
        num_decode_tokens=1,
        query_start_loc_p=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        has_initial_states_p=torch.tensor([True], device="cuda"),
        state_indices_tensor_p=torch.tensor([2], device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor([1], device="cuda", dtype=torch.int32),
        query_start_loc_d=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([5, 5], device="cuda"),
    )
    with set_forward_context(
        {prefix: mixed_metadata},
        get_current_vllm_config(),
        num_tokens=mixed.shape[0],
    ):
        mixed_output = module(mixed)

    expected_mixed = torch.vstack(
        (
            _causal_depthwise_reference(
                torch.vstack((request_a, decode[:1], next_a)), weight
            )[-1:],
            _causal_depthwise_reference(
                torch.vstack((request_b, decode[1:2], next_b)), weight
            )[-2:],
        )
    )
    torch.testing.assert_close(mixed_output, expected_mixed)
    torch.testing.assert_close(mixed, torch.vstack((next_a, next_b)), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hero_grug_moe_sconv_padded_decode_zeros_inactive_rows(dtype):
    prefix = "hero.sconv.padding"
    module = GrugMoeShortConv(2, 4, dtype, None, prefix).cuda()
    module.weight.data.fill_(1)
    module.weight.data[0].fill_(1.0625)
    module.kv_cache = (torch.zeros(2, 3, 2, device="cuda", dtype=torch.bfloat16),)
    metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_reqs=1,
        query_start_loc_p=None,
        has_initial_states_p=None,
        state_indices_tensor_p=torch.empty(0, device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor([1], device="cuda", dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([1], device="cuda"),
    )
    # Both operands fit BF16 exactly; their product needs FP32 output precision.
    for value, expected in ((1.0625, 1.12890625), (1.125, 2.2578125)):
        hidden = torch.full((4, 2), float("nan"), device="cuda", dtype=dtype)
        hidden[0].fill_(value)
        output = torch.full_like(hidden, float("nan"))
        with set_forward_context(
            {prefix: metadata}, get_current_vllm_config(), num_tokens=4
        ):
            torch.ops.vllm.grug_moe_short_conv(hidden, output, prefix)
        torch.testing.assert_close(
            output[0], torch.full_like(output[0], expected), rtol=0, atol=0
        )
        torch.testing.assert_close(
            output[1:], torch.zeros_like(output[1:]), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hero_grug_moe_sconv_full_graph_replay_matches_eager():
    """Replay new inputs and slots like eager, preserving inactive and null state."""
    prefix = "hero.sconv.fullgraph"
    padded = 4
    module = GrugMoeShortConv(2, 4, torch.float32, None, prefix).cuda()
    module.weight.data.copy_(
        torch.tensor([[1.0, 0.75], [0.5, -0.25], [-0.25, 0.125], [0.125, 0.5]])
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    initial_state = torch.randn(5, 3, 2, device="cuda", generator=generator)
    initial_state[0].zero_()  # slot 0 is vLLM's null block
    state_indices = torch.zeros(padded, device="cuda", dtype=torch.int32)
    hidden = torch.zeros(padded, 2, device="cuda")
    output = torch.zeros_like(hidden)

    def metadata(indices: torch.Tensor) -> ShortConvAttentionMetadata:
        return ShortConvAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=padded,
            num_decode_tokens=padded,
            num_reqs=padded,
            query_start_loc_p=None,
            has_initial_states_p=None,
            state_indices_tensor_p=torch.empty(0, device="cuda", dtype=torch.int32),
            state_indices_tensor_d=indices,
            num_accepted_tokens=None,
            query_start_loc_d=torch.arange(
                padded + 1, device="cuda", dtype=torch.int32
            ),
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
            num_computed_tokens_p=None,
            seq_lens=torch.full((padded,), 4, device="cuda"),
        )

    def step() -> None:
        with set_forward_context(
            {prefix: metadata(state_indices)},
            get_current_vllm_config(),
            num_tokens=padded,
        ):
            torch.ops.vllm.grug_moe_short_conv(hidden, output, prefix)

    module.kv_cache = (initial_state.clone(),)
    # Capture on vLLM's stream, as the runner does. A separate capture stream
    # leaves vLLM's tracked stream at the default stream for later tests.
    stream = current_stream()
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()

    for real_slots in ([1, 3], [4, 2, 1]):
        new_hidden = torch.randn(padded, 2, device="cuda", generator=generator)
        slots = real_slots + [0] * (padded - len(real_slots))

        module.kv_cache[0].copy_(initial_state)
        hidden.copy_(new_hidden)
        state_indices.copy_(torch.tensor(slots, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        graph_output = output.clone()
        graph_state = module.kv_cache[0].clone()

        module.kv_cache[0].copy_(initial_state)
        eager_indices = torch.tensor(slots, device="cuda", dtype=torch.int32)
        eager_output = torch.zeros_like(output)
        with set_forward_context(
            {prefix: metadata(eager_indices)},
            get_current_vllm_config(),
            num_tokens=padded,
        ):
            torch.ops.vllm.grug_moe_short_conv(new_hidden, eager_output, prefix)
        torch.cuda.synchronize()

        real = len(real_slots)
        torch.testing.assert_close(
            graph_output[:real], eager_output[:real], rtol=0, atol=0
        )
        torch.testing.assert_close(graph_state, module.kv_cache[0], rtol=0, atol=0)
        untouched = [slot for slot in range(5) if slot not in real_slots]
        torch.testing.assert_close(
            graph_state[untouched], initial_state[untouched], rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hero_profile", [False, True])
def test_grug_moe_cuda_accepts_bf16_expert_weights(hero_profile):
    cfg = GrugMoeRuntimeConfig(
        artifact_schema_version=2 if hero_profile else 1,
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
        if hero_profile:
            mlp.router.weight.copy_(mlp.router.weight.bfloat16().float())
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
        # FlashInfer packs weights in place; the oracle needs the logical layout.
        expected = _reference_moe(x, mlp)
        routed_experts.quant_method.process_weights_after_loading(routed_experts)

    # The fused-MoE kernels allocate through the workspace manager, which a real run
    # installs in GPUModelRunner.__init__. This test drives the layer directly, so it
    # has to do the same setup itself.
    init_workspace_manager(torch.device("cuda"))
    with set_forward_context(None, get_current_vllm_config(), num_tokens=x.shape[0]):
        actual = mlp(x)

    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("selective", [False, True])
def test_grug_moe_online_fp8_loading_matches_vllm(monkeypatch, selective):
    cfg = replace(
        _tiny_config(),
        hidden_dim=64,
        intermediate_dim=64,
        num_heads=4,
        head_dim=16,
    ).validate()
    monkeypatch.setattr(
        get_current_vllm_config(),
        "model_config",
        SimpleNamespace(dtype=torch.bfloat16),
    )
    # Online weight loading captures this device before it quantizes the experts.
    with torch.device("cuda"):
        quantized = GrugMoeMLP(
            cfg,
            params_dtype=torch.bfloat16,
            quant_config=(
                OnlineQuantizationConfig(QuantizationConfigArgs(moe="fp8_per_tensor"))
                if selective
                else Fp8Config()
            ),
            prefix="model.layers.0.mlp",
        )
        # Direct vLLM weights bypass Grug's checkpoint mapping.
        reference = FusedMoEFactory(
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_token,
            hidden_size=cfg.hidden_dim,
            intermediate_size=cfg.intermediate_dim,
            params_dtype=torch.bfloat16,
            renormalize=False,
            quant_config=Fp8Config(),
            prefix="reference.experts",
            router=GrugMoeRouter(
                top_k=cfg.num_experts_per_token,
                global_num_experts=cfg.num_experts,
                bias=quantized.router.bias,
            ),
            router_logits_dtype=torch.float32,
        )
    with torch.no_grad():
        _fill_parameter(quantized.router.weight, -0.1, 0.1)
        _fill_parameter(quantized.router.bias, 0.2, -0.2)
        experts = reference.routed_experts
        materialize_layer(experts, get_layerwise_info(experts))
        _fill_parameter(experts.w13_weight[:, : cfg.intermediate_dim], -0.15, 0.15)
        _fill_parameter(experts.w13_weight[:, cfg.intermediate_dim :], 0.12, -0.12)
        _fill_parameter(experts.w2_weight, -0.1, 0.1)
        params_dict = dict(quantized.named_parameters())
        for projection, weight in (
            ("gate_proj", experts.w13_weight[:, : cfg.intermediate_dim]),
            ("up_proj", experts.w13_weight[:, cfg.intermediate_dim :]),
            ("down_proj", experts.w2_weight),
        ):
            _try_load_grug_expert_weight(
                f"experts.{projection}.weight", weight, params_dict
            )
        experts.quant_method.process_weights_after_loading(experts)

    loaded = quantized.experts.routed_experts
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        actual_weight, expected_weight = getattr(loaded, name), getattr(experts, name)
        assert actual_weight.dtype == expected_weight.dtype
        torch.testing.assert_close(
            actual_weight.float(), expected_weight.float(), rtol=0, atol=0
        )

    x = torch.linspace(-0.6, 0.7, 3 * cfg.hidden_dim, device="cuda").reshape(
        3, cfg.hidden_dim
    ).to(torch.bfloat16)
    init_workspace_manager(torch.device("cuda"))
    with set_forward_context(None, get_current_vllm_config(), num_tokens=x.shape[0]):
        expected = reference(x, F.linear(x.float(), quantized.router.weight))
        actual = quantized(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    padding_rows = 5
    padded = F.pad(x, (0, 0, 0, padding_rows))
    is_padding = torch.arange(padded.shape[0], device=x.device) >= x.shape[0]
    with set_forward_context(
        None,
        get_current_vllm_config(),
        num_tokens=padded.shape[0],
        is_padding=is_padding,
    ):
        clean = quantized(padded.clone())[: x.shape[0]].clone()
        poisoned = padded.clone()
        poisoned[x.shape[0] :] = 32
        repeated = quantized(poisoned)[: x.shape[0]]

    torch.testing.assert_close(repeated, clean, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="requires CUDA with FP8 support",
)
@pytest.mark.parametrize("num_tokens", [1, 7])
@pytest.mark.parametrize("configuration", ["legacy", "online", "routed-experts-only"])
def test_hero_online_quantization_preserves_latent_and_shared_outputs(
    num_tokens, configuration, monkeypatch
):
    cfg = replace(
        _tiny_hero_config(),
        hidden_dim=128,
        latent_dim=64,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        head_dim=32,
    ).validate()
    monkeypatch.setattr(
        get_current_vllm_config(), "model_config", SimpleNamespace(dtype=torch.bfloat16)
    )
    modules = []
    # Online loaders record the construction device and materialize weights there.
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        selected_config = (
            Fp8Config()
            if configuration == "legacy"
            else OnlineQuantizationConfig(
                QuantizationConfigArgs(
                    moe="fp8_per_tensor",
                    linear="fp8_per_tensor" if configuration == "online" else None,
                )
            )
        )
        for profile, quant_config in (("reference", None), ("fp8", selected_config)):
            modules.append(
                nn.ModuleDict(
                    {
                        "mlp": GrugMoeMLP(
                            cfg, quant_config=quant_config, prefix=f"{profile}.mlp"
                        ),
                        "shared_experts": nn.ModuleList(
                            GrugMoeDenseMLP(
                                cfg.hidden_dim,
                                cfg.shared_expert_intermediate_dim,
                                torch.bfloat16,
                                quant_config=quant_config,
                                prefix=f"{profile}.shared_experts.{index}",
                            )
                            for index in range(cfg.num_shared_experts)
                        ),
                    }
                )
            )
    reference, quantized = modules
    with torch.no_grad():
        powers = torch.tensor([1, 2, 4, 8], device="cuda", dtype=torch.bfloat16)
        for index, (name, parameter) in enumerate(reference.named_parameters()):
            if name == "mlp.latent_norm.weight":
                parameter.fill_(1)
            elif name == "mlp.router.weight":
                _fill_parameter(parameter, -0.1, 0.1)
            elif name == "mlp.router.bias":
                _fill_parameter(parameter, 0.2, -0.2)
            else:
                rows = torch.arange(parameter.shape[-2], device="cuda") // 16
                columns = torch.arange(parameter.shape[-1], device="cuda") // 16
                pattern = powers[(rows[:, None] + 2 * columns + index) % 4]
                parameter.copy_(pattern / 128)
        # Distinguish gate/up halves, latent inputs and routed expert IDs.
        experts = reference["mlp"].experts.routed_experts
        experts.w13_weight[:, cfg.intermediate_dim :].mul_(2)
        experts.w13_weight[:, :, :16].mul_(2)
        for expert_id in range(cfg.num_experts):
            experts.w13_weight[expert_id].mul_(2**expert_id / 8)
            experts.w2_weight[expert_id].mul_(2**-expert_id)
        reference["mlp"].router.weight.copy_(
            reference["mlp"].router.weight.bfloat16().float()
        )

        # The 7:2 input blocks scale exactly to FP8 values 448:128. Negative rows
        # exercise the other routed experts; unequal blocks expose misloaded axes.
        inputs = torch.tensor([7 / 32, 2 / 32], device="cuda", dtype=torch.bfloat16)
        blocks = (torch.arange(cfg.hidden_dim, device="cuda") // 16) % 2
        x = inputs[blocks].repeat(num_tokens, 1)
        x[1::2].mul_(-4)
        routed_expected = _reference_moe(x, reference["mlp"])
        shared_expected = [
            _reference_dense_mlp(x, expert) for expert in reference["shared_experts"]
        ]
        expected = sum(shared_expected, routed_expected)

        params_dict = dict(quantized.named_parameters())
        for name, weight in reference.named_parameters():
            if name == "mlp.experts.routed_experts.w13_weight":
                for projection, shard in (
                    ("gate_proj", weight[:, : cfg.intermediate_dim]),
                    ("up_proj", weight[:, cfg.intermediate_dim :]),
                ):
                    _try_load_grug_expert_weight(
                        f"mlp.experts.{projection}.weight", shard, params_dict
                    )
            elif name == "mlp.experts.routed_experts.w2_weight":
                _try_load_grug_expert_weight(
                    "mlp.experts.down_proj.weight", weight, params_dict
                )
            else:
                parameter = params_dict[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, weight)

    init_workspace_manager(torch.device("cuda"))
    with set_forward_context(None, get_current_vllm_config(), num_tokens=num_tokens):
        actual = quantized["mlp"](x)
        torch.testing.assert_close(
            actual.float(), routed_expected.float(), atol=5e-2, rtol=5e-2
        )
        for expert, reference_output in zip(
            quantized["shared_experts"], shared_expected
        ):
            shared = expert(x)
            torch.testing.assert_close(
                shared.float(),
                reference_output.float(),
                atol=5e-2,
                rtol=5e-2,
            )
            actual = actual + shared

    assert quantized["mlp"].router.weight.dtype == torch.float32
    fp8_weights = [
        quantized["mlp"].experts.routed_experts.w13_weight,
        quantized["mlp"].experts.routed_experts.w2_weight,
    ]
    projection_weights = [
        quantized["mlp"].latent_down_proj.weight,
        quantized["mlp"].latent_up_proj.weight,
    ]
    for expert in quantized["shared_experts"]:
        projection_weights.extend(
            (expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight)
        )
    assert all(weight.dtype == torch.float8_e4m3fn for weight in fp8_weights)
    projection_dtype = (
        torch.bfloat16
        if configuration == "routed-experts-only"
        else torch.float8_e4m3fn
    )
    assert all(weight.dtype == projection_dtype for weight in projection_weights)
    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)


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


@pytest.mark.parametrize("split", [False, True], ids=["stacked", "split"])
def test_grug_moe_expert_weights_load_into_fused_moe_layout(split: bool):
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

    for projection, weight in (
        ("gate_proj", gate_weight),
        ("up_proj", up_weight),
        ("down_proj", down_weight),
    ):
        if split:
            shards = (
                (f"experts.{expert_id}.{projection}.weight", weight[expert_id])
                for expert_id in reversed(range(cfg.num_experts))
            )
        else:
            shards = ((f"experts.{projection}.weight", weight),)
        for name, shard in shards:
            _try_load_grug_expert_weight(name, shard, params_dict)

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


def test_grug_moe_registry_imports():
    model_cls = ModelRegistry._try_load_model_cls("GrugMoeForCausalLM")

    assert model_cls is not None
    assert model_cls.__name__ == "GrugMoeForCausalLM"


@pytest.mark.parametrize("rope_fused", [True, False])
def test_hero_yarn_config_roundtrip_and_partial_rotation_at_65k(rope_fused):
    scaling = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 16384,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "attention_factor": 1.0 + 0.1 * math.log(4.0),
    }
    config = GrugMoeConfig(
        vocab_size=32,
        hidden_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=65536,
        grugmoe_artifact_schema_version=2,
        rope_fused=rope_fused,
        rope_theta=10000.0,
        rope_scaling=scaling,
        partial_rotary_factor=0.5,
    )
    config = GrugMoeConfig.from_dict(config.to_dict())
    cfg = GrugMoeRuntimeConfig.from_hf_config(config)
    rotary = get_rope(
        cfg.inferred_head_dim,
        max_position=cfg.max_seq_len,
        rope_parameters=cfg.rope_parameters,
        is_neox_style=not cfg.rope_fused,
        dtype=torch.float32,
    )
    positions = torch.tensor([0, 16383, 32767, 65535])
    query = torch.linspace(-0.5, 0.5, 4 * 128).reshape(4, 128)
    actual, actual_key = rotary.forward_native(positions, query, query)

    # Independent complex rotation in float64. The production cache computes
    # phases in float32; at 65K its phase quantization contributes up to 0.002.
    rotary_dim = 64
    indices = torch.arange(rotary_dim // 2, dtype=torch.float64)
    low = max(
        math.floor(
            rotary_dim * math.log(16384 / (32 * 2 * math.pi)) / (2 * math.log(10000))
        ),
        0,
    )
    high = min(
        math.ceil(
            rotary_dim * math.log(16384 / (2 * math.pi)) / (2 * math.log(10000))
        ),
        rotary_dim - 1,
    )
    ramp = ((indices - low) / (high - low)).clamp(0, 1)
    inverse_frequency = torch.pow(10000.0, -2 * indices / rotary_dim) * (
        1 - ramp + ramp / 4
    )
    phase = positions.double()[:, None] * inverse_frequency[None, :]
    rotated = query[:, :rotary_dim].double()
    if rope_fused:
        pairs = torch.complex(rotated[:, 0::2], rotated[:, 1::2])
    else:
        pairs = torch.complex(rotated[:, :32], rotated[:, 32:])
    result = pairs * torch.polar(
        torch.full_like(phase, scaling["attention_factor"]), phase
    )
    expected = query.double().clone()
    if rope_fused:
        expected[:, :rotary_dim] = torch.stack(
            (result.real, result.imag), dim=-1
        ).flatten(-2)
    else:
        expected[:, :rotary_dim] = torch.cat((result.real, result.imag), dim=-1)
    torch.testing.assert_close(actual.double(), expected, atol=0.003, rtol=0)
    torch.testing.assert_close(actual_key, actual)
    torch.testing.assert_close(actual[:, 64:], query[:, 64:], atol=0, rtol=0)


def test_hero_yarn_rejects_full_dimension_amplitude_rescaling():
    # DeepSeek's all-dimension temperature convention would alter Hero's
    # unrotated half; importing it silently must fail at the config boundary.
    with pytest.raises(ValueError, match="partial Q/K amplitude"):
        GrugMoeConfig(
            max_position_embeddings=65536,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 16384,
                "mscale_all_dim": 1.0,
            },
        )


def _log_scaled_hero_config(n_floor: int, alpha: float, power: float):
    return replace(
        _tiny_hero_config(),
        artifact_schema_version=3,
        sconv=False,
        log_scaling_n_floor=n_floor,
        log_scaling_alpha=alpha,
        log_scaling_power=power,
    ).validate()


def test_log_scaling_factor_matches_the_training_model():
    # Marin's hero values: floor 65536, slope 0.1, power 2 give 1.296 at 262144.
    cfg = _log_scaled_hero_config(65536, 0.1, 2.0)
    visible_keys = torch.tensor([1, 65536, 131072, 262144])

    factor = _log_scaling_factor(visible_keys, cfg)

    expected = torch.tensor([1.0, 1.0, 1.143433966251171, 1.2964769927807063])
    torch.testing.assert_close(factor, expected, rtol=1e-6, atol=0)


def test_log_scaling_scales_each_global_query_by_its_own_position():
    cfg = _log_scaled_hero_config(4, 0.5, 2.0)
    q = torch.linspace(-1.0, 1.0, 4 * 16).view(4, 16)
    k = torch.linspace(1.0, 2.0, 4 * 8).view(4, 8)
    positions = torch.tensor([0, 3, 4, 7])

    layer_names = iter(range(100))

    def captured_q(module_cfg, is_global, rows):
        module = GrugMoeAttention(
            module_cfg,
            cache_config=None,
            params_dtype=torch.float32,
            sliding_window=None if is_global else module_cfg.sliding_window,
            use_rope=False,
            qk_mult_scale=1.0,
            is_global=is_global,
            prefix=f"log_scaling.{next(layer_names)}",
        )
        module.q_proj = _FixedProjection(q[rows])
        module.k_proj = _FixedProjection(k[rows])
        module.v_proj = _FixedProjection(k[rows])
        capture = _CaptureAttention(module.q_size)
        module.attn = capture
        module(positions[rows], torch.zeros(len(positions[rows]), cfg.hidden_dim))
        assert capture.q is not None
        return capture.q

    all_rows = slice(None)
    off = replace(
        cfg,
        log_scaling_n_floor=None,
        log_scaling_alpha=None,
        log_scaling_power=None,
    ).validate()
    # A query that sees n keys is scaled by (1 + 0.5 ln(max(n / 4, 1)))^2.
    factor = torch.tensor(
        [1.0, 1.0, (1 + 0.5 * math.log(5 / 4)) ** 2, (1 + 0.5 * math.log(2)) ** 2]
    )
    torch.testing.assert_close(
        captured_q(cfg, True, all_rows),
        captured_q(off, True, all_rows) * factor[:, None],
    )
    torch.testing.assert_close(
        captured_q(cfg, False, all_rows), captured_q(off, False, all_rows)
    )
    # The factor depends only on a token's own position, so a decode step or a
    # later prefill chunk gets the same query as the whole prompt at once.
    torch.testing.assert_close(
        captured_q(cfg, True, slice(3, 4)), captured_q(cfg, True, all_rows)[3:]
    )


def _write_hero_config(path, **fields) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GrugMoeForCausalLM"],
                "model_type": "grug_moe",
                "vocab_size": 64,
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
                "head_dim": 32,
                "max_position_embeddings": 128,
                "sliding_window": 8,
                "global_every": 3,
                "rope_fused": True,
                "sconv": False,
                "grugmoe_artifact_schema_version": 2,
                **fields,
            }
        )
    )


_LOG_SCALING_FIELDS = {
    "grugmoe_artifact_schema_version": 3,
    "log_scaling_n_floor": 8,
    "log_scaling_alpha": 0.5,
    "log_scaling_power": 2.0,
}


@pytest.mark.parametrize(
    "log_scaling",
    [
        dict(log_scaling_n_floor=8, log_scaling_alpha=0.5, log_scaling_power=2.0),
        # Schema 3 always carries the fields; null turns log scaling off.
        dict(log_scaling_n_floor=None, log_scaling_alpha=None, log_scaling_power=None),
    ],
)
def test_schema_3_config_loads_log_scaling(tmp_path, log_scaling):
    _write_hero_config(
        tmp_path / "model", grugmoe_artifact_schema_version=3, **log_scaling
    )

    cfg = GrugMoeRuntimeConfig.from_hf_config(
        get_config(tmp_path / "model", trust_remote_code=False)
    )

    assert cfg.is_hero
    assert {name: getattr(cfg, name) for name in log_scaling} == log_scaling


@pytest.mark.parametrize(
    "fields",
    [
        # Log scaling without schema 3 would load on a wheel that ignores it.
        {**_LOG_SCALING_FIELDS, "grugmoe_artifact_schema_version": 2},
        {**_LOG_SCALING_FIELDS, "log_scaling_power": None},
    ],
)
def test_log_scaling_requires_schema_3_and_all_three_fields(tmp_path, fields):
    _write_hero_config(tmp_path / "model", **fields)
    hf_config = get_config(tmp_path / "model", trust_remote_code=False)

    with pytest.raises(ValueError, match="log scaling|log_scaling"):
        GrugMoeRuntimeConfig.from_hf_config(hf_config)


def _scale_dummy_weights(model: nn.Module) -> None:
    # Dummy weights lie in [-1e-3, 1e-3]; widen them so attention, and with it
    # the query scale, moves the logits well above float32 noise.
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(300.0)


def _target_logprobs(output) -> list[float]:
    return [
        next(iter(entry.values())).logprob
        for entry in output.prompt_logprobs[1:]
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_log_scaling_is_identical_across_prefill_chunking_prefix_cache_and_decode(
    tmp_path, monkeypatch
):
    # apply_model sends a function to the engine-core process.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # flashinfer's sampler JIT-compiles with nvcc, which the CI image lacks.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _write_hero_config(tmp_path / "on", **_LOG_SCALING_FIELDS)
    _write_hero_config(
        tmp_path / "off",
        grugmoe_artifact_schema_version=3,
        log_scaling_n_floor=None,
        log_scaling_alpha=None,
        log_scaling_power=None,
    )
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 64, (40,), generator=generator).tolist()
    greedy = SamplingParams(temperature=0.0, max_tokens=8, logprobs=0, detokenize=False)
    scored = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False
    )

    def engine(model_dir, max_num_batched_tokens):
        llm = LLM(
            model=str(model_dir),
            skip_tokenizer_init=True,
            load_format="dummy",
            dtype="float32",
            # FlexAttention, the default for float32, recompiles per layer mask.
            attention_backend="TRITON_ATTN",
            # FlashInfer's SM100 MoE kernel does not support float32.
            moe_backend="triton",
            max_model_len=128,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.2,
        )
        llm.apply_model(_scale_dummy_weights)
        return llm

    llm = engine(tmp_path / "on", 128)
    decoded = llm.generate({"prompt_token_ids": prompt}, greedy)[0].outputs[0]
    sequence = prompt + list(decoded.token_ids[:-1])
    decode_logprobs = [
        entry[token].logprob
        for token, entry in zip(decoded.token_ids, decoded.logprobs)
    ]
    whole = _target_logprobs(
        llm.generate({"prompt_token_ids": sequence}, scored)[0]
    )
    # Its prompt starts with full blocks cached by the requests above.
    cached = llm.generate(
        {"prompt_token_ids": prompt + [decoded.token_ids[0]]}, greedy
    )[0]
    del llm
    cleanup_dist_env_and_memory()

    llm = engine(tmp_path / "on", 16)
    chunked = _target_logprobs(llm.generate({"prompt_token_ids": sequence}, scored)[0])
    del llm
    cleanup_dist_env_and_memory()

    llm = engine(tmp_path / "off", 128)
    unscaled = _target_logprobs(llm.generate({"prompt_token_ids": sequence}, scored)[0])
    del llm
    cleanup_dist_env_and_memory()

    tolerance = {"rtol": 0, "atol": 1e-4}
    # whole[j - 1] scores token j from the query at position j - 1.
    torch.testing.assert_close(
        torch.tensor(whole[len(prompt) - 1 :]),
        torch.tensor(decode_logprobs[:-1]),
        **tolerance,
    )
    torch.testing.assert_close(torch.tensor(chunked), torch.tensor(whole), **tolerance)
    assert cached.num_cached_tokens > 0
    assert list(cached.outputs[0].token_ids[:7]) == list(decoded.token_ids[1:])
    torch.testing.assert_close(
        torch.tensor(
            [
                entry[token].logprob
                for token, entry in zip(
                    cached.outputs[0].token_ids[:7], cached.outputs[0].logprobs
                )
            ]
        ),
        torch.tensor(decode_logprobs[1:]),
        **tolerance,
    )
    # Queries at positions below the floor of 8 keys are unscaled; later ones
    # are not, so the comparison above is sensitive to the factor.
    torch.testing.assert_close(
        torch.tensor(whole[:8]), torch.tensor(unscaled[:8]), **tolerance
    )
    assert (torch.tensor(whole[9:]) - torch.tensor(unscaled[9:])).abs().max() > 1e-2
