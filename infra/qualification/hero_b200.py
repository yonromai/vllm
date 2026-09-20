# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental full-Hero B200 qualification against retained Levanter goldens."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import re
import socket
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import boto3
import numpy as np
from botocore.config import Config

CHECKPOINT = (
    "s3://marin-us-east-02a/marin/grug/hero-ragged_a2a-nccl2307-ep-step81k/"
    "2026.08.19.2/checkpoints/step-108000"
)
BASELINE_GOLDEN_ROOT = (
    "s3://marin-us-east-02a/marin/reference/hero-forward/"
    "hero-535b-step108000-bf16-v1-dcfe4ced165a"
)
GOLDEN_ROOT = os.environ.get("HERO_GOLDEN_ROOT", BASELINE_GOLDEN_ROOT)
WEIGHT_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/hero-vllm-b200/"
    "hero-535b-step108000-bf16-split-v3"
)
RESULT_ROOT = os.environ.get(
    "HERO_RESULT_ROOT",
    "s3://marin-us-east-02a/marin/users/romain/hero-vllm-b200/"
    "qualification-9d1ccba766-v39-armed-all-mode-logits",
)
VLLM_REVISION = "9d1ccba766fc7cf7cda4a54ac826203052ccabd8"
WORLD_SIZE = 8
LOCAL_WORLD_SIZE = 4
MASTER_PORT = 43860
SIOCGIFADDR = 0x8915
TOP_LOGPROBS = 64
GATED_NORM_RANK = 128
QUALIFICATION_GPUS_PER_TASK = 4
QUALIFICATION_TASKS = WORLD_SIZE // QUALIFICATION_GPUS_PER_TASK
LAYER_PROBE_RANKS = (3, 4, 6, 7)
DECODE_LAYER_PROBE_RANK = 3
DECODE_LAYER_PROBE_POSITION = 2045
LAYER_PROBE_POSITIONS = (
    0, 1, 2, 3, 4, 5, 6, 7, 2044, 2045, 2046, 2047, 2048, 2049, 4094, 4095
)
PRECOMPILED_WHEEL = (
    "https://github.com/marin-community/vllm/releases/download/"
    "marin-vllm-gpu-candidate-70ea9ae8f260/"
    "vllm-0.0.0.dev20260916%2Bmarin.70ea9ae8f260.cu132-"
    "cp38-abi3-manylinux_2_28_aarch64.whl"
)
QUALIFICATION_SETUP = f"""\
set -e
cd "$IRIS_WORKDIR"
uv venv --python 3.12 "$IRIS_VENV"
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0+marin.{VLLM_REVISION[:12]} \\
VLLM_USE_PRECOMPILED=1 \\
VLLM_PRECOMPILED_WHEEL_LOCATION='{PRECOMPILED_WHEEL}' \\
uv pip install \\
  --python "$IRIS_VENV/bin/python" \\
  --constraint infra/release/gpu-constraints.txt \\
  --index-strategy unsafe-best-match \\
  --editable '.[runai]'
"""

# Fixed before the acceptance run. The comparison report applies these bounds.
TARGET_LOGPROB_TOLERANCE = 0.10
SAVED_TOP_PROBABILITY_TOLERANCE = 0.02
PREFILL_DECODE_TARGET_TOLERANCE = 0.05
PREFILL_DECODE_TOP_PROBABILITY_TOLERANCE = 0.02


def _s3_parts(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"s3://([^/]+)/(.*)", uri)
    if match is None:
        raise ValueError(f"Not an S3 URI: {uri}")
    return match.group(1), match.group(2).rstrip("/")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        config=Config(s3={"addressing_style": "virtual"}),
    )


def _download(client, uri: str, target: Path) -> None:
    bucket, key = _s3_parts(uri)
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))


def _put_json(client, uri: str, value: dict[str, Any]) -> None:
    bucket, key = _s3_parts(uri)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
        IfNoneMatch="*",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _task_index() -> int:
    task_id = os.environ["IRIS_TASK_ID"]
    match = re.search(r"/(\d+):\d+$", task_id)
    if match is None:
        raise ValueError(f"Cannot parse Iris task index from {task_id!r}")
    return int(match.group(1))


def _configure_node_network() -> None:
    host = os.environ["IRIS_ADVERTISE_HOST"]
    packed_host = socket.inet_aton(host)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, interface in socket.if_nameindex():
            request = struct.pack("256s", interface.encode()[:15])
            try:
                address = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, request)[20:24]
            except OSError:
                continue
            if address == packed_host:
                os.environ["VLLM_HOST_IP"] = host
                os.environ["GLOO_SOCKET_IFNAME"] = interface
                print(
                    f"vLLM node network: host={host} gloo_interface={interface}",
                    flush=True,
                )
                return
    raise RuntimeError(f"No local network interface owns advertised IP {host}")


def _configure_object_storage() -> None:
    # RunAI's S3 listing builds its own boto3 client. Botocore chooses path
    # addressing for this custom endpoint unless the shared profile says virtual.
    os.environ["AWS_CONFIG_FILE"] = str(Path(__file__).with_name("aws-config"))


def _job_key() -> str:
    return os.environ["IRIS_TASK_ID"].rsplit("/", 1)[0].strip("/").replace("/", "-")


def _master_address(client, task_index: int) -> str:
    rendezvous_uri = f"{RESULT_ROOT}/rendezvous-{_job_key()}.json"
    if task_index == 0:
        host = os.environ.get("IRIS_ADVERTISE_HOST") or socket.gethostbyname(
            socket.gethostname()
        )
        _put_json(client, rendezvous_uri, {"host": host, "port": MASTER_PORT})
        return host

    bucket, key = _s3_parts(rendezvous_uri)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            return json.loads(body)["host"]
        except client.exceptions.NoSuchKey:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {rendezvous_uri}")


def _rank_indices(arrays: dict[str, np.ndarray], rank: int) -> np.ndarray:
    return np.flatnonzero(arrays["score_case_indices"] == rank)


def _wait_for_all_ranks(client, global_rank: int) -> None:
    """Keep each EP engine alive until every distinct-length case has finished."""
    ready_uri = f"{RESULT_ROOT}/ready-rank-{global_rank}.json"
    _put_json(client, ready_uri, {"rank": global_rank})
    bucket, prefix = _s3_parts(f"{RESULT_ROOT}/ready-rank-")
    expected = {f"{prefix}{rank}.json" for rank in range(WORLD_SIZE)}
    deadline = time.monotonic() + 7 * 3600
    while time.monotonic() < deadline:
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        present = {item["Key"] for item in response.get("Contents", [])}
        if expected <= present:
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for all ranks at {RESULT_ROOT}")


def _top_by_rank(logprobs: dict[int, Any], count: int = 5) -> list[int]:
    ranked = sorted(
        (
            (entry.rank, int(token_id))
            for token_id, entry in logprobs.items()
            if entry.rank is not None and 1 <= entry.rank <= count
        ),
    )
    if [rank for rank, _ in ranked] != list(range(1, count + 1)):
        raise ValueError(f"Expected ranks 1..{count}, got {ranked}")
    return [token_id for _, token_id in ranked]


def _measure_mode(
    *,
    mode: str,
    request_logprobs: list[dict[int, Any] | None],
    array_index_for_prediction,
    arrays: dict[str, np.ndarray],
    rank: int,
    prediction_indices: np.ndarray,
) -> dict[str, Any]:
    positions: list[dict[str, Any]] = []
    for score_index in prediction_indices:
        prediction_position = int(arrays["prediction_positions"][score_index])
        output_index = array_index_for_prediction(prediction_position)
        observed = request_logprobs[output_index]
        if observed is None:
            raise ValueError(
                f"{mode} case {rank} position {prediction_position} has no logprobs"
            )
        target_id = int(arrays["target_token_ids"][score_index])
        target_entry = observed.get(target_id)
        if target_entry is None:
            raise ValueError(
                f"{mode} case {rank} target {target_id} missing at "
                f"{prediction_position}"
            )
        target_golden = float(arrays["target_logprobs"][score_index])
        target_observed = float(target_entry.logprob)
        golden_top_ids = [int(value) for value in arrays["top_token_ids"][score_index]]
        golden_top_logprobs = [
            float(value) for value in arrays["top_logprobs"][score_index]
        ]
        observed_saved_logprobs = [
            (
                None
                if observed.get(token_id) is None
                else float(observed[token_id].logprob)
            )
            for token_id in golden_top_ids
        ]
        saved_probability_errors = [
            math.inf
            if observed_logprob is None
            else abs(math.exp(observed_logprob) - math.exp(golden_logprob))
            for observed_logprob, golden_logprob in zip(
                observed_saved_logprobs, golden_top_logprobs, strict=True
            )
        ]
        observed_top_ids = _top_by_rank(observed)
        positions.append(
            {
                "prediction_position": prediction_position,
                "target_token_id": target_id,
                "target_logprob_golden": target_golden,
                "target_logprob_observed": target_observed,
                "target_logprob_abs_error": abs(target_observed - target_golden),
                "golden_top_token_ids": golden_top_ids,
                "observed_top_token_ids": observed_top_ids,
                "saved_top_logprobs_golden": golden_top_logprobs,
                "saved_top_logprobs_observed": observed_saved_logprobs,
                "saved_top_probability_abs_errors": saved_probability_errors,
                "top1_matches": observed_top_ids[0] == golden_top_ids[0],
                "ordered_top5_matches": observed_top_ids == golden_top_ids,
            }
        )

    return _summarize_mode(mode, rank, positions)


def _summarize_mode(
    mode: str, rank: int, positions: list[dict[str, Any]]
) -> dict[str, Any]:
    if not positions:
        raise ValueError(f"{mode} case {rank} has no saved positions")
    worst_target = max(positions, key=lambda row: row["target_logprob_abs_error"])
    worst_top = max(
        positions,
        key=lambda row: max(row["saved_top_probability_abs_errors"]),
    )
    return {
        "mode": mode,
        "case_index": rank,
        "positions": positions,
        "max_target_logprob_abs_error": worst_target["target_logprob_abs_error"],
        "worst_target_prediction_position": worst_target["prediction_position"],
        "max_saved_top_probability_abs_error": max(
            worst_top["saved_top_probability_abs_errors"]
        ),
        "worst_saved_top_prediction_position": worst_top["prediction_position"],
        "top1_mismatch_count": sum(not row["top1_matches"] for row in positions),
        "ordered_top5_mismatch_count": sum(
            not row["ordered_top5_matches"] for row in positions
        ),
        "missing_saved_top_count": sum(
            value is None
            for row in positions
            for value in row["saved_top_logprobs_observed"]
        ),
    }


def _run_rank(
    local_rank: int,
    global_rank: int,
    master_addr: str,
    config_dir: str,
    golden_path: str,
    output_path: str,
    input_evidence: dict[str, Any],
) -> None:
    # Iris tears down failed pods promptly; keep engine subprocess output in
    # the same S3 run so the original startup error survives the teardown.
    log_path = Path(output_path).with_suffix(".log")
    with log_path.open("w", buffering=1) as log_file:
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())
    os.environ.update(
        {
            "VLLM_DP_RANK": str(global_rank),
            "VLLM_DP_RANK_LOCAL": str(local_rank),
            "VLLM_DP_SIZE": str(WORLD_SIZE),
            "VLLM_DP_MASTER_IP": master_addr,
            "VLLM_DP_MASTER_PORT": str(MASTER_PORT),
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    layer0_audit_arm = None
    if global_rank in (6, 7):
        layer0_audit_path = Path(output_path).with_suffix(".layer0-moe.npz")
        layer0_boundary_path = Path(output_path).with_suffix(".layer0-boundary.npz")
        layer2_audit_path = Path(output_path).with_suffix(".layer2-moe.npz")
        layer2_boundary_path = Path(output_path).with_suffix(".layer2-boundary.npz")
        layer9_audit_path = Path(output_path).with_suffix(".layer9-moe.npz")
        layer9_boundary_path = Path(output_path).with_suffix(".layer9-boundary.npz")
        layer12_audit_path = Path(output_path).with_suffix(".layer12-moe.npz")
        layer12_boundary_path = Path(output_path).with_suffix(".layer12-boundary.npz")
        layer38_audit_path = Path(output_path).with_suffix(".layer38-moe.npz")
        layer38_boundary_path = Path(output_path).with_suffix(".layer38-boundary.npz")
        layer0_audit_arm = Path(output_path).with_suffix(".layer0-moe.armed")
        os.environ.update(
            {
                "HERO_LAYER0_MOE_AUDIT_PATH": str(layer0_audit_path),
                "HERO_LAYER0_BOUNDARY_AUDIT_PATH": str(layer0_boundary_path),
                "HERO_LAYER0_MOE_AUDIT_ARM_PATH": str(layer0_audit_arm),
                "HERO_LAYER0_MOE_AUDIT_POSITIONS": json.dumps([2046, 2047, 2048]),
                "HERO_LAYER2_MOE_AUDIT_PATH": str(layer2_audit_path),
                "HERO_LAYER2_BOUNDARY_AUDIT_PATH": str(layer2_boundary_path),
                "HERO_LAYER2_MOE_AUDIT_POSITIONS": json.dumps([6]),
                "HERO_LAYER9_MOE_AUDIT_PATH": str(layer9_audit_path),
                "HERO_LAYER9_BOUNDARY_AUDIT_PATH": str(layer9_boundary_path),
                "HERO_LAYER9_MOE_AUDIT_POSITIONS": json.dumps([0]),
                "HERO_LAYER12_MOE_AUDIT_PATH": str(layer12_audit_path),
                "HERO_LAYER12_BOUNDARY_AUDIT_PATH": str(layer12_boundary_path),
                "HERO_LAYER12_MOE_AUDIT_POSITIONS": json.dumps([6]),
                "HERO_LAYER38_MOE_AUDIT_PATH": str(layer38_audit_path),
                "HERO_LAYER38_BOUNDARY_AUDIT_PATH": str(layer38_boundary_path),
                "HERO_LAYER38_MOE_AUDIT_POSITIONS": json.dumps([6]),
            }
        )
        layer0_audit_arm.unlink(missing_ok=True)
    from vllm import LLM, SamplingParams

    with np.load(golden_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    valid_length = int(arrays["valid_lengths"][global_rank])
    prediction_indices = _rank_indices(arrays, global_rank)
    early_indices = prediction_indices[
        arrays["prediction_positions"][prediction_indices] < min(valid_length - 1, 31)
    ]
    short_logit_positions = [
        int(position) for position in arrays["prediction_positions"][early_indices]
    ]
    os.environ["HERO_SHORT_LOGITS_POSITIONS"] = json.dumps(short_logit_positions)
    os.environ["HERO_SHORT_LOGITS_PATH"] = str(
        Path(output_path).with_suffix(".short-logits.npz")
    )
    full_logit_positions = [
        int(position) for position in arrays["prediction_positions"][prediction_indices]
    ]
    os.environ["HERO_FULL_LOGITS_POSITIONS"] = json.dumps(full_logit_positions)
    os.environ["HERO_FULL_LOGITS_PATH"] = str(
        Path(output_path).with_suffix(".full-logits.npz")
    )
    windows = [(max(1, valid_length - 32), valid_length)]
    if valid_length >= 4095:
        windows.append((2040, 2052))
    decode_positions = sorted(
        {
            position
            for position in full_logit_positions
            if any(start - 1 <= position < end - 1 for start, end in windows)
        }
    )
    os.environ["HERO_DECODE_LOGITS_POSITIONS"] = json.dumps(decode_positions)
    os.environ["HERO_DECODE_LOGITS_DIR"] = str(
        Path(output_path).with_suffix(".decode-logits")
    )
    decode_capture_arm = Path(output_path).with_suffix(".decode-capture-armed")
    os.environ["HERO_DECODE_CAPTURE_ARM_PATH"] = str(decode_capture_arm)
    decode_capture_arm.unlink(missing_ok=True)
    tokens = [int(token) for token in arrays["tokens"][global_rank, :valid_length]]
    if global_rank == DECODE_LAYER_PROBE_RANK:
        os.environ.update(
            {
                "HERO_DECODE_LAYER_PROBE_PATH": str(
                    Path(output_path).with_suffix(".decode-layer.npz")
                ),
                "HERO_DECODE_LAYER_PROBE_POSITION": str(DECODE_LAYER_PROBE_POSITION),
                "HERO_DECODE_LAYER_PROBE_TOKEN_ID": str(
                    tokens[DECODE_LAYER_PROBE_POSITION]
                ),
            }
        )
    if global_rank in LAYER_PROBE_RANKS:
        os.environ.update(
            {
                "HERO_LAYER_PROBE_PATH": str(
                    Path(output_path).with_suffix(".layer.npz")
                ),
                "HERO_LAYER_PROBE_LENGTH": str(valid_length),
                "HERO_LAYER_PROBE_PREFIX_TOKENS": json.dumps(tokens[:8]),
                "HERO_LAYER_PROBE_POSITIONS": json.dumps(LAYER_PROBE_POSITIONS),
            }
        )

    llm = LLM(
        model=config_dir,
        model_weights=WEIGHT_ROOT,
        tokenizer=config_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4097,
        load_format="runai_streamer",
        model_loader_extra_config={
            "distributed": True,
            "concurrency": 8,
            "memory_limit": 2 * 1024**3,
        },
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_trace_replay=True,
        enable_return_routed_experts=True,
        max_logprobs=TOP_LOGPROBS,
        max_num_seqs=1,
        max_num_batched_tokens=4097,
        gpu_memory_utilization=0.95,
        disable_custom_all_reduce=True,
    )

    short_tokens = tokens[: min(valid_length, 32)]
    short_output = llm.generate(
        [{"prompt_token_ids": short_tokens}],
        SamplingParams(
            max_tokens=1,
            temperature=0,
            prompt_logprobs=TOP_LOGPROBS,
            detokenize=False,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )[0]
    short = _measure_mode(
        mode="short-prefill",
        request_logprobs=short_output.prompt_logprobs,
        array_index_for_prediction=lambda position: position + 1,
        arrays=arrays,
        rank=global_rank,
        prediction_indices=early_indices,
    )
    short_routes = short_output.outputs[0].routed_experts
    # Native arrays are [layer, case, position, expert], while vLLM returns
    # [position, layer, expert].
    expected_short_shape = (
        len(short_tokens),
        arrays["route_expert_ids"].shape[0],
        arrays["route_expert_ids"].shape[-1],
    )
    if short_routes is None or short_routes.shape != expected_short_shape:
        raise RuntimeError(
            "Short routed-expert capture has shape "
            f"{None if short_routes is None else short_routes.shape}; "
            f"expected {expected_short_shape}"
        )
    if layer0_audit_arm is not None:
        layer0_audit_arm.touch()

    prefill_output = llm.generate(
        [{"prompt_token_ids": tokens}],
        SamplingParams(
            max_tokens=1,
            temperature=0,
            prompt_logprobs=TOP_LOGPROBS,
            detokenize=False,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )[0]
    prefill = _measure_mode(
        mode="prefill",
        request_logprobs=prefill_output.prompt_logprobs,
        array_index_for_prediction=lambda position: position + 1,
        arrays=arrays,
        rank=global_rank,
        prediction_indices=prediction_indices,
    )
    prefill_routes = prefill_output.outputs[0].routed_experts
    expected_prefill_shape = (
        valid_length,
        arrays["route_expert_ids"].shape[0],
        arrays["route_expert_ids"].shape[-1],
    )
    if prefill_routes is None or prefill_routes.shape != expected_prefill_shape:
        raise RuntimeError(
            "Prefill routed-expert capture has shape "
            f"{None if prefill_routes is None else prefill_routes.shape}; "
            f"expected {expected_prefill_shape}"
        )
    np.savez_compressed(
        Path(output_path).with_suffix(".prefill-routes.npz"),
        short=short_routes,
        prefill=prefill_routes,
    )
    if global_rank in LAYER_PROBE_RANKS:
        trace_path = Path(output_path).with_suffix(".layer.npz")
        if not trace_path.exists():
            raise RuntimeError(f"Missing layer trace for rank {global_rank}")
        expected_positions = np.asarray(
            [position for position in LAYER_PROBE_POSITIONS if position < valid_length]
        )
        model_config = json.loads((Path(config_dir) / "config.json").read_text())
        layer_count = int(model_config["num_hidden_layers"])
        hidden_dim = int(model_config["hidden_size"])
        with np.load(trace_path, allow_pickle=False) as trace:
            if not np.array_equal(trace["positions"], expected_positions):
                raise ValueError(f"Wrong layer trace positions for rank {global_rank}")
            expected_tokens = np.asarray(tokens)[expected_positions]
            if not np.array_equal(trace["token_ids"], expected_tokens):
                raise ValueError(f"Wrong layer trace tokens for rank {global_rank}")
            names = ["model_input"] + [
                f"layer_{layer}_{site}"
                for layer in range(layer_count)
                for site in ("after_attn", "mlp_input", "after_block")
            ]
            names.extend(
                (
                    "embed_raw",
                    "embed_after_norm",
                    "embed_gate_up",
                    "embed_gate_sigmoid",
                )
            )
            for name in names:
                if trace[name].shape != (len(expected_positions), hidden_dim):
                    raise ValueError(
                        f"Bad layer trace {name} shape: {trace[name].shape}"
                    )
            for name in ("embed_gate_down", "embed_gate_silu"):
                if trace[name].shape != (len(expected_positions), GATED_NORM_RANK):
                    raise ValueError(
                        f"Bad layer trace {name} shape: {trace[name].shape}"
                    )

    # Only the saved positions require oracle comparisons. The tail window
    # covers every case; the long cases also score a short decode crossing the
    # 2,048-token attention boundary without replaying thousands of unscored
    # intermediate tokens.
    # Queue both windows together: offline DP/EP must keep every rank in the
    # same engine wave until all ranks have completed their collective steps.
    # The engine process is already running, so arm through a file it can see.
    # In particular, its startup probes can use position zero before this run.
    decode_capture_arm.touch()
    decode_outputs = llm.generate(
        [{"prompt_token_ids": tokens[:start]} for start, _ in windows],
        [
            SamplingParams(
                trace_decode_token_ids=tokens[start:end],
                max_tokens=end - start,
                logprobs=TOP_LOGPROBS,
                temperature=0,
                detokenize=False,
            )
            for start, end in windows
        ],
        use_tqdm=False,
    )
    decoded_positions: list[dict[str, Any]] = []
    for (prompt_length, end), output in zip(windows, decode_outputs, strict=True):
        continuation = tokens[prompt_length:end]
        indices = prediction_indices[
            (arrays["prediction_positions"][prediction_indices] >= prompt_length - 1)
            & (arrays["prediction_positions"][prediction_indices] < end - 1)
        ]
        if not len(indices):
            raise ValueError(
                f"Cached decode case {global_rank} window {prompt_length}:{end} "
                "has no saved positions"
            )
        decode_output = output.outputs[0]
        if list(decode_output.token_ids) != continuation:
            raise ValueError(
                f"Trace replay diverged for case {global_rank} "
                f"window {prompt_length}:{end}"
            )
        segment = _measure_mode(
            mode="cached-decode",
            request_logprobs=decode_output.logprobs,
            array_index_for_prediction=lambda position, start=prompt_length: (
                position - start + 1
            ),
            arrays=arrays,
            rank=global_rank,
            prediction_indices=indices,
        )
        decoded_positions.extend(segment["positions"])
    decoded_positions.sort(key=lambda row: row["prediction_position"])
    if len({row["prediction_position"] for row in decoded_positions}) != len(
        decoded_positions
    ):
        raise ValueError(f"Repeated cached-decode prediction for case {global_rank}")
    decode = _summarize_mode("cached-decode", global_rank, decoded_positions)
    decode["windows"] = [
        {"prompt_length": prompt_length, "continuation_length": end - prompt_length}
        for prompt_length, end in windows
    ]
    if global_rank == DECODE_LAYER_PROBE_RANK:
        decode_trace_path = Path(output_path).with_suffix(".decode-layer.npz")
        if not decode_trace_path.exists():
            raise RuntimeError("Missing Hero cached-decode layer trace")
        with np.load(decode_trace_path, allow_pickle=False) as trace:
            if not np.array_equal(trace["positions"], [DECODE_LAYER_PROBE_POSITION]):
                raise ValueError("Wrong cached-decode layer trace position")
            if not np.array_equal(
                trace["token_ids"], [tokens[DECODE_LAYER_PROBE_POSITION]]
            ):
                raise ValueError("Wrong cached-decode layer trace token")
    if decode_positions:
        capture_dir = Path(output_path).with_suffix(".decode-logits")
        captures = sorted(capture_dir.glob("*.npz")) if capture_dir.exists() else []
        if not captures:
            raise RuntimeError(f"No cached full logits captured for rank {global_rank}")
        captured = []
        for capture in captures:
            with np.load(capture) as source:
                captured.append({name: source[name] for name in source.files})
        np.savez_compressed(
            Path(output_path).with_suffix(".decode-logits.npz"),
            positions=np.concatenate([row["positions"] for row in captured]),
            forward_rows=np.concatenate([row["forward_rows"] for row in captured]),
            logits=np.concatenate([row["logits"] for row in captured]),
        )

    result = {
        "case_index": global_rank,
        "valid_length": valid_length,
        "checkpoint": CHECKPOINT,
        "golden_root": GOLDEN_ROOT,
        "weight_root": WEIGHT_ROOT,
        "vllm_revision": VLLM_REVISION,
        "qualification_revision": os.environ["HERO_QUALIFICATION_REVISION"],
        "iris_task_id": os.environ.get("IRIS_TASK_ID"),
        "host": socket.gethostname(),
        "local_rank": local_rank,
        "global_rank": global_rank,
        "runtime": {
            package: importlib.metadata.version(package)
            for package in (
                "vllm",
                "torch",
                "runai-model-streamer",
                "runai-model-streamer-s3",
                "numpy",
            )
        },
        "input_evidence": input_evidence,
        "serving_config": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "data_parallel_size": WORLD_SIZE,
            "expert_parallel_size": WORLD_SIZE,
            "moe_capacity_dropping": False,
            "pipeline_parallel_size": 1,
            "all2all_backend": "allgather_reducescatter",
            "batch_invariant": os.environ["VLLM_BATCH_INVARIANT"] == "1",
            "layer0_moe_audit": global_rank in (6, 7),
            "return_routed_experts": True,
            "short_prefix_route_audit_layers": [2, 12, 38]
            if global_rank in (6, 7)
            else [],
            "attention_backend": "FLASH_ATTN",
            "flash_attn_version": 2,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "model_runner": "v2",
            "trace_replay": True,
            "cached_decode_coverage": (
                "last 31-32 tokens of each case; plus positions around the "
                "2048-token boundary in the 4095- and 4096-token cases"
            ),
            "max_model_len": 4097,
            "max_model_len_note": (
                "4097 only lets the generate API return prompt logprobs for a "
                "4096-token prompt; the one output token is discarded and no "
                "score beyond the saved 4096-token prefix is used"
            ),
            "max_num_batched_tokens": 4097,
            "max_num_seqs_per_rank": 1,
            "load_format": "runai_streamer",
            "runai_distributed": True,
            "runai_memory_limit": 2 * 1024**3,
            "top_logprobs_returned": TOP_LOGPROBS,
        },
        "tolerances_fixed_before_run": {
            "target_logprob_abs": TARGET_LOGPROB_TOLERANCE,
            "saved_top_probability_abs": SAVED_TOP_PROBABILITY_TOLERANCE,
            "prefill_decode_target_logprob_abs": PREFILL_DECODE_TARGET_TOLERANCE,
            "prefill_decode_saved_top_probability_abs": (
                PREFILL_DECODE_TOP_PROBABILITY_TOLERANCE
            ),
        },
        "short": short,
        "prefill": prefill,
        "cached_decode": decode,
    }
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _wait_for_all_ranks(_s3_client(), global_rank)
    llm.llm_engine.engine_core.shutdown()


def _run_rank_safe(*args) -> None:
    try:
        _run_rank(*args)
    except BaseException:
        # vLLM's engine children otherwise keep multiprocessing's exit hook
        # waiting before Python prints the original rank exception.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)


def main() -> None:
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise RuntimeError(
            "Hero batch-invariance qualification requires VLLM_BATCH_INVARIANT=1"
        )
    task_index = _task_index()
    _configure_object_storage()
    _configure_node_network()
    client = _s3_client()
    master_addr = _master_address(client, task_index)
    local_root = Path("/tmp/hero-vllm-qualification")
    config_dir = local_root / "model-config"
    arrays_path = local_root / "golden-arrays.npz"
    golden_manifest_path = local_root / "golden-manifest.json"
    export_manifest_path = local_root / "export-manifest.json"
    _download(client, f"{WEIGHT_ROOT}/config.json", config_dir / "config.json")
    _download(client, f"{GOLDEN_ROOT}/arrays.npz", arrays_path)
    _download(client, f"{GOLDEN_ROOT}/manifest.json", golden_manifest_path)
    _download(client, f"{WEIGHT_ROOT}/export-manifest.json", export_manifest_path)

    golden_manifest = json.loads(golden_manifest_path.read_text())
    export_manifest = json.loads(export_manifest_path.read_text())
    arrays_sha256 = _sha256(arrays_path)
    if golden_manifest["asset_root"] != GOLDEN_ROOT:
        raise ValueError("Golden manifest asset root does not match the pinned root")
    if golden_manifest["checkpoint"]["uri"] != CHECKPOINT:
        raise ValueError(
            "Golden manifest checkpoint does not match the pinned checkpoint"
        )
    if golden_manifest["files"]["arrays.npz"]["sha256"] != arrays_sha256:
        raise ValueError("Downloaded golden arrays fail their retained SHA-256")
    required_export_fields = {
        "authoritative_weight_tree": "params",
        "authoritative_weight_dtype": "float32",
        "effective_weight_dtype": "bfloat16",
        "global_device_count": 32,
        "process_count": 32,
        # The split-v3 weights were exported against the original bundle;
        # new numerical reference bundles use the same checkpoint and weights.
        "golden_bundle": BASELINE_GOLDEN_ROOT,
    }
    for field, expected in required_export_fields.items():
        if export_manifest.get(field) != expected:
            raise ValueError(
                f"Export manifest {field} is {export_manifest.get(field)!r}, "
                f"expected {expected!r}"
            )
    if export_manifest["checkpoint"]["uri"] != CHECKPOINT:
        raise ValueError(
            "Export manifest checkpoint does not match the pinned checkpoint"
        )
    input_evidence = {
        "golden_manifest_sha256": _sha256(golden_manifest_path),
        "golden_arrays_sha256": arrays_sha256,
        "export_manifest_sha256": _sha256(export_manifest_path),
        "export_config_sha256": _sha256(config_dir / "config.json"),
        "export_source_revision": export_manifest["source_revision"],
        "pending_qb_rule": export_manifest["pending_qb_rule"],
        "pending_qb_betas_sha256": export_manifest["pending_qb_betas_sha256"],
        "expert_tensor_layout": export_manifest["expert_tensor_layout"],
    }

    context = mp.get_context("spawn")
    processes: list[tuple[int, Path, mp.Process]] = []
    for local_rank in range(LOCAL_WORLD_SIZE):
        global_rank = task_index * LOCAL_WORLD_SIZE + local_rank
        output_path = local_root / f"rank-{global_rank}.json"
        process = context.Process(
            target=_run_rank_safe,
            args=(
                local_rank,
                global_rank,
                master_addr,
                str(config_dir),
                str(arrays_path),
                str(output_path),
                input_evidence,
            ),
        )
        process.start()
        processes.append((global_rank, output_path, process))

    while True:
        failures = [
            (global_rank, process.exitcode)
            for global_rank, _, process in processes
            if process.exitcode is not None and process.exitcode != 0
        ]
        if failures or all(process.exitcode == 0 for _, _, process in processes):
            break
        time.sleep(2)
    if failures:
        for global_rank, output_path, process in processes:
            if process.exitcode in (None, 0):
                continue
            log_path = output_path.with_suffix(".log")
            if log_path.exists():
                log_uri = f"{RESULT_ROOT}/rank-{global_rank}.log"
                bucket, key = _s3_parts(log_uri)
                client.upload_file(str(log_path), bucket, key)
                print(f"uploaded failure log {log_uri}", flush=True)
        raise RuntimeError(f"Qualification ranks failed: {failures}")
    for global_rank, output_path, _ in processes:
        result_uri = f"{RESULT_ROOT}/rank-{global_rank}.json"
        _put_json(client, result_uri, json.loads(output_path.read_text()))
        print(f"uploaded {result_uri}", flush=True)
        short_path = output_path.with_suffix(".short-logits.npz")
        if not short_path.exists():
            raise RuntimeError(
                f"Missing short full-logit capture for rank {global_rank}"
            )
        short_uri = f"{RESULT_ROOT}/rank-{global_rank}.short-logits.npz"
        bucket, key = _s3_parts(short_uri)
        client.upload_file(str(short_path), bucket, key)
        print(f"uploaded {short_uri}", flush=True)
        logits_path = output_path.with_suffix(".full-logits.npz")
        if not logits_path.exists():
            raise RuntimeError(f"Missing full-logit capture for rank {global_rank}")
        logits_uri = f"{RESULT_ROOT}/rank-{global_rank}.full-logits.npz"
        bucket, key = _s3_parts(logits_uri)
        client.upload_file(str(logits_path), bucket, key)
        print(f"uploaded {logits_uri}", flush=True)
        routes_path = output_path.with_suffix(".prefill-routes.npz")
        if not routes_path.exists():
            raise RuntimeError(f"Missing routed-expert capture for rank {global_rank}")
        routes_uri = f"{RESULT_ROOT}/rank-{global_rank}.prefill-routes.npz"
        bucket, key = _s3_parts(routes_uri)
        client.upload_file(str(routes_path), bucket, key)
        print(f"uploaded {routes_uri}", flush=True)
        if global_rank in LAYER_PROBE_RANKS:
            trace_path = output_path.with_suffix(".layer.npz")
            if not trace_path.exists():
                raise RuntimeError(f"Missing layer trace for rank {global_rank}")
            trace_uri = f"{RESULT_ROOT}/rank-{global_rank}.layer.npz"
            bucket, key = _s3_parts(trace_uri)
            client.upload_file(str(trace_path), bucket, key)
            print(f"uploaded {trace_uri}", flush=True)
        if global_rank == DECODE_LAYER_PROBE_RANK:
            decode_trace_path = output_path.with_suffix(".decode-layer.npz")
            decode_trace_uri = f"{RESULT_ROOT}/rank-{global_rank}.decode-layer.npz"
            bucket, key = _s3_parts(decode_trace_uri)
            client.upload_file(str(decode_trace_path), bucket, key)
            print(f"uploaded {decode_trace_uri}", flush=True)
        if global_rank in (6, 7):
            audit_path = output_path.with_suffix(".layer0-moe.npz")
            if not audit_path.exists():
                raise RuntimeError(f"Missing layer-0 MoE audit for rank {global_rank}")
            audit_uri = f"{RESULT_ROOT}/rank-{global_rank}.layer0-moe.npz"
            bucket, key = _s3_parts(audit_uri)
            client.upload_file(str(audit_path), bucket, key)
            print(f"uploaded {audit_uri}", flush=True)
            boundary_path = output_path.with_suffix(".layer0-boundary.npz")
            if not boundary_path.exists():
                raise RuntimeError(
                    f"Missing layer-0 boundary audit for rank {global_rank}"
                )
            with np.load(boundary_path, allow_pickle=False) as boundary:
                if not np.array_equal(boundary["positions"], [2046, 2047, 2048]):
                    raise ValueError(
                        f"Wrong layer-0 boundary positions for rank {global_rank}"
                    )
                for name in ("model_input", "after_attn", "mlp_input"):
                    if boundary[name].shape != (3, 6144):
                        raise ValueError(
                            f"Bad layer-0 {name} shape: {boundary[name].shape}"
                        )
            boundary_uri = f"{RESULT_ROOT}/rank-{global_rank}.layer0-boundary.npz"
            bucket, key = _s3_parts(boundary_uri)
            client.upload_file(str(boundary_path), bucket, key)
            print(f"uploaded {boundary_uri}", flush=True)
            for layer, positions in ((2, [6]), (9, [0]), (12, [6]), (38, [6])):
                for kind in ("moe", "boundary"):
                    audit_path = output_path.with_suffix(f".layer{layer}-{kind}.npz")
                    if not audit_path.exists():
                        raise RuntimeError(
                            f"Missing layer-{layer} {kind} audit for rank {global_rank}"
                        )
                    with np.load(audit_path, allow_pickle=False) as audit:
                        if not np.array_equal(audit["positions"], positions):
                            raise ValueError(
                                f"Wrong layer {layer} {kind} positions "
                                f"for rank {global_rank}"
                            )
                        expected_shapes = (
                            {
                                "model_input": (1, 6144),
                                "after_attn": (1, 6144),
                                "mlp_input": (1, 6144),
                            }
                            if kind == "boundary"
                            else {
                                "mlp_input": (1, 6144),
                                "routed_input": (1, 3072),
                                "router_logits": (1, 384),
                                "selected_experts": (1, 8),
                            }
                        )
                        for name, expected_shape in expected_shapes.items():
                            if audit[name].shape != expected_shape:
                                raise ValueError(
                                    f"Layer {layer} {name} has shape "
                                    f"{audit[name].shape}"
                                )
                    audit_uri = (
                        f"{RESULT_ROOT}/rank-{global_rank}"
                        f".layer{layer}-{kind}.npz"
                    )
                    bucket, key = _s3_parts(audit_uri)
                    client.upload_file(str(audit_path), bucket, key)
                    print(f"uploaded {audit_uri}", flush=True)
        decode_path = output_path.with_suffix(".decode-logits.npz")
        if decode_path.exists():
            decode_uri = f"{RESULT_ROOT}/rank-{global_rank}.decode-logits.npz"
            bucket, key = _s3_parts(decode_uri)
            client.upload_file(str(decode_path), bucket, key)
            print(f"uploaded {decode_uri}", flush=True)


def submit(iris_config: Path, golden_root: str, result_root: str) -> None:
    """Submit the fixed two-node B200 qualification job through Iris."""
    from fray.iris_backend import (
        convert_constraints,
        convert_resources,
        resolve_coscheduling,
    )
    from fray.types import GpuConfig, ResourceConfig
    from iris.cli.connect import connect_controller
    from iris.client.client import IrisClient
    from iris.cluster.types import Entrypoint, EnvironmentSpec
    from iris.rpc import job_pb2
    from iris.rpc.proto_display import priority_band_value
    from rigging.timing import Duration

    repository = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--"], cwd=repository, check=True
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    name_digest = hashlib.sha256(
        json.dumps(
            {
                "qualification_revision": revision,
                "result_root": result_root,
                "golden_root": golden_root,
                "weight_root": WEIGHT_ROOT,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    resources = ResourceConfig(
        cpu=64,
        ram="400g",
        disk="1t",
        device=GpuConfig(variant="GB200", count=QUALIFICATION_GPUS_PER_TASK),
        replicas=QUALIFICATION_TASKS,
    )
    native_resources = convert_resources(resources)
    with (
        connect_controller(config_file=iris_config) as endpoint,
        IrisClient.remote(
            endpoint.url,
            credentials=endpoint.credentials,
            workspace=repository,
            bundle_exclude=re.compile(r"^(?:docs|tests)/"),
        ) as client,
    ):
        job = client.submit(
            entrypoint=Entrypoint.from_command(
                "python", "infra/qualification/hero_b200.py"
            ),
            name=f"hero-vllm-qualification-{name_digest}",
            user="hero-vllm",
            resources=native_resources,
            replicas=QUALIFICATION_TASKS,
            environment=EnvironmentSpec(
                env_vars={
                    "HERO_QUALIFICATION_REVISION": revision,
                    "HERO_GOLDEN_ROOT": golden_root,
                    "HERO_RESULT_ROOT": result_root,
                    "VLLM_BATCH_INVARIANT": "1",
                    "PYTHONUNBUFFERED": "1",
                },
                setup_scripts=[QUALIFICATION_SETUP],
            ),
            constraints=convert_constraints(resources),
            coscheduling=resolve_coscheduling(resources, QUALIFICATION_TASKS),
            scheduling_timeout=Duration.from_hours(24),
            timeout=Duration.from_hours(8),
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            priority_band=priority_band_value("interactive"),
            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        )
    print(job.job_id)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("submit")
        parser.add_argument("--iris-config", type=Path, required=True)
        parser.add_argument("--golden-root", required=True)
        parser.add_argument("--result-root", required=True)
        args = parser.parse_args()
        submit(args.iris_config, args.golden_root, args.result_root)
    else:
        main()
