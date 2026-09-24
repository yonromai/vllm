# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental full-Hero H100/B200 qualification against retained goldens."""

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

import numpy as np

CHECKPOINT = (
    "s3://marin-us-east-02a/marin/grug/hero-ragged_a2a-nccl2307-ep-step81k/"
    "2026.08.19.2/checkpoints/step-108000"
)
GOLDEN_ROOT = (
    "s3://marin-us-east-02a/marin/reference/hero-forward/"
    "hero-535b-step108000-bf16-v1-dcfe4ced165a"
)
EXPORT_GOLDEN_ROOT = GOLDEN_ROOT
FRESH_INPUT_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/hero-numerical-resolution/"
    "fp32-combine/hero-535b-step108000-bf16-fp32-combine-fresh-v1-123ec116ea42"
)
WEIGHT_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/hero-vllm-b200/"
    "hero-535b-step108000-bf16-split-v3"
)
GSM8K_INPUT_URI = (
    "s3://marin-us-east-02a/marin/users/romain/hero-gsm8k-async-01a0cc2c/"
    "inputs/train-0-31-v1.json"
)
GSM8K_INPUT_SHA256 = "5ea09a1a12757f00ab2a45bdd840a1dfadd5d378b4aeb8c3e8bb06b78a27ecd4"
SAVED_STOP_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/hero-4k-closure-01a0bca4/"
    "h100-eot-7755aaca7"
)
NATURAL_TRACE = os.environ.get("HERO_NATURAL_TRACE", "0")
PAD_KV_ROWS = int(os.environ.get("HERO_NUMERIC_PAD_KV_ROWS", "0"))
TRACE_TARGETS = {
    8: (1860, 102473, 1667),
    17: (3692, 4777, 3524),
}
CONTEXT_LIMIT = 4096
MAX_MODEL_LEN = CONTEXT_LIMIT + 1
# The pinned Marin tokenizer maps <|eot_id|> to 128009, but EOS to 128001.
# Hero commonly ends an assistant turn with EOT; EOS alone did not stop GSM8K.
HERO_TURN_END_TOKEN_ID = 128009
# Bound startup profiling and the full-prefix EP all-gather workspace on H100.
MAX_BATCHED_TOKENS = 1024
HARDWARE = os.environ.get("HERO_HARDWARE", "GB200")
if HARDWARE not in {"GB200", "H100"}:
    raise ValueError(f"Unsupported Hero pilot hardware {HARDWARE!r}")
PILOT = os.environ.get("HERO_PILOT", "1")
if PILOT not in {"0", "1"}:
    raise ValueError(f"HERO_PILOT must be 0 or 1, got {PILOT!r}")
RL_ROLLOUT = os.environ.get("HERO_RL_ROLLOUT", "0")
if RL_ROLLOUT not in {"0", "1"} or (PILOT == "1" and RL_ROLLOUT == "1"):
    raise ValueError("HERO_RL_ROLLOUT must be 0 or 1 and requires HERO_PILOT=0")
GSM8K_PILOT = os.environ.get("HERO_GSM8K_PILOT", "0")
if GSM8K_PILOT not in {"0", "1"} or (
    GSM8K_PILOT == "1" and (PILOT == "1" or RL_ROLLOUT == "1")
):
    raise ValueError("HERO_GSM8K_PILOT requires HERO_PILOT=0 and HERO_RL_ROLLOUT=0")
if NATURAL_TRACE not in {"0", "1", "2"} or (
    NATURAL_TRACE != "0" and GSM8K_PILOT != "1"
):
    raise ValueError("HERO_NATURAL_TRACE requires HERO_GSM8K_PILOT=1")
if PAD_KV_ROWS < 0:
    raise ValueError("HERO_NUMERIC_PAD_KV_ROWS must be nonnegative")
GPU_MEMORY_UTILIZATION = float(os.environ.get("HERO_GPU_MEMORY_UTILIZATION", "0.95"))
if not 0 < GPU_MEMORY_UTILIZATION < 1:
    raise ValueError("HERO_GPU_MEMORY_UTILIZATION must be between 0 and 1")
MODE_NAME = (
    "pilot"
    if PILOT == "1"
    else "rl-64"
    if RL_ROLLOUT == "1"
    else "eot-4k-gsm8k"
)
RESULT_ROOT = os.environ.get(
    "HERO_RESULT_ROOT",
    "s3://marin-us-east-02a/marin/users/romain/hero-gsm8k-async-01a0cc2c/"
    f"step108000-{HARDWARE.lower()}-"
    f"{MODE_NAME}"
    "-5a4a52329",
)
VLLM_REVISION = "5a4a52329468b6bd16b21d1f319fcb96d405dd36"
WORLD_SIZE = 32 if HARDWARE == "H100" else 8
LOCAL_WORLD_SIZE = 8 if HARDWARE == "H100" else 4
# Host-network tasks can share a node with other jobs. Derive a stable port
# from this run's output root so their vLLM DP listeners do not collide.
MASTER_PORT = 20000 + (
    int(hashlib.sha256(RESULT_ROOT.encode()).hexdigest()[:4], 16) % 20000
)
SIOCGIFADDR = 0x8915
TOP_LOGPROBS = 64
QUALIFICATION_GPUS_PER_TASK = LOCAL_WORLD_SIZE
QUALIFICATION_TASKS = WORLD_SIZE // QUALIFICATION_GPUS_PER_TASK
PRECOMPILED_WHEEL = (
    "https://github.com/marin-community/vllm/releases/download/"
    "marin-vllm-gpu-candidate-fb02daf1d713/"
    "vllm-0.0.0.dev20260920%2Bmarin.fb02daf1d713.cu132-"
    f"cp38-abi3-manylinux_2_28_{'x86_64' if HARDWARE == 'H100' else 'aarch64'}.whl"
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
    import boto3
    from botocore.config import Config

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
    fresh_path: str | None,
    gsm8k_inputs_path: str | None,
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
    from vllm import LLM, SamplingParams

    with np.load(golden_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    case_index = global_rank % len(arrays["valid_lengths"])
    valid_length = int(arrays["valid_lengths"][case_index])
    tokens = [int(token) for token in arrays["tokens"][case_index, :valid_length]]
    prediction_indices = _rank_indices(arrays, case_index)
    if RL_ROLLOUT == "1":
        slot = global_rank % 8
        bank = "original" if slot < 4 else "fresh"
        case_index = (0, 2, 4, 6)[slot % 4]
        if bank == "fresh":
            if fresh_path is None:
                raise ValueError("Fresh input bank was not downloaded")
            with np.load(fresh_path, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
        valid_length = int(arrays["valid_lengths"][case_index])
        prompt_length = min(valid_length, 4032)
        tokens = [
            int(token) for token in arrays["tokens"][case_index, :prompt_length]
        ]

    trace_root = Path(output_path).with_suffix(".numeric")
    trace_arm = Path(output_path).with_suffix(".numeric-arm")
    if NATURAL_TRACE != "0" and global_rank in TRACE_TARGETS:
        trace_position, trace_input_token, _ = TRACE_TARGETS[global_rank]
        trace_arm.write_text("sample\n")
        os.environ.update(
            {
                "HERO_NUMERIC_TRACE_ROOT": str(trace_root),
                "HERO_NUMERIC_TRACE_ARM": str(trace_arm),
                "HERO_NUMERIC_TRACE_POSITION": str(trace_position),
                "HERO_NUMERIC_TRACE_TOKEN": str(trace_input_token),
            }
        )

    llm = LLM(
        model=config_dir,
        model_weights=WEIGHT_ROOT,
        tokenizer=config_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
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
        enable_chunked_prefill=True,
        enable_trace_replay=True,
        enable_return_routed_experts=True,
        max_logprobs=TOP_LOGPROBS,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_custom_all_reduce=True,
    )

    if NATURAL_TRACE != "0":
        assert gsm8k_inputs_path is not None
        pilot_inputs = json.loads(Path(gsm8k_inputs_path).read_text())
        record = pilot_inputs["records"][global_rank]
        saved_path = Path(output_path).with_suffix(".saved.json")
        saved_uri = f"{SAVED_STOP_ROOT}/rank-{global_rank}.json"
        _download(_s3_client(), saved_uri, saved_path)
        saved = json.loads(saved_path.read_text())["gsm8k_pilot"]
        prompt_ids = record["hero_prompt_token_ids"]
        saved_response = saved["response_token_ids"]
        if prompt_ids != saved["prompt_token_ids"]:
            raise ValueError(f"Saved prompt differs at rank {global_rank}")
        if NATURAL_TRACE == "1":
            decode_params = SamplingParams(
                max_tokens=CONTEXT_LIMIT - len(prompt_ids),
                temperature=1.0,
                top_p=1.0,
                logprobs=5,
                detokenize=False,
                seed=17 + global_rank,
                stop_token_ids=[HERO_TURN_END_TOKEN_ID],
            )
        else:
            decode_params = SamplingParams(
                trace_decode_token_ids=saved_response,
                max_tokens=len(saved_response),
                temperature=0,
                logprobs=5,
                detokenize=False,
                ignore_eos=True,
            )
        decode_start = time.monotonic()
        rollout = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            decode_params,
            use_tqdm=False,
        )[0].outputs[0]
        decode_seconds = time.monotonic() - decode_start
        response_ids = [int(token) for token in rollout.token_ids]
        if response_ids != saved_response:
            raise ValueError(f"Decoded response differs at rank {global_rank}")
        if global_rank in TRACE_TARGETS:
            trace_arm.write_text("full\n")
        trajectory_ids = prompt_ids + saved_response
        prefill_start = time.monotonic()
        prefill = llm.generate(
            [{"prompt_token_ids": trajectory_ids}],
            SamplingParams(
                max_tokens=1,
                temperature=0,
                prompt_logprobs=1,
                detokenize=False,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )[0]
        prefill_seconds = time.monotonic() - prefill_start
        result: dict[str, Any] = {
            "rank": global_rank,
            "checkpoint": CHECKPOINT,
            "weight_root": WEIGHT_ROOT,
            "source_revision": os.environ["HERO_QUALIFICATION_REVISION"],
            "saved_capture_uri": saved_uri,
            "saved_capture_sha256": _sha256(saved_path),
            "decode_mode": "sampled" if NATURAL_TRACE == "1" else "forced_saved_ids",
            "sampled_prefix_matches": True if NATURAL_TRACE == "1" else None,
            "forced_response_matches_saved": NATURAL_TRACE == "2",
            "decode_tokens": len(response_ids),
            "decode_seconds": decode_seconds,
            "prefill_tokens": len(trajectory_ids),
            "prefill_seconds": prefill_seconds,
            "numeric_pad_kv_rows": PAD_KV_ROWS,
            "trace_position": TRACE_TARGETS[global_rank][0]
            if global_rank in TRACE_TARGETS
            else None,
        }
        if global_rank in TRACE_TARGETS:
            trace_position, trace_input_token, response_index = TRACE_TARGETS[
                global_rank
            ]
            if trajectory_ids[trace_position] != trace_input_token:
                raise ValueError("Saved trace input token differs")
            target_token = saved_response[response_index]
            sample_entry = rollout.logprobs[response_index][target_token]
            sample_score = float(sample_entry.logprob)
            prefill_entry = prefill.prompt_logprobs[
                len(prompt_ids) + response_index
            ][target_token]
            prefill_score = float(prefill_entry.logprob)
            sample_routes = rollout.routed_experts
            prefill_routes = prefill.outputs[0].routed_experts
            if sample_routes is None or prefill_routes is None:
                raise ValueError("Missing routed experts in natural trace")
            result.update(
                {
                    "target_token": target_token,
                    "decode_score": sample_score,
                    "saved_sample_score": saved["response_logprobs"][response_index],
                    "prefill_score": prefill_score,
                    "decode_route": sample_routes[trace_position].tolist(),
                    "prefill_route": prefill_routes[trace_position].tolist(),
                    "sample_trace_sha256": _sha256(Path(f"{trace_root}.sample.npz")),
                    "full_trace_sha256": _sha256(Path(f"{trace_root}.full.npz")),
                }
            )
            if NATURAL_TRACE == "1":
                result["sample_score"] = sample_score
                result["sample_route"] = result["decode_route"]
            trace_arm.unlink()
        Path(output_path).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        _wait_for_all_ranks(_s3_client(), global_rank)
        llm.llm_engine.engine_core.shutdown()
        return

    if RL_ROLLOUT == "1":
        rollout = llm.generate(
            [{"prompt_token_ids": tokens}],
            SamplingParams(
                max_tokens=64,
                temperature=0,
                logprobs=5,
                detokenize=False,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )[0]
        completion = rollout.outputs[0]
        response = [int(token_id) for token_id in completion.token_ids]
        routes = completion.routed_experts
        if len(response) != 64 or routes is None:
            raise ValueError("RL rollout did not return 64 tokens and full routes")
        expected_shape = (len(tokens) + len(response) - 1, 48, 8)
        if routes.shape != expected_shape:
            raise ValueError(f"RL route shape {routes.shape} != {expected_shape}")
        route_path = Path(output_path).with_suffix(".routes.npz")
        np.savez_compressed(route_path, routed_experts=routes.astype(np.int16))
        Path(output_path).write_text(
            json.dumps(
                {
                    "checkpoint": CHECKPOINT,
                    "weight_root": WEIGHT_ROOT,
                    "vllm_revision": VLLM_REVISION,
                    "qualification_revision": os.environ["HERO_QUALIFICATION_REVISION"],
                    "input_evidence": input_evidence,
                    "rank": global_rank,
                    "bank": bank,
                    "case_index": case_index,
                    "prompt_token_ids": tokens,
                    "response_token_ids": response,
                    "response_logprobs": [
                        float(logprobs[token_id].logprob)
                        for token_id, logprobs in zip(
                            response, completion.logprobs, strict=True
                        )
                    ],
                    "routed_experts_shape": list(routes.shape),
                    "routed_experts_sha256": _sha256(route_path),
                    "routed_experts_uri": (
                        f"{RESULT_ROOT}/rank-{global_rank}.routes.npz"
                    ),
                },
                sort_keys=True,
            ) + "\n"
        )
        _wait_for_all_ranks(_s3_client(), global_rank)
        llm.llm_engine.engine_core.shutdown()
        return

    if PILOT == "1":
        pilot_prompt = tokens[: min(valid_length, 32)]
        pilot_output = llm.generate(
            [{"prompt_token_ids": pilot_prompt}],
            SamplingParams(
                max_tokens=8,
                temperature=0,
                logprobs=5,
                detokenize=False,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )[0]
        completion = pilot_output.outputs[0]
        generated = [int(token_id) for token_id in completion.token_ids]
        routes = completion.routed_experts
        if len(generated) != 8 or routes is None:
            raise ValueError("Pilot did not return eight tokens and their routes")
        expected_shape = (len(pilot_prompt) + len(generated) - 1, 48, 8)
        if routes.shape != expected_shape:
            raise ValueError(f"Route shape {routes.shape} != {expected_shape}")
        scores = [
            float(logprobs[token_id].logprob)
            for token_id, logprobs in zip(generated, completion.logprobs, strict=True)
        ]
        Path(output_path).write_text(
            json.dumps(
                {
                    "checkpoint": CHECKPOINT,
                    "weight_root": WEIGHT_ROOT,
                    "vllm_revision": VLLM_REVISION,
                    "qualification_revision": os.environ["HERO_QUALIFICATION_REVISION"],
                    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                    "input_evidence": input_evidence,
                    "rank": global_rank,
                    "case_index": case_index,
                    "prompt_token_ids": pilot_prompt,
                    "response_token_ids": generated,
                    "response_logprobs": scores,
                    "routed_experts_shape": list(routes.shape),
                    "routed_experts": routes.tolist(),
                },
                sort_keys=True,
            ) + "\n"
        )
        _wait_for_all_ranks(_s3_client(), global_rank)
        llm.llm_engine.engine_core.shutdown()
        return

    early_indices = prediction_indices[
        arrays["prediction_positions"][prediction_indices] < min(valid_length - 1, 31)
    ]
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

    # Only the saved positions require oracle comparisons. The tail window
    # covers every case; the long cases also score a short decode crossing the
    # 2,048-token attention boundary without replaying thousands of unscored
    # intermediate tokens.
    windows = [(max(1, valid_length - 32), valid_length)]
    if valid_length >= 4095:
        windows.append((2040, 2052))
    # Queue all windows together: offline DP/EP must keep every rank in the
    # same engine wave until all ranks have completed their collective steps.
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

    gsm8k_pilot = None
    if gsm8k_inputs_path is not None:
        pilot_inputs = json.loads(Path(gsm8k_inputs_path).read_text())
        record = pilot_inputs["records"][global_rank]
        prompt_ids = record["hero_prompt_token_ids"]
        response_budget = CONTEXT_LIMIT - len(prompt_ids)
        if response_budget <= 0:
            raise ValueError(f"GSM8K prompt {global_rank} exceeds the context limit")
        start = time.monotonic()
        rollout = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            SamplingParams(
                max_tokens=response_budget,
                temperature=1.0,
                top_p=1.0,
                logprobs=5,
                detokenize=False,
                seed=17 + global_rank,
                stop_token_ids=[HERO_TURN_END_TOKEN_ID],
            ),
            use_tqdm=False,
        )[0]
        duration = time.monotonic() - start
        completion = rollout.outputs[0]
        response_ids = [int(token_id) for token_id in completion.token_ids]
        if not response_ids:
            raise ValueError(f"GSM8K rank {global_rank} returned no response tokens")
        response_scores = [
            float(logprobs[token_id].logprob)
            for token_id, logprobs in zip(
                response_ids, completion.logprobs, strict=True
            )
        ]
        routes = completion.routed_experts
        expected_shape = (len(prompt_ids) + len(response_ids) - 1, 48, 8)
        if routes is None or routes.shape != expected_shape:
            observed_shape = None if routes is None else routes.shape
            raise ValueError(f"GSM8K route shape {observed_shape} != {expected_shape}")
        route_path = Path(output_path).with_suffix(".gsm8k.routes.npz")
        np.savez_compressed(route_path, routed_experts=routes.astype(np.int16))
        gsm8k_capture = {
            "row_id": record["row_id"],
            "ground_truth": record["ground_truth"],
            "prompt_token_ids": prompt_ids,
            "response_token_ids": response_ids,
            "response_text": completion.text,
            "response_logprobs": response_scores,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "elapsed_seconds": duration,
            "routed_experts_shape": list(routes.shape),
            "routed_experts_sha256": _sha256(route_path),
            "routed_experts_uri": f"{RESULT_ROOT}/rank-{global_rank}.gsm8k.routes.npz",
            "sampling": {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": response_budget,
                "total_context_limit": CONTEXT_LIMIT,
                "seed": 17 + global_rank,
                "stop_token_ids": [HERO_TURN_END_TOKEN_ID],
            },
        }
        # A replay failure can kill an EP engine. Retain the natural response
        # before the optional same-prefix diagnostic starts.
        Path(output_path).with_suffix(".gsm8k.partial.json").write_text(
            json.dumps(
                {
                    "checkpoint": CHECKPOINT,
                    "weight_root": WEIGHT_ROOT,
                    "vllm_revision": VLLM_REVISION,
                    "qualification_revision": os.environ["HERO_QUALIFICATION_REVISION"],
                    "input_evidence": input_evidence,
                    "global_rank": global_rank,
                    "gsm8k_pilot": gsm8k_capture,
                },
                sort_keys=True,
            ) + "\n"
        )

        trajectory_ids = prompt_ids + response_ids
        replay_positions = sorted(
            {0, min(1, len(response_ids) - 1), len(response_ids) // 2}
            | set(range(max(0, len(response_ids) - 32), len(response_ids)))
        )
        prefill_start = time.monotonic()
        prefill_replay = llm.generate(
            [{"prompt_token_ids": trajectory_ids}],
            SamplingParams(
                max_tokens=1,
                temperature=0,
                prompt_logprobs=1,
                detokenize=False,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )[0]
        prefill_replay_seconds = time.monotonic() - prefill_start
        full_prefill_entries = prefill_replay.prompt_logprobs[
            len(prompt_ids) : len(trajectory_ids)
        ]
        if len(full_prefill_entries) != len(response_ids):
            raise ValueError("GSM8K full-prefix prefill logprobs are incomplete")
        full_prefill_scores = [
            float(entries[token_id].logprob)
            for entries, token_id in zip(
                full_prefill_entries, response_ids, strict=True
            )
        ]
        full_prefill_top_ids = [
            _top_by_rank(entries, count=1)[0] for entries in full_prefill_entries
        ]
        sampled_top_ids = [
            _top_by_rank(entries, count=1)[0] for entries in completion.logprobs
        ]
        full_prefill_routes = prefill_replay.outputs[0].routed_experts
        route_comparison = {
            "sampled_shape": list(routes.shape),
            "prefill_shape": (
                None if full_prefill_routes is None else list(full_prefill_routes.shape)
            ),
        }
        if (
            full_prefill_routes is not None
            and full_prefill_routes.shape[0] >= routes.shape[0]
        ):
            comparable = full_prefill_routes[: routes.shape[0]]
            ordered_different = np.any(comparable != routes, axis=-1)
            set_different = np.any(
                np.sort(comparable, axis=-1) != np.sort(routes, axis=-1), axis=-1
            )
            response_rows = slice(len(prompt_ids) - 1, None)
            route_comparison.update(
                ordered_mismatch_rows=int(np.any(ordered_different, axis=-1).sum()),
                set_mismatch_rows=int(np.any(set_different, axis=-1).sum()),
                ordered_mismatch_response_rows=int(
                    np.any(ordered_different[response_rows], axis=-1).sum()
                ),
                set_mismatch_response_rows=int(
                    np.any(set_different[response_rows], axis=-1).sum()
                ),
                compared_rows=int(routes.shape[0]),
                compared_response_rows=len(response_ids),
            )
        prefill_entries = [
            prefill_replay.prompt_logprobs[len(prompt_ids) + index]
            for index in replay_positions
        ]
        prefill_scores = [
            float(entries[response_ids[index]].logprob)
            for entries, index in zip(prefill_entries, replay_positions, strict=True)
        ]
        prefill_top_ids = [
            _top_by_rank(entries, count=1)[0] for entries in prefill_entries
        ]

        decode_start_index = max(len(prompt_ids), len(trajectory_ids) - 32)
        decode_start = time.monotonic()
        cached_replay = llm.generate(
            [{"prompt_token_ids": trajectory_ids[:decode_start_index]}],
            SamplingParams(
                trace_decode_token_ids=trajectory_ids[decode_start_index:],
                max_tokens=len(trajectory_ids) - decode_start_index,
                temperature=0,
                logprobs=1,
                detokenize=False,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )[0].outputs[0]
        cached_replay_seconds = time.monotonic() - decode_start
        if list(cached_replay.token_ids) != trajectory_ids[decode_start_index:]:
            raise ValueError(f"GSM8K cached replay diverged for rank {global_rank}")
        cached_scores = [
            float(logprobs[token_id].logprob)
            for token_id, logprobs in zip(
                cached_replay.token_ids, cached_replay.logprobs, strict=True
            )
        ]
        cached_top_ids = [
            _top_by_rank(logprobs, count=1)[0]
            for logprobs in cached_replay.logprobs
        ]
        cached_response_start = decode_start_index - len(prompt_ids)
        gsm8k_pilot = {
            **gsm8k_capture,
            "same_prefix": {
                "prefill_replay_seconds": prefill_replay_seconds,
                "cached_replay_seconds": cached_replay_seconds,
                "prefill_response_indices": replay_positions,
                "prefill_target_logprobs": prefill_scores,
                "full_prefill_target_logprobs": full_prefill_scores,
                "full_prefill_top1_mismatch_indices": [
                    index
                    for index, (sampled_top, prefill_top) in enumerate(
                        zip(sampled_top_ids, full_prefill_top_ids, strict=True)
                    )
                    if sampled_top != prefill_top
                ],
                "full_prefill_route_comparison": route_comparison,
                "sampled_top_token_ids": [
                    _top_by_rank(completion.logprobs[index], count=1)[0]
                    for index in replay_positions
                ],
                "prefill_top_token_ids": prefill_top_ids,
                "cached_response_start_index": cached_response_start,
                "cached_target_logprobs": cached_scores,
                "cached_top_token_ids": cached_top_ids,
            },
        }

    result = {
        "case_index": case_index,
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
            "attention_backend": "FLASH_ATTN",
            "flash_attn_version": 2,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "model_runner": "v2",
            "trace_replay": True,
            "cached_decode_coverage": (
                "last 31-32 tokens of each reference case; plus positions "
                "around the 2048-token boundary"
            ),
            "max_model_len": MAX_MODEL_LEN,
            "max_model_len_note": (
                "4097 lets the generate API return prompt logprobs for a "
                "4096-token prompt; the one output token is discarded and no "
                "score beyond the saved 4096-token prefix is used"
            ),
            "max_num_batched_tokens": MAX_BATCHED_TOKENS,
            "max_num_seqs_per_rank": 1,
            "load_format": "runai_streamer",
            "runai_distributed": True,
            "runai_memory_limit": 2 * 1024**3,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
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
        "gsm8k_pilot": gsm8k_pilot,
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
    task_index = _task_index()
    _configure_object_storage()
    _configure_node_network()
    client = _s3_client()
    master_addr = _master_address(client, task_index)
    local_root = Path("/tmp/hero-vllm-qualification")
    config_dir = local_root / "model-config"
    arrays_path = local_root / "golden-arrays.npz"
    golden_manifest_path = local_root / "golden-manifest.json"
    fresh_arrays_path = local_root / "fresh-arrays.npz"
    fresh_manifest_path = local_root / "fresh-manifest.json"
    gsm8k_inputs_path = local_root / "gsm8k-inputs.json"
    export_manifest_path = local_root / "export-manifest.json"
    _download(client, f"{WEIGHT_ROOT}/config.json", config_dir / "config.json")
    _download(client, f"{GOLDEN_ROOT}/arrays.npz", arrays_path)
    _download(client, f"{GOLDEN_ROOT}/manifest.json", golden_manifest_path)
    _download(client, f"{WEIGHT_ROOT}/export-manifest.json", export_manifest_path)
    if GSM8K_PILOT == "1":
        _download(client, GSM8K_INPUT_URI, gsm8k_inputs_path)
        if _sha256(gsm8k_inputs_path) != GSM8K_INPUT_SHA256:
            raise ValueError("GSM8K pilot inputs fail their retained SHA-256")
    if RL_ROLLOUT == "1":
        _download(client, f"{FRESH_INPUT_ROOT}/arrays.npz", fresh_arrays_path)
        _download(client, f"{FRESH_INPUT_ROOT}/manifest.json", fresh_manifest_path)

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
        "golden_bundle": EXPORT_GOLDEN_ROOT,
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
        "golden_training_sequence_length": golden_manifest["model"]["resolved_config"][
            "max_seq_len"
        ],
        "export_manifest_sha256": _sha256(export_manifest_path),
        "export_config_sha256": _sha256(config_dir / "config.json"),
        "export_eos_token_id": json.loads(
            (config_dir / "config.json").read_text()
        ).get("eos_token_id"),
        "export_source_revision": export_manifest["source_revision"],
        "pending_qb_rule": export_manifest["pending_qb_rule"],
        "pending_qb_betas_sha256": export_manifest["pending_qb_betas_sha256"],
        "expert_tensor_layout": export_manifest["expert_tensor_layout"],
    }
    if GSM8K_PILOT == "1":
        pilot_inputs = json.loads(gsm8k_inputs_path.read_text())
        if len(pilot_inputs["records"]) < WORLD_SIZE:
            raise ValueError("GSM8K pilot requires one distinct question per rank")
        input_evidence.update(
            {
                "gsm8k_inputs_uri": GSM8K_INPUT_URI,
                "gsm8k_inputs_sha256": GSM8K_INPUT_SHA256,
            }
        )
    if RL_ROLLOUT == "1":
        fresh_manifest = json.loads(fresh_manifest_path.read_text())
        fresh_sha256 = _sha256(fresh_arrays_path)
        if fresh_manifest["asset_root"] != FRESH_INPUT_ROOT:
            raise ValueError("Fresh input manifest root does not match")
        if fresh_manifest["checkpoint"]["uri"] != CHECKPOINT:
            raise ValueError("Fresh input bank uses a different checkpoint")
        if fresh_manifest["files"]["arrays.npz"]["sha256"] != fresh_sha256:
            raise ValueError("Fresh input arrays fail retained SHA-256")
        input_evidence.update(
            {
                "fresh_input_root": FRESH_INPUT_ROOT,
                "fresh_input_arrays_sha256": fresh_sha256,
                "fresh_input_manifest_sha256": _sha256(fresh_manifest_path),
                "fresh_scores_are_not_used": True,
            }
        )

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
                str(fresh_arrays_path) if RL_ROLLOUT == "1" else None,
                str(gsm8k_inputs_path) if GSM8K_PILOT == "1" else None,
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
            if NATURAL_TRACE != "0":
                for path in (
                    output_path,
                    output_path.with_suffix(".numeric.sample.npz"),
                    output_path.with_suffix(".numeric.full.npz"),
                ):
                    if path.exists():
                        uri = f"{RESULT_ROOT}/{path.name}"
                        bucket, key = _s3_parts(uri)
                        client.upload_file(str(path), bucket, key)
                        print(f"uploaded partial {uri}", flush=True)
            if GSM8K_PILOT == "1":
                for suffix in (".gsm8k.partial.json", ".gsm8k.routes.npz"):
                    partial_path = output_path.with_suffix(suffix)
                    if partial_path.exists():
                        partial_uri = f"{RESULT_ROOT}/rank-{global_rank}{suffix}"
                        bucket, key = _s3_parts(partial_uri)
                        client.upload_file(str(partial_path), bucket, key)
                        print(f"uploaded partial {partial_uri}", flush=True)
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
        if RL_ROLLOUT == "1":
            route_path = output_path.with_suffix(".routes.npz")
            route_uri = f"{RESULT_ROOT}/rank-{global_rank}.routes.npz"
            bucket, key = _s3_parts(route_uri)
            client.upload_file(str(route_path), bucket, key)
            print(f"uploaded {route_uri}", flush=True)
        if GSM8K_PILOT == "1" and NATURAL_TRACE == "0":
            route_path = output_path.with_suffix(".gsm8k.routes.npz")
            route_uri = f"{RESULT_ROOT}/rank-{global_rank}.gsm8k.routes.npz"
            bucket, key = _s3_parts(route_uri)
            client.upload_file(str(route_path), bucket, key)
            print(f"uploaded {route_uri}", flush=True)
        if NATURAL_TRACE != "0" and global_rank in TRACE_TARGETS:
            for mode in ("sample", "full"):
                trace_path = output_path.with_suffix(f".numeric.{mode}.npz")
                trace_uri = f"{RESULT_ROOT}/rank-{global_rank}.numeric.{mode}.npz"
                bucket, key = _s3_parts(trace_uri)
                client.upload_file(str(trace_path), bucket, key)
                print(f"uploaded {trace_uri}", flush=True)


def submit(iris_config: Path) -> None:
    """Submit the fixed-layout Hero qualification job through Iris."""
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
                "result_root": RESULT_ROOT,
                "weight_root": WEIGHT_ROOT,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    resources = ResourceConfig(
        cpu=64,
        ram="400g",
        disk="1t",
        device=GpuConfig(variant=HARDWARE, count=QUALIFICATION_GPUS_PER_TASK),
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
                "python", "infra/qualification/hero_rl_pilot.py"
            ),
            name=f"hero-vllm-qualification-{name_digest}",
            user="hero-vllm",
            resources=native_resources,
            replicas=QUALIFICATION_TASKS,
            environment=EnvironmentSpec(
                env_vars={
                    "HERO_QUALIFICATION_REVISION": revision,
                    "HERO_PILOT": PILOT,
                    "HERO_RL_ROLLOUT": RL_ROLLOUT,
                    "HERO_GSM8K_PILOT": GSM8K_PILOT,
                    "HERO_NATURAL_TRACE": NATURAL_TRACE,
                    "HERO_NUMERIC_PAD_KV_ROWS": str(PAD_KV_ROWS),
                    "HERO_HARDWARE": HARDWARE,
                    "HERO_RESULT_ROOT": RESULT_ROOT,
                    "HERO_GPU_MEMORY_UTILIZATION": str(GPU_MEMORY_UTILIZATION),
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
        args = parser.parse_args()
        submit(args.iris_config)
    else:
        main()
