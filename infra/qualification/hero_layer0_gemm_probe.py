# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental one-H100 replay of Hero's first differing BF16 GEMM row."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np
from hero_rl_pilot import QUALIFICATION_SETUP

TRACE_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/hero-4k-numerical-01a0bca4/"
    "h100-rank15-natural-trace-a8a2c3087"
)
WEIGHT_URI = (
    "s3://marin-us-east-02a/marin/users/romain/hero-vllm-b200/"
    "hero-535b-step108000-bf16-split-v3/model-layer-000.safetensors"
)
RESULT_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/hero-4k-numerical-01a0bca4/"
    "h100-rank15-key-gemm"
)
WEIGHT_NAME = "model.layers.0.self_attn.k_proj.weight"
WEIGHT_SHA256 = "6f64f41853be63e22fa85b9fd6c2bf329837fb2263b1417b46696d267b5e1e8c"
CASES = ((15, "sample", "full"),)


def _s3_parts(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"s3://([^/]+)/(.*)", uri)
    if match is None:
        raise ValueError(f"Not an S3 URI: {uri}")
    return match.group(1), match.group(2).rstrip("/")


def run() -> None:
    import boto3
    import torch
    import torch.nn.functional as F
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url="https://cwobject.com",
        config=Config(s3={"addressing_style": "virtual"}),
    )
    bucket, key = _s3_parts(WEIGHT_URI)

    def read_range(start: int, end: int) -> bytes:
        value = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")[
            "Body"
        ].read()
        if len(value) != end - start + 1:
            raise ValueError("Incomplete S3 range")
        return value

    header_size = struct.unpack("<Q", read_range(0, 7))[0]
    header = json.loads(read_range(8, header_size + 7))
    info = header[WEIGHT_NAME]
    if info["dtype"] != "BF16" or info["shape"] != [1536, 6144]:
        raise ValueError("Unexpected layer-0 key-projection weight")
    start, end = info["data_offsets"]
    weight_bytes = read_range(header_size + 8 + start, header_size + 7 + end)
    if hashlib.sha256(weight_bytes).hexdigest() != WEIGHT_SHA256:
        raise ValueError("Layer-0 weight hash differs from retained export")
    weight_bits = np.frombuffer(weight_bytes, dtype="<u2").copy()
    weight = (
        torch.from_numpy(weight_bits)
        .view(torch.bfloat16)
        .reshape(1536, 6144)
        .to("cuda")
    )

    rows = []
    for rank, left_mode, right_mode in CASES:
        modes = {}
        for mode in (left_mode, right_mode):
            trace_uri = f"{TRACE_ROOT}/rank-{rank}.numeric.{mode}.npz"
            trace_bucket, trace_key = _s3_parts(trace_uri)
            path = Path(f"/tmp/hero-rank-{rank}-{mode}.npz")
            s3.download_file(trace_bucket, trace_key, str(path))
            with np.load(path, allow_pickle=False) as trace:
                position = int(trace["position"])
                first = int(trace["first_forward_position"])
                batch_size = int(trace["forward_tokens"])
                row = position - first
                if not 0 <= row < batch_size:
                    raise ValueError("Trace position outside forward batch")
                input_row = trace["layer_0_attention_input"].copy()
                observed = trace["layer_0_k_projection"].copy()
            x = torch.from_numpy(input_row).to(device="cuda", dtype=torch.bfloat16)
            variants = {}
            for fill in ("zeros", "repeat"):
                batch = torch.zeros(
                    (batch_size, 6144), device="cuda", dtype=torch.bfloat16
                )
                if fill == "repeat":
                    batch[:] = x
                else:
                    batch[row] = x
                output = F.linear(batch, weight)[row].float().cpu().numpy()
                variants[fill] = {
                    "equal_elements": int(np.count_nonzero(output == observed)),
                    "max_abs_delta": float(np.max(np.abs(output - observed))),
                    "output": output,
                }
            modes[mode] = {
                "trace_uri": trace_uri,
                "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "position": position,
                "forward_tokens": batch_size,
                "row": row,
                "input_row": input_row,
                "observed": observed,
                "variants": variants,
            }
        left = modes[left_mode]
        right = modes[right_mode]
        if not np.array_equal(left["input_row"], right["input_row"]):
            raise ValueError("Key-projection inputs differ across serving modes")
        observed_different = left["observed"] != right["observed"]
        reproduced_different = (
            left["variants"]["zeros"]["output"] != right["variants"]["zeros"]["output"]
        )
        rows.append(
            {
                "rank": rank,
                "position": left["position"],
                "modes": {
                    mode: {
                        key: value
                        for key, value in data.items()
                        if key not in {"input_row", "observed", "variants"}
                    }
                    | {
                        "variants": {
                            fill: {
                                key: value
                                for key, value in variant.items()
                                if key != "output"
                            }
                            for fill, variant in data["variants"].items()
                        }
                    }
                    for mode, data in modes.items()
                },
                "observed_different_elements": int(
                    np.count_nonzero(observed_different)
                ),
                "reproduced_different_elements": int(
                    np.count_nonzero(reproduced_different)
                ),
                "different_element_masks_equal": bool(
                    np.array_equal(observed_different, reproduced_different)
                ),
                "both_modes_exactly_reproduced": bool(
                    left["variants"]["zeros"]["equal_elements"] == 1536
                    and right["variants"]["zeros"]["equal_elements"] == 1536
                ),
            }
        )
    result = {
        "source_revision": __import__("os").environ["HERO_PROBE_REVISION"],
        "weight_uri": WEIGHT_URI,
        "weight_name": WEIGHT_NAME,
        "weight_sha256": WEIGHT_SHA256,
        "trace_root": TRACE_ROOT,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "matrix_operation": "torch.nn.functional.linear with BF16 input and weight",
        "rows": rows,
    }
    output_bucket, output_key = _s3_parts(f"{RESULT_ROOT}/report.json")
    s3.put_object(
        Bucket=output_bucket,
        Key=output_key,
        Body=(json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
        IfNoneMatch="*",
    )
    print(json.dumps({"result_uri": f"{RESULT_ROOT}/report.json", "rows": rows}))


def submit(iris_config: Path) -> None:
    from fray.iris_backend import convert_constraints, convert_resources
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
    resources = ResourceConfig(
        cpu=16,
        ram="80g",
        disk="100g",
        device=GpuConfig(variant="H100", count=1),
        replicas=1,
    )
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
                "python", "infra/qualification/hero_layer0_gemm_probe.py"
            ),
            name=f"hero-layer0-gemm-{revision[:12]}",
            user="hero-vllm",
            resources=convert_resources(resources),
            replicas=1,
            environment=EnvironmentSpec(
                env_vars={"HERO_PROBE_REVISION": revision, "PYTHONUNBUFFERED": "1"},
                setup_scripts=[QUALIFICATION_SETUP],
            ),
            constraints=convert_constraints(resources),
            timeout=Duration.from_hours(2),
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            priority_band=priority_band_value("interactive"),
            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
        )
    print(job.job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=("run", "submit"), default="run")
    parser.add_argument("--iris-config", type=Path)
    args = parser.parse_args()
    if args.action == "submit":
        if args.iris_config is None:
            parser.error("--iris-config is required for submit")
        submit(args.iris_config)
    else:
        run()
