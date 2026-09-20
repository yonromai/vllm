# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate the retained full-Hero B200 qualification results."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from hero_b200 import (
    PREFILL_DECODE_TARGET_TOLERANCE,
    PREFILL_DECODE_TOP_PROBABILITY_TOLERANCE,
    RESULT_ROOT,
    SAVED_TOP_PROBABILITY_TOLERANCE,
    TARGET_LOGPROB_TOLERANCE,
    WORLD_SIZE,
)


def _s3_parts(uri: str) -> tuple[str, str]:
    match = re.fullmatch(r"s3://([^/]+)/(.*)", uri)
    if match is None:
        raise ValueError(f"Not an S3 URI: {uri}")
    return match.group(1), match.group(2).rstrip("/")


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
        config=Config(s3={"addressing_style": "virtual"}),
    )


def _read_json(client, uri: str) -> dict[str, Any]:
    bucket, key = _s3_parts(uri)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def _put(client, uri: str, body: bytes, content_type: str) -> None:
    bucket, key = _s3_parts(uri)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        IfNoneMatch="*",
    )


def _finite(value: Any) -> bool:
    if isinstance(value, bool | str) or value is None:
        return True
    if isinstance(value, int | float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _worst_saved(mode: dict[str, Any]) -> tuple[float, int, int, int]:
    candidates = []
    for position in mode["positions"]:
        for top_index, error in enumerate(position["saved_top_probability_abs_errors"]):
            candidates.append(
                (
                    error,
                    position["prediction_position"],
                    top_index,
                    position["golden_top_token_ids"][top_index],
                )
            )
    return max(candidates)


def _mode_summary(mode: dict[str, Any]) -> dict[str, Any]:
    worst_saved_error, worst_saved_position, top_index, token_id = _worst_saved(mode)
    return {
        "max_target_logprob_abs_error": mode["max_target_logprob_abs_error"],
        "worst_target_prediction_position": mode["worst_target_prediction_position"],
        "max_saved_top_probability_abs_error": worst_saved_error,
        "worst_saved_top_prediction_position": worst_saved_position,
        "worst_saved_top_index": top_index,
        "worst_saved_top_token_id": token_id,
        "top1_mismatch_count": mode["top1_mismatch_count"],
        "ordered_top5_mismatch_count": mode["ordered_top5_mismatch_count"],
        "missing_saved_top_count": mode["missing_saved_top_count"],
        "position_count": len(mode["positions"]),
    }


def _paired_summary(
    left: dict[str, Any], right: dict[str, Any], *, left_name: str, right_name: str
) -> dict[str, Any]:
    left_positions = {row["prediction_position"]: row for row in left["positions"]}
    right_positions = {row["prediction_position"]: row for row in right["positions"]}
    if not left_positions or left_positions.keys() - right_positions.keys():
        raise ValueError(f"{left_name} positions are not covered by {right_name}")

    target_candidates = []
    saved_candidates = []
    for prediction_position, left_row in left_positions.items():
        right_row = right_positions[prediction_position]
        if left_row["golden_top_token_ids"] != right_row["golden_top_token_ids"]:
            raise ValueError(f"Saved top IDs differ at position {prediction_position}")
        target_candidates.append(
            (
                abs(
                    left_row["target_logprob_observed"]
                    - right_row["target_logprob_observed"]
                ),
                prediction_position,
            )
        )
        for top_index, (left_logprob, right_logprob, token_id) in enumerate(
            zip(
                left_row["saved_top_logprobs_observed"],
                right_row["saved_top_logprobs_observed"],
                left_row["golden_top_token_ids"],
                strict=True,
            )
        ):
            difference = (
                math.inf
                if left_logprob is None or right_logprob is None
                else abs(math.exp(left_logprob) - math.exp(right_logprob))
            )
            saved_candidates.append(
                (difference, prediction_position, top_index, token_id)
            )

    worst_target = max(target_candidates)
    worst_saved = max(saved_candidates)
    return {
        "left_mode": left_name,
        "right_mode": right_name,
        "position_count": len(left_positions),
        "max_target_logprob_abs_difference": worst_target[0],
        "worst_target_prediction_position": worst_target[1],
        "max_saved_top_probability_abs_difference": worst_saved[0],
        "worst_saved_top_prediction_position": worst_saved[1],
        "worst_saved_top_index": worst_saved[2],
        "worst_saved_top_token_id": worst_saved[3],
    }


def _gate(name: str, value: Any, expected: Any, relation: str) -> dict[str, Any]:
    if relation == "le":
        passed = value <= expected
    elif relation == "eq":
        passed = value == expected
    else:
        raise ValueError(relation)
    return {
        "name": name,
        "observed": value,
        "expected": expected,
        "relation": relation,
        "passed": passed,
    }


def _serving_config_provenance(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare shared settings and retain rank-specific capture instrumentation."""
    audit_fields = {"layer0_moe_audit", "short_prefix_route_audit_layers"}
    common = {
        key: value
        for key, value in results[0]["serving_config"].items()
        if key not in audit_fields
    }
    for result in results[1:]:
        other = {
            key: value
            for key, value in result["serving_config"].items()
            if key not in audit_fields
        }
        if other != common:
            raise ValueError("Ranks disagree on serving_config")
    provenance = {
        **common,
        "layer0_moe_audit_ranks": [
            result["case_index"]
            for result in results
            if result["serving_config"].get("layer0_moe_audit", False)
        ],
    }
    short_prefix_audits = {
        str(result["case_index"]): result["serving_config"].get(
            "short_prefix_route_audit_layers", []
        )
        for result in results
        if result["serving_config"].get("short_prefix_route_audit_layers")
    }
    if short_prefix_audits:
        provenance["short_prefix_route_audit_layers_by_rank"] = short_prefix_audits
    return provenance


def aggregate(results: list[dict[str, Any]], result_root: str) -> dict[str, Any]:
    results = sorted(results, key=lambda row: row["case_index"])
    if [row["case_index"] for row in results] != list(range(WORLD_SIZE)):
        raise ValueError("Expected exactly one result for each case 0..7")

    invariant_fields = (
        "checkpoint",
        "golden_root",
        "weight_root",
        "vllm_revision",
        "qualification_revision",
        "runtime",
        "input_evidence",
        "tolerances_fixed_before_run",
    )
    provenance = {field: results[0][field] for field in invariant_fields}
    for result in results[1:]:
        for field in invariant_fields:
            if result[field] != provenance[field]:
                raise ValueError(f"Ranks disagree on {field}")
    provenance["serving_config"] = _serving_config_provenance(results)

    cases = []
    gates = []
    for result in results:
        mode_summaries = {
            name: _mode_summary(result[key])
            for name, key in (
                ("short-prefill", "short"),
                ("prefill", "prefill"),
                ("cached-decode", "cached_decode"),
            )
        }
        prefill_decode = _paired_summary(
            result["cached_decode"],
            result["prefill"],
            left_name="cached-decode",
            right_name="prefill",
        )
        repeat_prefill = _paired_summary(
            result["short"],
            result["prefill"],
            left_name="short-prefill",
            right_name="prefill",
        )
        case_index = result["case_index"]
        case_prefix = f"case-{case_index}"
        for mode_name, summary in mode_summaries.items():
            gates.extend(
                (
                    _gate(
                        f"{case_prefix}/{mode_name}/target-logprob",
                        summary["max_target_logprob_abs_error"],
                        TARGET_LOGPROB_TOLERANCE,
                        "le",
                    ),
                    _gate(
                        f"{case_prefix}/{mode_name}/saved-top-probability",
                        summary["max_saved_top_probability_abs_error"],
                        SAVED_TOP_PROBABILITY_TOLERANCE,
                        "le",
                    ),
                    _gate(
                        f"{case_prefix}/{mode_name}/top1-mismatches",
                        summary["top1_mismatch_count"],
                        0,
                        "eq",
                    ),
                    _gate(
                        f"{case_prefix}/{mode_name}/missing-saved-top",
                        summary["missing_saved_top_count"],
                        0,
                        "eq",
                    ),
                )
            )
        for pair_name, pair in (
            ("prefill-vs-cached-decode", prefill_decode),
            ("short-vs-repeat-prefill", repeat_prefill),
        ):
            gates.extend(
                (
                    _gate(
                        f"{case_prefix}/{pair_name}/target-logprob",
                        pair["max_target_logprob_abs_difference"],
                        PREFILL_DECODE_TARGET_TOLERANCE,
                        "le",
                    ),
                    _gate(
                        f"{case_prefix}/{pair_name}/saved-top-probability",
                        pair["max_saved_top_probability_abs_difference"],
                        PREFILL_DECODE_TOP_PROBABILITY_TOLERANCE,
                        "le",
                    ),
                )
            )
        cases.append(
            {
                "case_index": case_index,
                "valid_length": result["valid_length"],
                "iris_task_id": result["iris_task_id"],
                "host": result["host"],
                "modes": mode_summaries,
                "prefill_vs_cached_decode": prefill_decode,
                "short_vs_repeat_prefill": repeat_prefill,
            }
        )

    all_finite = all(_finite(result) for result in results)
    gates.append(_gate("all-observations-finite", all_finite, True, "eq"))
    lengths = [case["valid_length"] for case in cases]
    required_lengths = [32, 128, 512, 2047, 2048, 2049, 4095, 4096]
    gates.append(
        _gate(
            "eight-distinct-required-lengths",
            sorted(lengths),
            required_lengths,
            "eq",
        )
    )
    job_ids = {result["iris_task_id"].rsplit("/", 1)[0] for result in results}
    gates.append(_gate("one-concurrent-dp-ep-job", len(job_ids), 1, "eq"))

    return {
        "passed": all(gate["passed"] for gate in gates),
        "result_root": result_root,
        **provenance,
        "case_count": len(cases),
        "mixed_length_evidence": {
            "valid_lengths": lengths,
            "iris_job_ids": sorted(job_ids),
            "note": (
                "All eight distinct lengths ran concurrently as the eight ranks "
                "of one data-parallel/expert-parallel group."
            ),
        },
        "repeat_state_evidence": {
            "note": (
                "Each rank ran short prefill, full prefill, then cached trace "
                "decode over recorded tail positions (and the 2,048 boundary "
                "for 4K cases) in one engine with prefix caching disabled."
            )
        },
        "cases": cases,
        "gates": gates,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full-Hero BF16 vLLM B200 qualification",
        "",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        (
            "| case | length | mode | max target Δlogp (position) | "
            "max saved-top Δp (position) | top1 misses |"
        ),
        "|---:|---:|---|---:|---:|---:|",
    ]
    for case in report["cases"]:
        for mode, summary in case["modes"].items():
            lines.append(
                f"| {case['case_index']} | {case['valid_length']} | {mode} | "
                f"{summary['max_target_logprob_abs_error']:.6g} "
                f"({summary['worst_target_prediction_position']}) | "
                f"{summary['max_saved_top_probability_abs_error']:.6g} "
                f"({summary['worst_saved_top_prediction_position']}) | "
                f"{summary['top1_mismatch_count']} |"
            )
    lines.extend(
        (
            "",
            (
                "| case | paired modes | max target Δlogp (position) | "
                "max saved-top Δp (position) |"
            ),
            "|---:|---|---:|---:|",
        )
    )
    for case in report["cases"]:
        for label, key in (
            ("prefill vs cached decode", "prefill_vs_cached_decode"),
            ("short vs repeat prefill", "short_vs_repeat_prefill"),
        ):
            pair = case[key]
            lines.append(
                f"| {case['case_index']} | {label} | "
                f"{pair['max_target_logprob_abs_difference']:.6g} "
                f"({pair['worst_target_prediction_position']}) | "
                f"{pair['max_saved_top_probability_abs_difference']:.6g} "
                f"({pair['worst_saved_top_prediction_position']}) |"
            )
    failed = [gate for gate in report["gates"] if not gate["passed"]]
    lines.extend(("", f"Failed gates: {len(failed)}", ""))
    lines.extend(
        f"- `{gate['name']}`: observed `{gate['observed']}`, expected "
        f"`{gate['relation']} {gate['expected']}`"
        for gate in failed
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    client = _client()
    results = [
        _read_json(client, f"{args.result_root}/rank-{rank}.json")
        for rank in range(WORLD_SIZE)
    ]
    report = aggregate(results, args.result_root)
    report_json = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    report_markdown = _markdown(report).encode()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_bytes(report_json)
    (args.output_dir / "report.md").write_bytes(report_markdown)
    if args.upload:
        _put(client, f"{args.result_root}/report.json", report_json, "application/json")
        _put(client, f"{args.result_root}/report.md", report_markdown, "text/markdown")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output_dir / 'report.md'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
