# SPDX-License-Identifier: Apache-2.0
"""Compare retained Hero full-vocabulary logits and verify score-capture parity."""

import argparse
import json
from pathlib import Path

import numpy as np


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _distribution_metrics(native: np.ndarray, observed: np.ndarray) -> dict:
    p = _probabilities(native)
    q = _probabilities(observed)
    return {
        "top1_native": int(np.argmax(p)),
        "top1_observed": int(np.argmax(q)),
        "total_variation": float(np.abs(p - q).sum() / 2),
        "max_probability_difference": float(np.abs(p - q).max()),
        "kl_native_to_observed": float(np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0]))),
    }


def _matching_native_logits(
    arrays: dict[str, np.ndarray], case: int, position: int
) -> list[int]:
    fields = ("tokens", "token_validity", "segment_ids", "positions")
    return [
        index
        for index, (other, other_position) in enumerate(
            zip(
                arrays["full_logit_case_indices"],
                arrays["full_logit_prediction_positions"],
                strict=True,
            )
        )
        if int(other_position) == position
        and all(
            np.array_equal(
                arrays[field][case, : position + 1],
                arrays[field][int(other), : position + 1],
            )
            for field in fields
        )
    ]


def audit(
    native_arrays: dict[str, np.ndarray],
    original_ranks: list[dict],
    captured_ranks: list[dict],
    captured_logits: list[dict[str, np.ndarray]],
) -> dict:
    score_parity = {}
    identity_parity = {}
    for case, (original, captured) in enumerate(
        zip(original_ranks, captured_ranks, strict=True)
    ):
        score_parity[str(case)] = {
            mode: original[mode] == captured[mode]
            for mode in ("short", "prefill", "cached_decode")
        }
        identity_parity[str(case)] = {
            field: original[field] == captured[field]
            for field in (
                "checkpoint",
                "golden_root",
                "weight_root",
                "vllm_revision",
                "input_evidence",
                "serving_config",
                "tolerances_fixed_before_run",
            )
        }

    positions = []
    for index, (case, position) in enumerate(
        zip(
            native_arrays["full_logit_case_indices"],
            native_arrays["full_logit_prediction_positions"],
            strict=True,
        )
    ):
        case = int(case)
        position = int(position)
        captured = captured_logits[case]
        hits = np.flatnonzero(captured["positions"] == position)
        if len(hits) != 1:
            raise ValueError(f"Expected one captured logit row for {case}/{position}")
        alternatives = _matching_native_logits(native_arrays, case, position)
        observed_probabilities = _probabilities(captured["logits"][hits[0]])
        scored_row = next(
            row
            for row in captured_ranks[case]["prefill"]["positions"]
            if int(row["prediction_position"]) == position
        )
        target = int(scored_row["target_token_id"])
        comparisons = [
            {
                "native_case": int(native_arrays["full_logit_case_indices"][other]),
                **_distribution_metrics(
                    native_arrays["full_logits"][other],
                    captured["logits"][hits[0]],
                ),
            }
            for other in alternatives
        ]
        native_variation_tv = max(
            (
                _distribution_metrics(
                    native_arrays["full_logits"][left],
                    native_arrays["full_logits"][right],
                )["total_variation"]
                for left in alternatives
                for right in alternatives
            ),
            default=0.0,
        )
        positions.append(
            {
                "case": case,
                "position": position,
                "api_target_logprob_difference": float(
                    np.log(observed_probabilities[target])
                    - float(scored_row["target_logprob_observed"])
                ),
                "same_shape": comparisons[alternatives.index(index)],
                "closest_native": min(
                    comparisons, key=lambda value: value["total_variation"]
                ),
                "native_shape_variation_tv": native_variation_tv,
            }
        )
    return {
        "score_parity": score_parity,
        "identity_parity": identity_parity,
        "all_scores_identical": all(
            same for modes in score_parity.values() for same in modes.values()
        ),
        "all_inputs_identical": all(
            same for fields in identity_parity.values() for same in fields.values()
        ),
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-arrays", type=Path, required=True)
    parser.add_argument("--original-ranks", type=Path, required=True)
    parser.add_argument("--captured-ranks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.native_arrays) as source:
        native = {name: source[name] for name in source.files}
    original = [
        json.loads((args.original_ranks / f"rank-{rank}.json").read_text())
        for rank in range(8)
    ]
    captured = [
        json.loads((args.captured_ranks / f"rank-{rank}.json").read_text())
        for rank in range(8)
    ]
    logits = []
    for rank in range(8):
        with np.load(args.captured_ranks / f"rank-{rank}.full-logits.npz") as source:
            logits.append({name: source[name] for name in source.files})
    result = json.dumps(audit(native, original, captured, logits), indent=2) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result)


if __name__ == "__main__":
    main()
