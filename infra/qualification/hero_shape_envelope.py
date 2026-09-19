# SPDX-License-Identifier: Apache-2.0
"""Audit Hero score gates against native same-prefix shape variation."""

import argparse
import json
from pathlib import Path

import numpy as np


class NativeEnvelope:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self._matches: dict[tuple[int, int], tuple[int, ...]] = {}
        self.scores = {
            (int(case), int(position)): index
            for index, (case, position) in enumerate(
                zip(
                    arrays["score_case_indices"],
                    arrays["prediction_positions"],
                    strict=True,
                )
            )
        }
        self.full_logits = {
            (int(case), int(position)): index
            for index, (case, position) in enumerate(
                zip(
                    arrays["full_logit_case_indices"],
                    arrays["full_logit_prediction_positions"],
                    strict=True,
                )
            )
        }

    def matched_scores(self, case: int, position: int) -> tuple[int, ...]:
        cached = self._matches.get((case, position))
        if cached is not None:
            return cached
        arrays = self.arrays
        fields = ("tokens", "token_validity", "segment_ids", "positions")
        matches = []
        for other in range(arrays["tokens"].shape[0]):
            index = self.scores.get((other, position))
            if index is None:
                continue
            if all(
                np.array_equal(
                    arrays[field][case, : position + 1],
                    arrays[field][other, : position + 1],
                )
                for field in fields
            ):
                matches.append(index)
        if not matches:
            raise ValueError(f"No native score for case {case}, position {position}")
        result = tuple(matches)
        self._matches[(case, position)] = result
        return result

    def target_values(self, case: int, position: int, token: int) -> tuple[float, ...]:
        arrays = self.arrays
        indices = self.matched_scores(case, position)
        values = tuple(
            float(arrays["target_logprobs"][index])
            for index in indices
            if int(arrays["target_token_ids"][index]) == token
        )
        if not values:
            raise ValueError(
                f"No native target {token} at case {case}, position {position}"
            )
        return values

    def top_probabilities(
        self, case: int, position: int, token: int
    ) -> tuple[float, ...]:
        arrays = self.arrays
        values = []
        for index in self.matched_scores(case, position):
            hits = np.flatnonzero(arrays["top_token_ids"][index] == token)
            if hits.size:
                values.append(float(np.exp(arrays["top_logprobs"][index, hits[0]])))
                continue
            other_case = int(arrays["score_case_indices"][index])
            logits_index = self.full_logits.get((other_case, position))
            if logits_index is None:
                continue
            logits = arrays["full_logits"][logits_index].astype(np.float64)
            shifted = logits - np.max(logits)
            values.append(float(np.exp(shifted[token]) / np.exp(shifted).sum()))
        if not values:
            raise ValueError(
                f"No native top token {token} at case {case}, position {position}"
            )
        return tuple(values)


def _outside(value: float, reference: tuple[float, ...]) -> float:
    return max(min(reference) - value, value - max(reference), 0.0)


def _span(reference: tuple[float, ...]) -> float:
    return max(reference) - min(reference)


def _mode_metrics(native: NativeEnvelope, case: int, mode: dict) -> tuple[float, float]:
    max_target = 0.0
    max_top_probability = 0.0
    for row in mode["positions"]:
        position = int(row["prediction_position"])
        target = int(row["target_token_id"])
        max_target = max(
            max_target,
            _outside(
                float(row["target_logprob_observed"]),
                native.target_values(case, position, target),
            ),
        )
        for token, observed in zip(
            row["golden_top_token_ids"],
            row["saved_top_logprobs_observed"],
            strict=True,
        ):
            max_top_probability = max(
                max_top_probability,
                _outside(
                    float(np.exp(observed)),
                    native.top_probabilities(case, position, int(token)),
                ),
            )
    return max_target, max_top_probability


def _pair_metrics(
    native: NativeEnvelope, case: int, left: dict, right: dict
) -> tuple[float, float]:
    right_by_position = {
        int(row["prediction_position"]): row for row in right["positions"]
    }
    max_target = 0.0
    max_top_probability = 0.0
    for left_row in left["positions"]:
        position = int(left_row["prediction_position"])
        right_row = right_by_position.get(position)
        if right_row is None:
            continue
        target = int(left_row["target_token_id"])
        if target != int(right_row["target_token_id"]):
            raise ValueError(f"Target changed at case {case}, position {position}")
        max_target = max(
            max_target,
            abs(
                float(left_row["target_logprob_observed"])
                - float(right_row["target_logprob_observed"])
            )
            - _span(native.target_values(case, position, target)),
        )
        left_top = dict(
            zip(
                left_row["golden_top_token_ids"],
                left_row["saved_top_logprobs_observed"],
                strict=True,
            )
        )
        right_top = dict(
            zip(
                right_row["golden_top_token_ids"],
                right_row["saved_top_logprobs_observed"],
                strict=True,
            )
        )
        for token in left_top.keys() & right_top.keys():
            max_top_probability = max(
                max_top_probability,
                abs(float(np.exp(left_top[token]) - np.exp(right_top[token])))
                - _span(native.top_probabilities(case, position, int(token))),
            )
    return max(max_target, 0.0), max(max_top_probability, 0.0)


def evaluate(native: NativeEnvelope, report: dict, ranks: list[dict]) -> dict:
    adjusted = {}
    for rank in ranks:
        case = int(rank["case_index"])
        for mode_name, source_name in (
            ("short-prefill", "short"),
            ("prefill", "prefill"),
            ("cached-decode", "cached_decode"),
        ):
            target, top_probability = _mode_metrics(native, case, rank[source_name])
            adjusted[f"case-{case}/{mode_name}/target-logprob"] = target
            adjusted[f"case-{case}/{mode_name}/saved-top-probability"] = top_probability
        for pair_name, left, right in (
            ("prefill-vs-cached-decode", "prefill", "cached_decode"),
            ("short-vs-repeat-prefill", "short", "prefill"),
        ):
            target, top_probability = _pair_metrics(
                native, case, rank[left], rank[right]
            )
            adjusted[f"case-{case}/{pair_name}/target-logprob"] = target
            adjusted[f"case-{case}/{pair_name}/saved-top-probability"] = top_probability
    gates = []
    for gate in report["gates"]:
        changed = gate["name"] in adjusted
        observed = adjusted.get(gate["name"], gate["observed"])
        expected = gate["expected"]
        passed = (
            observed <= expected if gate["relation"] == "le" else observed == expected
        )
        gates.append(
            {
                "name": gate["name"],
                "observed": observed,
                "original_observed": gate["observed"],
                "expected": expected,
                "passed": passed,
                "shape_adjusted": changed,
            }
        )
    return {
        "original_failed": [
            gate["name"] for gate in report["gates"] if not gate["passed"]
        ],
        "adjusted_failed": [gate["name"] for gate in gates if not gate["passed"]],
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-arrays", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ranks", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.native_arrays) as arrays:
        native = NativeEnvelope({key: arrays[key] for key in arrays.files})
    report = json.loads(args.report.read_text())
    ranks = [
        json.loads((args.ranks / f"rank-{rank}.json").read_text()) for rank in range(8)
    ]
    print(json.dumps(evaluate(native, report, ranks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
