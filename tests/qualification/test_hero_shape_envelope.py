# SPDX-License-Identifier: Apache-2.0

import numpy as np

from infra.qualification.hero_shape_envelope import NativeEnvelope, evaluate


def _native_arrays() -> dict[str, np.ndarray]:
    return {
        "tokens": np.array([[42, 1, 0], [42, 1, 2]], dtype=np.int32),
        "token_validity": np.array(
            [[True, True, False], [True, True, True]], dtype=np.bool_
        ),
        "segment_ids": np.zeros((2, 3), dtype=np.int32),
        "positions": np.array([[0, 1, 0], [0, 1, 2]], dtype=np.int32),
        "score_case_indices": np.array([0, 1], dtype=np.int32),
        "prediction_positions": np.array([0, 0], dtype=np.int32),
        "target_token_ids": np.array([7, 7], dtype=np.int32),
        "target_logprobs": np.array([-1.0, -1.2], dtype=np.float32),
        "top_token_ids": np.array([[10, 11], [10, 11]], dtype=np.int32),
        "top_logprobs": np.array([[-0.5, -1.0], [-0.6, -1.1]], dtype=np.float32),
        "full_logit_case_indices": np.empty(0, dtype=np.int32),
        "full_logit_prediction_positions": np.empty(0, dtype=np.int32),
        "full_logits": np.empty((0, 16), dtype=np.float32),
    }


def test_native_envelope_matches_semantic_prefix_only() -> None:
    arrays = _native_arrays()
    native = NativeEnvelope(arrays)
    assert native.target_values(0, 0, 7) == (-1.0, -1.2000000476837158)

    changed_segment = _native_arrays()
    changed_segment["segment_ids"][1, 0] = 1
    assert NativeEnvelope(changed_segment).target_values(0, 0, 7) == (-1.0,)


def test_shape_envelope_keeps_top1_gate_while_accepting_native_range() -> None:
    row = {
        "prediction_position": 0,
        "target_token_id": 7,
        "target_logprob_observed": -1.18,
        "golden_top_token_ids": [10, 11],
        "saved_top_logprobs_observed": [-0.55, -1.05],
    }
    modes = {"positions": [row]}
    rank = {
        "case_index": 0,
        "short": modes,
        "prefill": modes,
        "cached_decode": modes,
    }
    report = {
        "gates": [
            {
                "name": "case-0/prefill/target-logprob",
                "observed": 0.18,
                "expected": 0.05,
                "relation": "le",
                "passed": False,
            },
            {
                "name": "case-0/prefill/top1-mismatches",
                "observed": 1,
                "expected": 0,
                "relation": "eq",
                "passed": False,
            },
        ]
    }
    result = evaluate(NativeEnvelope(_native_arrays()), report, [rank])
    assert result["adjusted_failed"] == ["case-0/prefill/top1-mismatches"]
    assert result["gates"][0]["observed"] == 0.0
