# SPDX-License-Identifier: Apache-2.0

import numpy as np

from infra.qualification.hero_full_logit_audit import native_bundle_parity


def test_native_bundle_parity_checks_scores_and_reordered_full_rows() -> None:
    original = {
        "target_logprobs": np.array([-1.0], dtype=np.float32),
        "route_expert_ids": np.array([3], dtype=np.int32),
        "full_logit_case_indices": np.array([6], dtype=np.int32),
        "full_logit_prediction_positions": np.array([2048], dtype=np.int32),
        "full_logits": np.array([[1.0, 2.0]], dtype=np.float32),
    }
    diagnostic = {
        **original,
        "full_logit_case_indices": np.array([6, 6], dtype=np.int32),
        "full_logit_prediction_positions": np.array([2047, 2048], dtype=np.int32),
        "full_logits": np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32),
    }

    assert native_bundle_parity(original, diagnostic)["all_original_fields_identical"]
    assert native_bundle_parity(original, diagnostic)[
        "all_original_full_rows_identical"
    ]

    diagnostic["target_logprobs"] = np.array([-1.1], dtype=np.float32)
    assert not native_bundle_parity(original, diagnostic)[
        "all_original_fields_identical"
    ]
