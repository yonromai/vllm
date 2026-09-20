# SPDX-License-Identifier: Apache-2.0
"""Small route-axis and mismatch tests for the full-Hero audit."""

import numpy as np

from infra.qualification.hero_route_audit import compare_routes


def test_order_only_and_expert_set_mismatch() -> None:
    routes = np.tile(np.array([0, 1], dtype=np.int32), (2, 2, 4, 1))
    native = {"route_expert_ids": routes, "valid_lengths": np.array([4, 4])}
    observed = routes[:, 0].transpose(1, 0, 2).copy()
    observed[1, 0] = [1, 0]
    observed[2, 1] = [1, 2]
    result = compare_routes(native, {0: {"prefill": observed, "short": observed[:2]}})
    case = result["cases"]["0"]
    assert case["ordered_mismatch_count"] == 2
    assert case["expert_set_mismatch_count"] == 1
    assert case["first_ordered_mismatch"] == {"position": 1, "layer": 0}
    assert case["first_expert_set_mismatch"] == {"position": 2, "layer": 1}
    assert case["first_layer_with_set_mismatch"] == 1
    assert case["token_zero_first_set_mismatch_layer"] is None
    assert case["short_vs_prefill_exact"]
