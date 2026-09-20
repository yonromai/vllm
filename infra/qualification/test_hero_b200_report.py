"""Checks for rank-specific Hero capture settings in qualification reports."""

import pytest
from hero_b200_report import _serving_config_provenance


def test_rank_specific_audits_are_preserved_without_changing_shared_settings():
    results = [
        {
            "case_index": rank,
            "serving_config": {
                "dtype": "bfloat16",
                "layer0_moe_audit": rank >= 6,
                "short_prefix_route_audit_layers": [2, 12, 38] if rank >= 6 else [],
            },
        }
        for rank in range(8)
    ]

    config = _serving_config_provenance(results)

    assert config == {
        "dtype": "bfloat16",
        "layer0_moe_audit_ranks": [6, 7],
        "short_prefix_route_audit_layers_by_rank": {"6": [2, 12, 38], "7": [2, 12, 38]},
    }


def test_shared_serving_setting_difference_is_rejected():
    results = [
        {
            "case_index": 0,
            "serving_config": {"dtype": "bfloat16", "layer0_moe_audit": False},
        },
        {
            "case_index": 1,
            "serving_config": {"dtype": "float16", "layer0_moe_audit": True},
        },
    ]

    with pytest.raises(ValueError, match="Ranks disagree on serving_config"):
        _serving_config_provenance(results)
