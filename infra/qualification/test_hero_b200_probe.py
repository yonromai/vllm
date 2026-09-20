"""Checks for the full-prefill versus cached-decode embedding capture."""

from pathlib import Path

import numpy as np
from hero_b200 import _collect_embedding_history


def test_embedding_history_joins_decode_prefix_and_steps(tmp_path: Path):
    output = tmp_path / "rank-3.json"
    history = tmp_path / "rank-3.embedding-history"
    history.mkdir()
    tokens = list(range(12))
    sites = ("embed_raw", "embed_after_norm", "embed_gate_down", "model_input")

    def save(name: str, positions: np.ndarray, offset: float) -> None:
        arrays = {
            "positions": positions,
            "token_ids": np.asarray(tokens)[positions],
        }
        for site in sites:
            arrays[site] = (positions + offset)[:, None].astype(np.float32)
        np.savez_compressed(history / f"{name}.npz", **arrays)

    save("full", np.arange(12), 0)
    save("prefix", np.arange(10), 100)
    save("step-10", np.asarray([10]), 100)
    save("step-11", np.asarray([11]), 100)

    aggregate = _collect_embedding_history(output, tokens, 10, 11)

    with np.load(aggregate, allow_pickle=False) as captured:
        np.testing.assert_array_equal(captured["positions"], np.arange(12))
        np.testing.assert_array_equal(captured["token_ids"], tokens)
        for site in sites:
            np.testing.assert_array_equal(
                captured[f"full_{site}"].flatten(), np.arange(12)
            )
            np.testing.assert_array_equal(
                captured[f"decode_{site}"].flatten(), np.arange(12) + 100
            )
