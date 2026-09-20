# SPDX-License-Identifier: Apache-2.0
"""Compare captured vLLM Hero expert IDs with the native golden routes."""

import argparse
import json
from pathlib import Path

import numpy as np


def _first_mismatch(mask: np.ndarray) -> dict[str, int] | None:
    hits = np.argwhere(mask)
    if not len(hits):
        return None
    position, layer = hits[0]
    return {"position": int(position), "layer": int(layer)}


def compare_routes(
    native: dict[str, np.ndarray], captured: dict[int, dict[str, np.ndarray]]
) -> dict:
    """Compare [layer, case, position, expert] to vLLM [position, layer, expert]."""
    native_routes = native["route_expert_ids"]
    if native_routes.ndim != 4:
        raise ValueError(f"Expected four native route axes, got {native_routes.shape}")
    results = {}
    for case, bundle in sorted(captured.items()):
        length = int(native["valid_lengths"][case])
        expected = native_routes[:, case, :length, :].transpose(1, 0, 2)
        observed = bundle.get("prefill", bundle.get("routed_experts"))
        if observed is None or observed.shape != expected.shape:
            raise ValueError(
                f"Case {case} routes have shape "
                f"{None if observed is None else observed.shape}; expected {expected.shape}"
            )
        ordered_mismatch = np.any(observed != expected, axis=-1)
        set_mismatch = np.any(
            np.sort(observed, axis=-1) != np.sort(expected, axis=-1), axis=-1
        )
        token_zero = np.flatnonzero(set_mismatch[0])
        layers_with_mismatch = np.flatnonzero(np.any(set_mismatch, axis=0))
        short = bundle.get("short")
        if short is not None and short.shape != observed[: len(short)].shape:
            raise ValueError(f"Case {case} short routes have shape {short.shape}")
        results[str(case)] = {
            "valid_length": length,
            "ordered_mismatch_count": int(np.count_nonzero(ordered_mismatch)),
            "expert_set_mismatch_count": int(np.count_nonzero(set_mismatch)),
            "first_ordered_mismatch": _first_mismatch(ordered_mismatch),
            "first_expert_set_mismatch": _first_mismatch(set_mismatch),
            "first_layer_with_set_mismatch": (
                int(layers_with_mismatch[0]) if len(layers_with_mismatch) else None
            ),
            "token_zero_first_set_mismatch_layer": (
                int(token_zero[0]) if len(token_zero) else None
            ),
            "position_2048_set_mismatch_layers": (
                np.flatnonzero(set_mismatch[2048]).astype(int).tolist()
                if length > 2048
                else None
            ),
            "short_vs_prefill_exact": (
                bool(np.array_equal(short, observed[: len(short)]))
                if short is not None
                else None
            ),
        }
    paired = None
    if 6 in captured and 7 in captured:
        left = captured[6].get("prefill", captured[6].get("routed_experts"))
        right = captured[7].get("prefill", captured[7].get("routed_experts"))
        paired = bool(np.array_equal(left, right[: len(left)]))
    return {"cases": results, "case_6_vs_7_common_prefix_exact": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-arrays", type=Path, required=True)
    parser.add_argument("--captured-ranks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.native_arrays) as source:
        native = {name: source[name] for name in source.files}
    captured = {}
    for path in sorted(args.captured_ranks.glob("rank-*.prefill-routes.npz")):
        case = int(path.name.split(".")[0].removeprefix("rank-"))
        with np.load(path) as source:
            captured[case] = {name: source[name] for name in source.files}
    if not captured:
        raise ValueError(f"No captured routes in {args.captured_ranks}")
    report = compare_routes(native, captured)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
