"""Shared by the figure tests: the synthetic results files, and small changes made to them in memory.

The files under `tests/fixtures/results/` are whole imagined experiments in the shape of the analysis results
(`protocol/data-contracts.md`), made by `tests/fixtures/make_results_fixtures.py` from simulated trials. They are drafts,
they say that they are synthetic, and nothing in them is a result.
"""

import copy
import json
from functools import cache
from pathlib import Path

FOLDER = Path(__file__).resolve().parent / "fixtures" / "results"

#: The files, by what each is there to show.
CASES = {
    "base": "every block ran; 20 placements in trained cells; distances corrected for scale",
    "no_spread_threshold": "the spread policy has no block fit; distances as read",
    "open_above": "the clustered policy's interval is open above",
    "threshold_blocks_only": "no workspace test; the replay's interval is open below",
    "held_out_placements": "13 placements in trained cells and 13 in held-out cells, counted by region",
}


@cache
def _read(name: str) -> str:
    return (FOLDER / f"{name}.json").read_text(encoding="utf-8")


def path(name: str) -> Path:
    """Where a case's file is."""
    return FOLDER / f"{name}.json"


def load(name: str = "base") -> dict:
    """A case's results, as a fresh copy that a test may change."""
    return json.loads(_read(name))


def changed(name: str = "base", **paths) -> dict:
    """A case's results with some fields replaced: `changed("base", **{"workspace.lookup": None})`."""
    results = load(name)
    for dotted, value in paths.items():
        target = results
        *parents, last = dotted.split(".")
        for key in parents:
            target = target[key]
        target[last] = copy.deepcopy(value)
    return results


def contract_only(name: str = "base") -> dict:
    """A case's results without the fields a figure can use that the contract's example does not have: what the
    distances were divided by, each policy's own placement in a pair, a pair's region, and the counts by region.

    A figure must draw from this as well: those fields are used when they are there and done without otherwise.
    """
    results = load(name)
    results["distance_scale"].pop("divisor", None)
    workspace = results["workspace"]
    if workspace is not None:
        workspace.pop("by_region", None)
        for pair in workspace["pairs"]:
            pair.pop("region", None)
            for policy in ("act_clustered", "act_spread"):
                pair[policy].pop("x_mm", None)
                pair[policy].pop("y_mm", None)
    return results
