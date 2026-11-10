"""Shared by the tests: the repository's protocol and grid, and start-state audits as a camera might measure them."""

from functools import cache

from tolerance_radius import config, layouts
from tolerance_radius.geometry import Grid
from tolerance_radius.schema import DemoStart, Point
from tolerance_radius.seeds import derived_rng

PROTOCOL = config.load_protocol()
GRID = Grid.from_protocol(PROTOCOL)


@cache
def audit(dataset: str, noise_mm: float = 1.5, seed: int = 5) -> tuple[DemoStart, ...]:
    """Every demonstration of a set, measured up to `noise_mm` away from its planned start in x and in y.

    A stand-in for the camera audit: the cube is never exactly on its target, so the five episodes of a clustered cell
    start at five slightly different places.
    """
    rng = derived_rng(seed, "audit", dataset)
    starts = []
    for episode in layouts.layout(dataset, PROTOCOL):
        dx, dy = ((rng.random(2) * 2.0 - 1.0) * noise_mm).tolist()
        measured = Point(episode.target.x_mm + dx, episode.target.y_mm + dy)
        start = DemoStart(dataset, episode.episode_index, episode.cell, measured=measured, planned=episode.target)
        starts.append(start)
    return tuple(starts)


def demos_at(*positions: tuple[float, float], dataset: str = "clustered") -> list[DemoStart]:
    """Demonstrations measured exactly at the given positions."""
    return [
        DemoStart(dataset, index, GRID.cell_of(Point(x, y)) or "A2", measured=Point(x, y))
        for index, (x, y) in enumerate(positions)
    ]
