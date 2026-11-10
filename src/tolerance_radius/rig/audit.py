"""The start-state audit: where the cube really was at the start of every recorded episode.

Every distance in the study is taken to these measured positions, never to the plan. The audit reads the first frame
of each episode from the recorded dataset, measures the cube with the same detector that placed it, and compares the
result with the layout's target. An episode that cannot be measured, that sits in the wrong cell or that is farther
from its target than the limit is a problem to be looked at before training, not a row to be dropped quietly.

The table frame is taken from the two floor tags as each episode's own first frame shows them. The camera of this
rig moves by a millimetre or two in an hour (its pipe hangs on the light box's walls) and the tags are taped to the
floor: through one conversion for the whole dataset, every start would carry the camera's movement since that
conversion was made. How far the camera's picture stood from the stored calibration, and how bright the picture was,
is written down for every episode.

The measuring part (`audit_frames`) takes pictures and a plan and has no hardware or dataset dependency, so it is
tested on synthetic pictures. `audit_dataset` is the thin part that reads the frames from a LeRobot dataset on disk.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from tolerance_radius.geometry import Grid
from tolerance_radius.rig import lerobot101
from tolerance_radius.rig.vision import (
    TableView,
    VisionError,
    _fit_residual_mm,
    floor_levels,
    grid_probes_px,
    WRIST_OTHER_ABOVE,
    WRIST_STEP,
    load_geometry,
    tag_outlines_over,
    wrist_contrast,
    wrist_levels,
    wrist_reads,  # noqa: F401 -- for the audit's script, which prints the two states' readings
)
from tolerance_radius.schema import DemoStart, Point

#: A measured start farther than this from its planned target is reported. The placement was accepted within the
#: protocol's tolerance by the same instrument; the margin allows for the difference between a live frame and a
#: recorded, compressed one.
DEFAULT_LIMIT_MM = 3.0
#: The camera's picture may stand this far from the stored calibration's in an episode, as for a session
#: (`camera_tool.VIEW_MOVED_LIMIT_MM`): beyond it the episode was recorded through another view than the rest.
VIEW_LIMIT_MM = 5.0
#: An episode's frame is a single one, and compressed. Its tags' outline corners are averaged with those of the
#: episodes recorded next to it, this many either side, as long as theirs lie within `SAME_VIEW_PX` of its own (the
#: largest distance between two corners that should be the same one): the camera drifts slowly inside a sitting and
#: may jump between two.
NEIGHBOURS = 2
SAME_VIEW_PX = 0.4
#: Inside a take the front camera's picture is looked at once in this many frames (once a second) and at its end.
THROUGH_EVERY = 30
#: The floor tags' outline corners may lie this far, in pixels, from where the take's first frame has them. The
#: camera's creep and what one compressed frame's corners are worth are well under a pixel; the camera gone back to
#: its own zoom moved them by 70 (episode 3 of the set that was put aside, 2026-10-06).
TAKE_TAGS_PX = 3.0
#: ... and the floor tags' black may be this share of what it is in the take's first frame. It is 28 to 32 in the
#: takes recorded with nothing wrong; with the tone range stretched it is 18 to 20, with the exposure back on
#: automatic 86. The floor's own level is not used: the arm's shadow lowers it by a tenth while it is over the grid.
TAKE_BLACK_RANGE = (0.7, 1.4)

@dataclass(frozen=True, slots=True)
class AuditResult:
    """What the audit found for one dataset."""

    dataset: str
    #: One entry per episode whose cube was measured, in episode order.
    starts: tuple[DemoStart, ...]
    #: Plain sentences, one per thing that needs a person's eyes. Empty means the dataset is as planned.
    problems: tuple[str, ...]
    #: Distance from each measured start to its planned target, by episode index.
    errors_mm: dict[int, float]
    #: The largest distance, over the episodes, between where an episode's own frame puts the grid and where the
    #: stored calibration puts it; None when the tags were not solved from the frames.
    view_disagreement_mm: float | None = None
    #: For every episode whose tags could be measured: how far its picture stood from the stored calibration's
    #: (`camera_moved_mm`), over how many episodes' frames its tags were averaged (`frames`), and the picture's
    #: levels in its first frame (`grey`, `tag_black`, `red_over_blue`, as `vision.floor_levels` gives them).
    camera: dict[int, dict[str, float]] | None = None
    #: How far the tags' outline corners lie apart from one episode to the next, in pixels (root mean square over
    #: neighbours in the same view): what one compressed frame's corners are worth. None with fewer than two.
    tag_noise_px: float | None = None
    #: Things to know that stop nothing: said, not counted as problems.
    notes: tuple[str, ...] = ()
    #: What was seen inside every take (`takes_through`), by episode; None when the takes were not looked through.
    takes: dict[int, dict[str, float]] | None = None
    #: The wrist camera's first frame of every episode: the floor over the jaws' green (`contrast`, None when the
    #: frame is not the rest picture), and its darkest and brightest hundredth and its mean.
    wrist: dict[int, dict[str, float | None]] | None = None

    @property
    def clean(self) -> bool:
        return not self.problems

    @property
    def mean_error_mm(self) -> float:
        return float(np.mean(list(self.errors_mm.values()))) if self.errors_mm else float("nan")

    @property
    def max_error_mm(self) -> float:
        return max(self.errors_mm.values()) if self.errors_mm else float("nan")


def episode_views(
    frames_bgr: Sequence[tuple[int, np.ndarray]], geometry: dict[str, Any]
) -> tuple[dict[int, TableView], dict[int, dict[int, np.ndarray]], float | None]:
    """A view of the table for every episode whose first frame shows both floor tags, solved from their outlines
    there (averaged with the neighbouring episodes' while those lie in the same place); every such episode's own
    corners; and how far corners lie apart from one episode to the next (px, rms), or None.
    """
    tags = {int(tag_id): corners for tag_id, corners in geometry["tags"].items()}
    ordered = sorted(frames_bgr, key=lambda item: item[0])
    corners: dict[int, dict[int, np.ndarray]] = {}
    for index, frame in ordered:
        quads, _found, measured, _count = tag_outlines_over([frame], tags)
        if all(measured[tag_id] for tag_id in tags):
            corners[index] = quads
    indices = [index for index, _ in ordered if index in corners]

    def apart(a: int, b: int) -> float:
        return max(float(np.linalg.norm(corners[a][tag_id] - corners[b][tag_id], axis=1).max()) for tag_id in tags)

    solve = lerobot101.load("placement").homography_from_tag_geometry
    views: dict[int, TableView] = {}
    for place, index in enumerate(indices):
        near = [other for other in indices[max(0, place - NEIGHBOURS): place + NEIGHBOURS + 1]
                if apart(index, other) <= SAME_VIEW_PX]
        mean = {tag_id: np.mean([corners[other][tag_id] for other in near], axis=0) for tag_id in tags}
        matrix = solve(mean, tags, geometry["tag_mm"])
        if matrix is not None:
            views[index] = TableView(matrix, residual_mm=_fit_residual_mm(matrix, mean, tags), frames_used=len(near),
                                     geometry_created=geometry.get("created"))
    steps = [np.concatenate([np.linalg.norm(corners[a][tag_id] - corners[b][tag_id], axis=1) for tag_id in tags])
             for a, b in zip(indices, indices[1:], strict=False) if apart(a, b) <= SAME_VIEW_PX]
    noise = float(np.sqrt(np.mean(np.square(np.concatenate(steps))))) if steps else None
    return views, corners, noise


def audit_frames(
    dataset: str,
    frames_bgr: Sequence[tuple[int, np.ndarray]],
    planned: Sequence[Any],
    view: TableView,
    grid: Grid,
    *,
    limit_mm: float = DEFAULT_LIMIT_MM,
    geometry: dict[str, Any] | None = None,
) -> AuditResult:
    """Measure the cube in each episode's first frame and compare with the plan.

    `frames_bgr` holds `(episode index, first frame)` pairs. `planned` holds the layout's episodes (anything with
    `episode_index`, `cell` and `target`). `view` is the stored calibration. With `geometry` (the table geometry),
    every episode is measured through a view solved from the floor tags in its own frame, and `view` is what those
    are held against: how far the camera's picture stood from it, and whether the picture's levels were its own.
    Without `geometry` every episode is measured through `view` itself.
    """
    problems: list[str] = []
    notes: list[str] = []
    plan = {episode.episode_index: episode for episode in planned}
    indices = [index for index, _ in frames_bgr]
    if sorted(indices) != sorted(plan):
        missing = sorted(set(plan) - set(indices))
        extra = sorted(set(indices) - set(plan))
        problems.append(f"the dataset's episodes do not match the plan: missing {missing or 'none'}, unplanned {extra or 'none'}")

    # The empty scene is the per-pixel median of the first frames: the cube is somewhere else in most of them.
    background = lerobot101.load("placement").empty_scene([frame for _, frame in frames_bgr])
    views: dict[int, TableView] = {}
    camera: dict[int, dict[str, float]] | None = None
    noise = None
    if geometry is not None:
        views, _corners, noise = episode_views(frames_bgr, geometry)
        camera = {}
        probes = grid_probes_px(view, geometry)
        for index, frame in sorted(frames_bgr, key=lambda item: item[0]):
            if index not in views:
                continue
            own = views[index]
            camera[index] = {"camera_moved_mm": round(view.disagreement_mm(own, probes), 3),
                             "frames": float(own.frames_used or 1)}
            try:
                quads = [[own.to_px(Point(float(x), float(y))) for x, y in tag["corners"]]
                         for tag in geometry["tags"].values()]
                camera[index].update(floor_levels(frame, grid_probes_px(own, geometry)[:4], quads))
            except VisionError:
                pass  # the levels are a record beside the measurement; an episode without them is still measured
        unsolved = sorted(index for index in indices if index not in views)
        if unsolved and not views:
            problems.append("the floor tags could not be measured in any episode's first frame: every start is "
                            "measured through the stored calibration, which the camera has left behind since")
        elif unsolved:
            problems.append(f"the floor tags could not be measured in the first frame of episodes {unsolved}: each "
                            "is measured through the view of the nearest episode that has them")

    def measuring_view(index: int) -> TableView:
        """The view an episode is measured through, with the empty scene; with the table's geometry it looks for
        the cube only on the grid and around it, as the sessions do: the arm does not start every episode from the
        same pose to the pixel, and in the median picture of the empty scene it is a blur."""
        if not views:
            chosen = view
        else:
            chosen = views[min(views, key=lambda other: abs(other - index))]
        chosen = chosen.with_background(background)
        return chosen if geometry is None else chosen.with_cube_region(geometry["grid"])

    starts: list[DemoStart] = []
    errors: dict[int, float] = {}
    for index, frame in sorted(frames_bgr, key=lambda item: item[0]):
        episode = plan.get(index)
        measurement = measuring_view(index).measure(frame)
        if measurement is None:
            problems.append(f"episode {index}: no single clean cube in its first frame")
            continue
        cell = grid.cell_of(measurement.position)
        if cell is None:
            problems.append(f"episode {index}: the cube was measured off the grid at "
                            f"({measurement.position.x_mm:.1f}, {measurement.position.y_mm:.1f}) mm")
            continue
        target = None if episode is None else episode.target
        starts.append(DemoStart(dataset=dataset, episode_index=index, cell=cell, measured=measurement.position,
                                planned=target))
        if not grid.is_trained(cell):
            problems.append(f"episode {index}: the cube started in held-out cell {cell}")
        if episode is None:
            continue
        errors[index] = measurement.position.distance_to(target)
        if cell != episode.cell:
            problems.append(f"episode {index}: planned for cell {episode.cell}, measured in cell {cell}")
        if errors[index] > limit_mm:
            problems.append(f"episode {index}: {errors[index]:.1f} mm from its target (limit {limit_mm:g} mm)")

    disagreement = None
    if camera:
        moved = {index: entry["camera_moved_mm"] for index, entry in camera.items()}
        disagreement = max(moved.values())
        notes.append(f"the camera's picture stood {min(moved.values()):.2f} to {disagreement:.2f} mm from the stored "
                     "calibration's over the episodes; every start is measured through the floor tags as its own "
                     "frame shows them")
        far = sorted(index for index, value in moved.items() if value > VIEW_LIMIT_MM)
        if far:
            problems.append(f"in episodes {far} the camera's picture stood more than {VIEW_LIMIT_MM} mm from the "
                            "stored calibration's: they were recorded through another view than the rest")
        problems.extend(_other_pictures(camera, view))
    return AuditResult(dataset, tuple(starts), tuple(problems), errors, disagreement, camera, noise, tuple(notes))


def _other_pictures(camera: dict[int, dict[str, float]], view: TableView) -> list[str]:
    """Episodes recorded on another picture than the calibration's, by the limits a session is held to."""
    from tolerance_radius.rig.camera_tool import PICTURE_CONTRAST_RANGE, PICTURE_GREY_LIMIT

    then = view.floor_levels
    if not then or "tag_black" not in then:
        return []
    low, high = PICTURE_CONTRAST_RANGE
    darker, stretched = [], []
    for index, now in sorted(camera.items()):
        if "grey" not in now or "tag_black" not in now:
            continue
        contrast = (now["grey"] / max(now["tag_black"], 1.0)) / (then["grey"] / max(then["tag_black"], 1.0))
        if not low <= contrast <= high:
            stretched.append(index)
        elif abs(now["grey"] - then["grey"]) > PICTURE_GREY_LIMIT:
            darker.append(index)
    told = []
    if stretched:
        told.append(f"episodes {stretched} were recorded in another picture state than the calibration's (the "
                    f"contrast, white floor over the tags' black, outside {low} to {high} of it)")
    if darker:
        told.append(f"episodes {darker} were recorded more than {PICTURE_GREY_LIMIT:.0f} grey levels brighter or "
                    "darker than the calibration")
    return told


def first_frames(repo_id: str, camera: str = "front") -> list[tuple[int, np.ndarray]]:
    """The first frame of every episode of a recorded dataset, BGR, read with the arm tooling's own loader.

    As the reader gives them, which is as the camera gave them to the recorder: LeRobot's writing and reading of a
    video keep the picture's levels (*measured* 2026-10-06: an episode's floor read 199 to 202 from the saved video
    and 203 in the frame its session started on; frames fed into a dataset came back 1.5 grey levels apart on
    average). Until that evening this function undid a "stretch by the readers" that does not exist: the videos it
    was measured on held a stretched picture because the camera's stream was in that state while they were recorded.
    """
    return list(lerobot101.load("placement")._load_episode_first_frames(repo_id, camera))


def sampled_frames(root: str | Path, camera: str = "front", every: int = THROUGH_EVERY
                   ) -> Iterator[tuple[int, int, np.ndarray]]:
    """`(episode, frame within it, frame BGR)` for one frame in `every` of each take of a dataset on disk, and each
    take's last frame, episode by episode. Read from the video files themselves; every frame is decoded once."""
    import json

    import av
    import pandas as pd

    root = Path(root)
    fps = float(json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"])
    # use_threads=False as in the replay's reader: with the reader's thread pool a process in the rig's environment
    # that has read parquet files failed to exit about half the time (measured 2026-10-05).
    tables = [pd.read_parquet(path, use_threads=False) for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))]
    episodes = pd.concat(tables, ignore_index=True)
    key = f"videos/observation.images.{camera}"
    for (chunk, number), group in episodes.groupby([f"{key}/chunk_index", f"{key}/file_index"]):
        wanted: dict[int, tuple[int, int]] = {}
        for _, row in group.iterrows():
            start, length = int(round(float(row[f"{key}/from_timestamp"]) * fps)), int(row["length"])
            for within in sorted({*range(0, length, max(1, every)), length - 1}):
                wanted[start + within] = (int(row["episode_index"]), within)
        path = root / "videos" / f"observation.images.{camera}" / f"chunk-{int(chunk):03d}" / f"file-{int(number):03d}.mp4"
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    yield (*wanted[index], frame.to_ndarray(format="bgr24"))


def takes_through(frames: Iterable[tuple[int, int, np.ndarray]], geometry: dict[str, Any]
                  ) -> tuple[dict[int, dict[str, float]], list[str]]:
    """What the front camera's picture did inside every take: by episode, and the takes that need eyes.

    `frames` are `(episode, frame within it, frame)` in a take's own order, as `sampled_frames` gives them. Every
    frame that shows both floor tags is held against the first such frame of its take: how far the tags' outline
    corners lie from theirs (`tags_moved_px`, the largest), and the tags' black as a share of theirs
    (`black_low`, `black_high`). A frame in which the arm hides a tag says nothing and is passed over.

    The start of an episode is all the audit measured before 2026-10-06. That evening the front camera went back to
    its own zoom and exposure 3.4 seconds into a take, and the take was saved: its first frame was as it should be.
    """
    tags = {int(tag_id): corners for tag_id, corners in geometry["tags"].items()}
    solve = lerobot101.load("placement").homography_from_tag_geometry
    first: dict[int, tuple[dict[int, np.ndarray], float | None]] = {}
    seen: dict[int, dict[str, float]] = {}
    for episode, _within, frame in frames:
        entry = seen.setdefault(episode, {"frames": 0.0, "frames_with_tags": 0.0})
        entry["frames"] += 1
        quads, _found, measured, _count = tag_outlines_over([frame], tags)
        if not all(measured[tag_id] for tag_id in tags):
            continue
        black = None
        matrix = solve(quads, tags, geometry["tag_mm"])
        if matrix is not None:
            own = TableView(matrix)
            try:
                outlines = [[own.to_px(Point(float(x), float(y))) for x, y in tag["corners"]]
                            for tag in geometry["tags"].values()]
                black = floor_levels(frame, grid_probes_px(own, geometry)[:4], outlines).get("tag_black")
            except VisionError:
                black = None
        entry["frames_with_tags"] += 1
        if episode not in first:
            first[episode] = (quads, black)
            entry.update(tags_moved_px=0.0)
            if black:
                entry.update(black_low=1.0, black_high=1.0)
            continue
        then, black_then = first[episode]
        moved = max(float(np.linalg.norm(quads[tag_id] - then[tag_id], axis=1).max()) for tag_id in tags)
        entry["tags_moved_px"] = max(entry["tags_moved_px"], moved)
        if black and black_then:
            share = float(black) / float(black_then)
            entry["black_low"] = min(entry.get("black_low", share), share)
            entry["black_high"] = max(entry.get("black_high", share), share)
    for entry in seen.values():
        for name in ("tags_moved_px", "black_low", "black_high"):
            if name in entry:
                entry[name] = round(entry[name], 3)
    problems = []
    unseen = sorted(episode for episode, entry in seen.items() if entry["frames_with_tags"] < 2)
    if unseen:
        problems.append(f"episodes {unseen}: the floor tags are in sight in fewer than two of the frames looked at; "
                        "whether the front camera's picture changed inside these takes cannot be said")
    moved = sorted(episode for episode, entry in seen.items() if entry.get("tags_moved_px", 0.0) > TAKE_TAGS_PX)
    if moved:
        most = max(seen[episode]["tags_moved_px"] for episode in moved)
        problems.append(f"episodes {moved}: the front camera's picture changes inside the take (the floor tags lie "
                        f"up to {most:.0f} px from where the take's first frame has them, allowed {TAKE_TAGS_PX:g}): "
                        "the camera's zoom or mode, or the camera itself")
    low, high = TAKE_BLACK_RANGE
    toned = sorted(episode for episode, entry in seen.items() if episode not in moved
                   and (entry.get("black_low", 1.0) < low or entry.get("black_high", 1.0) > high))
    if toned:
        problems.append(f"episodes {toned}: the front camera's tone range changes inside the take (the floor tags' "
                        f"black goes outside {low} to {high} of what it is in the take's first frame)")
    return seen, problems


def wrist_states(first_frames_bgr: Iterable[tuple[int, np.ndarray]]
                 ) -> tuple[dict[int, dict[str, float | None]], list[str]]:
    """The wrist camera's first frame of every episode by its numbers, and what they say of its tone state.

    The wrist camera has no tag. On 2026-10-06 its picture was in one of two states from episode to episode, the
    second the first with its tone range stretched, and nothing said so. Each first frame, the arm at rest, is read
    by `vision.wrist_contrast`: the episodes above the line between the two states are named; so are first frames
    that are not the rest picture; and, when every frame reads on one side of the line, two groups more than a
    step apart (the line was measured in one rest pose; a step is a step in any).
    """
    levels: dict[int, dict[str, float | None]] = {}
    for episode, frame in first_frames_bgr:
        contrast = wrist_contrast(frame)
        levels[episode] = {**wrist_levels(frame), "contrast": None if contrast is None else round(contrast, 3)}
    judged = {episode: entry["contrast"] for episode, entry in levels.items() if entry["contrast"] is not None}

    def span(group: Sequence[int]) -> str:
        return f"{min(judged[e] for e in group):.2f} to {max(judged[e] for e in group):.2f}"

    problems = []
    other = sorted(episode for episode, value in judged.items() if value > WRIST_OTHER_ABOVE)
    usual = sorted(episode for episode, value in judged.items() if value <= WRIST_OTHER_ABOVE)
    if other:
        rest = f", {span(usual)} in the rest" if usual else ""
        problems.append(f"the wrist camera's picture is in its other tone state in {len(other)} of {len(levels)} "
                        f"episodes (the floor over the jaws' green reads {span(other)} there{rest}; the line is "
                        f"{WRIST_OTHER_ABOVE:g}): {other}")
    elif len(judged) > 1:
        ordered = sorted(judged, key=lambda episode: judged[episode])
        step, place = max((judged[b] / judged[a], place)
                          for place, (a, b) in enumerate(zip(ordered, ordered[1:], strict=False)))
        if step > WRIST_STEP:
            lower, upper = sorted(ordered[: place + 1]), sorted(ordered[place + 1:])
            problems.append(f"the wrist camera's first frames fall into two groups: the floor over the jaws' green "
                            f"reads {span(lower)} in episodes {lower} and {span(upper)} in episodes {upper}")
    unjudged = sorted(episode for episode, entry in levels.items() if entry["contrast"] is None)
    if unjudged:
        problems.append(f"the wrist camera's first frame could not be judged in {len(unjudged)} of {len(levels)} "
                        f"episodes (it is not the rest picture, the jaws over the floor): {unjudged}")
    return levels, problems


def contact_sheet(frames_bgr: Sequence[tuple[int, np.ndarray]], view: TableView, out_path: Any) -> Any:
    """Every episode's first frame with its detection drawn, tiled into one picture for a person to look through."""
    placement = lerobot101.load("placement")
    background = placement.empty_scene([frame for _, frame in frames_bgr])
    items = [(index, frame, placement.detect_object(frame, background)) for index, frame in frames_bgr]
    return placement.contact_sheet(items, out_path)


def audit_dataset(dataset: str, repo_id: str, planned: Sequence[Any], view: TableView, grid: Grid,
                  *, limit_mm: float = DEFAULT_LIMIT_MM, root: str | Path | None = None
                  ) -> tuple[AuditResult, list[tuple[int, np.ndarray]]]:
    """Audit a recorded dataset on disk. Returns the result and the frames (for the contact sheet).

    With `root`, the dataset's folder, every take is also looked through (`takes_through`) and the wrist camera's
    first frames are held against each other (`wrist_states`); what they find joins the problems.
    """
    geometry = load_geometry()
    frames = first_frames(repo_id)
    result = audit_frames(dataset, frames, planned, view, grid, limit_mm=limit_mm, geometry=geometry)
    if root is None:
        return result, frames
    takes, inside = takes_through(sampled_frames(root, "front"), geometry)
    wrist, states = wrist_states(first_frames(repo_id, "wrist"))
    return replace(result, problems=(*result.problems, *inside, *states), takes=takes, wrist=wrist), frames
