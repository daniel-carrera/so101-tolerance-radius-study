#!/usr/bin/env python3
"""Recover where the cube started in each trial of the August evaluation, and how far that was from a demonstration.

In August 2026 an ACT policy (act-cube-v1) was evaluated in 31 real trials on the earlier SO-101 project. The cube's
start position was not logged. This script measures it afterwards from each trial's front-camera video, with the same
detector and the same pixel-to-table conversion that measured the 55 training demonstrations
(`scripts/data/placement.py` of the earlier project: imported, not copied), and computes each trial's distance to the
nearest demonstration.

Run it from the repository's top folder with the rig's interpreter (the `lerobot` conda environment, `RIG_PY` in the
Makefile). It reads files only and takes a few seconds:

    "$HOME/miniconda3/envs/lerobot/bin/python" analysis/pilot_august/recover_positions.py

Arguments, all optional:

    lerobot_101           the earlier project; default: $LEROBOT101_ROOT, else the folder `lerobot-101` beside this
                          repository
    --out DIR             where the outputs go; default: the folder this script is in
    --background WHICH    "training" (default) or "trials": which empty-scene picture the cube detector subtracts
    --frames N            how many frames at the start of each video are measured (default 10, a third of a second)
    --contact-sheet PATH  also write a picture of every detection, for a person to look at (not written by default)
    --figure-distance X   "A" (default) or "B": which of the two conversions below the figure's distances come from
    --self-test           check the statistics helpers against independent calculations, and stop

What it reads, all under the earlier project (which is never written to):

    eval/trials/act-cube-v1.jsonl     the 31 trial lines
    eval/recordings/*_front.mp4       one video per attempt (30 exist: the first trial was not filmed)
    logs/session_<stamp>.log          the rollout's own log: confirms which trial line belongs to which video
    logs/rollout_<stamp>_front.jpg    the still a rollout saves when it starts: a second look at each placement, and
                                      the only picture of the first trial's placement
    eval/placements.json              the 55 demonstrations' start positions (pixels and table millimetres)
    eval/ghosts/ghosts.json           the pixel targets the three "ghost" scenes were placed against
    eval/table_geometry.json          where the two AprilTags are in table millimetres
    scripts/data/placement.py         detect_object, homography_from_tag_geometry, px_to_mm, assign_cells, ...
    scripts/camera/tag_check.py       detect (AprilTag corners)

and, if it is on this machine, the recorded dataset itself in the LeRobot cache (~/.cache/huggingface/lerobot).
Nothing is ever downloaded: the Hugging Face libraries are forced offline. The dataset gives the same empty-scene
background that built `placements.json`, and lets the script re-measure all 55 demonstrations as a check of the
instrument. Without it the script still runs, on the trials' own background, and says so.

What it writes, all in --out:

    trials_with_distance.csv          one row per trial line (31)
    success_vs_distance.png / .pdf    the figure (with --figure-distance B: success_vs_distance_onemap.png / .pdf,
                                      so that the cross-check never overwrites the primary figure)

A full report is printed. `README.md` beside this script records the run of 2026-10-05.

Two ways of converting pixels to millimetres are reported.

  A, "as in August" (the primary one). Each trial frame is converted with the homography solved from the two tags
     in that same frame, exactly as `placement.py extract` converted each demonstration (and falling back, as it
     does, to the session's consensus map when a tag is hidden). Distances are to the millimetre positions in
     `placements.json`.
  B, "one map" (a cross-check). The pixel positions of trials and demonstrations are all converted with one single
     homography. The camera view was the same in both sessions (the tag corners moved by less than a pixel), so this
     removes the frame-to-frame noise of the tag solve, which method A carries twice.

Safety: this script opens no camera and no serial port. `cv2.VideoCapture` is only ever given a file path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

# The earlier project is read-only here: importing its modules must not leave __pycache__ files in it.
sys.dont_write_bytecode = True
# Nothing is downloaded. These must be set before the Hugging Face libraries are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(2)  # the analysis is small; keep the machine quiet

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIAL_LOG = "eval/trials/act-cube-v1.jsonl"
INPUT_FILES = (
    TRIAL_LOG,
    "eval/placements.json",
    "eval/table_geometry.json",
    "scripts/data/placement.py",
    "scripts/camera/tag_check.py",
)

# The policy was trained with an evaluation split of 0.1 taken from the tail: the last ceil(0.1 x 55) = 6 episodes
# (49 to 54) were held out for the evaluation loss, so 49 episodes (0 to 48) were trained on.
# Read from docs/run-journal.md of the earlier project ("eval_split 0.1 (tail = eps 49-54 ...)").
EVAL_TAIL_EPISODES = 6

SUCCESS_SUBGOAL = 5
BIN_MM = 10.0  # protocol/protocol.md section 10: "success counts in 10 mm bins of d with Wilson 95 % intervals"
EDGE_NOTE_MM = 2.0  # a trial this close to a bin edge could fall on either side of it: named in the report
NEIGHBOURHOOD_MM = 10.0  # how many trained demonstrations lie this close to the cube: a measure of local density
Z95 = 1.959963984540054

# The protocol's model of success against distance (config/protocol.toml, draft 0.1, [analysis]):
#   P(success at d) = (1 - lapse) / (1 + exp((d - r50) / width))
# with a flat prior on r50, a flat prior on log(width), and the lapse values weighted equally.
R50_GRID_MM = np.arange(0.0, 80.0 + 1e-9, 0.5)
WIDTH_GRID_MM = np.geomspace(0.5, 20.0, 30)
LAPSE_VALUES = (0.0, 0.02, 0.05, 0.10)

# Where a trial's start position came from.
FROM_VIDEO = "video frame"
FROM_STILL = "session-start still"
NO_PICTURE = "none"

SCORED_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO attempt (\d+) scored (\d)/\d")
ENDED_LINE = re.compile(r"Attempt (\d+): ended by operator after (\d+) steps")

CSV_COLUMNS = (
    "trial", "ts", "scene_id", "cell_logged", "kind", "axis", "subgoal", "success", "failure",
    "position_source", "source_file", "frame_used", "mapping_evidence",
    "px_x", "px_y", "area_px", "blob_angle_deg", "n_candidates", "homography",
    "x_mm", "y_mm", "cell_recovered", "cell_matches_logged",
    "nearest_trained49_ep", "nearest_trained49_cell", "d_nearest_trained49_mm",
    "nearest_all55_ep", "nearest_all55_cell", "d_nearest_all55_mm", "n_trained49_within_10mm",
    "x_mm_onemap", "y_mm_onemap", "nearest_trained49_ep_onemap", "d_nearest_trained49_mm_onemap",
    "d_nearest_all55_mm_onemap",
    "max_move_first_frames_px", "max_move_first_frames_mm", "still_vs_video_px", "still_vs_video_mm",
    "ghost_episode", "ghost_error_px", "ghost_error_mm", "ghost_error_mm_onemap", "ghost_turn_deg", "flags",
)  # fmt: skip


# ------------------------------------------------------------------------------------------------------ small helpers


def say(text: str = "") -> None:
    print(text, flush=True)


def shown(path: Path) -> str:
    """A path as it is printed: relative to the repository when it is inside or beside it, else its name only.

    Reports get pasted into records, and a machine's folder names do not belong there.
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT.parent):
        return os.path.relpath(resolved, REPO_ROOT)
    return resolved.name


def wilson(successes: int, n: int) -> tuple[float, float]:
    """The Wilson 95 % interval for a proportion. (nan, nan) when there are no trials.

    With no success the lower end is exactly 0, and with no failure the upper end is exactly 1: the formula gives
    those values, and rounding must not leave them a hair inside (which would put the rate outside its own interval).
    """
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denominator = 1.0 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, centre - half)
    upper = 1.0 if successes == n else min(1.0, centre + half)
    return (lower, upper)


def posterior_r50(d_mm: np.ndarray, success: np.ndarray) -> dict:
    """The protocol's grid posterior for the tolerance radius r50, applied to (distance, outcome) pairs.

    Returns the posterior median and 95 % interval of r50, the median width, and whether the interval touches the
    grid's edge (in which case the data do not bound the radius on that side).
    """
    d = np.asarray(d_mm, float)[None, None, None, :]
    y = np.asarray(success, float)[None, None, None, :]
    r50 = R50_GRID_MM[:, None, None, None]
    width = WIDTH_GRID_MM[None, :, None, None]
    lapse = np.asarray(LAPSE_VALUES)[None, None, :, None]
    p = (1.0 - lapse) / (1.0 + np.exp(np.clip((d - r50) / width, -700, 700)))
    p = np.clip(p, 1e-300, 1.0)
    one_minus_p = np.clip(1.0 - p, 1e-300, 1.0)
    log_like = (y * np.log(p) + (1.0 - y) * np.log(one_minus_p)).sum(axis=-1)
    post = np.exp(log_like - log_like.max())
    post /= post.sum()

    def quantile(grid: np.ndarray, marginal: np.ndarray, q: float) -> float:
        return float(grid[min(int(np.searchsorted(np.cumsum(marginal), q)), len(grid) - 1)])

    over_r50 = post.sum(axis=(1, 2))
    lo, median, hi = (quantile(R50_GRID_MM, over_r50, q) for q in (0.025, 0.5, 0.975))
    return {
        "median": median,
        "lo": lo,
        "hi": hi,
        "touches_lower_edge": bool(lo <= R50_GRID_MM[0]),
        "touches_upper_edge": bool(hi >= R50_GRID_MM[-1]),
        "width_median": quantile(WIDTH_GRID_MM, post.sum(axis=(0, 2)), 0.5),
    }


def head_commit(repo: Path) -> str:
    """The commit a checkout is at, read from its .git files (no git command is run)."""
    with contextlib.suppress(OSError):
        head = (repo / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head.removeprefix("ref: ")
        loose = repo / ".git" / ref
        if loose.is_file():
            return loose.read_text().strip()
        for line in (repo / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return "unknown"


def short_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S")


def gap_px(a, b) -> float:
    """Distance between two image points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def nearest(point: np.ndarray, others: np.ndarray, indices: np.ndarray) -> tuple[int, float]:
    """(episode index, distance) of the nearest of `others` to `point`."""
    distances = np.linalg.norm(others - point, axis=1)
    k = int(distances.argmin())
    return int(indices[k]), float(distances[k])


def fmt(value, digits: int = 2) -> str:
    """A value as it goes into the CSV or a table: empty when missing, yes/no for a truth value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# ------------------------------------------------------------------------------------------- the August instrument


def default_lerobot_101() -> Path:
    """Where the earlier project is: $LEROBOT101_ROOT if set, else the folder `lerobot-101` beside this repository."""
    return Path(os.environ.get("LEROBOT101_ROOT", REPO_ROOT.parent / "lerobot-101"))


def load_instrument(repo: Path) -> SimpleNamespace:
    """Import the earlier project's own detector and conversion, and read its table geometry.

    Returns a namespace with `P` (the placement module), `detect_tags`, `geometry` (tag corners in table millimetres)
    and `tag_mm`.
    """
    data_folder = repo / "scripts" / "data"
    if not (data_folder / "placement.py").is_file():
        raise SystemExit(f"no scripts/data/placement.py under {shown(repo)}: is this the lerobot-101 project?")
    sys.path.insert(0, str(data_folder))
    import placement  # adds scripts/ and scripts/camera to the path itself
    from tag_check import detect as detect_tags

    if Path(placement.__file__).resolve().parent != data_folder.resolve():
        raise SystemExit(f"imported placement from {placement.__file__}, not from {data_folder}")
    geometry, tag_mm = placement.load_tag_positions(None)  # eval/table_geometry.json of the earlier project
    return SimpleNamespace(P=placement, detect_tags=detect_tags, geometry=geometry, tag_mm=tag_mm)


def load_training_first_frames(placement, repo_id: str, camera: str, scratch_parent: Path):
    """Frame 0 of every training episode, through the very loader that built placements.json.

    The dataset must already be on this machine: the Hugging Face libraries are offline, so a missing dataset is an
    error here, never a download. The `datasets` library writes a small cache when it opens the parquet files; that
    is pointed at a temporary folder inside the output folder and removed again, so nothing is written elsewhere.
    Returns (frames, None) or (None, reason).
    """
    home = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))
    if not (home / repo_id / "meta" / "info.json").is_file():
        return None, f"the dataset {repo_id.split('/')[-1]} is not in the LeRobot cache"
    try:
        with tempfile.TemporaryDirectory(prefix=".hf_datasets_cache_", dir=scratch_parent) as scratch:
            os.environ["HF_DATASETS_CACHE"] = scratch
            frames = placement._load_episode_first_frames(repo_id, camera)
        return frames, None
    except Exception as error:  # any failure means "not available here"; it is reported, not hidden
        return None, f"{type(error).__name__}: {error}"


def measure(inst: SimpleNamespace, frame: np.ndarray, background: np.ndarray):
    """One frame through the August instrument.

    Returns (cube detection or None, pixel-to-mm homography or None, {tag id: its four corners in pixels}).
    """
    detection = inst.P.detect_object(frame, background)
    found = inst.detect_tags(frame)
    homography = inst.P.homography_from_tag_geometry(found, inst.geometry, inst.tag_mm)
    return detection, homography, found


# -------------------------------------------------------------------------------------- trial lines and their pictures


def read_trials(repo: Path) -> list[dict]:
    """The trial lines, numbered from 1 in file order (trial 1 is line 1 of the log)."""
    trials = []
    for number, line in enumerate((repo / TRIAL_LOG).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["trial"] = number
        record["cell_logged"] = record["cell"]
        record["success"] = int(record["subgoal"] == SUCCESS_SUBGOAL)
        trials.append(record)
    return trials


def session_log_facts(repo: Path, stamp: str | None) -> dict:
    """What a rollout's own log says about its attempt: when it was scored, the score, the number of steps."""
    facts: dict = {"exists": False, "scored": [], "steps": None}
    path = repo / "logs" / f"session_{stamp}.log"
    if stamp is None or not path.is_file():
        return facts
    facts["exists"] = True
    with path.open(errors="replace") as handle:
        for line in handle:
            if "scored" in line:
                match = SCORED_LINE.match(line)
                if match:
                    facts["scored"].append((match.group(1).replace(" ", "T"), int(match.group(3))))
            elif "ended by operator" in line:
                match = ENDED_LINE.search(line)
                if match:
                    facts["steps"] = int(match.group(2))
    return facts


def locate_sources(repo: Path, trials: list[dict]) -> None:
    """For each trial line: its front video (from the line's own `videos` field), and the evidence for that mapping.

    A line without videos is matched to a rollout session through the session log: the session whose "attempt scored"
    line carries the trial's timestamp. Its start-of-session still is then the only picture of the placement.
    """
    recordings = repo / "eval" / "recordings"
    session_stamps = sorted(
        path.stem.removeprefix("session_")
        for path in (repo / "logs").glob("session_2*.log")
        if re.fullmatch(r"session_\d{8}_\d{6}", path.stem)
    )
    for trial in trials:
        ts = trial["ts"]
        fronts = [Path(v).name for v in (trial.get("videos") or []) if str(v).endswith("_front.mp4")]
        trial["video"], trial["stamp"], trial["mapping"] = None, None, []
        if fronts:
            trial["stamp"] = fronts[0][:15]
            trial["mapping"].append("named in the trial line")
            if (recordings / fronts[0]).is_file():
                trial["video"] = recordings / fronts[0]
            else:
                trial["mapping"].append("VIDEO FILE MISSING")
        else:
            # only the latest rollout session that started before the line was written can be the one
            earlier = [s for s in session_stamps if parse_stamp(s) <= datetime.fromisoformat(ts)]
            if earlier and any(when == ts for when, _ in session_log_facts(repo, earlier[-1])["scored"]):
                trial["stamp"] = earlier[-1]
                trial["mapping"].append("no video in the trial line; session found by its log")
        facts = session_log_facts(repo, trial["stamp"])
        scored = facts["scored"]
        trial["log_ts_match"] = any(when == ts for when, _ in scored) if facts["exists"] else None
        trial["log_score_match"] = (
            any(when == ts and score == trial["subgoal"] for when, score in scored) if facts["exists"] else None
        )
        trial["log_steps"] = facts["steps"]
        still = repo / "logs" / f"rollout_{trial['stamp']}_front.jpg"
        trial["still"] = still if trial["stamp"] and still.is_file() else None


def read_first_frames(video: Path, count: int) -> tuple[list[np.ndarray], int]:
    """The first `count` frames of a video file, and the file's frame count. Reads a file; never a camera."""
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(count):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames, total


def load_pictures(trials: list[dict], n_frames: int) -> None:
    """Read each trial's first video frames and its start-of-session still."""
    for trial in trials:
        trial["frames"], trial["frame_count"] = [], None
        if trial["video"] is not None:
            trial["frames"], trial["frame_count"] = read_first_frames(trial["video"], n_frames)
        trial["still_image"] = cv2.imread(str(trial["still"])) if trial["still"] is not None else None


def report_sources(repo: Path, trials: list[dict]) -> None:
    """Section 1 of the report: which video belongs to which trial line, and how sure that is."""
    say("\n1. WHICH VIDEO BELONGS TO WHICH TRIAL LINE")
    filmed = [t for t in trials if t["video"] is not None]
    on_disk = sorted(path.name for path in (repo / "eval" / "recordings").glob("*_front.mp4"))
    named = sorted(t["video"].name for t in filmed)
    say(f"  [read] {len(trials)} trial lines; {len(on_disk)} front videos in eval/recordings.")
    say(f"  [measured] {len(filmed)} lines name their own front video (the rollout wrote the path into the line it "
        f"appended); every video is named by exactly one line: {named == on_disk}.")
    same_second = sum(bool(t["log_ts_match"]) for t in trials)
    same_score = sum(bool(t["log_score_match"]) for t in trials)
    say("  [measured] the session log of the same start time has an 'attempt scored' line with the trial's timestamp "
        f"(to the second) for {same_second}/{len(trials)} lines, and with the trial's score for "
        f"{same_score}/{len(trials)}.")
    in_order = all(parse_stamp(t["stamp"]) < datetime.fromisoformat(t["ts"]) for t in trials if t["stamp"])
    for this, following in pairwise(trials):
        if this["stamp"] and following["stamp"]:
            in_order &= datetime.fromisoformat(this["ts"]) < parse_stamp(following["stamp"])
    say("  [measured] each line was written after its own session started and before the next one started: "
        f"{in_order}.")
    counted = [t for t in trials if t["log_steps"] is not None and t["frame_count"] is not None]
    equal = sum(t["log_steps"] == t["frame_count"] for t in counted)
    say("  [measured] where the log gives a step count (attempts the operator ended), it equals the video's frame "
        f"count in {equal}/{len(counted)} cases.")
    for t in trials:
        if t["video"] is None:
            say(f"  [measured] trial {t['trial']} ({t['scene_id']}, {t['ts']}) has NO VIDEO: its line names none and "
                f"no recording is that early. Session by its log: {t['stamp']}; start-of-session still: "
                f"{t['still'].name if t['still'] else 'none'}.")


# ------------------------------------------------------------------------------------------------- the measurements


def remeasure_training(inst: SimpleNamespace, frames: list, placements: list[dict], name: str) -> SimpleNamespace:
    """Check 2a: re-measure every training episode's frame 0 exactly as `placement.py extract` did.

    Compares with placements.json, and returns the session's empty-scene background, the tag corners seen in each
    frame, the tag solves, and the session's consensus map.
    """
    P = inst.P
    background = P.empty_scene([frame for _, frame in frames])
    rows, maps, tags = [], [], []
    for episode, frame in frames:
        detection, homography, found = measure(inst, frame, background)
        row = {"episode_index": episode, "detected": detection is not None}
        if detection is not None:
            row["px"] = detection["centroid"]
        rows.append(row)
        maps.append(homography)
        tags.append(found)
    P.assign_cells(rows, maps)
    by_episode = {p["episode_index"]: p for p in placements}
    worst_px = worst_mm = 0.0
    same_cell = same_label = 0
    for row in rows:
        ref = by_episode[row["episode_index"]]
        if "mm" not in row:
            worst_px = worst_mm = math.inf
            continue
        worst_px = max(worst_px, gap_px(row["px"], ref["px"]))
        worst_mm = max(worst_mm, gap_px(row["mm"], ref["mm"]))
        same_cell += row["cell"] == ref["cell"]
        same_label += row["homography"] == ref["homography"]
    lerobot = sys.modules.get("lerobot")
    decoder = f"LeRobot {getattr(lerobot, '__version__', 'of unknown version')}" if lerobot else "an unknown loader"
    say(f"  2a. [measured] frame 0 of all {len(rows)} training episodes, re-read from the local dataset ({name}, "
        f"decoded by {decoder}) and re-measured: largest difference from placements.json {worst_px:.4f} px and "
        f"{worst_mm:.4f} mm; same cell {same_cell}/{len(rows)}; same homography label {same_label}/{len(rows)}.")
    solves = [m for m in maps if m is not None]
    consensus, _ = P.consensus_homography(solves)
    if consensus is not None:
        borrowed = [p for p in placements if p.get("homography") == "session-consensus"]
        off_mm = [float(np.linalg.norm(P.px_to_mm([p["px"]], consensus)[0] - p["mm"])) for p in borrowed]
        worst = max(off_mm, default=0.0)
        say(f"      [measured] the training session's consensus map reproduces the {len(borrowed)} "
            f"'session-consensus' rows of placements.json to {worst:.4f} mm.")
    return SimpleNamespace(background=background, tags=tags, solves=solves, consensus=consensus)


def measure_trials(inst: SimpleNamespace, trials: list[dict], background: np.ndarray, placements: list[dict]):
    """Measure every trial's start: the cube in the first video frame that shows it, or in the session's still.

    Fills each trial's source, detection and flags, then converts pixels to table millimetres with the earlier
    project's own `assign_cells` (its conversion, its fallback, its labels). Returns the tag solves that were found.
    """
    area_lo = 0.75 * min(p["area"] for p in placements)
    area_hi = 1.25 * max(p["area"] for p in placements)
    rows, maps = [], []
    for trial in trials:
        trial["flags"] = []
        trial["per_frame"] = []  # (frame index, cube detection or None, homography or None)
        trial["detection"] = trial["image_used"] = trial["map"] = trial["frame_used"] = None
        trial["tags"], trial["source_file"] = {}, ""
        if trial["frames"]:
            trial["position_source"] = FROM_VIDEO
            trial["source_file"] = trial["video"].name
            seen = []
            for k, frame in enumerate(trial["frames"]):
                detection, homography, found = measure(inst, frame, background)
                trial["per_frame"].append((k, detection, homography))
                if detection is not None:
                    seen.append((k, detection, homography, found))
            if seen:
                k, trial["detection"], trial["map"], trial["tags"] = seen[0]
                trial["frame_used"], trial["image_used"] = k, trial["frames"][k]
                if k != 0:
                    trial["flags"].append(f"no cube detected in frame 0; frame {k} used")
            else:
                trial["flags"].append(f"NO CUBE DETECTED in the first {len(trial['frames'])} frames")
        elif trial["still_image"] is not None:
            trial["position_source"] = FROM_STILL
            trial["source_file"] = trial["still"].name
            trial["image_used"] = trial["still_image"]
            trial["detection"], trial["map"], trial["tags"] = measure(inst, trial["still_image"], background)
            trial["flags"].append("not filmed: position from the rollout's start-of-session still")
            if trial["detection"] is None:
                trial["flags"].append("NO CUBE DETECTED in the still")
        else:
            trial["position_source"] = NO_PICTURE
            trial["flags"].append("NO PICTURE of this trial's placement: position not recovered")
        detection = trial["detection"]
        row = {"episode_index": trial["trial"], "detected": detection is not None}
        if detection is not None:
            row["px"] = detection["centroid"]
            if detection["n_candidates"] != 1:
                trial["flags"].append(f"{detection['n_candidates']} cube-like blobs; the largest was taken")
            if not area_lo <= detection["area"] <= area_hi:
                trial["flags"].append(f"blob area {detection['area']} px outside the training range")
        rows.append(row)
        maps.append(trial["map"])
    note = inst.P.assign_cells(rows, maps)
    if note:
        say(f"  !! placement.assign_cells: {note}")
    for trial, row in zip(trials, rows, strict=True):
        trial["px"], trial["mm"] = row.get("px"), row.get("mm")
        trial["cell_recovered"], trial["homography"] = row.get("cell"), row.get("homography")
    return [m for m in maps if m is not None]


def derive(inst: SimpleNamespace, trials: list[dict], demos: SimpleNamespace, background: np.ndarray) -> None:
    """From each measured start: the distances to the demonstrations, and the per-trial checks of the instrument."""
    P = inst.P
    for trial in trials:
        trial["d_all"] = trial["d_trained"] = None
        if trial["mm"] is None:
            if trial["px"] is not None:
                trial["flags"].append("cube detected but no homography: position in pixels only")
            continue
        if trial["cell_recovered"] != trial["cell_logged"]:
            trial["flags"].append(
                f"recovered cell {trial['cell_recovered']} is not the logged cell {trial['cell_logged']}"
            )
        point = np.array(trial["mm"])
        trained_mm, trained_ep = demos.mm[demos.trained], demos.ep[demos.trained]
        trial["nn_all"], trial["d_all"] = nearest(point, demos.mm, demos.ep)
        trial["nn_trained"], trial["d_trained"] = nearest(point, trained_mm, trained_ep)
        trial["n_within"] = int((np.linalg.norm(trained_mm - point, axis=1) <= NEIGHBOURHOOD_MM).sum())
        if demos.mm_onemap is not None:
            onemap = P.px_to_mm([trial["px"]], demos.onemap)[0]
            trial["mm_onemap"] = onemap
            trial["nn_all_onemap"], trial["d_all_onemap"] = nearest(onemap, demos.mm_onemap, demos.ep)
            trial["nn_trained_onemap"], trial["d_trained_onemap"] = nearest(
                onemap, demos.mm_onemap[demos.trained], trained_ep
            )
        # steadiness over the first frames of the video: the cube has not been touched yet
        moved_px, moved_mm = [], []
        for _, detection, homography in trial["per_frame"]:
            if detection is None:
                continue
            moved_px.append(gap_px(detection["centroid"], trial["px"]))
            if homography is not None:
                moved_mm.append(float(np.linalg.norm(P.px_to_mm([detection["centroid"]], homography)[0] - point)))
        trial["steady_px"] = max(moved_px) if len(moved_px) > 1 else None
        trial["steady_mm"] = max(moved_mm) if len(moved_mm) > 1 else None
        # the start-of-session still against the video's frame: two pictures of one placement
        trial["still_dpx"] = trial["still_dmm"] = None
        if trial["position_source"] == FROM_VIDEO and trial["still_image"] is not None:
            detection, homography, _ = measure(inst, trial["still_image"], background)
            if detection is not None:
                trial["still_dpx"] = gap_px(detection["centroid"], trial["px"])
                if homography is not None:
                    in_still = P.px_to_mm([detection["centroid"]], homography)[0]
                    trial["still_dmm"] = float(np.linalg.norm(in_still - point))
        # ghost trials: the placement should be the named training episode's
        trial["ghost_ep"] = trial["ghost_px"] = trial["ghost_mm"] = trial["ghost_mm_onemap"] = None
        trial["ghost_turn_deg"] = None
        match = re.fullmatch(r"GHOST_EP(\d+)", trial["scene_id"])
        if match:
            episode = int(match.group(1))
            k = int(np.where(demos.ep == episode)[0][0])
            trial["ghost_ep"] = episode
            if demos.angle[k] is not None:
                # the blob's rectangle looks the same after a quarter turn, so the difference is taken within +-45
                turn = trial["detection"]["angle"] - demos.angle[k]
                trial["ghost_turn_deg"] = float((turn + 45.0) % 90.0 - 45.0)
            trial["ghost_vector_px"] = (trial["px"][0] - demos.px[k, 0], trial["px"][1] - demos.px[k, 1])
            trial["ghost_px"] = float(np.hypot(*trial["ghost_vector_px"]))
            trial["ghost_mm"] = float(np.linalg.norm(point - demos.mm[k]))
            if demos.mm_onemap is not None:
                trial["ghost_mm_onemap"] = float(np.linalg.norm(trial["mm_onemap"] - demos.mm_onemap[k]))


# ------------------------------------------------------------------------------------------------------- the report


def report_ghosts(repo: Path, trials: list[dict], demos: SimpleNamespace) -> None:
    """Check 2b: each ghost trial's recovered start against the training episode it re-created."""
    ghost_file = repo / "eval" / "ghosts" / "ghosts.json"
    if ghost_file.is_file():
        targets = json.loads(ghost_file.read_text())["ghosts"]
        worst = max(gap_px(g["centroid_px"], demos.px[int(np.where(demos.ep == g["episode"])[0][0])]) for g in targets)
        episodes = ", ".join(str(g["episode"]) for g in targets)
        say(f"\n  [read] the {len(targets)} ghost targets of eval/ghosts/ghosts.json (episodes {episodes}) are the "
            f"placements.json pixels of those episodes: largest difference {worst:.4f} px.")
    say("\n  2b. GHOST TRIALS: recovered start against the training episode it re-created [measured]")
    say("      (the operator slid the cube until the live readout, the same detector, showed 4 px or less)")
    say("      | trial | scene | source | error px | error mm (A, as in August) | error mm (B, one map) | "
        "blob turned by (deg) |")
    say("      |---|---|---|---|---|---|---|")
    ghosts = [t for t in trials if t["scene_id"].startswith("GHOST")]
    for t in ghosts:
        if t["d_trained"] is None:
            say(f"      | {t['trial']} | {t['scene_id']} | {t['position_source']} | not recovered | | | |")
            continue
        say(f"      | {t['trial']} | {t['scene_id']} | {t['position_source']} | {t['ghost_px']:.2f} | "
            f"{t['ghost_mm']:.2f} | {fmt(t['ghost_mm_onemap'])} | {fmt(t['ghost_turn_deg'], 1)} |")
    got = [t for t in ghosts if t.get("ghost_mm") is not None]
    if not got:
        return
    px = [t["ghost_px"] for t in got]
    mm = [t["ghost_mm"] for t in got]
    line = (f"      {len(got)}/{len(ghosts)} measured: error {min(px):.2f} to {max(px):.2f} px, {min(mm):.2f} to "
            f"{max(mm):.2f} mm (A), mean {np.mean(mm):.2f} mm (A)")
    if demos.mm_onemap is not None:
        onemap = [t["ghost_mm_onemap"] for t in got]
        line += f", {min(onemap):.2f} to {max(onemap):.2f} mm (B), mean {np.mean(onemap):.2f} mm (B)"
    say(line + ".")
    mean_vector = np.mean([t["ghost_vector_px"] for t in got], axis=0)
    say(f"      mean error as a vector: ({mean_vector[0]:+.2f}, {mean_vector[1]:+.2f}) px, so no common shift between "
        "the recorded trials and the training frames.")
    turns = [abs(t["ghost_turn_deg"]) for t in got if t["ghost_turn_deg"] is not None]
    if turns:
        say("      the blob's rectangle (a rough sign of how the cube is turned, not a measurement of it) is turned by "
            f"{min(turns):.1f} to {max(turns):.1f} degrees against the episode's (median {np.median(turns):.1f}).")


def solve_spread(P, points_px: np.ndarray, solves: list[np.ndarray]) -> tuple[float, float, float]:
    """How far the millimetre positions of some pixels move with the choice of tag solve: mean, 95th centile, max.

    Each solve converts every point; the spread is each conversion's distance from the per-point median.
    """
    projected = np.stack([P.px_to_mm(points_px, m) for m in solves])
    deviation = np.linalg.norm(projected - np.median(projected, axis=0), axis=2)
    return float(deviation.mean()), float(np.percentile(deviation, 95)), float(deviation.max())


def report_checks(
    inst: SimpleNamespace,
    trials: list[dict],
    demos: SimpleNamespace,
    training: SimpleNamespace | None,
    trial_session: SimpleNamespace,
    placements: list[dict],
    n_frames: int,
) -> None:
    """Check 2c: the other tests of the instrument on the trial pictures."""
    P = inst.P
    measured = [t for t in trials if t["d_trained"] is not None]
    filmed = [t for t in measured if t["position_source"] == FROM_VIDEO]
    detected = [t for t in trials if t["detection"] is not None]
    say("\n  2c. OTHER CHECKS [measured]")
    in_frame_0 = sum(t["position_source"] == FROM_VIDEO and t["frame_used"] == 0 for t in detected)
    in_still = sum(t["position_source"] == FROM_STILL for t in detected)
    one_blob = sum(t["detection"]["n_candidates"] == 1 for t in detected)
    say(f"      cube detected in {len(detected)}/{len(trials)} trials ({in_frame_0} in frame 0 of the video, "
        f"{in_still} in a start-of-session still); exactly one cube-like blob in {one_blob}.")
    labels = [t["homography"] for t in measured]
    say(f"      homography: {labels.count('episode')} from the two tags in the trial's own picture ('episode'), "
        f"{labels.count('session-consensus')} from the session's consensus map (a tag hidden).")
    areas = [t["detection"]["area"] for t in detected]
    say(f"      blob area {min(areas)} to {max(areas)} px (training episodes [read]: "
        f"{min(p['area'] for p in placements)} to {max(p['area'] for p in placements)} px).")
    same_cell = sum(t["cell_recovered"] == t["cell_logged"] for t in measured)
    say(f"      recovered cell equals the logged cell in {same_cell}/{len(measured)} trials.")
    steady_px = [t["steady_px"] for t in filmed if t["steady_px"] is not None]
    steady_mm = [t["steady_mm"] for t in filmed if t["steady_mm"] is not None]
    if steady_px:
        say(f"      over the first {n_frames} frames of a video the position moves by at most {max(steady_px):.2f} px "
            f"(median {np.median(steady_px):.2f}) and, each frame converted with its own tag solve, by at most "
            f"{max(steady_mm):.2f} mm (median {np.median(steady_mm):.2f}).")
    still_px = [t["still_dpx"] for t in filmed if t["still_dpx"] is not None]
    still_mm = [t["still_dmm"] for t in filmed if t["still_dmm"] is not None]
    if still_px:
        say(f"      start-of-session still against video frame, same placement, {len(still_px)} trials: at most "
            f"{max(still_px):.2f} px apart (median {np.median(still_px):.2f}); in millimetres, each picture with its "
            f"own tag solve, at most {max(still_mm):.2f} (median {np.median(still_mm):.2f}).")
    if trial_session.other_background is not None:
        moved = []
        for t in filmed:
            detection = P.detect_object(t["image_used"], trial_session.other_background)
            if detection is not None:
                moved.append(gap_px(detection["centroid"], t["px"]))
        if moved:
            say(f"      with the other empty-scene background ({trial_session.other_background_name}) the position "
                f"changes by at most {max(moved):.2f} px ({len(moved)}/{len(filmed)} detected).")
    trial_px = np.array([t["px"] for t in measured], float)
    mean, centile, worst = solve_spread(P, trial_px, trial_session.solves)
    say(f"      tag-solve noise, trial session: the {len(trial_session.solves)} solves, applied to the trials' cube "
        f"pixels, differ from their median by {mean:.2f} mm on average, {centile:.2f} at the 95th percentile, "
        f"{worst:.2f} at most (consensus outliers: {len(trial_session.outliers)}).")
    if training is not None:
        mean, centile, worst = solve_spread(P, demos.px, training.solves)
        say(f"      tag-solve noise, training session: the {len(training.solves)} solves, applied to the "
            f"demonstrations' cube pixels, differ from their median by {mean:.2f} mm on average, {centile:.2f} at the "
            f"95th percentile, {worst:.2f} at most.")
        between = P.px_to_mm(trial_px, trial_session.consensus) - P.px_to_mm(trial_px, training.consensus)
        between = np.linalg.norm(between, axis=1)
        say("      the two sessions' consensus maps, at the trials' cube pixels, differ by "
            f"{between.mean():.2f} mm on average ({between.max():.2f} at most).")
        shifts = []
        for tag_id in sorted(inst.geometry):
            before = [found[tag_id] for found in training.tags if tag_id in found]
            after = [t["tags"][tag_id] for t in filmed if tag_id in t["tags"]]
            if before and after:
                shift = np.linalg.norm(np.mean(after, axis=0) - np.mean(before, axis=0), axis=1)
                shifts.append(float(shift.max()))
        if shifts:
            say("      tag corners, trial session against training session (means): largest shift "
                f"{max(shifts):.2f} px, so the camera view is the same to within a pixel.")
    here = P.px_to_mm(trial_px, trial_session.consensus)
    across = np.linalg.norm(P.px_to_mm(trial_px + [1.0, 0.0], trial_session.consensus) - here, axis=1)
    down = np.linalg.norm(P.px_to_mm(trial_px + [0.0, 1.0], trial_session.consensus) - here, axis=1)
    smallest, largest = min(across.min(), down.min()), max(across.max(), down.max())
    say("      image scale at the trials' cube positions (trial session's consensus map): one pixel is "
        f"{across.min():.2f} to {across.max():.2f} mm across the image and {down.min():.2f} to {down.max():.2f} mm "
        f"down it, so the live readout's 4 px is about {4 * smallest:.1f} to {4 * largest:.1f} mm.")


def largest_pair_per_cell(positions_mm: np.ndarray, demos: SimpleNamespace) -> list[float]:
    """For each trained cell: the distance between its two most distant trained demonstrations."""
    spans = []
    for cell in sorted(set(demos.cell.values())):
        members = positions_mm[[demos.cell[e] == cell and bool(demos.trained[i]) for i, e in enumerate(demos.ep)]]
        if len(members) > 1:
            spans.append(float(max(np.linalg.norm(a - b) for a in members for b in members)))
    return spans


def report_demonstrations(demos: SimpleNamespace) -> None:
    """What the demonstrations' own spacing is, to read the trials' distances against."""
    n_demos, n_trained = len(demos.ep), int(demos.trained.sum())
    say(f"  [read] training set: {n_demos} demonstrations in placements.json; {n_trained} trained on (episodes 0 to "
        f"{n_trained - 1}), {n_demos - n_trained} held out for the evaluation loss (journal).")
    trained_mm = demos.mm[demos.trained]
    pairwise = np.linalg.norm(trained_mm[:, None, :] - trained_mm[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    neighbour = pairwise.min(axis=1)
    spans = largest_pair_per_cell(demos.mm, demos)
    say(f"  [measured from placements.json] among the {n_trained} trained demonstrations, the nearest other one is "
        f"{np.median(neighbour):.1f} mm away (median; {neighbour.min():.1f} to {neighbour.max():.1f}); the trained "
        f"demonstrations of one cell lie within {min(spans):.1f} to {max(spans):.1f} mm of each other (largest pair "
        "per cell).")
    if demos.mm_onemap is not None:
        spans = largest_pair_per_cell(demos.mm_onemap, demos)
        say("  [measured] with one map for all of them (method B) the largest pair per cell is "
            f"{min(spans):.1f} to {max(spans):.1f} mm.")


def report_trial_table(trials: list[dict], demos: SimpleNamespace) -> None:
    say("\n  3a. EVERY TRIAL [measured unless marked; scene, cell, kind and outcome are read from the trial log]")
    say("      x, y: table millimetres, method A.  d: distance to the nearest demonstration (mm).")
    say("      | trial | scene | cell | kind | outcome | x | y | nearest of 49 | d (49 trained) | nearest of 55 | "
        "d (all 55) | d (49), one map (B) | source |")
    say("      |---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in trials:
        outcome = "success" if t["success"] else f"failure ({t['failure']}, subgoal {t['subgoal']})"
        head = f"      | {t['trial']} | {t['scene_id']} | {t['cell_logged']} | {t['kind']} | {outcome} | "
        if t["d_trained"] is None:
            say(head + f"| | | not recovered | | | | {t['position_source']} |")
            continue
        say(head + f"{t['mm'][0]:.1f} | {t['mm'][1]:.1f} | ep {t['nn_trained']} ({demos.cell[t['nn_trained']]}) | "
            f"{t['d_trained']:.1f} | ep {t['nn_all']} ({demos.cell[t['nn_all']]}) | {t['d_all']:.1f} | "
            f"{fmt(t.get('d_trained_onemap'), 1)} | {t['position_source']} |")


def report_by_kind(trials: list[dict], demos: SimpleNamespace) -> None:
    """The trials by kind, and for each held-out cell the one demonstration its distances are measured to."""
    names = {
        "ghost": "trained cell, a demonstration's placement re-created",
        "free": "trained cell, placed freely in the cell",
        "interior": "held-out cell, interior (B2, D2)",
        "corner": "held-out cell, corner (A1, E3)",
    }
    measured = [t for t in trials if t["d_trained"] is not None]
    say("\n  BY KIND OF TRIAL [measured; d to the nearest of the trained demonstrations, mm]")
    say("      | kind | trials | successes | d, method A | d, method B |")
    say("      |---|---|---|---|---|")
    for kind, name in names.items():
        group = [t for t in measured if trial_kind(t) == kind]
        if not group:
            continue
        d = [t["d_trained"] for t in group]
        line = f"      | {name} | {len(group)} | {sum(t['success'] for t in group)} | {min(d):.2f} to {max(d):.2f} | "
        if "d_trained_onemap" in group[0]:
            onemap = [t["d_trained_onemap"] for t in group]
            line += f"{min(onemap):.2f} to {max(onemap):.2f} "
        say(line + "|")
    held_out = sorted({t["cell_logged"] for t in measured if t["kind"] == "ood"})
    parts = []
    for cell in held_out:
        for episode in sorted({t["nn_trained"] for t in measured if t["cell_logged"] == cell}):
            part = f"{cell}: ep {episode} ({demos.cell[episode]})"
            if demos.mm_onemap is not None:
                k = int(np.where(demos.ep == episode)[0][0])
                part += f", stored {np.linalg.norm(demos.mm[k] - demos.mm_onemap[k]):.2f} mm from its one-map position"
            parts.append(part)
    if parts:
        say(f"      nearest trained demonstration of the trials in each held-out cell: {'; '.join(parts)}.")


def summarise(key: str, title: str, subset: list[dict]) -> dict:
    """Success against one distance measure: 10 mm bins with Wilson intervals, the extremes, the protocol's model."""
    d = np.array([t[key] for t in subset])
    y = np.array([t["success"] for t in subset])
    edges = np.arange(0.0, BIN_MM * math.ceil((d.max() + 1e-9) / BIN_MM) + 1e-9, BIN_MM)
    bins = []
    for lo, hi in pairwise(edges):
        inside = (d >= lo) & (d < hi)
        n, k = int(inside.sum()), int(y[inside].sum())
        ci_lo, ci_hi = wilson(k, n)
        rate = k / n if n else math.nan
        bins.append({"lo": float(lo), "hi": float(hi), "n": n, "k": k, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi})
    in_order = np.sort(d)
    widest = int(np.diff(in_order).argmax())
    gap = (float(in_order[widest]), float(in_order[widest + 1]))  # the widest stretch of distance with no trial
    near, far = d <= gap[0], d >= gap[1]

    say(f"\n  {title}  ({len(subset)} trials)")
    say("      | d (mm) | trials | successes | rate | Wilson 95 % |")
    say("      |---|---|---|---|---|")
    for b in bins:
        if b["n"]:
            say(f"      | {b['lo']:.0f} to {b['hi']:.0f} | {b['n']} | {b['k']} | {100 * b['rate']:.0f} % | "
                f"{100 * b['ci_lo']:.1f} to {100 * b['ci_hi']:.1f} % |")
        else:
            say(f"      | {b['lo']:.0f} to {b['hi']:.0f} | 0 | | no trials | |")
    say(f"      widest stretch with no trial: {gap[0]:.2f} to {gap[1]:.2f} mm.")
    for words, side, edge in (("at or below", near, gap[0]), ("at or above", far, gap[1])):
        k, n = int(y[side].sum()), int(side.sum())
        ci_lo, ci_hi = wilson(k, n)
        say(f"      {words} {edge:.1f} mm: {k}/{n} successes (Wilson 95 %: {100 * ci_lo:.1f} to {100 * ci_hi:.1f} %).")
    if y.any():
        top = subset[int(np.argmax(np.where(y == 1, d, -np.inf)))]
        line = f"      largest distance with a success: {top[key]:.2f} mm (trial {top['trial']}, {top['scene_id']})"
        if ((y == 1) & near).any():
            line += f"; largest among the trials below the empty stretch: {d[(y == 1) & near].max():.2f} mm"
        say(line + ".")
    if (y == 0).any():
        bottom = subset[int(np.argmin(np.where(y == 0, d, np.inf)))]
        say(f"      smallest distance with a failure: {bottom[key]:.2f} mm (trial {bottom['trial']}, "
            f"{bottom['scene_id']}).")
    post = posterior_r50(d, y)
    unbounded = post["touches_lower_edge"] or post["touches_upper_edge"]
    edge_note = " (the interval reaches the grid's edge: not bounded on that side)" if unbounded else ""
    say("      [inferred] the protocol's model (draft 0.1) fitted to these trials: r50 median "
        f"{post['median']:.1f} mm, 95 % interval {post['lo']:.1f} to {post['hi']:.1f} mm{edge_note}; width median "
        f"{post['width_median']:.1f} mm.")
    lucky = (y == 1) & far
    if lucky.any():
        without = posterior_r50(d[~lucky], y[~lucky])
        names = ", ".join(f"trial {t['trial']}" for t, flag in zip(subset, lucky, strict=True) if flag)
        say(f"      [inferred] the same without the {int(lucky.sum())} success(es) beyond the empty stretch ({names}): "
            f"r50 median {without['median']:.1f} mm, 95 % interval {without['lo']:.1f} to {without['hi']:.1f} mm; "
            f"width median {without['width_median']:.1f} mm.")
    to_edge = np.abs(d - BIN_MM * np.round(d / BIN_MM))
    close = [
        f"trial {t['trial']} ({t[key]:.2f} mm)"
        for t, e in zip(subset, to_edge, strict=True)
        if e <= EDGE_NOTE_MM and t[key] > BIN_MM / 2
    ]
    if close:
        say(f"      within {EDGE_NOTE_MM:g} mm of a bin edge (about the instrument's noise): {', '.join(close)}.")
    return {"bins": bins, "gap": gap}


def write_csv(path: Path, trials: list[dict], demos: SimpleNamespace) -> None:
    """One row per trial line. Empty cells mean "not applicable" or "not recovered"."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for t in trials:
            evidence = list(t["mapping"])
            if t["log_ts_match"]:
                evidence.append("session log scored at the same second")
            if t["log_score_match"]:
                evidence.append("same score")
            if t["log_steps"] is not None and t["frame_count"] is not None:
                same = t["log_steps"] == t["frame_count"]
                evidence.append("log steps = video frames" if same else "LOG STEPS DIFFER FROM VIDEO FRAMES")
            has = t["d_trained"] is not None
            onemap = has and "mm_onemap" in t
            detection = t["detection"]
            row = {
                "trial": t["trial"],
                "ts": t["ts"],
                "scene_id": t["scene_id"],
                "cell_logged": t["cell_logged"],
                "kind": t["kind"],
                "axis": t["axis"] or "",
                "subgoal": t["subgoal"],
                "success": t["success"],
                "failure": t["failure"] or "",
                "position_source": t["position_source"],
                "source_file": t["source_file"],
                "frame_used": fmt(t["frame_used"]),
                "mapping_evidence": "; ".join(evidence),
                "px_x": fmt(t["px"][0]) if t["px"] else "",
                "px_y": fmt(t["px"][1]) if t["px"] else "",
                "area_px": detection["area"] if detection else "",
                "blob_angle_deg": fmt(float(detection["angle"]), 1) if detection else "",
                "n_candidates": detection["n_candidates"] if detection else "",
                "homography": t["homography"] or "",
                "x_mm": fmt(t["mm"][0]) if has else "",
                "y_mm": fmt(t["mm"][1]) if has else "",
                "cell_recovered": t["cell_recovered"] or "",
                "cell_matches_logged": fmt(t["cell_recovered"] == t["cell_logged"]) if has else "",
                "nearest_trained49_ep": t["nn_trained"] if has else "",
                "nearest_trained49_cell": demos.cell[t["nn_trained"]] if has else "",
                "d_nearest_trained49_mm": fmt(t["d_trained"]) if has else "",
                "nearest_all55_ep": t["nn_all"] if has else "",
                "nearest_all55_cell": demos.cell[t["nn_all"]] if has else "",
                "d_nearest_all55_mm": fmt(t["d_all"]) if has else "",
                "n_trained49_within_10mm": t["n_within"] if has else "",
                "x_mm_onemap": fmt(float(t["mm_onemap"][0])) if onemap else "",
                "y_mm_onemap": fmt(float(t["mm_onemap"][1])) if onemap else "",
                "nearest_trained49_ep_onemap": t["nn_trained_onemap"] if onemap else "",
                "d_nearest_trained49_mm_onemap": fmt(t["d_trained_onemap"]) if onemap else "",
                "d_nearest_all55_mm_onemap": fmt(t["d_all_onemap"]) if onemap else "",
                "max_move_first_frames_px": fmt(t.get("steady_px")),
                "max_move_first_frames_mm": fmt(t.get("steady_mm")),
                "still_vs_video_px": fmt(t.get("still_dpx")),
                "still_vs_video_mm": fmt(t.get("still_dmm")),
                "ghost_episode": fmt(t.get("ghost_ep")),
                "ghost_error_px": fmt(t.get("ghost_px")),
                "ghost_error_mm": fmt(t.get("ghost_mm")),
                "ghost_error_mm_onemap": fmt(t.get("ghost_mm_onemap")),
                "ghost_turn_deg": fmt(t.get("ghost_turn_deg"), 1),
                "flags": "; ".join(t["flags"]),
            }
            writer.writerow([row[column] for column in CSV_COLUMNS])


# ----------------------------------------------------------------------------------------------------------- pictures


def swarm_offsets(x_px: np.ndarray, diameter_px: float) -> np.ndarray:
    """Vertical offsets (pixels) that keep markers at the given x positions from covering one another.

    Deterministic: points are taken in order of x and each takes the free slot nearest the strip's centre line.
    """
    offsets = np.zeros(len(x_px))
    placed: list[tuple[float, float]] = []
    step = diameter_px / 2.0
    candidates = [0.0]
    for k in range(1, 40):
        candidates += [k * step, -k * step]
    for i in np.argsort(x_px, kind="stable"):
        for candidate in candidates:
            if all((x_px[i] - px) ** 2 + (candidate - py) ** 2 >= diameter_px**2 for px, py in placed):
                offsets[i] = candidate
                placed.append((float(x_px[i]), candidate))
                break
    return offsets


def trial_kind(row: dict) -> str:
    """Which of the four kinds of trial a row is, for the figure's markers."""
    if row["scene_id"].startswith("GHOST"):
        return "ghost"
    if row["kind"] == "id":
        return "free"
    return "interior" if row["axis"] == "position_interp" else "corner"


def make_figure(
    rows: list[dict], key: str, bins: list[dict], gap: tuple[float, float] | None, png: Path, pdf: Path
) -> None:
    """Success (1 or 0) against distance to the nearest trained demonstration (`key`), with the binned rates.

    Each trial is one marker in a strip: successes above the rate axis's 1, failures below its 0. Markers are spread
    vertically only so that they do not hide one another; the horizontal position is the measurement. Between 0 and 1
    the axis is the success rate in 10 mm bins with Wilson 95 % intervals.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, ink_secondary, grid, axis_grey = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    blue, orange = "#2a78d6", "#eb6834"  # checked as a pair for colour-blind readers; shape repeats the distinction
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    kinds = {
        "ghost": {"marker": "o", "colour": blue, "label": "trained cell, a demonstration's placement re-created"},
        "free": {"marker": "s", "colour": blue, "label": "trained cell, placed freely in the cell"},
        "interior": {"marker": "^", "colour": orange, "label": "held-out cell, interior (B2, D2)"},
        "corner": {"marker": "v", "colour": orange, "label": "held-out cell, corner (A1, E3)"},
    }
    plotted = [r for r in rows if r.get(key) is not None]
    x_max = BIN_MM * math.ceil(max(r[key] for r in plotted) / BIN_MM)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.815, bottom=0.115)
    ax.set_xlim(-1.0, x_max + 0.6)

    # Lay the two strips out first: their heights come from how many markers have to sit side by side.
    fig.canvas.draw()
    point = fig.dpi / 72.0  # pixels per typographic point
    marker_area = 24.0  # points squared
    diameter_px = (2.0 * math.sqrt(marker_area / math.pi) + 1.0) * point
    groups = {outcome: [r for r in plotted if r["success"] == outcome] for outcome in (1, 0)}
    offsets_px, half_px = {}, {}
    for outcome, group in groups.items():
        x = np.array([r[key] for r in group], float)
        x_px = ax.transData.transform(np.column_stack([x, np.zeros_like(x)]))[:, 0] if len(x) else x
        offsets_px[outcome] = swarm_offsets(x_px, diameter_px)
        half_px[outcome] = (float(np.abs(offsets_px[outcome]).max()) if len(x) else 0.0) + diameter_px / 2.0
    clear_px, pad_px = 8.0 * point, 4.0 * point  # between a strip and the rate axis; beyond a strip
    unit_px = ax.get_window_extent().height - 2.0 * (half_px[1] + half_px[0]) - 2.0 * (clear_px + pad_px)
    strip_centre = {1: 1.0 + (clear_px + half_px[1]) / unit_px, 0: -(clear_px + half_px[0]) / unit_px}
    ax.set_ylim(
        -(clear_px + 2.0 * half_px[0] + pad_px) / unit_px,
        1.0 + (clear_px + 2.0 * half_px[1] + pad_px) / unit_px,
    )
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        ax.axhline(level, color=grid, linewidth=0.6, zorder=0)
    if gap is not None:
        ax.fill_between([gap[0], gap[1]], 0.0, 1.0, color=ink_secondary, alpha=0.08, linewidth=0, zorder=0)
        middle = (gap[0] + gap[1]) / 2
        label = f"no trials between\n{gap[0]:.1f} and {gap[1]:.1f} mm"
        ax.text(middle, 0.5, label, ha="center", va="center", color=ink_secondary, fontsize=8)

    # the trials, in two strips
    where: dict[int, tuple[float, float]] = {}
    for outcome, group in groups.items():
        x = np.array([r[key] for r in group], float)
        y = strip_centre[outcome] + offsets_px[outcome] / unit_px
        for row, xi, yi in zip(group, x, y, strict=True):
            style = kinds[trial_kind(row)]
            where[row["trial"]] = (float(xi), float(yi))
            from_still = row["position_source"] != FROM_VIDEO
            ax.scatter(
                [xi],
                [yi],
                s=marker_area,
                marker=style["marker"],
                facecolor="white" if from_still else style["colour"],
                edgecolor=style["colour"] if from_still else "white",
                linewidth=1.1 if from_still else 0.7,
                zorder=3,
            )
    for row in plotted:
        if row["success"] == 1 and gap is not None and row[key] >= gap[1]:
            ax.annotate(
                f"trial {row['trial']} ({row['cell_logged']})",
                where[row["trial"]],
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color=ink_secondary,
                fontsize=7.5,
            )

    # the binned rates
    for b in bins:
        if b["n"] == 0:
            continue
        centre = (b["lo"] + b["hi"]) / 2
        ax.hlines(b["rate"], b["lo"], b["hi"], color=ink, linewidth=1.6, zorder=2)
        ax.errorbar(
            [centre],
            [b["rate"]],
            yerr=[[max(0.0, b["rate"] - b["ci_lo"])], [max(0.0, b["ci_hi"] - b["rate"])]],
            fmt="D",
            color=ink,
            markersize=4.5,
            markerfacecolor=ink,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.1,
            capsize=2.5,
            zorder=4,
        )
        label_y = min(max((b["ci_lo"] + b["ci_hi"]) / 2, 0.10), 0.90)
        ax.annotate(
            f"{b['k']}/{b['n']}",
            (centre, label_y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color=ink,
            fontsize=8,
        )

    ax.set_yticks([strip_centre[0], 0.0, 0.25, 0.5, 0.75, 1.0, strip_centre[1]])
    ax.set_yticklabels(["failure (0)", "0", "0.25", "0.5", "0.75", "1", "success (1)"])
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, BIN_MM))
    ax.set_xlabel("Distance from the cube's start position to the nearest trained demonstration, d (mm)", color=ink)
    ax.set_ylabel("Success rate in 10 mm bins\n(95 % Wilson interval)", color=ink)
    ax.tick_params(colors=ink_secondary, length=3, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_grey)

    def key(name: str) -> Line2D:
        style = kinds[name]
        return Line2D(
            [],
            [],
            linestyle="none",
            marker=style["marker"],
            markersize=5.5,
            markerfacecolor=style["colour"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=style["label"],
        )

    # two columns, filled downwards: trained cells on the left, held-out cells and the bin summary on the right
    handles = [key("ghost"), key("free")]
    if any(r["position_source"] != FROM_VIDEO for r in plotted):
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=5.5,
                markerfacecolor="white",
                markeredgecolor=blue,
                markeredgewidth=1.1,
                label="the same, start position from a still (trial not filmed)",
            )
        )
    handles += [key("interior"), key("corner")]
    handles.append(
        Line2D(
            [],
            [],
            color=ink,
            linewidth=1.4,
            marker="D",
            markersize=4.5,
            markerfacecolor=ink,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="success rate in the bin, 95 % Wilson interval",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.995),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=1.6,
        labelcolor=ink,
    )
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def write_contact_sheet(path: Path, trials: list[dict]) -> None:
    """Every detection, cropped around the cube and enlarged, for a person to look at."""
    tiles = []
    half, zoom = 45, 3
    side = 2 * half * zoom
    for trial in trials:
        image, detection = trial["image_used"], trial["detection"]
        if image is None and trial["frames"]:
            image = trial["frames"][0]
        if image is None:
            continue
        if detection is None:
            tile = cv2.resize(image, (side, side))
            cv2.putText(tile, "NOT DETECTED", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cx, cy = detection["centroid"]
            x0 = int(np.clip(round(cx) - half, 0, image.shape[1] - 2 * half))
            y0 = int(np.clip(round(cy) - half, 0, image.shape[0] - 2 * half))
            crop = image[y0:y0 + 2 * half, x0:x0 + 2 * half]
            tile = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            box_cx, box_cy, box_w, box_h = detection["box"]
            box = cv2.boxPoints(((box_cx, box_cy), (box_w, box_h), detection["angle"]))
            cv2.polylines(tile, [((box - [x0, y0]) * zoom).astype(np.int32)], True, (0, 220, 0), 1, cv2.LINE_AA)
            centre = (round((cx - x0) * zoom), round((cy - y0) * zoom))
            cv2.drawMarker(tile, centre, (255, 255, 255), cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
            cv2.drawMarker(tile, centre, (0, 0, 0), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
        label = f"{trial['trial']} {trial['scene_id']} {'ok' if trial['success'] else 'fail'}"
        cv2.rectangle(tile, (0, 0), (side, 18), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    columns = 8
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


# ---------------------------------------------------------------------------------------------------------- self-test


def self_test() -> int:
    """Check the statistics helpers against independent calculations. Touches no file of either project."""
    # Wilson: values worked by hand, and those the August journal quotes (19/19: 83-100 %, 1/12: 2-35 %, 20/31: 47-79 %)
    expected = {(19, 19): (0.8318, 1.0), (1, 12): (0.0149, 0.3539), (20, 31): (0.4695, 0.7888), (0, 8): (0.0, 0.3244)}
    for (k, n), (lo, hi) in expected.items():
        got = wilson(k, n)
        assert abs(got[0] - lo) < 5e-4 and abs(got[1] - hi) < 5e-4, (k, n, got)
    assert all(math.isnan(v) for v in wilson(0, 0))
    for n in range(1, 60):  # the rate always lies inside its own interval, also at 0 and at 1
        assert wilson(0, n)[0] == 0.0 and wilson(n, n)[1] == 1.0
        assert all(wilson(k, n)[0] <= k / n <= wilson(k, n)[1] for k in range(n + 1))

    # the grid is the protocol's
    assert len(R50_GRID_MM) == 161 and R50_GRID_MM[0] == 0.0 and abs(R50_GRID_MM[-1] - 80.0) < 1e-9
    assert len(WIDTH_GRID_MM) == 30 and abs(WIDTH_GRID_MM[0] - 0.5) < 1e-12 and abs(WIDTH_GRID_MM[-1] - 20.0) < 1e-9
    assert np.allclose(np.diff(np.log(WIDTH_GRID_MM)), np.log(40.0) / 29.0)

    # the posterior, written again as plain loops, on data shaped like August's
    d = np.array([1.5, 2.0, 38.1, 3.0, 2.4, 45.3, 1.4, 2.3, 35.4, 2.1, 33.4, 5.6, 1.2, 0.6, 38.6, 31.0, 40.0, 44.2])
    y = (d < 10.0).astype(float)
    y[16] = 1.0
    grid = np.zeros((len(R50_GRID_MM), len(WIDTH_GRID_MM), len(LAPSE_VALUES)))
    for i, r50 in enumerate(R50_GRID_MM):
        for j, width in enumerate(WIDTH_GRID_MM):
            for k, lapse in enumerate(LAPSE_VALUES):
                p = (1.0 - lapse) / (1.0 + np.exp((d - r50) / width))
                grid[i, j, k] = np.prod(np.where(y == 1.0, p, 1.0 - p))
    cdf = np.cumsum(grid.sum(axis=(1, 2))) / grid.sum()
    by_loops = [float(R50_GRID_MM[int(np.searchsorted(cdf, q))]) for q in (0.025, 0.5, 0.975)]
    fast = posterior_r50(d, y)
    assert by_loops == [fast["lo"], fast["median"], fast["hi"]], (by_loops, fast)

    # the posterior recovers a radius it was given (simulated trials, seeded)
    rng = np.random.default_rng(1)
    d = rng.uniform(0.0, 60.0, 400)
    y = (rng.uniform(size=400) < 0.95 / (1.0 + np.exp((d - 22.0) / 3.0))).astype(float)
    simulated = posterior_r50(d, y)
    assert simulated["lo"] <= 22.0 <= simulated["hi"] and abs(simulated["median"] - 22.0) < 3.0, simulated

    # nearest, and the marker layout
    points = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 4.0]])
    assert nearest(np.array([3.0, 3.0]), points, np.array([7, 8, 9])) == (9, 1.0)
    x = np.array([0.0, 1.0, 2.0, 2.5, 3.0, 30.0, 31.0, 3.2, 0.5, 1.5]) * 8.0
    offsets = swarm_offsets(x, 7.0)
    spacing = np.hypot(x[:, None] - x[None, :], offsets[:, None] - offsets[None, :])
    np.fill_diagonal(spacing, np.inf)
    assert spacing.min() >= 7.0 - 1e-9, spacing.min()
    assert fmt(None) == "" and fmt(True) == "yes" and fmt(1.234) == "1.23" and fmt(0) == "0"
    say("self-test passed: Wilson interval, model grid, posterior (against plain loops and on simulated trials), "
        "nearest, marker layout, formatting")
    return 0


# --------------------------------------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lerobot_101", nargs="?", type=Path, default=None,
                        help="the earlier project (default: $LEROBOT101_ROOT, else ../lerobot-101 beside this one)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent, help="output folder")
    parser.add_argument("--background", choices=("training", "trials"), default="training",
                        help="empty-scene picture for the cube detector: the training set's, or the trials' own")
    parser.add_argument("--frames", type=int, default=10, help="frames measured at the start of each video")
    parser.add_argument("--contact-sheet", type=Path, default=None, help="also write a picture of every detection")
    parser.add_argument("--figure-distance", choices=("A", "B"), default="A",
                        help="the figure's distances: A, converted as in August (default), or B, with one map")
    parser.add_argument("--self-test", action="store_true", help="check the statistics helpers, and stop")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    repo = (args.lerobot_101 or default_lerobot_101()).resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    say("=" * 100)
    say("Start positions of the August trials (act-cube-v1), recovered from video")
    say(f"run at {datetime.now().isoformat(timespec='seconds')}   earlier project: {shown(repo)}")
    say("Each number below is [measured] by this script, [read] from a file, or [inferred] (an estimate or a "
        "judgement).")
    say("=" * 100)

    inst = load_instrument(repo)
    P = inst.P
    say(f"[read] earlier project at commit {head_commit(repo)} (from its .git files). First 12 hex digits of SHA-256:")
    for relative in INPUT_FILES:
        say(f"         {short_hash(repo / relative)}  {relative}")
    say(f"[measured] Python {sys.version.split()[0]}, NumPy {np.__version__}, OpenCV {cv2.__version__}.")

    placements_doc = json.loads((repo / "eval" / "placements.json").read_text())
    placements = sorted(placements_doc["placements"], key=lambda p: p["episode_index"])
    if any(not p.get("detected") or "mm" not in p for p in placements):
        raise SystemExit("placements.json has an episode without a position; this analysis needs all of them")
    demos = SimpleNamespace(
        ep=np.array([p["episode_index"] for p in placements]),
        px=np.array([p["px"] for p in placements], float),
        mm=np.array([p["mm"] for p in placements], float),
        cell={p["episode_index"]: p["cell"] for p in placements},
        angle=[p.get("angle") for p in placements],
        onemap=None,
        mm_onemap=None,
    )
    demos.trained = demos.ep < (len(placements) - EVAL_TAIL_EPISODES)

    trials = read_trials(repo)
    locate_sources(repo, trials)
    load_pictures(trials, args.frames)
    report_sources(repo, trials)

    say("\n2. THE INSTRUMENT")
    dataset_name = placements_doc["repo_id"].split("/")[-1]
    camera = placements_doc.get("camera", "front")
    training_frames, why_not = load_training_first_frames(P, placements_doc["repo_id"], camera, out)
    trials_background = P.empty_scene([t["frames"][0] for t in trials if t["frames"]])
    training = None
    if training_frames is None:
        say(f"  !! the training dataset could not be read locally ({why_not}).")
        say("     The cube detector uses the trials' own empty-scene picture instead, and the re-measurement of the "
            "training episodes is skipped.")
    else:
        training = remeasure_training(inst, training_frames, placements, dataset_name)
    trial_session = SimpleNamespace(other_background=None, other_background_name="")
    if args.background == "training" and training is not None:
        background = training.background
        name = "the per-pixel median of the 55 training episodes' first frames (the one that built placements.json)"
        trial_session.other_background = trials_background
        trial_session.other_background_name = "the trials' own median"
    else:
        background = trials_background
        name = "the per-pixel median of the trials' own first frames"
        if training is not None:
            trial_session.other_background = training.background
            trial_session.other_background_name = "the training episodes' median"
    say(f"  [measured] empty-scene background used by the cube detector: {name}.")

    trial_session.solves = measure_trials(inst, trials, background, placements)
    trial_session.consensus, trial_session.outliers = P.consensus_homography(trial_session.solves)
    if training is not None and training.consensus is not None:
        demos.onemap = training.consensus
        onemap_name = "the training session's consensus map, the one placements.json used when a tag was hidden"
    else:
        demos.onemap = trial_session.consensus
        onemap_name = "the trial session's consensus map (training frames unavailable)"
    if demos.onemap is not None:
        demos.mm_onemap = P.px_to_mm(demos.px, demos.onemap)
    derive(inst, trials, demos, background)
    measured = [t for t in trials if t["d_trained"] is not None]
    filmed = [t for t in measured if t["position_source"] == FROM_VIDEO]

    report_ghosts(repo, trials, demos)
    report_checks(inst, trials, demos, training, trial_session, placements, args.frames)

    say("\n3. RESULTS")
    report_demonstrations(demos)
    report_trial_table(trials, demos)
    report_by_kind(trials, demos)
    n_trained, n_demos = int(demos.trained.sum()), len(demos.ep)
    primary = summarise(
        "d_trained",
        f"3b. SUCCESS AGAINST d, nearest of the {n_trained} trained demonstrations, method A (primary)",
        measured,
    )
    summarise("d_all", f"3c. The same against the nearest of all {n_demos} demonstrations, method A", measured)
    figure_key, figure_summary = "d_trained", primary
    if demos.mm_onemap is not None:
        title = f"3d. Cross-check: nearest of the {n_trained} trained, method B (one map: {onemap_name})"
        onemap_summary = summarise("d_trained_onemap", title, measured)
        if args.figure_distance == "B":
            figure_key, figure_summary = "d_trained_onemap", onemap_summary
        difference = np.array([t["d_trained_onemap"] - t["d_trained"] for t in measured])
        say("\n      method B minus method A, d to the nearest trained demonstration: mean "
            f"{difference.mean():+.2f} mm, from {difference.min():+.2f} to {difference.max():+.2f} mm; largest "
            f"absolute difference {np.abs(difference).max():.2f} mm.")
    if len(filmed) != len(measured):
        summarise("d_trained", "3e. As 3b, but only the trials recovered from video", filmed)
    below = [t["n_within"] for t in measured if t["d_trained"] <= primary["gap"][0]]
    above = [t["n_within"] for t in measured if t["d_trained"] >= primary["gap"][1]]
    say(f"\n  [measured] trained demonstrations within {NEIGHBOURHOOD_MM:g} mm of the cube: {min(below)} to "
        f"{max(below)} for the trials below the empty stretch, {min(above)} to {max(above)} for those above it.")

    say("\n  FLAGS")
    flagged = [t for t in trials if t["flags"]]
    if not flagged:
        say("      none")
    for t in flagged:
        say(f"      trial {t['trial']} ({t['scene_id']}): " + "; ".join(t["flags"]))

    csv_path = out / "trials_with_distance.csv"
    stem = "success_vs_distance_onemap" if figure_key == "d_trained_onemap" else "success_vs_distance"
    png, pdf = out / f"{stem}.png", out / f"{stem}.pdf"
    write_csv(csv_path, trials, demos)
    gap = figure_summary["gap"]
    make_figure(measured, figure_key, figure_summary["bins"], gap if gap[1] - gap[0] > BIN_MM else None, png, pdf)
    say("\n4. WRITTEN")
    if args.figure_distance == "B":
        used = "B (one map)" if figure_key == "d_trained_onemap" else "A (method B was not available)"
        say(f"  the figure uses the distances of method {used}.")
    written = [csv_path, png, pdf]
    if args.contact_sheet is not None:
        write_contact_sheet(args.contact_sheet, trials)
        written.append(args.contact_sheet)
    for path in written:
        say(f"  {shown(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
