"""Write the trial map's data file from the study's own records. No camera, no arm, nothing outside this folder.

Run from the repository root with the project's interpreter:

    .venv/bin/python scripts/export_data.py

It reads, and changes none of:

- `data/trials/main.jsonl`: the 108 counted trials (condition, measured start, nearest demonstration, outcome, note)
- `data/audit/clustered.json`, `data/audit/spread.json`: where each demonstration's cube started, measured
- `config/protocol.toml`: the grid, the held-out cells and the blocks' titles
- `config/table_geometry.json`: the two AprilTags on the table, and the front camera's view used to turn the map
- `data/instrument/byeye_op1_20261008_172808.csv`: 20 resets made by eye with no overlay, each measured afterwards
- `data/camera_views.json`: for each session, the front camera's view of the table (table millimetres
  to picture pixels), solved from the frame the session started on with the session's own code;
  the page draws the grid over each trial's video through it

and writes `docs/data/trials.json`, plus `docs/data/trials.js`, the same object assigned to
`window.TRIAL_DATA`, for a browser that opens the page from a file URL and will not fetch a JSON file from disk.

Every value is copied from those files or computed from them with the study's own code (`tolerance_radius.geometry`).
Before writing, it checks each trial's logged distance against the demonstration starts it names, and stops if one
differs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
#: The trial map page, served by GitHub Pages.
PAGE = REPO_ROOT / "docs"
sys.path.insert(0, str(REPO_ROOT / "src"))

from tolerance_radius import geometry, schema  # noqa: E402
from tolerance_radius.geometry import Grid, Point  # noqa: E402

TRIALS = REPO_ROOT / "data" / "trials" / "main.jsonl"
AUDITS = {"clustered": REPO_ROOT / "data" / "audit" / "clustered.json",
          "spread": REPO_ROOT / "data" / "audit" / "spread.json"}
PROTOCOL = REPO_ROOT / "config" / "protocol.toml"
TABLE_GEOMETRY = REPO_ROOT / "config" / "table_geometry.json"
BY_EYE = REPO_ROOT / "data" / "instrument" / "byeye_op1_20261008_172808.csv"
VIEWS = REPO_ROOT / "data" / "camera_views.json"

#: The demonstration set each condition's distance is measured in. A replay plays back demonstrations of the
#: clustered set (protocol.toml, [blocks.T1b]); its distance is to the replayed demonstration itself.
DEMO_SET = {"act_clustered": "clustered", "act_spread": "spread", "replay": "clustered"}

#: The three clips Daniel approved for the portfolio on 2026-11-09, copied unchanged from
#: the portfolio site's assets. Each is byte-identical to the copy that was reviewed for its trial. The page suggests
#: them first.
#: Every other trial's clip is `assets/trials/<trial>.mp4`: its front-camera recording re-encoded for a browser the
#: same way, added at Daniel's request on 2026-11-10 (see the trial map's notes).
APPROVED_CLIPS = {
    "T1a-025": "arm-policy-far",
    "T1b-002": "arm-replay-far-miss",
    "T1b-008": "arm-replay-own-start",
}

#: Spelling fixed in the scorer's notes at Daniel's request on 2026-11-10: one word each, nothing reworded. The log
#: keeps every note as typed; the page marks a fixed note as "spelling corrected". Each pair must match its note
#: exactly once, as whole words, or the export stops. "mid air" (three notes) is an accepted spelling and stays.
NOTE_SPELLING = {
    "T1a-013": [("agian", "again")],
    "T1a-014": [("little but", "little bit")],
    "T1b-012": [("procedure a if", "procedure as if")],
    "T1b-018": [("god", "good")],
    "T2-012": [("tthen", "then")],
    "T2-013": [("went to low", "went too low")],
    "T2-024": [("grabed", "grabbed")],
    "T2-027": [("nuging", "nudging")],
    "T2-029": [("grabed", "grabbed")],
    "T2-033": [("re grasp", "regrasp")],
    "T2-035": [("re grasp", "regrasp")],
    "T2-040": [("god", "good")],
    "T2-049": [("nuged", "nudged")],
}


def fixed_note(trial_id: str, note: str, problems: list[str]) -> str:
    """The scorer's note with its listed spelling fixes applied, each checked to match exactly once."""
    for wrong, right in NOTE_SPELLING.get(trial_id, []):
        pattern = r"\b" + re.escape(wrong) + r"\b"
        if len(re.findall(pattern, note)) != 1:
            problems.append(f"{trial_id}: the note does not hold {wrong!r} exactly once")
            continue
        note = re.sub(pattern, right, note)
    return note


#: How far a logged distance may sit from the one recomputed here before the export stops.
DISTANCE_CHECK_MM = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_turn(geometry_file: dict, grid: Grid) -> dict:
    """Which way the grid's x and y run in the front camera's picture, from the view stored at registration."""
    mm_to_px = np.linalg.inv(np.array(geometry_file["registration"]["px_to_mm"], dtype=float))

    def px(x: float, y: float) -> tuple[float, float]:
        u, v, w = mm_to_px @ np.array([x, y, 1.0])
        return u / w, v / w

    width, height = grid.cols * grid.cell_mm, grid.rows * grid.cell_mm
    return {
        "x_runs_left": bool(px(width, height / 2)[0] < px(0, height / 2)[0]),
        "y_runs_down": bool(px(width / 2, height)[1] > px(width / 2, 0)[1]),
    }


def main() -> None:
    protocol = tomllib.loads(PROTOCOL.read_text(encoding="utf-8"))
    grid = Grid.from_protocol(protocol)
    table_geometry = json.loads(TABLE_GEOMETRY.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in TRIALS.read_text(encoding="utf-8").splitlines() if line.strip()]
    views = json.loads(VIEWS.read_text(encoding="utf-8"))

    demos = {name: geometry.trained_demos(schema.read_demo_starts(path), name) for name, path in AUDITS.items()}
    demo_by_episode = {name: {d.episode_index: d for d in ds} for name, ds in demos.items()}

    trials, problems = [], []
    for record in records:
        planned = record["planned"]
        condition = planned["condition"]
        if record["outcome"] not in ("success", "failure"):
            problems.append(f"{planned['trial_id']}: outcome {record['outcome']!r} is not a scored trial")
            continue
        demo_set = DEMO_SET[condition]
        placed = Point(record["placed"]["x_mm"], record["placed"]["y_mm"])
        nearest = demo_by_episode[demo_set].get(record["nearest_demo_episode"])
        if nearest is None:
            problems.append(f"{planned['trial_id']}: episode {record['nearest_demo_episode']} not in the {demo_set} set")
            continue
        recomputed = placed.distance_to(nearest.measured)
        if abs(recomputed - record["nearest_demo_distance_mm"]) > DISTANCE_CHECK_MM:
            problems.append(f"{planned['trial_id']}: logged {record['nearest_demo_distance_mm']} mm, "
                            f"recomputed {recomputed} mm")
        if condition == "replay":
            if record["nearest_demo_episode"] != planned["replay_episode"]:
                problems.append(f"{planned['trial_id']}: distance not taken to the replayed demonstration")
        else:
            index, distance = geometry.nearest(placed, demos[demo_set])
            if demos[demo_set][index].episode_index != record["nearest_demo_episode"]:
                problems.append(f"{planned['trial_id']}: logged nearest episode is not the nearest")
        cell = grid.cell_of(placed)
        trials.append({
            "id": planned["trial_id"],
            "block": planned["block"],
            "condition": condition,
            "demo_set": demo_set,
            "x_mm": placed.x_mm,
            "y_mm": placed.y_mm,
            "target_mm": [planned["target"]["x_mm"], planned["target"]["y_mm"]],
            "cell": cell,
            "held_out": cell in grid.held_out_cells,
            "d_mm": record["nearest_demo_distance_mm"],
            "nearest_episode": record["nearest_demo_episode"],
            "nearest_xy_mm": [nearest.measured.x_mm, nearest.measured.y_mm],
            "replay_episode": planned["replay_episode"],
            "level_mm": planned["level_mm"],
            "staircase_id": planned["staircase_id"],
            "pair_id": planned["pair_id"],
            "outcome": record["outcome"],
            "failure": record["failure"],
            "subgoal": record["subgoal"],
            "notes": fixed_note(planned["trial_id"], record["notes"], problems),
            "notes_spelling_fixed": [list(pair) for pair in NOTE_SPELLING.get(planned["trial_id"], [])],
            "ts_start": record["ts_start"],
            "session": Path(record["media"]["front"]).parent.name,
            "clip": APPROVED_CLIPS.get(planned["trial_id"], f"trials/{planned['trial_id']}"),
            "featured": planned["trial_id"] in APPROVED_CLIPS,
        })

    for trial_id in NOTE_SPELLING:
        if trial_id not in {t["id"] for t in trials}:
            problems.append(f"spelling fix for {trial_id}: the trial is not in the log")
    for trial_id, name in APPROVED_CLIPS.items():
        if trial_id not in {t["id"] for t in trials}:
            problems.append(f"approved clip {name}: trial {trial_id} is not in the log")
    for trial in trials:
        for suffix in (".mp4", ".jpg"):
            if not (PAGE / "assets" / f"{trial['clip']}{suffix}").is_file():
                problems.append(f"{trial['id']}: assets/{trial['clip']}{suffix} is missing from docs/assets")

    for trial in trials:
        if trial["session"] not in views:
            problems.append(f"{trial['id']}: no camera view for session {trial['session']} in {VIEWS.name}")
    by_eye = [float(row["error_mm"]) for row in csv.DictReader(BY_EYE.open(encoding="utf-8"))]
    if not by_eye:
        problems.append(f"{BY_EYE.name}: no resets in it")

    if problems:
        print("export stopped, nothing written:", *problems, sep="\n  ")
        sys.exit(1)

    data = {
        "about": {
            "study": "SO-101 arm, ACT policy and open-loop replay, counted trials of 2026-10-08",
            "sources": {
                "trials": {"path": str(TRIALS.relative_to(REPO_ROOT)), "sha256": sha256(TRIALS)},
                "demonstrations": {name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)}
                                   for name, path in AUDITS.items()},
                "protocol": {"path": str(PROTOCOL.relative_to(REPO_ROOT)), "sha256": sha256(PROTOCOL)},
                "table_geometry": {"path": str(TABLE_GEOMETRY.relative_to(REPO_ROOT)),
                                   "sha256": sha256(TABLE_GEOMETRY)},
                "resets_by_eye": {"path": str(BY_EYE.relative_to(REPO_ROOT)), "sha256": sha256(BY_EYE)},
                "camera_views": {"path": str(VIEWS.relative_to(REPO_ROOT)), "sha256": sha256(VIEWS)},
            },
            "written_by": "scripts/export_data.py",
            "lengths": "millimetres in the table frame of the AprilTags; positions are the cube's measured centre",
        },
        "table": {
            "cols": grid.cols,
            "rows": grid.rows,
            "cell_mm": grid.cell_mm,
            "held_out_cells": list(grid.held_out_cells),
            "cube_edge_mm": grid.cube_edge_mm,
            # A reset is accepted when the camera measures the cube's center within this of its target.
            "placement_tolerance_mm": float(protocol["placement"]["tolerance_mm"]),
        },
        # Resets made by eye from a reference picture, with no overlay: how far each landed from its target.
        "resets_by_eye_mm": by_eye,
        # Each session's front-camera view: a 3 x 3 matrix from table millimetres to pixels of the 640 x 480 picture.
        "camera_views": {session: {"mm_to_px": entry["mm_to_px"], "frame_px": entry["frame_px"]}
                         for session, entry in sorted(views.items())
                         if session in {t["session"] for t in trials}},
        "tags": [{"id": tag_id, "corners_mm": tag["corners"], "center_mm": tag["centre"]}
                 for tag_id, tag in sorted(table_geometry["tags"].items())],
        "camera_view": camera_turn(table_geometry, grid),
        # `planned` is where a start was aimed: in the clustered set five starts share each planned point, and the
        # page draws them as one cross at the average of their measured starts.
        "demos": {name: [{"episode": d.episode_index, "cell": d.cell, "x_mm": d.measured.x_mm,
                          "y_mm": d.measured.y_mm,
                          "planned_mm": None if d.planned is None else [d.planned.x_mm, d.planned.y_mm]}
                         for d in ds] for name, ds in demos.items()},
        "blocks": {name: block["title"] for name, block in protocol["blocks"].items()},
        "trials": trials,
    }
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    out = PAGE / "data"
    out.mkdir(exist_ok=True)
    (out / "trials.json").write_text(text, encoding="utf-8")
    script = ("// Written by export_data.py with trials.json, for pages opened from a file URL. Do not edit.\n"
              "window.TRIAL_DATA = " + text.replace("</", "<\\/").rstrip("\n") + ";\n")
    (out / "trials.js").write_text(script, encoding="utf-8")

    print(f"wrote data/trials.json and data/trials.js: {len(trials)} trials, "
          f"{sum(len(d) for d in demos.values())} demonstration starts, {len(trials)} clips")
    for condition in ("act_clustered", "act_spread", "replay"):
        mine = [t for t in trials if t["condition"] == condition]
        far = [t for t in mine if t["d_mm"] > 20]
        own = [t for t in mine if t["level_mm"] == 0.0]
        print(f"  {condition}: {len(mine)} trials, {sum(t['outcome'] == 'success' for t in mine)} succeeded; "
              f"beyond 20 mm {sum(t['outcome'] == 'success' for t in far)} of {len(far)}; "
              f"at a demonstration's own start {sum(t['outcome'] == 'success' for t in own)} of {len(own)}")
    turn = data["camera_view"]
    print(f"  map turned as the front camera sees it: x runs {'left' if turn['x_runs_left'] else 'right'}, "
          f"y runs {'down' if turn['y_runs_down'] else 'up'}")
    largest = max(math.hypot(t["x_mm"] - t["nearest_xy_mm"][0], t["y_mm"] - t["nearest_xy_mm"][1]) for t in trials)
    print(f"  largest distance to the named demonstration: {largest:.1f} mm")


if __name__ == "__main__":
    main()
