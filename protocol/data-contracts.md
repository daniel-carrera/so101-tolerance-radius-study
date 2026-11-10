# Data contracts

The records this project writes and reads. `src/tolerance_radius/schema.py` defines them and checks them; this page
explains them. Where the two disagree, the code is right.

## Conventions

- A position is where the front camera measures the cube to be, table frame, millimetres (`x_mm`, `y_mm`): the
  centre of the cube's visible outline mapped through the table plane, a few millimetres from the centre of its
  footprint. Every position is measured the same way, so the offset cancels in distances (`protocol/protocol.md`,
  section 7).
- A name ending in `_mm`, `_s`, `_deg` or `_hz` carries that unit.
- Times are ISO 8601 with a UTC offset.
- People are role codes (`op1`, `sc1`), never names. Paths are relative to the repository.

## Files

| File | Written by | Read by | Kept in git |
|---|---|---|---|
| `config/protocol.toml` | hand, before the freeze | planner, tools, analysis | yes |
| `config/layouts/<dataset>.json` | `scripts/make_layouts.py` | the recording tool | yes |
| `config/table_geometry.json` | `scripts/rig/measure.py register-table` (once per table layout) | every tool that ties the camera's picture to the table | yes |
| `data/audit/<dataset>.json` | the start-state audit | planner, analysis | yes |
| `data/audit/<dataset>_camera.json` | the start-state audit: per episode, the camera's distance from the calibration and the picture's levels in its first frame (`episodes`); what the front camera's picture did inside the take, one frame a second (`takes`: frames looked at, frames with both floor tags, how far the tags lay from the take's first frame, the range of their black); and the wrist camera's first frame by the number that tells its tone state, with its darkest and brightest hundredth and its mean (`wrist`) | people, the paper's rig table | yes |
| `data/audit/<dataset>_rate.json` | `scripts/rig/recording_rate.py`, from the sessions' logs | people, the paper's rig table | yes |
| `data/audit/scene_<log>.json` | `scripts/rig/scene_audit.py --log <log>`, from the trials' own videos: for every record of the log, where the basket and the wrist camera stood at its first and last frames, and the two limits beyond which a trial is named | analysis (`results.json`, `scene`) | yes |
| `data/trials/<session>.jsonl` | the trial runner | planner, analysis | yes |
| `data/recordings/` (trial videos), datasets, checkpoints | the rig tools, training | people | no (their hashes go in the journal) |
| `data/instrument/table_view.json` | `scripts/rig/measure.py calibrate` | every tool that measures a position, as the reference it holds the live picture against | yes |
| `data/instrument/*.csv` | `scripts/rig/measure.py log` and `byeye` | `scripts/check_instrument.py`, analysis | yes |
| `data/instrument/not_used/` | moved there by hand, each file with its reason in the folder's `README.md` | nothing: the analysis reads the files of `data/instrument/` itself and not this folder | yes |
| `data/instrument/*.png` | `scripts/rig/measure.py`: `view_check_<time>` (a check's marked picture) and `view_raw_<time>` (the same frame as it came), `view_live_<time>`, `table_registration_<time>` and its three `_raw` frames, `calibration_<time>` (the frame a calibration kept, stored or refused), `session_<time>` (the frame a session or a logging command started on, as it came) | people, `analysis/instrument/` | yes |
| `data/recording/<dataset>_placements.jsonl` | the recording tool | analysis (re-records, placement errors) | yes |
| `analysis/report/results.json` | `scripts/make_report.py` | the figure scripts, people | yes |
| `analysis/report/report.md` | `scripts/make_report.py` | people | yes |
| `paper/generated/` | `scripts/make_report.py` | the paper | yes |
| `paper/figures/` | `scripts/make_figures.py` | the paper | yes |

## The table geometry (`config/table_geometry.json`)

Written by one registration (`src/tolerance_radius/rig/registration.py`). The first four keys are what the rest of the
instrument reads; `registration` is the record of how they were made.

| Key | Meaning |
|---|---|
| `tag_mm` | The floor tags' nominal size, 60 |
| `tags` | For floor tags `"0"` and `"1"`: `centre` and the four `corners` in table millimetres, in the detector's order: **the tag's outline as measured from the picture**, put through the table frame. Every session measures the floor tags the same way |
| `grid` | `cols`, `rows`, `cell_mm`, from the protocol |
| `note`, `created` | The operator's line and the time |
| `registration.sheet_tags`, `frames` | The sheet tags the frame was fitted through, and how many frames were averaged |
| `registration.sheet_scale`, `sheet_outline_measured_mm` | The sheet's printed scale across and up, and the caliper reading of the grid's outline it came from (`null`: taken as designed) |
| `registration.sheet_fit_mean_mm`, `sheet_fit_max_mm` | How far the sheet tags' outline corners are from one flat plane seen through an ideal lens |
| `registration.leave_one_out_mean_mm`, `leave_one_out_max_mm` | Each sheet tag left out of the fit in turn, its corners predicted by the others |
| `registration.sheet_tag_scale`, `sheet_tag_scale_each` | The sheet tags' size read through the frame, as a share of their printed size, across and up: the mean over the tags, and each tag by its id |
| `registration.detector_inset_px` | How far inside a tag's outline the detector puts its corners, mean over the tags in view |
| `registration.floor_tags_stated_mm` | Each floor tag's size by calipers, across and up the tag as printed (60 x 60 when not given) |
| `registration.floor_tag_sides_mm`, `floor_tag_detector_sides_mm` | Each floor tag's four sides through the frame: by its outline (what is stored), and by the detector's own corners, about a millimetre smaller (sides 1 and 3 are the tag's width, 2 and 4 its height) |
| `registration.floor_tag_scale` | Each floor tag's outline size as a share of its stated size: across, up |
| `registration.single_frame_wander_mean_mm`, `single_frame_wander_max_mm` | The frame solved from one frame's floor tags against the registration's frame: its largest gap anywhere on the grid, averaged over the frames, and the largest of all |
| `registration.px_to_mm` | The table frame itself: the 3 x 3 conversion from pixels to table millimetres fitted through the sheet. The calibration says how far its own solve lies from it |
| `registration.floor_tag_corners_px` | Where each floor tag's outline corners were in the picture, averaged over the frames. The calibration compares where they are now: tags that moved the same way mean the camera moved; tags that did not move alike mean a camera that turned or a tag that was moved, which the two tags cannot tell apart (noted beyond 0.5 px, refused beyond 1.5 mm on the floor) |
| `registration.notes` | Sentences the registration printed as `NOTE` and stored: figures beyond what a flat sheet and flat tags give, but not beyond use |

## The stored calibration (`data/instrument/table_view.json`)

Since the afternoon of 2026-10-06 no position is measured through this file's conversion: every session solves
its own from the two floor tags as it finds them, and holds this file against it (how far the camera's picture
has moved, the picture's levels, the registration it belongs to).

| Key | Meaning |
|---|---|
| `px_to_mm` | The 3 x 3 conversion from the front camera's pixels to table millimetres, solved at the calibration from the two floor tags' outlines averaged over its frames |
| `residual_mm`, `frames_used` | How well the two floor tags fitted the table geometry in that solve, and from how many frames |
| `geometry_created` | The `created` stamp of the table geometry the solve was made against. A session refuses a stored conversion whose stamp is not that of the table geometry in place |
| `floor_levels` | The picture the calibration saw. `grey`: the median grey level (0 to 255) of the floor inside the grid; `tag_black`: the grey level of the floor tags' black; `red_over_blue`: the median of red over blue on the floor. A session holds the picture it opens on against them and refuses to start when `grey` is more than 25 off or the contrast, `grey` over `tag_black`, is outside 0.7 to 1.4 of the stored one; the colour only warns. `null` in a file stored before 2026-10-05 |
| `created`, `note` | The time and the operator's line |
| `record` | What the calibration saw, read by no tool. `picture`: the frame it kept, as it came (`data/instrument/calibration_<time>.png`). `floor_tag_corners_px`: where each floor tag's outline corners were in the picture. `since_registration`: each tag's shift in pixels since the table was registered (`shift_px`), how far the two shifts differ (`differ_px`), the turn of the pair in degrees (`turn_deg`) and the change of the distance between them as a share (`stretch`), their movement against each other on the floor through the registered frame (`relative_mm`), how far apart the tags lie (`tags_apart_mm`), and the largest gap between this solve and the registered frame on the grid (`frames_apart_mm`) |

## The camera and the picture, episode by episode (`data/audit/<dataset>_camera.json`)

Written by the start-state audit beside the starts. Every start is measured through a conversion solved from the
two floor tags as that episode's own first frame shows them (averaged with up to two episodes either side whose
tags lie within 0.4 px of its own), because the camera moves and the tags do not.

| Key | Meaning |
|---|---|
| `calibration_created` | The stored calibration the episodes are held against |
| `tag_noise_px` | How far the tags' outline corners lie apart from one episode to the next (px, rms over neighbours in the same view): what one recorded frame's corners are worth |
| `episodes.<index>.camera_moved_mm` | How far that episode's conversion puts the grid's corners and centre from where the stored calibration puts them, at most: the camera's movement since the calibration, on the floor |
| `episodes.<index>.frames` | Over how many episodes' first frames its tags' corners were averaged (1 to 5) |
| `episodes.<index>.grey`, `tag_black`, `red_over_blue` | The picture's levels in that frame, as the calibration stores its own. Frames are read as LeRobot's reader gives them, which is as the camera gave them to the recorder (until the evening of 2026-10-06 a "stretch by the readers" was undone here that does not exist) |
| `takes.<index>.frames`, `frames_with_tags` | Inside the take: how many frames of the front camera's video were looked at (one a second, and the last), and in how many both floor tags were in sight |
| `takes.<index>.tags_moved_px` | The largest distance, over those frames, of a floor tag's outline corner from where the take's first such frame has it. Over 3 px the audit names the episode: the picture changed inside the take |
| `takes.<index>.black_low`, `black_high` | The range of the floor tags' black over those frames, as a share of what it is in the take's first. Outside 0.7 to 1.4 the audit names the episode: the tone range changed inside the take |
| `wrist.<index>.contrast` | The wrist camera's first frame of the episode by the number that tells its tone state: the floor's grey over the green of the jaws, the green brought to a floor of 195 along the line the usual state keeps (`vision.wrist_contrast`). 2.50 to 2.64 in the usual state and 3.08 to 3.35 in the other, *measured* on 2,893 rest frames of the takes of 2026-10-06; the line is 2.85 (`vision.WRIST_OTHER_ABOVE`). `null` when the frame is not the rest picture (the jaws over the floor). The audit names the episodes above the line, the episodes that could not be judged, and a set whose readings fall into two groups on one side of the line |
| `wrist.<index>.darkest`, `brightest`, `mean` | The same frame's darkest and brightest hundredth and its mean, for the record: they move with the exposure, and no rule goes by them |

An episode whose floor tags could not be measured has no entry and is measured through the nearest episode's
conversion; the audit names it.

## A demonstration's start (`DemoStart`)

One per recorded episode, from the start-state audit.

| Field | Meaning |
|---|---|
| `dataset` | `clustered` or `spread` |
| `episode_index` | The episode's index in its dataset |
| `cell` | The grid cell the measured position falls in |
| `measured` | Where the cube was, from the episode's first frame |
| `planned` | Where the layout asked for it |
| `trained` | False for an episode that was recorded but left out of training |

Distances in the analysis are always taken to `measured` positions of episodes with `trained = true`.

## A planned trial (`PlannedTrial`)

What the planner asks for before a trial runs. It is stored inside the trial's record.

| Field | Meaning |
|---|---|
| `trial_id` | Unique in a log, readable: `T1a-017` |
| `block` | `T1a`, `T1b`, `T2` or `T3` |
| `condition` | `act_clustered`, `act_spread` or `replay` |
| `blind_code` | What the scorer sees in place of the condition |
| `target` | Where to put the cube |
| `level_mm` | The requested distance to the nearest demonstration; absent when the placement was drawn uniformly |
| `staircase_id`, `staircase_index` | Which staircase asked, and its trial number from 0 |
| `pair_id` | Shared by the two trials that test one placement with two policies |
| `replay_episode` | The demonstration replayed (replay only) |

## A trial (`TrialRecord`)

One line of a trial log. A log is appended to and never rewritten.

| Field | Meaning |
|---|---|
| `planned` | The plan, as above |
| `placed` | Where the cube was when the placement was accepted, as measured |
| `nearest_demo_episode`, `nearest_demo_distance_mm` | The nearest trained demonstration to `placed` in the condition's own set, and the distance. The analysis recomputes both and stops if they differ |
| `outcome` | `success`, `failure` or `void` |
| `subgoal` | Graded progress, 0 to 5; success exactly at 5 |
| `failure` | A label from the failure taxonomy; absent on success |
| `void_reason` | Why a void trial does not count |
| `notes` | Free text from the scorer |
| `operator`, `scorer` | Role codes |
| `policy_repo`, `checkpoint` | The policy tested; both absent for a replay |
| `execution` | Every setting of the protocol's `[execution]` section as the session ran it (the same list in every record), and what belongs to this trial alone: `steps_sent`; `loop_s`, the seconds from the first step sent to the last on the runner's monotonic clock, to a millisecond (0 when fewer than two steps were sent); `device`; `handoff_tolerance_used_deg` (5 for a replay, 8 for a policy); `start_pose_deg`, the arm's joint positions when the attempt began: the pose the handoff gate passed at (since the night of 2026-10-07; until then the pose read before the gate's first look, which after a correction is the pose that was refused: 8 of the 21 scored trials of the replay pilot). A trial void because no attempt was made (the gate refused to the end, or the session was stopped during a correction) holds the pose read before the gate's first look; and `view_crept_mm`, how far the front camera's picture stood from the session's conversion at the look that let the attempt go, to a hundredth of a millimetre (since the same night; not in the records of the replay pilot). It is at most the session's limit (1 mm), since a reading over it or one that could not be taken sends the trial back to its placement; `null`, or a figure over the limit, only in a pilot run with another limit or with the stop switched off: the limit is the protocol's `view_crept_limit_mm`, which a counted session cannot change, and every record carries it |
| `versions` | Commit ids of this repository, the arm tooling and LeRobot; the protocol version |
| `media` | The trial's videos |
| `ts_start`, `ts_end` | When it ran |

Rules the code enforces: a scored trial has a placement and a nearest demonstration; success goes with subgoal 5 and no
failure label; a failure names its kind; a void trial states its reason and has no score; a replay has no policy and
names its episode; unknown fields are refused, so a misspelt field cannot pass silently.

## The analysis results (`analysis/report/results.json`)

One file, written by one run of `scripts/make_report.py`, holds everything the paper's numbers and figures are made
from: the same run writes `paper/generated/numbers.tex`, and the figure scripts read this file and nothing else from
the trials. A number in a figure and the same number in the text therefore cannot come from two analyses.

Only counted trials enter it: logs named `pilot_*` are never read, and a record whose `versions.protocol_status` is
not `frozen` is refused unless the run is asked for a draft, in which case `draft` is true, the output goes to
`analysis/report-draft/` and nothing under `paper/` is written.

Distances are in millimetres **as used by the analysis**: when the scale correction of `protocol/protocol.md`, section 7,
applies, they are already divided by the factor.

```
{
  "schema_version": 1,
  "draft": false,
  "provenance": {
    "protocol_version": "1.0", "protocol_status": "frozen", "protocol_sha256": "...",
    "project_commit": "...", "logs": ["data/trials/main.jsonl"],
    "n_records": 131, "n_scored": 120, "n_void": 11
  },
  "distance_scale": {"applied": true, "factor": 1.031, "factor_min": 1.027, "factor_max": 1.034, "n_runs": 4},
  "bin_width_mm": 10.0,
  "interval_level": 0.95,

  "conditions": {
    "act_clustered": {
      "trials": [
        {"trial_id": "T1a-001", "block": "T1a", "part": "zero", "d_mm": 1.3, "success": true, "subgoal": 5,
         "failure": null, "x_mm": 124.6, "y_mm": 75.9, "level_mm": 0.0, "pair_id": null}
      ],
      "n_void": 2,
      "bins": [{"lo_mm": 0.0, "hi_mm": 10.0, "n": 9, "k": 9, "wilson_lo": 0.70, "wilson_hi": 1.0}],
      "max_success_d_mm": 27.9,
      "min_failure_d_mm": 14.2,
      "block_fit": {
        "blocks": ["T1a"], "n": 34, "k": 19,
        "r50_median_mm": 21.5, "r50_lo_mm": 16.0, "r50_hi_mm": 27.5,
        "open_below": false, "open_above": false,
        "width_median_mm": 3.1, "lapse_mean": 0.03, "floor_mean": 0.04,
        "curve": {"d_mm": [0.0, 0.5], "mean": [0.97, 0.97], "lo": [0.85, 0.85], "hi": [1.0, 1.0]}
      },
      "pooled_fit": {"blocks": ["T1a", "T2"], "...": "the same fields as block_fit"}
    },
    "replay": {"...": "the same; pooled_fit is null"},
    "act_spread": {"...": "the same; block_fit is null when T3 was not run"}
  },

  "comparisons": {
    "clustered_minus_replay": {"fits": "block", "median_mm": 9.5, "lo_mm": 1.0, "hi_mm": 18.5,
                               "prob_greater": 0.98, "verdict": "larger"},
    "spread_minus_clustered": null
  },

  "workspace": {
    "n_pairs": 20,
    "pairs": [
      {"pair_id": "T2-p03", "x_mm": 131.2, "y_mm": 61.0,
       "act_clustered": {"trial_id": "T2-005", "success": false, "d_mm": 18.4, "p_lookup": 0.71},
       "act_spread": {"trial_id": "T2-006", "success": true, "d_mm": 6.2, "p_lookup": 0.96}}
    ],
    "table": {"both": 9, "neither": 2, "only_spread": 8, "only_clustered": 1},
    "mcnemar_p": 0.039,
    "success": {"act_clustered": {"k": 10, "n": 20, "wilson_lo": 0.30, "wilson_hi": 0.70},
                "act_spread": {"k": 17, "n": 20, "wilson_lo": 0.64, "wilson_hi": 0.95}},
    "lookup": {"fit": "act_clustered block_fit",
               "act_clustered": {"expected": 11.3, "lo": 6, "hi": 16, "observed": 10, "verdict": "matches"},
               "act_spread": {"expected": 18.6, "lo": 15, "hi": 20, "observed": 17, "verdict": "matches"}}
  },

  "reset_sensitivity": {"fit": "act_clustered block_fit", "placed": 0.96, "by_eye": 0.81,
                        "n_placed": 120, "n_by_eye": 20},

  "demonstrations": {
    "clustered": [{"episode_index": 0, "cell": "B1", "x_mm": 75.4, "y_mm": 24.8, "trained": true}],
    "spread": []
  }
}
```

- `part` is `zero`, `far`, a staircase's id, or `pair`.
- `curve` gives the posterior-mean success probability and its pointwise band on the distances `d_mm`, which run from 0
  to at least the largest distance tested, in steps no larger than 0.5 mm.
- `verdict` of a comparison is `larger`, `smaller` or `not told apart`; of a lookup prediction `matches`,
  `falls short` or `exceeds` (`protocol/protocol.md`, sections 9 and 10).
- A number is a JSON number, never a formatted string: rounding belongs to the paper's macros and to the figures.
- Whatever the data do not support is `null`, never a made-up value: a block that was not run, a fit with no trials,
  a largest success when there was none.
- The values in the example above are invented to show the shape. They are not results.

### How the fields above are filled where the example leaves a choice

- **Rounding.** Every number that is not whole is kept to 9 significant digits. That is far finer than anything
  measured; it keeps the last digits of a sum, which differ from one build of NumPy to another, out of the file, so
  the same inputs give the same bytes with either of the project's interpreters. Distances are rounded before they are
  binned and fitted: the trials listed are exactly the numbers the analysis used.
- **What is divided by the scale factor.** Every distance between two measured positions: `d_mm`, placement errors,
  by-eye errors, the scatter and spacing of the demonstrations. Not divided: positions (`x_mm`, `y_mm`), the planner's
  requested `level_mm`, and everything under `instrument.scale_check` and `instrument.repeatability`, which describe
  the instrument as it reads. `distance_scale.divisor` is the number distances were divided by: `factor` when
  `applied` is true, 1 when it is false. `factor`, `factor_min` and `factor_max` are given whenever there is a scale
  run, applied or not; all three are `null` and `n_runs` is 0 when there is none.
- **`bins`** lists only bins that hold a trial. A trial is binned by its `d_mm`.
- **`pooled_fit`** is the fit of every scored trial of a policy, and `blocks` names the blocks that contributed. When
  the policy's workspace trials do not exist it is the block fit again, with `blocks` naming the one block. When the
  threshold block was not run it is the fit of the workspace trials alone. It is `null` for the replay, and for a
  policy without a scored trial.
- **`provenance.project_commit`** is the commit the trials ran on, from the records; `null` when the records name
  more than one (`project_commits` lists them) or when there is no record.
- **`workspace`** is `null` when its block has no scored trial. A pair is a placement at which both policies have a
  scored trial; its `x_mm`, `y_mm` are the mean of the two measured placements. `lookup` is `null` without the
  clustered policy's block fit, and `mcnemar_p` and the Wilson bounds are `null` without a pair.
- **`reset_sensitivity`** is `null` without the clustered policy's block fit; `by_eye` is `null` without a by-eye file.
  `n_placed` counts the scored trials of every condition.
- **`demonstrations`** holds an empty list for a set without an audit file (`provenance.audits` says which were read).
- **`comparisons.*.prob_greater`** counts a tie half for each side: on the grid two radii can be equal
  (`prob_equal`). The verdict is the rule of prediction 2 applied to `prob_greater` as written here.

### Fields the analysis adds

Every field of the file that the example above does not show is in the block below, at its place. The values are
invented to show the shape. `tests/test_report.py` reads both blocks and fails if the file gains a field that is in
neither, or loses one that is in either.

```
{
  "banner": null,
  "provenance": {
    "protocol_file": "config/protocol.toml", "project_commits": ["0123abc456"],
    "pilot_logs_not_read": ["data/trials/pilot_replay.jsonl"],
    "audits": {"clustered": "data/audit/clustered.json", "spread": "data/audit/spread.json"},
    "scale_runs": ["data/instrument/ruler_x_20261006_081000.csv"],
    "repeat_sets": ["data/instrument/repeat_20261006_082500.csv"],
    "by_eye": ["data/instrument/byeye_op1_20261007_160000.csv"],
    "placement_logs": {"clustered": "data/recording/clustered_placements.jsonl", "spread": null},
    "decisions": "docs/decisions.md", "scene_audits": ["data/audit/scene_main.json"],
    "second_scoring": "data/second_scoring/main/agreement.json", "note": null
  },
  "distance_scale": {"divisor": 1.031, "spread_points": 0.7, "why": "the 4 scale runs agree within 1 percentage point"},

  "conditions": {
    "act_clustered": {
      "trials": [
        {"nearest_demo_episode": 26, "replay_episode": null, "cell": "B1", "region": "trained_cells",
         "in_hull": false, "placement_error_mm": 1.3, "duration_s": 13.0, "steps_sent": 300,
         "loop_s": 10.064, "achieved_hz": 29.71, "checkpoint": "act-tolrad-clustered-v1@080000"}
      ],
      "n_scored": 60, "threshold_block": "T1a",
      "anchors": {"zero": {"n": 6, "k": 6}, "far": {"n": 4, "k": 0}},
      "block_fit": {
        "width_lo_mm": 0.9, "width_hi_mm": 9.3,
        "max_success_d_mm": 25.2, "min_failure_d_mm": 17.7, "zero_k": 6, "zero_n": 6,
        "sensitivity": {
          "lapse_values": [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5],
          "r50_median_mm": 21.5, "r50_lo_mm": 17.0, "r50_hi_mm": 26.5, "open_below": false, "open_above": false,
          "lapse_mean": 0.054, "median_shift_mm": 0.5, "limit_mm": 5.0, "plateau_in_doubt": false
        }
      },
      "replay_episodes": null,
      "descriptive": {
        "hull": {"inside": {"n": 28, "k": 16}, "outside": {"n": 32, "k": 16}},
        "by_region": {
          "trained_cells": {"n": 41, "k": 27, "bins": [
            {"lo_mm": 0.0, "hi_mm": 10.0, "n": 9, "k": 9, "wilson_lo": 0.70, "wilson_hi": 1.0}]},
          "held_out_cells": {"...": "the same"}
        },
        "n_off_grid": 0,
        "failures": [{"label": "grasp", "n": 15, "d_min_mm": 17.7, "d_median_mm": 25.9, "d_max_mm": 39.6}],
        "duration_median_s": 20.5, "steps_sent_median": 525.0
      }
    }
  },

  "comparisons": {"clustered_minus_replay": {"prob_equal": 0.0005, "caveats": []}},

  "workspace": {
    "block": "T2", "n_unpaired": 0,
    "pairs": [
      {"region": "trained_cells",
       "act_clustered": {"x_mm": 131.9, "y_mm": 60.2, "checkpoint": "act-tolrad-clustered-v1@080000"},
       "act_spread": {"x_mm": 130.5, "y_mm": 61.8, "checkpoint": "act-tolrad-spread-v1@080000"}}
    ],
    "distances": {"act_clustered": {"median_mm": 21.5, "max_mm": 38.2},
                  "act_spread": {"median_mm": 10.8, "max_mm": 23.3}},
    "lookup": {"act_clustered": {"pmf": [0.0, 0.01]}, "act_spread": {"pmf": [0.0, 0.01]}, "caveats": []},
    "by_region": {
      "trained_cells": {
        "n_pairs": 13,
        "act_clustered": {"k": 8, "n": 13, "wilson_lo": 0.36, "wilson_hi": 0.82,
                          "lookup_expected": 8.1, "lookup_lo": 5, "lookup_hi": 11},
        "act_spread": {"k": 11, "n": 13, "wilson_lo": 0.58, "wilson_hi": 0.96,
                       "lookup_expected": 12.0, "lookup_lo": 10, "lookup_hi": 13}
      },
      "held_out_cells": {
        "n_pairs": 13,
        "act_clustered": {"k": 2, "n": 13, "wilson_lo": 0.04, "wilson_hi": 0.42,
                          "lookup_expected": 1.6, "lookup_lo": 0, "lookup_hi": 5},
        "act_spread": {"k": 9, "n": 13, "wilson_lo": 0.42, "wilson_hi": 0.87,
                       "lookup_expected": 8.8, "lookup_lo": 5, "lookup_hi": 12}
      }
    },
    "by_run": [
      {"runs": {"act_clustered": "act-tolrad-clustered-v1", "act_spread": "act-tolrad-spread-v1"}, "n_pairs": 13,
       "act_clustered": {"k": 5, "n": 13, "wilson_lo": 0.18, "wilson_hi": 0.64,
                         "lookup_expected": 4.9, "lookup_lo": 2, "lookup_hi": 8},
       "act_spread": {"k": 10, "n": 13, "wilson_lo": 0.50, "wilson_hi": 0.92,
                      "lookup_expected": 10.2, "lookup_lo": 7, "lookup_hi": 13}}
    ]
  },

  "blocks": {
    "T1a": {"kind": "threshold", "conditions": ["act_clustered"], "planned": 34, "n_scored": 34, "n_void": 3,
            "by_condition": {"act_clustered": {"n_scored": 34, "n_void": 3}}}
  },

  "instrument": {
    "scale_check": {
      "true_step_mm": 10.0, "n_runs": 4, "n_steps": 20,
      "mean_abs_step_error_mm": 0.3, "max_abs_step_error_mm": 0.33,
      "pass_line": {"mean_abs_mm": 1.5, "max_abs_mm": 3.0}, "passes": true,
      "runs": [
        {"file": "data/instrument/ruler_x_20261006_081000.csv", "label": "ruler_x", "n_steps": 5,
         "true_span_mm": 50.0, "measured_span_mm": 51.5, "factor": 1.03,
         "mean_abs_step_error_mm": 0.3, "max_abs_step_error_mm": 0.3, "max_off_line_mm": 0.0}
      ]
    },
    "repeatability": {"n": 10, "n_sets": 1, "scatter_max_mm": 0.23, "scatter_rms_mm": 0.16,
                      "files": ["data/instrument/repeat_20261006_082500.csv"]},
    "by_eye": {"n": 20, "n_operators": 1, "operators": ["op1"],
               "error_median_mm": 4.5, "error_p95_mm": 11.4, "error_max_mm": 11.5, "errors_mm": [3.8, 11.4]}
  },

  "datasets": {
    "clustered": {
      "n_episodes": 55, "n_trained": 55,
      "placement_errors_mm": [0.54, 0.86], "placement_error_median_mm": 0.81, "placement_error_max_mm": 1.37,
      "scatter_max_mm": 1.46, "spacing_min_mm": null,
      "recording": {"n_lines": 58, "n_kept": 55, "n_superseded": 3, "superseded_sessions": ["20261006_083000"],
                    "n_rerecorded_episodes": 7, "n_rerecords": 8, "days": ["2026-10-06"],
                    "sessions": ["20261006_090100"], "matches_audit": true}
    },
    "spread": {"...": "the same; scatter_max_mm is null and spacing_min_mm is a number"},
    "both": {"placement_error_median_mm": 0.86, "placement_error_max_mm": 1.55, "recording_days": ["2026-10-06"]}
  },

  "conduct": {
    "void_reasons": [{"reason": "bumped the table", "n": 2}],
    "void_trials": [{"trial_id": "T1a-015", "block": "T1a", "condition": "act_clustered", "reason": "bumped the table"}],
    "reset_errors_mm": [1.3, 1.4], "reset_error_median_mm": 1.05, "reset_error_max_mm": 1.55,
    "trial_minutes_median": 2.1, "n_trial_intervals": 144,
    "trial_days": ["2026-10-07"], "people": ["op1"],
    "execution": {
      "by_condition": {"act_clustered": {"clamp_deg": 15.0, "control_hz": 30}},
      "common": {"clamp_deg": 15.0, "control_hz": 30},
      "per_trial": {"act_clustered": {"device": ["mps"], "handoff_tolerance_used_deg": [8.0]}},
      "achieved_hz": {
        "shortfall_limit": 0.02,
        "policy": {"requested": 30, "median": 29.7, "min": 29.6, "max": 29.8, "n": 109, "n_without": 1,
                   "short_of_requested": false},
        "replay": {"...": "the same"},
        "all": {"...": "the same"}
      }
    },
    "view_crept": {"n": 132, "n_without": 0, "median_mm": 0.48, "max_mm": 0.79, "limit_mm": 1.0, "n_over_limit": 0},
    "start_pose_deg": {"n": 132, "joints": {"shoulder_pan.pos": {"min": -4.5, "median": -3.0, "max": -1.5}}},
    "checkpoints": {"by_condition": {"act_clustered": ["act-tolrad-clustered-v1@080000"]},
                    "runs": {"act_clustered": ["act-tolrad-clustered-v1"]},
                    "step": "080000", "evaluation_step": 80000},
    "code": {"versions": {"lerobot": ["lerobot 0.6.1 at 6adf515116"]}, "lerobot_commit": "6adf515116"}
  },

  "scene": {
    "files": ["data/audit/scene_main.json"],
    "basket_named_mm": 2.0, "wrist_named_px": 2.0, "mm_per_px": 1.08, "basket_moved_limit_mm": 5.0,
    "n_scored": 132, "n_audited": 132, "not_audited": ["T3-024"],
    "n_basket_read": 131, "n_wrist_read": 131, "basket_unread": ["T1a-012"], "wrist_unread": ["T1a-013"],
    "n_basket_moved": 1,
    "basket_moved": [{"trial_id": "T1a-008", "condition": "act_clustered", "outcome": "success", "moved_mm": 31.3}],
    "moved_unknown": ["T1a-010"],
    "n_named_basket": 22, "named_basket": ["T1a-005"], "named_basket_min_mm": 2.1,
    "n_named_wrist": 1, "named_wrist": ["T1a-004"],
    "basket_start_max_mm": 2.4, "wrist_start_max_px": 2.4,
    "basket_start_bins": [{"over_mm": 0.0, "up_to_mm": 1.0, "n": 50}, {"over_mm": 3.0, "up_to_mm": null, "n": 0}],
    "success": {"named": {"n": 23, "k": 12}, "unnamed": {"n": 107, "k": 60}, "start_not_read": {"n": 2, "k": 1}}
  },

  "second_scoring": {
    "file": "data/second_scoring/main/agreement.json", "rehearsal": false,
    "logs": ["data/trials/main.jsonl"], "seed": 20261006, "one_in": 5, "n_scored": 132, "n_drawn": 29,
    "interval_level": 0.95, "why_not": null,
    "all": {
      "n_items": 29, "n_agree": 27, "n_disagree": 1, "n_cannot_tell": 1, "n_compared": 28,
      "agreement": 0.964285714, "wilson_lo": 0.822878023, "wilson_hi": 0.99366748,
      "table": {"success_success": 13, "success_failure": 1, "failure_success": 0, "failure_failure": 14},
      "cannot_tell_first": {"success": 0, "failure": 1}
    },
    "by_kind": {"policy": {"...": "the same counts"}, "replay": {"...": "the same counts"}},
    "scorers": {"first": ["op1"], "second": ["sc2"], "n_same_role": 0},
    "same_person": false,
    "disagreements": [{"trial_id": "T1b-013", "first": "success", "second": "failure"}],
    "cannot_tell": [{"trial_id": "T2-031", "first": "failure"}]
  },

  "deviations": {"n": 1, "rows": [{"date": "2026-10-07", "decision": "...", "why": "..."}]},

  "simulation": {
    "files": ["analysis/simulation/procedure.csv"],
    "protocol_digest_simulated": "d06f765f859a", "protocol_digest_now": "d06f765f859a",
    "stale": false, "why_no_case": null,
    "case": {
      "block": "T1a", "r50_true_mm": 20.0, "width_true_mm": 3.0, "lapse_true": 0.0, "replications": 300,
      "bias_mm": -0.24, "rmse_mm": 1.63, "interval_width_mm": 7.8, "coverage": 0.97,
      "rmse_fixed_mm": 2.32, "interval_width_fixed_mm": 9.01,
      "paired": {"pairs": 20, "rate_low": 0.4, "rate_high": 0.9, "power": 0.88}
    },
    "blocks": {
      "T1a": {
        "n_trials": 34,
        "by_width": {
          "cliff": {"width_true_mm": 1.0, "n_truths": 12, "rmse_mm": 0.79, "interval_width_mm": 4.19},
          "middling": {"...": "the same"},
          "slope": {"...": "the same"}
        },
        "n_truths_falling_at_zero": 12, "open_below_falling_at_zero": 0.5
      }
    },
    "open_below_falling_at_zero": 0.55,
    "comparison_rule": {
      "protocol_digest_simulated": "d06f765f859a", "stale": false, "pairs": 300, "replay_r50_mm": 15.0,
      "cases": [{"width_mm": 3.0, "difference_mm": 5.0, "larger": 0.38, "not_told_apart": 0.62, "smaller": 0.0}]
    }
  },

  "reset_sensitivity": {"caveats": []},
  "caveats": ["the lookup prediction uses the block fit of act_clustered, and its interval is open above, ..."],
  "predictions": [{"id": 1, "name": "the curve", "held": true, "why": "...", "caveats": [],
                   "numbers": {"r50_median_mm": 21.0}}],
  "not_computed": [{"what": "the number of practice episodes", "why": "..."}],
  "notes": ["..."]
}
```

- **`banner`** is on the file's first line, ahead of the sorted keys. It is `null` in the counted report and a
  sentence in anything else: a draft, a synthetic example. `provenance.note` is the note the run was given, if any.
- **A trial** also carries the demonstration its distance is measured to (`nearest_demo_episode`), the grid `cell`
  of its placement and that cell's `region` (`trained_cells` or `held_out_cells`; both `null` off the grid), whether
  the placement lies inside the convex hull of its policy's demonstrations (`in_hull`, `null` for a replay), its
  `placement_error_mm`, its `duration_s`, its `steps_sent`, its `loop_s`, its `achieved_hz` and the `checkpoint`
  its record names (`null` for a replay). A duration runs from the record's `ts_start` to its `ts_end`: from just
  before the handoff gate to after the videos were closed, in whole seconds. It is longer than the attempt, so no
  control rate is made from it. `loop_s` is the control loop's own duration, as the record gives it (`null` when the
  record has none).
- **The achieved control rate** of a trial (`achieved_hz`) is `(steps_sent - 1) / loop_s`: the steps sent after the
  first, over the time from the first step sent to the last. The handoff gate and the policy's first inference come
  before the first step and are outside it, and so is everything after the last step. It is `null` when fewer than
  two steps were sent, when `loop_s` is not above zero, or when the record carries no `loop_s`. The loop sends its
  second step as soon as it is predicted, without waiting out a period, so the figure can read high by up to one
  interval of the attempt (a third of a percent for 300 steps).
- **A region is where the cube was, not where the plan asked for it.** A trial's region is the grid cell its
  measured position falls in; a pair's is the cell of the mean of its two measured positions. A position exactly on
  the border between two cells belongs to the cell that starts there (`Grid.cell_of`).
- **`anchors`** counts the trials of the condition's threshold block at `d = 0` (`zero`) and at the far distance
  (`far`); `threshold_block` names that block.
- **Inside a fit**, `max_success_d_mm`, `min_failure_d_mm`, `zero_k` and `zero_n` are computed over that fit's own
  trials: for a block fit the threshold block alone, for a pooled fit every scored trial of the policy. `zero_n`
  counts the trials asked at a demonstration's own start (`part` is `zero`), `zero_k` the successes among them. The
  condition's own `max_success_d_mm` and `min_failure_d_mm`, beside its `bins`, cover all its scored trials. The
  paper's table prints the block fit's.
- **`sensitivity`**, inside a fit, is the plateau check of `protocol/protocol.md`, section 10: the same trials fitted a
  second time with the longer list `analysis.lapse_values_sensitivity` (`lapse_values`), everything else the same.
  It gives that fit's radius, interval and open sides, its posterior-mean lapse (`lapse_mean`), and
  `median_shift_mm`, its median minus the main fit's. `plateau_in_doubt` is true when the shift is larger in size
  than `limit_mm` (`analysis.sensitivity_limit_mm`); a shift of exactly the limit is not. `sensitivity` is `null`
  when the protocol file has no such list; `limit_mm` and `plateau_in_doubt` are `null` when it has no limit. The
  comparisons, the lookup prediction, the reset sensitivity and the predictions use the main fit.
- **`descriptive`** is what protocol section 10 calls descriptive only. `by_region` splits the condition's scored
  trials by the region of each placement and gives, for each region, the count and the same 10 mm bins as `bins`
  (only bins that hold a trial). `n_off_grid` counts the trials measured off the grid, which are in neither region.
- **`workspace.pairs[].region`** is the region of the pair's position. `workspace.by_region` gives, for each
  region, each policy's successes with a Wilson interval and what the lookup prediction expects there: the count
  (`lookup_expected`, the sum of `p_lookup`) and the interval of the predicted count (`lookup_lo`, `lookup_hi`, made
  as `lookup.*.lo` and `hi` are, from the placements of that region alone); all three are `null` without the fit or
  without a pair in the region. The prediction's verdict is made over all pairs together, and so is the paired
  test, whatever the region. `n_unpaired` counts scored trials of the block whose
  partner is missing: they are in the trial lists and in the pooled fits, and in nothing under `workspace`.
  `lookup.*.pmf` is the whole distribution of the predicted count, entry `k` being the probability of `k` successes.
- **`workspace.by_run`** gives the same counts as `by_region` for each pair of training runs the two trials of a
  pair were run with. A record names its checkpoint `<run>@<step>`, or by the step alone; `runs` holds the run of
  each policy's trial, `null` when the record does not name one. It is a list, ordered by the runs' names, with one
  entry when every pair was run with the same two runs. Each pair's two trials carry their `checkpoint`.
- **`blocks`** has an entry for every block of the protocol, run or not, and is `null` when no trial log was read.
  `planned` comes from the protocol file.
- **`instrument`**: each of its three entries is `null` when its files are missing. `scale_check` is the ruler check
  as read (`protocol/protocol.md`, section 7); a run's `factor` is its measured span over its true span.
  `by_eye.errors_mm` are the distances from where the cube was for the reference picture.
- **`datasets`**: an entry is `null` without the set's audit. `recording` is `null` without the recorder's placement
  log; `n_superseded` counts its lines whose episode was saved again later (a smoke test or a restart),
  `n_rerecorded_episodes` the kept episodes with at least one earlier take, `n_rerecords` those takes.
  `matches_audit` says whether the kept lines are the audited episodes. The entries of `both` are `null` unless both
  sets have what they need.
- **`conduct`** is `null` when no trial log was read. `trial_minutes_median` is the median time from one record's
  start to the next one's in the same log, void trials included. `void_reasons` counts the void trials by reason;
  reasons that differ only in capitals, spacing or a final full stop are one, and so are all trials that a key
  ended at once, whose reason carries the time (`ended by a key 0.4 s after it began (under 1 s)`); `void_trials`
  gives each trial's reason as written.
- **`conduct.execution`**: `by_condition` holds the settings the scored trials of each condition ran with. The
  analysis has refused the logs unless they are the same in every scored trial of a condition and hold every
  setting of the protocol's `[execution]` section with the protocol's value. `common` holds those that every
  condition shares. `per_trial` lists, for each condition, the values that occur among its scored trials of two
  entries that belong to the single trial and are held to nothing: the `device` and the tolerance the handoff gate
  used. `start_pose_deg` summarises the arm's recorded joint positions at the start of the scored attempts, joint
  by joint, in the records' units; it is `null` when no record carries a start pose.
- **`conduct.execution.achieved_hz`** gives the achieved control rate of the scored trials: of the policy trials
  (`policy`), of the replay trials (`replay`) and of both together (`all`). For each: the rate the trials were
  asked to run at (`requested`, their `control_hz`; `null` if they do not share one), the `median`, the smallest
  (`min`) and the largest (`max`) achieved rate, the number of trials these rest on (`n`) and the number of scored
  trials of that kind without a rate (`n_without`). The three figures are `null` when `n` is 0.
  `short_of_requested` is true when the median lies more than `shortfall_limit` (2 %) below the requested rate, and
  `null` without a median or a requested rate; the report then says so in words. That line is the analysis's own,
  not the protocol's. The paper's `\execControlRateAchieved` is the median of the policy trials.
- **`conduct.view_crept`** is the camera's creep before an attempt: how far the front camera's picture stood from
  where its session solved the conversion from pixels to millimetres, read just before the attempt and written into
  the record (`execution.view_crept_mm`). Over the scored trials: how many carry a reading (`n`), how many do not
  (`n_without`), the median and the largest reading, the protocol's limit beyond which a session makes no attempt
  (`limit_mm`, its `execution.view_crept_limit_mm`; `null` when the protocol file states none) and how many
  readings lie beyond it (`n_over_limit`, `null` without a limit). The readings are as the sessions read them and
  are not divided by the scale factor: they were held against the limit as read. The whole entry is `null` when no
  scored record carries a reading. The paper's `\viewCreptMedian` and `\viewCreptMax` print the two figures with
  two decimals, because one decimal can print a reading under the limit as the limit.
- **`conduct.checkpoints`**: for each policy the checkpoint names (`by_condition`) and the training runs (`runs`)
  that occur among its scored trials. `step` is the training step as the records name it, and `evaluation_step`
  that name as a number, `null` when it is not one. The analysis has refused the logs unless every scored policy
  trial names the same step.
- **`scene`** is the scene at the start of the scored trials, from the scene audits of the counted logs
  (`data/audit/scene_<log>.json`, which `scripts/rig/scene_audit.py` writes from the trials' own videos;
  `provenance.scene_audits` names the files read). The protocol (section 8) names a trial that began with the
  basket more than `basket_named_mm` or the wrist camera more than `wrist_named_px` from its place; both limits are
  the audit file's. An offset is the larger of across and down, as the audit reads it: millimetres on the floor for
  the basket, pixels of the wrist camera's picture for that camera. Nothing here is divided by the scale factor,
  which belongs to the instrument that measures the cube.
  `n_audited` counts the scored trials that are in an audit and `not_audited` names the others;
  `n_basket_read` and `n_wrist_read` count the trials whose first frames gave a reading, and `basket_unread` and
  `wrist_unread` name those whose video did not. `basket_moved` lists the trials the audit marks as having moved
  the basket (it stood more than `basket_moved_limit_mm` from where the trial began; the audit's rule, repeated
  here for the text), each with its condition and outcome from the trial log; `moved_unknown` names the trials
  whose end says nothing of the basket. `named_basket` and `named_wrist` name the trials beyond a limit at their
  start, `named_basket_min_mm` is the smallest offset among those named for the basket (`null` when none is), and
  `basket_start_max_mm` and `wrist_start_max_px` are the largest offsets at a start. `basket_start_bins` counts
  the trials by the basket's offset at the start: an offset on a bin's upper end belongs to that bin, the first bin
  holds an offset of zero as well, and the last has no upper end (`up_to_mm` is `null`). `success` gives plain
  counts and no test: the successes of the trials named for the basket or for the wrist camera (`named`), of those
  read at their start and named for neither (`unnamed`), and of those whose start could not be read
  (`start_not_read`); the three together are the audited trials. `mm_per_px` is the size of one pixel of the front
  camera at the basket, as the audit's code states it. The whole entry is `null` when no counted log has a scene
  audit, when the audits hold none of the scored trials, and when no trial log was read. An audit that was made
  from another log is refused.
- **`second_scoring`** is what the report gives of the second scoring from video, read from the agreement file
  that `scripts/second_scoring.py compare` wrote (the section on the second scoring, below, explains its counts;
  `provenance.second_scoring` names the file). `all` and `by_kind` are that file's tallies as they stand: the items
  scored again, those in which the two scores agree, differ and could not be told, the share that agree among the
  compared with its Wilson interval at `interval_level`, and the table of first score against second. `scorers`
  holds the role codes, and `same_person` is true when every item's two scores carry the same code: the operator
  scored again from the video, with the trials' identities hidden. `disagreements` and `cannot_tell` name the trials
  concerned. The counts are given only when the agreement is the second scoring of exactly the trials the report
  reads: `why_not` is then `null`. Otherwise it says why not (a rehearsal; drawn from other logs; a log that has
  changed since the draw, which the logs' digest shows) and `all`, `by_kind`, `scorers`, `same_person` and the two
  lists are `null`. The whole entry is `null` without an agreement file, and when no trial log was read. The first
  score stands in the analysis whatever the second one says.
- **`deviations`** counts the rows of `docs/decisions.md` marked **DEVIATION**; `null` without that file.
- **`simulation`** is what the paper quotes of the procedure simulation. `case` is the one case fixed before the
  data; it is `null`, with the reason in `why_no_case`, when the simulation was made from another protocol file
  than the present one (`stale`) or lacks a row. `blocks` has an entry for the threshold block of each condition, as
  the block is written: under `by_width`, for a cliff, a middling width and a slope, the error and the mean interval
  width of the middle one of the simulated truths with that width (the median over those `n_truths` truths, each
  quantity on its own: the first number of a cell of the simulation report's Table 1), and
  `open_below_falling_at_zero`, the share of simulated blocks whose interval is open below, averaged over the
  truths whose radius is under two widths. The top-level `open_below_falling_at_zero` is that share over all the
  blocks' truths together. `blocks` and it are `null` when the simulation is stale; an entry is `null` where the
  table has no row. `comparison_rule` is the simulation of the rule of prediction 2
  (`analysis/simulation/comparison_rule.csv`), which has a report and a digest of its own: for every simulated case
  (the width of both curves, the policy's radius minus the replay's) the share of the simulated pairs of blocks
  with each verdict. It is `null` when that table is not there, and its `cases`, `pairs` and `replay_r50_mm` are
  `null` when its report names another protocol file than the present one (`stale`).
- **`caveats`** are sentences that say which results rest on a block fit that does not establish a radius: a fit
  whose plateau is in doubt (`protocol/protocol.md`, section 10), or whose interval is open on a side. The results that
  can rest on one are the two comparisons, the lookup prediction, the reset sensitivity and prediction 1. Each is
  computed by its rule all the same and carries its sentences beside it, never without: `comparisons.*.caveats`,
  `workspace.lookup.caveats`, `reset_sensitivity.caveats` and `predictions[].caveats` (prediction 2 carries those
  of its comparison, prediction 4 those of the lookup prediction, prediction 5 those of the comparison of the two
  policies; prediction 3 rests on no fit). The top-level `caveats` holds every such sentence once. Each list is
  empty when there is nothing to say; a result that was not computed is `null` and carries none.
- **`predictions`** has one entry for each prediction of `protocol/protocol.md`, section 9. `held` is `null` where a
  prediction is not judged, and `why` says why, or by which rule it was judged. What `numbers` holds differs from
  one prediction to the next. For prediction 3, `numbers.outcome` is `spread ahead` (the prediction holds: the exact
  paired test gives p below 0.05 with the spread policy ahead), `clustered ahead` (p below 0.05 the other way) or
  `not decided`.
- **`not_computed`** lists what the analysis could not compute from what it was given, each with the reason.
  `notes` are sentences the report prints: trials left out of the paired table, a void trial run with other settings.

## The second scoring (`data/second_scoring/<name>/`)

A part of the counted trials is scored again from video by a person who is shown neither the condition nor the
distance (`protocol/protocol.md`, section 8; `docs/runbook/05-second-scoring.md`). `scripts/second_scoring.py` writes one
folder for each draw, `<name>` being `main` for the counted trials. `src/tolerance_radius/second_scoring.py` defines
the files and checks them; where this page and the code disagree, the code is right.

| File | Written by | Read by | Kept in git |
|---|---|---|---|
| `items.csv` | `draw` | `score`, `compare` | yes |
| `videos/<item>_<view>.mp4` | `draw` (copies of the trials' videos) | `score` | no (`*.mp4` is ignored) |
| `scores.csv` | `score`, one line at a time | `score` (to carry on), `compare` | yes |
| `key/key.csv` | `draw` | `compare` | yes |
| `key/draw.json` | `draw` | `compare` | yes |
| `agreement.json` | `compare` | people; the report (`results.json`, `second_scoring`) | yes |

What the scorer's tool reads is `items.csv` and `videos/`, and nothing else. `key/` says which trial each item is: it
is kept in a folder of its own so that the rest can be handed to a scorer without it, and nobody who scores opens it
before every item has a score.

### Which trials are drawn

- **From:** the scored trials (not void) of the counted logs. A pilot log and a record that was not run under a
  frozen protocol are refused, as in the analysis.
- **How many:** the trials are grouped by block and condition, and each group gives one trial in five, **rounded
  up**: every condition of every block is there, and the whole is a fifth or slightly more.
- **Which:** each group is put in a random order and gives the first trials of that order. A trial whose videos
  cannot be used is passed over, listed with the reason, and replaced by the next trial of the same order; a group
  that cannot be filled stops the draw. A video cannot be used when it is missing, does not decode, or holds fewer
  frames than the trial sent steps (`execution.steps_sent`): the trial runner records a frame for every step, so
  such a video has lost some and may not show how the attempt ended.
- **Pairs:** a trial whose pair partner (same `pair_id`) is already drawn is passed over too, so that the two trials
  of one placement are not both items. The groups are drawn in the order of the blocks and, inside a block, of the
  conditions. With every pair complete and every video usable, every trial of a group has the same chance of being
  drawn. A partner is taken after all only when its group has no other trial left, and `whole_pairs` then names
  the pair.
- **What the random orders depend on:** the seed **and the logs' own bytes**. Every random stream is named by the
  seed and by `logs_digest`, one SHA-256 of the logs. The seed is fixed before the first trial, so nobody chooses
  the draw afterwards; the logs' digest does not exist before the last trial, so nobody knows the draw beforehand
  and scores the trials that will be checked with more care.
- **Order and codes:** the drawn trials are shuffled by another stream of the same kind and numbered in that order:
  `item-01`, `item-02`, ... Without the logs' digest (which is in `key/`), an item's code and place say nothing
  about its trial.
- The same logs, seed and list of trials passed over always give the same items.

### `items.csv`

One line for each item, in the order shown, under the heading `item,front,wrist` (an item and then one camera name
for each view the draw was asked for).

| Column | Meaning |
|---|---|
| `item` | The item's code |
| `front`, `wrist` | The copy of that view's video: a path **relative to the folder that holds `items.csv`** (`videos/item-07_front.mp4`), so that the folder can be scored wherever it is put. Everywhere else in this project a path is relative to the repository |

It holds nothing else: no trial, block, condition, distance or score. The copies are made in the order of the items,
so their dates say nothing either. A video's own content is what it is: it shows where the cube started and how the
arm moved.

### `scores.csv`

Appended to, one line for each item scored, forced to disk before the next item is shown. Heading
`item,score,scorer,watched_again,videos_sha256`.

| Column | Meaning |
|---|---|
| `item` | The item's code |
| `score` | `success` (the cube was released in the basket: the first scoring's success, grade 5), `failure` (it was not: any of the first scoring's failures, grades 0 to 4), or `cannot_tell` (the video does not show which; the first scoring has no such answer, and it is not a void) |
| `scorer` | The role code of the person at the keys: `op` or `sc` and a number (`sc2`, `op1`). Anything else is refused, so that neither a name nor initials reach the file |
| `watched_again` | How many times the scorer had the whole video shown again after it had ended, in the sitting the score was recorded in. Called so because "replay" is the name of a condition here |
| `videos_sha256` | One SHA-256 of the video files that were shown for the item: of the files' own SHA-256s, in the order of the views. The comparison holds it against the one `key/key.csv` has for the item |

An item has one line. A line is never changed: a score recorded by a slip stays, and the journal says so.

### `key/key.csv` and `key/draw.json`

`key.csv` has one line for each item under `item,trial_id,log,videos_sha256`: the item's code, the trial it is, the
log that trial is in, and the digest of the copies the draw made for it (as in `scores.csv`).

`draw.json` is the draw's own record:

```json
{
  "schema_version": 1,
  "rehearsal": false,
  "seed": 20261006,
  "one_in": 5,
  "views": ["front", "wrist"],
  "logs": [{"file": "data/trials/main.jsonl", "sha256": "...", "n_records": 140, "n_scored": 132}],
  "logs_digest": "...",
  "n_scored": 132,
  "n_drawn": 29,
  "groups": [{"block": "T1a", "condition": "act_clustered", "n_scored": 34, "n_drawn": 7}],
  "passed_over": [{"trial_id": "T1a-026", "log": "data/trials/main.jsonl", "reason": "its wrist video is not there"}],
  "whole_pairs": []
}
```

- The values in the examples of this section are invented to show the shape. They are not results.
  `tests/test_second_scoring.py` reads both examples and fails if a file gains a field that its example lacks, or
  loses one it has. (The two examples are fenced as `json`: the tests of the analysis take every plain fenced
  object after the heading of `results.json` for a part of that file's contract.)
- `rehearsal` is true for a draw made with `--rehearsal`, from pilot trials, to try the tool. Nothing made from such
  a draw is a result, and `agreement.json` carries the mark on.
- `logs[].sha256` is the log's hash when the draw was made. The comparison refuses a log whose hash has changed: a
  draw is made after the last trial. `logs_digest` is one SHA-256 of all the logs together (of their own hashes,
  in name order): it names the draw's random streams beside the seed.
- `groups` has an entry for each block and condition that has a scored trial, in the order drawn.
- `passed_over` lists the trials that came up and whose videos could not be used, with the reason in words.
- `whole_pairs` lists the pairs of which both trials were drawn because a group had no other trial left. It is
  empty in an ordinary draw.
- The comparison makes the draw again from the logs, the seed and `passed_over`, and refuses a record or a key that
  says anything else.

### `agreement.json`

Written by `compare`. It is refused until every item has a score; when a log is not the file the draw recorded;
when `key.csv` or `draw.json` is not what the recorded seed draws from the logs; and when a score's `videos_sha256`
is not the key's for its item (the scores belong to another draw, or a copy was changed).

```json
{
  "schema_version": 1,
  "rehearsal": false,
  "provenance": {
    "folder": "data/second_scoring/main", "logs": ["data/trials/main.jsonl"], "logs_digest": "...",
    "seed": 20261006, "one_in": 5, "views": ["front", "wrist"], "n_scored": 132, "n_drawn": 29,
    "groups": ["as in draw.json"], "passed_over": [], "whole_pairs": []
  },
  "scorers": {"first": ["op1"], "second": ["sc2"], "n_same_role": 0},
  "interval_level": 0.95,
  "all": {
    "n_items": 29, "n_agree": 27, "n_disagree": 1, "n_cannot_tell": 1, "n_compared": 28,
    "agreement": 0.964285714, "wilson_lo": 0.822878023, "wilson_hi": 0.99366748,
    "table": {"success_success": 13, "success_failure": 1, "failure_success": 0, "failure_failure": 14},
    "cannot_tell_first": {"success": 0, "failure": 1}
  },
  "by_kind": {"policy": {"...": "the same counts"}, "replay": {"...": "the same counts"}},
  "disagreements": [
    {"trial_id": "T1b-013", "item": "item-08", "block": "T1b", "condition": "replay", "kind": "replay",
     "first": "success", "first_subgoal": 5, "first_failure": null, "second": "failure", "watched_again": 1}
  ],
  "cannot_tell": [{"...": "the same fields, with second = cannot_tell"}],
  "watched_again": {"n_items": 4, "total": 6}
}
```

- **The counts.** `n_agree` items have the same second score as first score, `n_disagree` have the other one, and in
  `n_cannot_tell` the second scorer could not tell. `n_compared` is `n_agree + n_disagree`.
- **`agreement`** is `n_agree / n_compared`: the share that agree **among the items the second scorer could tell**.
  `wilson_lo` and `wilson_hi` are its Wilson interval at `interval_level` (`tolerance_radius.intervals`). All three
  are `null` when `n_compared` is 0. Numbers that are not whole are kept to 9 significant digits.
- **`table`** counts the compared items by first score and then second score: `success_failure` is scored a success
  at the rig and a failure from the video. `cannot_tell_first` counts the items the second scorer could not tell by
  their first score.
- **`by_kind`** gives the same for the policy trials and for the replay trials. Both entries are always there.
- **`disagreements`** and **`cannot_tell`** list the items concerned by trial, in name order, with the first
  scoring's grade and failure label. The first score stands in the analysis whatever the second one says.
- **`scorers`** holds the role codes of the first scoring (from the drawn trials' records) and of the second, and
  `n_same_role`, the number of items whose two scores carry the same code: the operator scoring again.
- **`watched_again`** counts the items whose video was shown again at least once, and those showings in all.
- **`provenance`** holds what the draw's record says (seed, views, the logs' digest, counts, groups, trials passed
  over, whole pairs), after the comparison has made the draw again from the logs and found the record to say the
  same, with the names of the folder and of the logs.
- There is no kappa: the four counts of `table` say what it would, and let a reader compute it.
