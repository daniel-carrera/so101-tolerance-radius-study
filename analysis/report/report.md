# Analysis report

This is the counted analysis, written by `scripts/make_report.py` together with `results.json`. Every number below is computed from the files named in this section, by the rules of `protocol/protocol.md`, section 10. A number is followed by where it comes from.

- **Protocol:** `config/protocol.toml`, version 1.0, status frozen, SHA-256 `78fd2bfa905d`.
- **Trial logs read:** `data/trials/main.jsonl`: 108 records, 108 scored and 0 void.
- **Pilot logs left alone:** `data/trials/pilot_policy.jsonl`, `data/trials/pilot_replay.jsonl`, `data/trials/pilot_shakedown.jsonl`. Pilot trials are never analysed.
- **The trials ran on** commit `57946a8c28`, `ccf9c7da57`, `ccf9c7da57+changes` of this repository (from the records).
- **Audits:** `data/audit/clustered.json`, `data/audit/spread.json`. **Scene audits:** `data/audit/scene_main.json`.
- **Instrument files:** scale runs `data/instrument/ruler_x_20261006_142823.csv`, `data/instrument/ruler_x_corner_20261006_143546.csv`, `data/instrument/ruler_y_20261006_133638.csv`, `data/instrument/ruler_y_corner_20261006_144151.csv`; repeated placements `data/instrument/repeat_20261008_174537.csv`; resets by eye `data/instrument/byeye_op1_20261008_172808.csv`.
- **Placement logs of the recorder:** `data/recording/clustered_placements.jsonl`, `data/recording/spread_placements.jsonl`.
- **Decision log:** `docs/decisions.md`.
- **Second scoring from video:** `data/second_scoring/main/agreement.json`.

## The radii at a glance

| Condition | Fit | Successes | At d = 0 | Radius (median) | 95 % interval | Do the data bound it? | Radius, longer lapse list | Its interval | Shift |
|---|---|---|---|---|---|---|---|---|---|
| the clustered policy | block (T1a) | 21 of 34 | 6 of 6 | 39.0 mm | 23.0 mm to 52.0 mm | yes, on both sides | 42.5 mm | 26.0 mm to 64.0 mm | +3.5 mm |
| the clustered policy | pooled (T1a, T2) | 37 of 60 | 6 of 6 | 36.0 mm | 25.0 mm to 48.5 mm | yes, on both sides | 42.0 mm | 27.5 mm to 58.0 mm | +6.0 mm |
| the replay | block (T1b) | 13 of 22 | 4 of 4 | 16.5 mm | 13.0 mm to 21.5 mm | yes, on both sides | 17.0 mm | 13.0 mm to 23.5 mm | +0.5 mm |
| the spread policy | pooled (T2) | 22 of 26 | none asked | 28.5 mm | 19.5 mm to 64.5 mm | open above: report "at least" | 29.0 mm | 20.0 mm to 69.0 mm | +0.5 mm |

The radius is the posterior median of `r50`. Comparisons and predictions use block fits only; a pooled fit is every scored trial of a policy and is given as a check.

**The plateau check** (protocol section 10): every fit is made a second time with the lapse values 0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, everything else the same. The first fit is the radius that is reported; the shift is the second fit's median minus the first's. The counts at d = 0 are the trials asked at a demonstration's own start among the trials of that fit.

- **The plateau of the clustered policy is in doubt** (pooled (T1a, T2) fit): with the longer list the radius is 42.0 mm (27.5 mm to 58.0 mm) against 36.0 mm, a shift of +6.0 mm, more than the protocol's 5 mm. At d = 0 it succeeded in 6 of 6 trials. By protocol section 10 the paper says that this condition's radius is not established.

## Distances and the scale correction

**Distances are left as the camera reads them.** The 4 scale run(s) read a true length as longer by factors from 1.0123 to 1.0248 (mean 1.0186); they differ by 1.25 percentage points. Reason: the 4 scale runs differ by more than 1 percentage point. The spread of the factor is the instrument's accuracy.

## The instrument (blocks V0 and V1)

**Scale check:** over 31 steps of 10 mm in 4 run(s), a measured step was off by 0.25 mm on average and by 0.79 mm at worst, as read. It passes the protocol's line (mean at most 1.5 mm, no step off by more than 3 mm).

| Run | Steps | Measured span | Factor | Mean step error | Worst step | Off the line |
|---|---|---|---|---|---|---|
| `data/instrument/ruler_x_20261006_142823.csv` | 6 | 61.20 of 60 mm | 1.0200 | 0.23 mm | 0.37 mm | 0.72 mm |
| `data/instrument/ruler_x_corner_20261006_143546.csv` | 5 | 50.85 of 50 mm | 1.0170 | 0.17 mm | 0.31 mm | 0.09 mm |
| `data/instrument/ruler_y_20261006_133638.csv` | 15 | 151.85 of 150 mm | 1.0123 | 0.28 mm | 0.70 mm | 0.69 mm |
| `data/instrument/ruler_y_corner_20261006_144151.csv` | 5 | 51.24 of 50 mm | 1.0248 | 0.26 mm | 0.79 mm | 0.83 mm |

**Repeatability:** 11 placements in 1 set(s) (`data/instrument/repeat_20261008_174537.csv`) lay at most 0.26 mm from the mean of their set, as read.

**Resets by eye:** 20 resets by 1 person(s) (op1) landed a median of 15.95 mm from where the cube was for the reference picture; 95th percentile 31.03 mm (interpolated between ranks), largest 35.53 mm. From `data/instrument/byeye_op1_20261008_172808.csv`.

Not in any committed record: the disagreement of each session's camera solve with the stored one (the tools print it).

## The two datasets

| Set | Trained on | Start to target, median | Largest | Layout as measured | Recording |
|---|---|---|---|---|---|
| clustered | 55 of 55 | 0.92 mm | 2.07 mm | scatter about a point at most 1.41 mm | 2 episodes recorded again (2 extra takes); 0 earlier lines recorded over; days 2026-10-07 |
| spread | 55 of 55 | 0.69 mm | 2.71 mm | closest two starts 17.51 mm apart | 1 episodes recorded again (1 extra takes); 0 earlier lines recorded over; days 2026-10-07 |

Both sets together: a start lay a median of 0.72 mm from its planned target, at most 2.71 mm (from the audits).

"Recorded again" counts kept episodes that had at least one earlier take in the same sitting. "Recorded over" counts lines of the placement log whose episode number was saved again later: a smoke test or a restart. No record says which of those were practice episodes.

Not in any committed record: torn frames and frame pacing of the new datasets (the checks print them).

## Trials

| Block | Condition | Planned | Scored | Void |
|---|---|---|---|---|
| T1b | replay | 22 | 22 | 0 |
| T1a | act_clustered | 34 | 34 | 0 |
| T2 | act_clustered and act_spread | 52 | 52 | 0 |
| T3 | act_spread | 24 | 0 | 0 |
| all |  | 132 | 108 | 0 |

No trial was void.

## Clustered policy (`act_clustered`)

60 scored trials and 0 void, from the counted logs. In block T1a: 6 of 6 succeeded at d = 0, 1 of 4 at the far anchors. Over all its scored trials, the largest distance with a success is 44.8 mm and the smallest with a failure 10.1 mm.

| Distance (closed below, open above) | Successes | Wilson 95 % interval |
|---|---|---|
| 0 to 10 mm | 11 of 11 | 74 % to 100 % |
| 10 to 20 mm | 6 of 10 | 31 % to 83 % |
| 20 to 30 mm | 7 of 13 | 29 % to 77 % |
| 30 to 40 mm | 9 of 12 | 47 % to 91 % |
| 40 to 50 mm | 4 of 11 | 15 % to 65 % |
| 50 to 60 mm | 0 of 3 | 0 % to 56 % |

**Block fit** (T1a; 21 successes in 34 trials): radius 39.0 mm, 95 % interval 23.0 mm to 52.0 mm. Bounded by the data: yes, on both sides. Width 13.66 mm (median), lapse 0.058 and floor 0.051 (posterior means). Its trials: 6 of 6 at d = 0; farthest success 44.8 mm, nearest failure 10.1 mm. With lapse values up to 0.5: radius 42.5 mm (26.0 mm to 64.0 mm; open above: report "at least"), a shift of +3.5 mm; lapse 0.177 (posterior mean).

**Pooled fit** (T1a, T2; 37 successes in 60 trials): radius 36.0 mm, 95 % interval 25.0 mm to 48.5 mm. Bounded by the data: yes, on both sides. Width 15.51 mm (median), lapse 0.046 and floor 0.051 (posterior means). Its trials: 6 of 6 at d = 0; farthest success 44.8 mm, nearest failure 10.1 mm. With lapse values up to 0.5: radius 42.0 mm (27.5 mm to 58.0 mm; yes, on both sides), a shift of +6.0 mm; lapse 0.185 (posterior mean). **The plateau is in doubt.**

## Replay (`replay`)

22 scored trials and 0 void, from the counted logs. In block T1b: 4 of 4 succeeded at d = 0. Over all its scored trials, the largest distance with a success is 19.8 mm and the smallest with a failure 14.7 mm.

| Distance (closed below, open above) | Successes | Wilson 95 % interval |
|---|---|---|
| 0 to 10 mm | 7 of 7 | 65 % to 100 % |
| 10 to 20 mm | 6 of 11 | 28 % to 79 % |
| 20 to 30 mm | 0 of 4 | 0 % to 49 % |

**Block fit** (T1b; 13 successes in 22 trials): radius 16.5 mm, 95 % interval 13.0 mm to 21.5 mm. Bounded by the data: yes, on both sides. Width 2.03 mm (median), lapse 0.037 and floor 0.051 (posterior means). Its trials: 4 of 4 at d = 0; farthest success 19.8 mm, nearest failure 14.7 mm. With lapse values up to 0.5: radius 17.0 mm (13.0 mm to 23.5 mm; yes, on both sides), a shift of +0.5 mm; lapse 0.063 (posterior mean).

Demonstrations replayed: episodes 0, 5, 15, 17, 18, 23, 25, 28, 35, 48, 51.

## Spread policy (`act_spread`)

26 scored trials and 0 void, from the counted logs. In block T3: 0 of 0 succeeded at d = 0. Over all its scored trials, the largest distance with a success is 26.0 mm and the smallest with a failure 16.1 mm.

| Distance (closed below, open above) | Successes | Wilson 95 % interval |
|---|---|---|
| 0 to 10 mm | 11 of 11 | 74 % to 100 % |
| 10 to 20 mm | 9 of 11 | 52 % to 95 % |
| 20 to 30 mm | 2 of 3 | 21 % to 94 % |
| 30 to 40 mm | 0 of 1 | 0 % to 79 % |

**Block fit:** none, block T3 has no scored trial.

**Pooled fit** (T2; 22 successes in 26 trials): radius 28.5 mm, 95 % interval 19.5 mm to 64.5 mm. Bounded by the data: open above: report "at least". Width 4.94 mm (median), lapse 0.043 and floor 0.050 (posterior means). Its trials: 0 of 0 at d = 0; farthest success 26.0 mm, nearest failure 16.1 mm. With lapse values up to 0.5: radius 29.0 mm (20.0 mm to 69.0 mm; open above: report "at least"), a shift of +0.5 mm; lapse 0.059 (posterior mean).

## Comparison of radii

**The clustered policy's radius minus the replay's:** 22.0 mm (95 % interval 5.5 mm to 35.5 mm). Posterior probability that the first is larger: 0.9947. Verdict by the rule of prediction 2: **larger**.

**The spread policy's radius minus the clustered policy's:** not computed, it needs both block fits.

From the two block fits, taken as independent, with every pair of grid values weighted exactly. "Larger" is a probability of at least 0.95, "smaller" at most 0.05. On the grid two radii can be equal; such a tie counts half for each side.

## The workspace test

26 placements were tested with both policies (block T2). Both succeeded at 14, neither at 2, only the spread policy at 8, only the clustered policy at 2. Exact McNemar test over all pairs together: p = 0.1094.

| Policy | Successes | Wilson 95 % interval | Median d | Largest d | Lookup prediction (expected, 95 % interval of the count) |
|---|---|---|---|---|---|
| the clustered policy | 16 of 26 | 43 % to 78 % | 27.1 mm | 50.5 mm | 17.67 (10 to 24): matches |
| the spread policy | 22 of 26 | 66 % to 94 % | 10.8 mm | 38.3 mm | 21.04 (15 to 25): matches |

The lookup prediction is the clustered policy's block fit, which has not seen these trials, at each trial's measured distance to its own policy's nearest demonstration. The interval is that of the predicted count: it carries the uncertainty of the fit and the chance in the trials. An observed count inside it, ends included, matches.

By region. A pair's region is the grid cell of its measured position (the mean of its two placements). The lookup prediction's verdict is made over all pairs; the expected count and the interval of the predicted count are given for each group besides.

| Region | Pairs | Policy | Successes | Wilson 95 % interval | Lookup expects |
|---|---|---|---|---|---|
| trained cells | 13 | the clustered policy | 8 of 13 | 36 % to 82 % | 10.13 (6 to 13) |
| trained cells | 13 | the spread policy | 13 of 13 | 77 % to 100 % | 11.11 (8 to 13) |
| held-out cells | 13 | the clustered policy | 8 of 13 | 36 % to 82 % | 7.54 (3 to 12) |
| held-out cells | 13 | the spread policy | 9 of 13 | 42 % to 87 % | 9.94 (6 to 13) |

By training run: every pair was run with one pair of runs, `act-tolrad-clustered-v1` for the clustered policy and `act-tolrad-spread-v1` for the spread policy (from the records' `checkpoint`).

## Reset sensitivity (a calculation, not a measurement)

The clustered policy's block fit, at a demonstration's own start, with a reset that is off by each of the 108 placement errors of this study's scored trials: 88 % success. With each of the 20 by-eye errors of block V1 instead: 79 %.

## The predictions of protocol section 9

1. **The curve.** Predicted: at least 5 successes in 6 at d = 0, at most 1 in 4 at the far anchors, and a radius between 5 and 40 mm. Found: 6 of 6 at d = 0, 1 of 4 at the far anchors, radius 39.0 mm (block fit, posterior median). **held**.
2. **Hardware.** Predicted: the clustered policy's radius is larger than the replay's, by a posterior probability of at least 0.95. Found: the difference is 22.0 mm (5.5 mm to 35.5 mm), and the probability that the policy's radius is larger is 0.9947: larger. **held**.
3. **Layout.** Predicted: at the placements of the workspace test the spread policy succeeds more often than the clustered one, by the exact paired test over all pairs: p below 0.05 with the spread policy ahead. Found: the spread policy succeeded at 22 of 26 placements and the clustered policy at 16 (in trained cells 13 against 8 of 13; in held-out cells 9 against 8 of 13); exact McNemar p = 0.1094. **did not hold**. The test did not decide: the counts are what there is to report.
4. **Lookup.** Predicted: each policy's count in the workspace test lies inside the 95 % interval of its lookup prediction. Found: the clustered policy 16 against 17.67 (10 to 24): matches; the spread policy 22 against 21.04 (15 to 25): matches. **held**.
5. **Layout and radius.** No prediction was made; reported as an estimate. Found: the threshold block of the spread policy or of the clustered policy was not run.

## Descriptive only

| Condition | Successes inside and outside the hull of the demonstrations | Failure labels (count, range of d) | Median duration |
|---|---|---|---|
| the clustered policy | 28 of 36 inside, 9 of 24 outside | grasp 23 (10.1 mm to 50.6 mm) | 24 s |
| the replay | does not apply | grasp 9 (14.7 mm to 25.4 mm) | 24 s |
| the spread policy | 21 of 22 inside, 1 of 4 outside | grasp 4 (16.1 mm to 38.3 mm) | 20 s |

A duration runs from a record's `ts_start` to its `ts_end`: from just before the handoff gate to after the videos were closed, in whole seconds. It is longer than the attempt. The control loop's own duration is `loop_s`, and the rate made from it is under "How the trials were run".

**Success by the cell of the placement, at each distance.** A trial counts as in a trained cell or in a held-out cell by the grid cell its measured position falls in. Beyond the largest gap of a layout every placement is in a held-out cell, so on the far side of a curve distance and a cell without demonstrations cannot be told apart (protocol section 10).

Clustered policy (`act_clustered`), successes of trials with a Wilson 95 % interval.

| Distance | In trained cells | In held-out cells |
|---|---|---|
| 0 to 10 mm | 11 of 11 (74 % to 100 %) | no trial |
| 10 to 20 mm | 6 of 10 (31 % to 83 %) | no trial |
| 20 to 30 mm | 4 of 9 (19 % to 73 %) | 3 of 4 (30 % to 95 %) |
| 30 to 40 mm | no trial | 9 of 12 (47 % to 91 %) |
| 40 to 50 mm | no trial | 4 of 11 (15 % to 65 %) |
| 50 to 60 mm | no trial | 0 of 3 (0 % to 56 %) |
| all distances | 21 of 30 | 16 of 30 |

Replay (`replay`), successes of trials with a Wilson 95 % interval.

| Distance | In trained cells | In held-out cells |
|---|---|---|
| 0 to 10 mm | 7 of 7 (65 % to 100 %) | no trial |
| 10 to 20 mm | 6 of 11 (28 % to 79 %) | no trial |
| 20 to 30 mm | 0 of 4 (0 % to 49 %) | no trial |
| all distances | 13 of 22 | no trial |

Spread policy (`act_spread`), successes of trials with a Wilson 95 % interval.

| Distance | In trained cells | In held-out cells |
|---|---|---|
| 0 to 10 mm | 11 of 11 (74 % to 100 %) | no trial |
| 10 to 20 mm | 2 of 2 (34 % to 100 %) | 7 of 9 (45 % to 94 %) |
| 20 to 30 mm | no trial | 2 of 3 (21 % to 94 %) |
| 30 to 40 mm | no trial | 0 of 1 (0 % to 79 %) |
| all distances | 13 of 13 | 9 of 13 |

## The second scoring from video

24 of the 108 scored trials were scored again from their videos alone, with neither the condition nor the distance shown to the scorer: one trial in 5 of each condition of each block, rounded up, drawn with seed 20261006 from `data/trials/main.jsonl` after its last trial (protocol section 8). From `data/second_scoring/main/agreement.json`, written by `scripts/second_scoring.py compare`. The first score, given at the rig, is the one the analysis uses whatever the second says.

- **Agreement:** the two scores agree in 24 of the 24 trials the second scorer could tell: 100 % (Wilson 95 % interval 86 % to 100 %). They differ in 0, and in 0 the second scorer answered "cannot tell".
- **Who scored:** the first score was given at the rig by op1, the second from the video by op1 (role codes, from the trial records and from the scores).
- **Both scores of all 24 trials are by the same person**, the second given from the video with the trials' identities hidden: no second person scored (protocol section 8 provides for this).

First score against second score, in trials:

| First score (at the rig) | Second: success | Second: failure | Second: cannot tell |
|---|---|---|---|
| success | 17 | 0 | 0 |
| failure | 0 | 7 | 0 |

By kind of trial:

| Kind of trial | Scored again | Agree | Differ | Cannot tell | Wilson 95 % interval of the share that agree |
|---|---|---|---|---|---|
| policy | 19 | 19 of 19 | 0 | 0 | 83 % to 100 % |
| replay | 5 | 5 of 5 | 0 | 0 | 57 % to 100 % |

## The scene at each trial's start

The basket stands on tape and the arm can carry it off; the same catch can move the wrist camera in its mount. So the first and the last frames of every trial's two videos were read for the basket's place and for the wrist camera's, each against its reference picture, and a trial that began with the basket more than 2 mm or the wrist camera more than 2 px from its place is named here (protocol section 8). From `data/audit/scene_main.json`, written by `scripts/rig/scene_audit.py`; a trial's kind and outcome are those of its record. An offset is the larger of across and down, as the audit reads it: millimetres on the floor for the basket, pixels of the wrist camera's picture for that camera.

- **Trials read:** 108 of the 108 scored trials are in the audit. The basket's place at the start was read in 108 of them and the wrist camera's in 108. The first frames of every video in the audit could be read.
- **The basket was moved in 2 trial(s)** (it stood more than 5 mm from where the trial began, at the trial's last frames or, where the arm hid it then, when the next trial of the session began: the audit's rule): T1a-008 (the clustered policy, success, 35.0 mm); T1a-013 (the clustered policy, success, 45.0 mm). A trial in which the arm moves the basket stands as scored (protocol section 8). The end of 1 trial(s) says nothing of the basket (its last frames did not show the basket, and the audit found no later trial of the session to judge it by): T1a-005.
- **Named for the basket:** 49 trial(s) began with it more than 2 mm from its place: T1b-013, T1b-014, T1b-015, T1b-016, T1a-020, T1a-021, T1a-022, T1a-023, T1a-024, T1b-017, T1b-018, T1b-019, T1a-025, T1b-020, T1a-026, T1b-021, T1a-027, T1a-028, T1a-029, T1a-030, T1a-031, T1a-032, T1a-033, T1b-022, T1a-034, T2-001, T2-002, T2-003, T2-030, T2-032, T2-033, T2-034, T2-035, T2-037, T2-038, T2-039, T2-040, T2-041, T2-042, T2-043, T2-044, T2-045, T2-046, T2-047, T2-048, T2-049, T2-050, T2-051, T2-052.
- **Named for the wrist camera:** no trial began with it more than 2 px from its place.
- **Largest offsets at a start:** the basket 2.3 mm, the wrist camera 1.9 px.
- **Successes, as plain counts and with no test:** 36 of the 49 trials named for the basket or for the wrist camera succeeded, and 36 of the 59 trials named for neither succeeded.

| The basket at a trial's start, from its place | Trials |
|---|---|
| up to 1 mm | 17 |
| over 1 mm, up to 2 mm | 42 |
| over 2 mm, up to 3 mm | 49 |
| over 3 mm | 0 |
| all trials whose start was read | 108 |

One pixel of the front camera is about 1.08 mm at the basket (`MM_PER_PX` in `src/tolerance_radius/rig/basket_place.py`: the two floor tags are 526.9 mm and 486 px apart), so the 2 mm line lies 1.9 pixels from the basket's place, and an offset of 2 whole pixels, about 2.2 mm, is the first pixel step past it. The trials named for the basket began 2.1 mm to 2.3 mm from it: at most 0.3 pixels past the line.

## How the trials were run

- **Resets:** over the 108 scored trials the accepted placement lay a median of 0.34 mm from its target, at most 0.62 mm (placed to target, from the records).
- **Pace:** a median of 1.27 minutes from one record's start to the next one's in the same log (107 intervals; void trials included, no pause left out).
- **Days:** 2026-10-08. **People** (operator and scorer codes): op1.
- **Execution settings of the clustered policy** (the same in every scored trial, and every setting of the protocol's `[execution]` section with the protocol's value): `clamp_deg` = 15.0, `control_hz` = 30, `handoff_tolerance_deg` = 8.0, `max_handoff_retries` = 3, `min_attempt_s` = 1.0, `replay_handoff_tolerance_deg` = 5.0, `stall_fraction` = 0.9, `stall_progress_deg` = 1.5, `stall_steps` = 30, `temporal_ensemble_coeff` = 0.01, `time_limit_s` = 45.0, `view_crept_limit_mm` = 1.0. Belonging to the single trial and not held to anything: `device` mps; `handoff_tolerance_used_deg` 8.0.
- **Execution settings of the replay** (the same in every scored trial, and every setting of the protocol's `[execution]` section with the protocol's value): `clamp_deg` = 15.0, `control_hz` = 30, `handoff_tolerance_deg` = 8.0, `max_handoff_retries` = 3, `min_attempt_s` = 1.0, `replay_handoff_tolerance_deg` = 5.0, `stall_fraction` = 0.9, `stall_progress_deg` = 1.5, `stall_steps` = 30, `temporal_ensemble_coeff` = 0.01, `time_limit_s` = 45.0, `view_crept_limit_mm` = 1.0. Belonging to the single trial and not held to anything: `device` mps; `handoff_tolerance_used_deg` 5.0.
- **Execution settings of the spread policy** (the same in every scored trial, and every setting of the protocol's `[execution]` section with the protocol's value): `clamp_deg` = 15.0, `control_hz` = 30, `handoff_tolerance_deg` = 8.0, `max_handoff_retries` = 3, `min_attempt_s` = 1.0, `replay_handoff_tolerance_deg` = 5.0, `stall_fraction` = 0.9, `stall_progress_deg` = 1.5, `stall_steps` = 30, `temporal_ensemble_coeff` = 0.01, `time_limit_s` = 45.0, `view_crept_limit_mm` = 1.0. Belonging to the single trial and not held to anything: `device` mps; `handoff_tolerance_used_deg` 8.0.
- **Control rate**, asked and achieved, from the records (`control_hz`, `steps_sent`, `loop_s`). Achieved is the rate between the first step sent and the last: the steps sent after the first, over the time from the first step to the last (the record's `loop_s`, taken on the trial runner's monotonic clock and kept to a millisecond). The handoff gate and the policy's first inference come before the first step and are outside it, and so is everything after the last step. The loop sends its second step as soon as it is predicted, without waiting out a period, so a trial's figure can read high by up to one interval of its attempt: a third of a percent for 300 steps.
  - policy trials: asked 30 Hz; achieved a median of 29.1 Hz (29.1 to 29.2 Hz over 86 trials).
  - replay trials: asked 30 Hz; achieved a median of 29.2 Hz (29.2 to 29.3 Hz over 22 trials).
  - all scored trials: asked 30 Hz; achieved a median of 29.2 Hz (29.1 to 29.3 Hz over 108 trials).
  - **The policy trials ran more than 2 % below the rate they were asked for: a median of 29.1 Hz against 30 Hz, 2.9 % below.**
  - **The replay trials ran more than 2 % below the rate they were asked for: a median of 29.2 Hz against 30 Hz, 2.6 % below.**
- **The camera's creep before an attempt:** 108 of the 108 scored trials record how far the front camera's picture stood from where its session solved the conversion, read just before the attempt (`execution.view_crept_mm`, from the records): a median of 0.57 mm, at most 0.99 mm. These are as the sessions read them, not divided by the scale factor. None lies beyond the protocol's limit of 1 mm (`execution.view_crept_limit_mm`), beyond which a session makes no attempt.
- **Start pose of the arm** when the attempt began, over the 108 scored trials that record it, in the records' own units: `elbow_flex.pos` 95.8 to 96.8 (median 96.5); `gripper.pos` 0.5 to 10.0 (median 0.8); `shoulder_lift.pos` -103.4 to -100.2 (median -102.9); `shoulder_pan.pos` -9.3 to 1.4 (median -2.7); `wrist_flex.pos` 56.0 to 76.0 (median 75.3); `wrist_roll.pos` -100.1 to -84.7 (median -91.8).
- **Checkpoints** (from the records): the clustered policy `act-tolrad-clustered-v1@080000`; the spread policy `act-tolrad-spread-v1@080000`. Training step: 80,000; every scored policy trial names the same one.
- **`versions.lerobot`:** `lerobot 0.6.1 at 6adf51511b`.
- **`versions.lerobot101_commit`:** `02b3ced82f`.
- **`versions.project_commit`:** `57946a8c28`, `ccf9c7da57`, `ccf9c7da57+changes`.
- **`versions.protocol_status`:** `frozen`.
- **`versions.protocol_version`:** `1.0`.
- **Deviations:** the decision log marks 3:
  - 2026-10-08: **DEVIATION** from protocol 1.0, sections 8 and 12: block 3 (T3, the spread policy's threshold block, 24 trials) was not run, and this is counted as a deviation. The row of this day that records the omission called it "a decision and not the rule"; from here on the paper lists it with the deviations Why: The protocol's rule for leaving the block out (section 12, the third stop line) applies when blocks T1a and T1b are not complete by 15:00; they were complete at 14:54, and block T2 at 16:49, before its 19:00. Section 8 says the block is run "if time allows": the operator decided at about 17:07 that the evening went to the by-eye resets, the repeatability check, the rig's facts and the second scoring, before any result had been computed by kind of trial. Two of the blind readers of the paper found "not a deviation" unconvincing for these reasons, and the author decided to count it (2026-10-08, "yes to all three")
  - 2026-10-08: **DEVIATION** from protocol 1.0, section 7 (block V0, item 2): the repeatability check, which the protocol places before any trial, was run after the last counted trial, from 17:45 to 17:48 on the day of the trials (`data/instrument/repeat_20261008_174537.csv`) Why: The day's order at the rig put the pilot, the freeze and the blocks first, and the check stood on the list of what was left for the rig after block 2 (journal, 16:51); nobody held that list against section 7. No trial's measurement depends on the check's result (11 readings within 0.26 mm of their mean). Found on 2026-10-08 by the independent review of the paper against the protocol
  - 2026-10-08: **DEVIATION** from protocol 1.0, sections 7 (item 3) and 8: the tag check at the end of block 1 (T1a with T1b) and at the end of block 2 (T2) was not run as a step of its own. The next session's start made the same check, about 47 and 39 minutes after each block's last trial (15:40 and 17:28) Why: A session begins by solving the conversion from the floor tags and holding the stored calibration against it; both of those sessions went on, and their figures were shown, not logged. Read again on 2026-10-08 from each of the two sessions' kept start frames with the session's own check (`camera_tool.since_calibration`; one frame where a session averages 20): the camera's picture 0.82 and 0.71 mm from the calibration (said beyond 1.5 mm, refused beyond 5), the two floor tags 1.7 and 1.6 mm against each other since the registration (said beyond 1.5, refused beyond 3). Inside the blocks every session began with this check, and no attempt was made with the picture more than 1.0 mm from the session's own solve. Found by the same review

## What the paper quotes of the simulations of the design (simulated, not measured)

Each threshold block as written. A cell is the error of the estimated radius (RMSE) / the mean width of its 95 % interval, in millimetres, for the middle one of the simulated truths with that width (the simulation report's Table 1). The last column is the share of simulated blocks whose interval is open below, over the truths whose curve is already falling at 0 mm (radius under two widths).

| Block | Trials | Cliff (width 1 mm) | Middling (width 3 mm) | Slope (width 10 mm) | Open below when falling at 0 mm |
|---|---|---|---|---|---|
| T1b (the replay) | 22 | 1.01 / 6.29 | 2.07 / 12.57 | 5.16 / 23.62 | 60 % |
| T1a (the clustered policy) | 34 | 0.79 / 4.19 | 1.64 / 8.11 | 4.23 / 18.40 | 50 % |
| T3 (the spread policy) | 24 | 0.94 / 5.29 | 1.92 / 11.27 | 4.81 / 21.71 | 57 % |

Over the blocks together, the interval is open below in 55 % of such simulated blocks.

Fixed before any data: block T1a as written, against a curve with radius 20 mm, width 3 mm and lapse 0; 300 simulated blocks (`analysis/simulation/procedure.csv`, `analysis/simulation/paired_power.csv`, `analysis/simulation/procedure.md`, `analysis/simulation/comparison_rule.csv`, `analysis/simulation/comparison_rule.md`, made from protocol file `78fd2bfa905d`).

- As written: bias -0.2 mm, root-mean-square error 1.63 mm, mean interval width 7.80 mm, the interval held the truth in 97 % of blocks.
- The same trials at fixed, equally spaced distances: error 2.32 mm, interval width 9.01 mm.
- Paired test, 26 pairs, success rates 0.4 against 0.9: the exact McNemar test tells them apart 96 % of the time (computed exactly).

The comparison rule of prediction 2, simulated on the clustered policy's block and the replay's as written (`analysis/simulation/comparison_rule.csv`, made from protocol file `78fd2bfa905d`): the share of 300 simulated pairs of blocks with each verdict, both curves of one width, the replay's radius 15 mm.

| Width of both curves | True difference (policy minus replay) | larger | not told apart | smaller |
|---|---|---|---|---|
| 1 mm | 0 mm | 3 % | 95 % | 2 % |
| 1 mm | 3 mm | 56 % | 44 % | 0 % |
| 1 mm | 5 mm | 92 % | 8 % | 0 % |
| 1 mm | 8 mm | 100 % | 0 % | 0 % |
| 1 mm | 10 mm | 100 % | 0 % | 0 % |
| 1 mm | 15 mm | 100 % | 0 % | 0 % |
| 3 mm | 0 mm | 2 % | 95 % | 3 % |
| 3 mm | 3 mm | 17 % | 82 % | 1 % |
| 3 mm | 5 mm | 38 % | 62 % | 0 % |
| 3 mm | 8 mm | 72 % | 28 % | 0 % |
| 3 mm | 10 mm | 86 % | 14 % | 0 % |
| 3 mm | 15 mm | 99 % | 1 % | 0 % |
| 6 mm | 0 mm | 3 % | 94 % | 3 % |
| 6 mm | 3 mm | 10 % | 88 % | 2 % |
| 6 mm | 5 mm | 20 % | 80 % | 0 % |
| 6 mm | 8 mm | 38 % | 62 % | 0 % |
| 6 mm | 10 mm | 55 % | 45 % | 0 % |
| 6 mm | 15 mm | 83 % | 17 % | 0 % |

## What the analysis could not compute

- **The block fit of act_spread:** block T3 has no scored trial.
- **The comparison spread_minus_clustered:** it needs both block fits.
- **The largest disagreement of a session's camera solve with the stored one:** the tools print it and no committed record holds it.
- **Torn frames and mispaced episodes of the new datasets:** the dataset checks print their findings and no committed record holds them.
- **The number of practice episodes:** no record says which episodes were practice; the placement logs only show lines that were later recorded over.

## The paper's number macros

200 of 217 have a value; the others print `??` in the paper. A value is written as the paper prints it; the source is the comment beside the macro in `paper/generated/numbers.tex`.

| Macro | Value | From |
|---|---|---|
| `\simTruthRadius` | `20.0` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simTruthWidth` | `3.0` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simReplications` | `300` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simBias` | `\ensuremath{-}0.2` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simRmse` | `1.6` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simIntervalWidth` | `7.8` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simCoverage` | `97` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simRmseConstant` | `2.3` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simIntervalWidthConstant` | `9.0` | analysis/simulation/procedure.csv, the case fixed before the data (simulated, not measured) |
| `\simPairedRateLow` | `40` | analysis/simulation/paired_power.csv, the case fixed before the data (computed exactly, not measured) |
| `\simPairedRateHigh` | `90` | analysis/simulation/paired_power.csv, the case fixed before the data (computed exactly, not measured) |
| `\simPairedPower` | `96` | analysis/simulation/paired_power.csv, the case fixed before the data (computed exactly, not measured) |
| `\simWidthCliff` | `1.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simWidthMiddling` | `3.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simWidthSlope` | `10.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseReplayCliff` | `1.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthReplayCliff` | `6.3` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseReplayMiddling` | `2.1` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthReplayMiddling` | `12.6` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseReplaySlope` | `5.2` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthReplaySlope` | `23.6` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simOpenBelowReplay` | `60` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simRmseClusteredCliff` | `0.8` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthClusteredCliff` | `4.2` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseClusteredMiddling` | `1.6` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthClusteredMiddling` | `8.1` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseClusteredSlope` | `4.2` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthClusteredSlope` | `18.4` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simOpenBelowClustered` | `50` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simRmseSpreadCliff` | `0.9` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthSpreadCliff` | `5.3` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseSpreadMiddling` | `1.9` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthSpreadMiddling` | `11.3` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseSpreadSlope` | `4.8` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthSpreadSlope` | `21.7` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simOpenBelowSpread` | `57` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simOpenBelow` | `55` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simRuleReplayRadius` | `15.0` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRulePairs` | `300` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffEqual` | `3` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffThree` | `56` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffFive` | `92` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffEight` | `100` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffTen` | `100` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffFifteen` | `100` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleSmallerWidthOneDiffEqual` | `2` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffEqual` | `2` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffThree` | `17` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffFive` | `38` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffEight` | `72` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffTen` | `86` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffFifteen` | `99` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleSmallerWidthThreeDiffEqual` | `3` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffEqual` | `3` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffThree` | `10` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffFive` | `20` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffEight` | `38` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffTen` | `55` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffFifteen` | `83` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleSmallerWidthSixDiffEqual` | `3` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\nScaleRuns` | `4` | data/instrument/ruler_*.csv, as the camera reads |
| `\nScaleSteps` | `31` | data/instrument/ruler_*.csv, as the camera reads |
| `\instrumentMeanAbsError` | `0.3` | data/instrument/ruler_*.csv, as the camera reads |
| `\instrumentMaxStepError` | `0.8` | data/instrument/ruler_*.csv, as the camera reads |
| `\scaleFactor` | `1.019` | data/instrument/ruler_*.csv, as the camera reads: mean of the runs |
| `\scaleFactorMin` | `1.012` | data/instrument/ruler_*.csv, as the camera reads: smallest of the runs |
| `\scaleFactorMax` | `1.025` | data/instrument/ruler_*.csv, as the camera reads: largest of the runs |
| `\nRepeatability` | `11` | data/instrument/repeat_*.csv |
| `\repeatabilityScatter` | `0.3` | data/instrument/repeat_*.csv, as the camera reads |
| `\nByEye` | `20` | data/instrument/byeye_*.csv |
| `\nOperatorsByEye` | `1` | data/instrument/byeye_*.csv: the labels |
| `\byEyeErrorMedian` | `15.9` | data/instrument/byeye_*.csv |
| `\byEyeErrorNinetyFive` | `31.0` | data/instrument/byeye_*.csv: 95th percentile, interpolated between ranks |
| `\byEyeErrorMax` | `35.5` | data/instrument/byeye_*.csv |
| `\nDemosTrainedClustered` | `55` | data/audit/clustered.json |
| `\nDemosTrainedSpread` | `55` | data/audit/spread.json |
| `\nRerecordedClustered` | `2` | data/recording/clustered_placements.jsonl: kept episodes with at least one earlier take |
| `\nRerecordedSpread` | `1` | data/recording/spread_placements.jsonl: kept episodes with at least one earlier take |
| `\demoPlacementErrorMedian` | `0.7` | data/audit/clustered.json and data/audit/spread.json: measured start to planned target |
| `\demoPlacementErrorMax` | `2.7` | data/audit/clustered.json and data/audit/spread.json: measured start to planned target |
| `\clusteredScatterMax` | `1.4` | data/audit/clustered.json: trained starts about the mean of their point |
| `\spreadSpacingMin` | `17.5` | data/audit/spread.json: trained starts |
| `\nRecordingDays` | `1` | data/recording/*_placements.jsonl: dates of the kept episodes |
| `\nTrialsReplay` | `22` | the counted trial logs, replay, scored, every block |
| `\nVoidReplay` | `0` | the counted trial logs, replay, void, every block |
| `\successZeroReplay` | `4` | the counted trial logs, replay, block T1b, anchors at d = 0 |
| `\nZeroReplay` | `4` | the counted trial logs, replay, block T1b, anchors at d = 0 |
| `\radiusReplay` | `16.5` | block fit of replay: posterior median of r50 |
| `\radiusReplayLow` | `13.0` | block fit of replay: lower end of the interval |
| `\radiusReplayHigh` | `21.5` | block fit of replay: upper end of the interval |
| `\maxSuccessDistReplay` | `19.8` | the counted trial logs, replay, scored, block T1b alone |
| `\minFailureDistReplay` | `14.7` | the counted trial logs, replay, scored, block T1b alone |
| `\radiusReplayWide` | `17.0` | block fit of replay made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusReplayWideLow` | `13.0` | block fit of replay made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusReplayWideHigh` | `23.5` | block fit of replay made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftReplay` | `0.5` | block fit of replay: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nReplayEpisodes` | `11` | the counted trial logs, replay, scored: distinct replayed episodes |
| `\nTrialsClustered` | `60` | the counted trial logs, act_clustered, scored, every block |
| `\nVoidClustered` | `0` | the counted trial logs, act_clustered, void, every block |
| `\successZeroClustered` | `6` | the counted trial logs, act_clustered, block T1a, anchors at d = 0 |
| `\nZeroClustered` | `6` | the counted trial logs, act_clustered, block T1a, anchors at d = 0 |
| `\radiusClustered` | `36.0` | pooled fit of act_clustered: posterior median of r50 |
| `\radiusClusteredLow` | `25.0` | pooled fit of act_clustered: lower end of the interval |
| `\radiusClusteredHigh` | `48.5` | pooled fit of act_clustered: upper end of the interval |
| `\maxSuccessDistClustered` | `44.8` | the counted trial logs, act_clustered, scored, block T1a alone |
| `\minFailureDistClustered` | `10.1` | the counted trial logs, act_clustered, scored, block T1a alone |
| `\radiusClusteredWide` | `42.0` | pooled fit of act_clustered made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusClusteredWideLow` | `27.5` | pooled fit of act_clustered made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusClusteredWideHigh` | `58.0` | pooled fit of act_clustered made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftClustered` | `6.0` | pooled fit of act_clustered: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nTrialsClusteredThreshold` | `34` | the counted trial logs, block T1a, scored |
| `\nVoidClusteredThreshold` | `0` | the counted trial logs, block T1a, void |
| `\successFarClustered` | `1` | the counted trial logs, block T1a, far anchors |
| `\nFarClustered` | `4` | the counted trial logs, block T1a, far anchors |
| `\radiusClusteredThreshold` | `39.0` | block fit of act_clustered: posterior median of r50 |
| `\radiusClusteredThresholdLow` | `23.0` | block fit of act_clustered: lower end of the interval |
| `\radiusClusteredThresholdHigh` | `52.0` | block fit of act_clustered: upper end of the interval |
| `\radiusClusteredThresholdWide` | `42.5` | block fit of act_clustered made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusClusteredThresholdWideLow` | `26.0` | block fit of act_clustered made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusClusteredThresholdWideHigh` | `64.0` | block fit of act_clustered made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftClusteredThreshold` | `3.5` | block fit of act_clustered: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nTrialsSpread` | `26` | the counted trial logs, act_spread, scored, every block |
| `\nVoidSpread` | `0` | the counted trial logs, act_spread, void, every block |
| `\successZeroSpread` | ?? | the counted trial logs, act_spread, block T3, anchors at d = 0 |
| `\nZeroSpread` | `0` | the counted trial logs, act_spread, block T3, anchors at d = 0 |
| `\radiusSpread` | `28.5` | pooled fit of act_spread: posterior median of r50 |
| `\radiusSpreadLow` | `19.5` | pooled fit of act_spread: lower end of the interval |
| `\radiusSpreadHigh` | `64.5` | pooled fit of act_spread: upper end of the interval |
| `\maxSuccessDistSpread` | ?? | the counted trial logs, act_spread, scored, block T3 alone |
| `\minFailureDistSpread` | ?? | the counted trial logs, act_spread, scored, block T3 alone |
| `\radiusSpreadWide` | `29.0` | pooled fit of act_spread made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusSpreadWideLow` | `20.0` | pooled fit of act_spread made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusSpreadWideHigh` | `69.0` | pooled fit of act_spread made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftSpread` | `0.5` | pooled fit of act_spread: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nTrialsSpreadThreshold` | `0` | the counted trial logs, block T3, scored |
| `\nVoidSpreadThreshold` | `0` | the counted trial logs, block T3, void |
| `\radiusSpreadThreshold` | ?? | block fit of act_spread: posterior median of r50 |
| `\radiusSpreadThresholdLow` | ?? | block fit of act_spread: lower end of the interval |
| `\radiusSpreadThresholdHigh` | ?? | block fit of act_spread: upper end of the interval |
| `\radiusSpreadThresholdWide` | ?? | block fit of act_spread made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusSpreadThresholdWideLow` | ?? | block fit of act_spread made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusSpreadThresholdWideHigh` | ?? | block fit of act_spread made again with the protocol's longer list of lapse values: upper end of the interval |
| `\nPairs` | `26` | the counted trial logs, block T2: placements with a scored trial of both policies |
| `\nTrialsWorkspace` | `52` | the counted trial logs, block T2, scored |
| `\nVoidWorkspace` | `0` | the counted trial logs, block T2, void |
| `\pairsBoth` | `14` | the paired table of block T2 |
| `\pairsNeither` | `2` | the paired table of block T2 |
| `\pairsOnlySpread` | `8` | the paired table of block T2 |
| `\pairsOnlyClustered` | `2` | the paired table of block T2 |
| `\pairsDiscordant` | `10` | the paired table of block T2 |
| `\mcnemarP` | `0.11` | exact McNemar test on the discordant pairs of block T2, all pairs together |
| `\workspaceSuccessClustered` | `16` | the pairs of block T2, act_clustered |
| `\workspaceClusteredLow` | `43` | the pairs of block T2, act_clustered: Wilson interval |
| `\workspaceClusteredHigh` | `78` | the pairs of block T2, act_clustered: Wilson interval |
| `\workspaceDistClusteredMedian` | `27.1` | the pairs of block T2, act_clustered: measured d |
| `\workspaceDistClusteredMax` | `50.5` | the pairs of block T2, act_clustered: measured d |
| `\lookupPredictedClustered` | `17.7` | block fit of act_clustered at the distances of the pairs of block T2, act_clustered |
| `\lookupClusteredLow` | `10` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_clustered |
| `\lookupClusteredHigh` | `24` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_clustered |
| `\workspaceSuccessSpread` | `22` | the pairs of block T2, act_spread |
| `\workspaceSpreadLow` | `66` | the pairs of block T2, act_spread: Wilson interval |
| `\workspaceSpreadHigh` | `94` | the pairs of block T2, act_spread: Wilson interval |
| `\workspaceDistSpreadMedian` | `10.8` | the pairs of block T2, act_spread: measured d |
| `\workspaceDistSpreadMax` | `38.3` | the pairs of block T2, act_spread: measured d |
| `\lookupPredictedSpread` | `21.0` | block fit of act_clustered at the distances of the pairs of block T2, act_spread |
| `\lookupSpreadLow` | `15` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_spread |
| `\lookupSpreadHigh` | `25` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_spread |
| `\nPairsTrained` | `13` | the pairs of block T2 whose measured position lies in trained cells |
| `\workspaceSuccessClusteredTrained` | `8` | the pairs of block T2 whose measured position lies in trained cells, act_clustered |
| `\lookupPredictedClusteredTrained` | `10.1` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in trained cells, act_clustered |
| `\workspaceSuccessSpreadTrained` | `13` | the pairs of block T2 whose measured position lies in trained cells, act_spread |
| `\lookupPredictedSpreadTrained` | `11.1` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in trained cells, act_spread |
| `\nPairsHeldOut` | `13` | the pairs of block T2 whose measured position lies in held-out cells |
| `\workspaceSuccessClusteredHeldOut` | `8` | the pairs of block T2 whose measured position lies in held-out cells, act_clustered |
| `\lookupPredictedClusteredHeldOut` | `7.5` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in held-out cells, act_clustered |
| `\workspaceSuccessSpreadHeldOut` | `9` | the pairs of block T2 whose measured position lies in held-out cells, act_spread |
| `\lookupPredictedSpreadHeldOut` | `9.9` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in held-out cells, act_spread |
| `\radiusDiffClusteredReplay` | `22.0` | block fits, act_clustered minus replay, taken as independent: median of the difference |
| `\radiusDiffClusteredReplayLow` | `5.5` | block fits, act_clustered minus replay, taken as independent: lower end of the interval |
| `\radiusDiffClusteredReplayHigh` | `35.5` | block fits, act_clustered minus replay, taken as independent: upper end of the interval |
| `\probClusteredExceedsReplay` | `>0.99` | block fits of act_clustered and replay, taken as independent; a tie on the grid counts half |
| `\radiusDiffSpreadClustered` | ?? | block fits, act_spread minus act_clustered, taken as independent: median of the difference |
| `\radiusDiffSpreadClusteredLow` | ?? | block fits, act_spread minus act_clustered, taken as independent: lower end of the interval |
| `\radiusDiffSpreadClusteredHigh` | ?? | block fits, act_spread minus act_clustered, taken as independent: upper end of the interval |
| `\probSpreadExceedsClustered` | ?? | block fits of act_spread and act_clustered, taken as independent; a tie on the grid counts half |
| `\resetSensitivityPlaced` | `88` | block fit of act_clustered at the placement errors of the scored trials (a calculation) |
| `\resetSensitivityByEye` | `79` | block fit of act_clustered at the by-eye errors of data/instrument/byeye_*.csv (a calculation) |
| `\nTrialsBasketNamed` | `49` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials whose first frames show the basket beyond the audit's limit from its place |
| `\nTrialsWristNamed` | `0` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials whose first frames show the wrist camera beyond the audit's limit from its place |
| `\nTrialsBasketMoved` | `2` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials that the audit marks as having moved the basket |
| `\basketStartMax` | `2.3` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials: the basket at the first frames against its reference, the larger of across and down |
| `\wristStartMax` | `1.9` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials: the fixed jaw in the wrist camera's first frames against its reference, the larger of across and down |
| `\nSecondScored` | `24` | data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare: the trials scored again |
| `\nSecondScoredAgree` | `24` | data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare: the trials whose second score is the first score |
| `\nSecondScoredCannotTell` | `0` | data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare: the trials the second scorer could not tell |
| `\nTrialsTotal` | `108` | the counted trial logs, scored |
| `\nVoidTotal` | `0` | the counted trial logs, void |
| `\voidReasons` | `none` | the counted trial logs: void_reason, as typed |
| `\resetErrorMedian` | `0.3` | the counted trial logs, scored: placed to target |
| `\resetErrorMax` | `0.6` | the counted trial logs, scored: placed to target |
| `\viewCreptMedian` | `0.57` | the counted trial logs, scored: execution.view_crept_mm, as the session read it before the attempt |
| `\viewCreptMax` | `0.99` | the counted trial logs, scored: execution.view_crept_mm, as the session read it before the attempt |
| `\trialMinutesMedian` | `1.3` | the counted trial logs: one record's ts_start to the next one's in the same log |
| `\nTrialDays` | `1` | the counted trial logs: dates of ts_start |
| `\nOperators` | `1` | the counted trial logs: operator and scorer codes |
| `\nDeviations` | `3` | docs/decisions.md: rows marked DEVIATION |
| `\evalCheckpoint` | `80,000` | the counted trial logs: checkpoint of the policy trials |
| `\execClamp` | `15` | the counted trial logs: execution.clamp_deg |
| `\execEnsembleCoeff` | `0.01` | the counted trial logs: execution.temporal_ensemble_coeff of the policy trials |
| `\execControlRate` | `30` | the counted trial logs: execution.control_hz |
| `\execControlRateAchieved` | `29.1` | the counted trial logs, scored policy trials: the steps sent after the first over execution.loop_s, the time from the first step sent to the last; median over the trials |
| `\execTimeLimit` | `45` | the counted trial logs: execution.time_limit_s |
| `\lerobotCommit` | `6adf51511b` | the counted trial logs: versions.lerobot |
| `\gridDriftMax` | ?? | cannot be produced: the tools print the disagreement of a session's camera solve and no committed record holds it |
| `\nPracticeEpisodes` | ?? | cannot be produced: no record says which episodes were practice |
| `\nTornFramesNew` | ?? | cannot be produced: the torn-frame scan prints its findings and no committed record holds them |
| `\nMispacedEpisodesNew` | ?? | cannot be produced: the pacing check prints its findings and no committed record holds them |
