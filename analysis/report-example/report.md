**SYNTHETIC: made by tests/fixtures_experiment.py from invented success curves. No trial was run. Nothing in this file is a result. DRAFT: written with --draft. It may hold trials that were not run under the frozen protocol. Nothing in it goes into the paper.**

# Analysis report

This is a draft, written by `scripts/make_report.py` together with `results.json`. Every number below is computed from the files named in this section, by the rules of `protocol/protocol.md`, section 10. A number is followed by where it comes from.

- **Protocol:** `config/protocol.toml`, version 1.0, status frozen, SHA-256 `647d0a7f49ef`.
- **Trial logs read:** `data/trials/main.jsonl`: 145 records, 132 scored and 13 void.
- **Pilot logs left alone:** `data/trials/pilot_replay.jsonl`. Pilot trials are never analysed.
- **The trials ran on** commit `0123abc456` of this repository (from the records).
- **Audits:** `data/audit/clustered.json`, `data/audit/spread.json`. **Scene audits:** `data/audit/scene_main.json`.
- **Instrument files:** scale runs `data/instrument/ruler_x_20261006_081000.csv`, `data/instrument/ruler_x_corner_20261006_081200.csv`, `data/instrument/ruler_y_20261006_081100.csv`, `data/instrument/ruler_y_corner_20261006_081300.csv`; repeated placements `data/instrument/repeat_20261006_082500.csv`; resets by eye `data/instrument/byeye_op1_20261007_160000.csv`.
- **Placement logs of the recorder:** `data/recording/clustered_placements.jsonl`, `data/recording/spread_placements.jsonl`.
- **Decision log:** `docs/decisions.md`.
- **Second scoring from video:** `data/second_scoring/main/agreement.json`.

## The radii at a glance

| Condition | Fit | Successes | At d = 0 | Radius (median) | 95 % interval | Do the data bound it? | Radius, longer lapse list | Its interval | Shift |
|---|---|---|---|---|---|---|---|---|---|
| the clustered policy | block (T1a) | 18 of 34 | 6 of 6 | 21.0 mm | 17.0 mm to 26.0 mm | yes, on both sides | 21.5 mm | 17.0 mm to 26.5 mm | +0.5 mm |
| the clustered policy | pooled (T1a, T2) | 27 of 60 | 6 of 6 | 21.5 mm | 18.5 mm to 24.0 mm | yes, on both sides | 21.5 mm | 18.5 mm to 24.0 mm | 0.0 mm |
| the replay | block (T1b) | 12 of 22 | 4 of 4 | 12.0 mm | 9.0 mm to 14.5 mm | yes, on both sides | 12.0 mm | 9.0 mm to 15.0 mm | 0.0 mm |
| the spread policy | block (T3) | 14 of 24 | 4 of 4 | 20.0 mm | 11.0 mm to 31.0 mm | yes, on both sides | 21.0 mm | 12.0 mm to 51.0 mm | +1.0 mm |
| the spread policy | pooled (T2, T3) | 30 of 50 | 4 of 4 | 18.0 mm | 13.5 mm to 25.0 mm | yes, on both sides | 19.0 mm | 13.5 mm to 26.5 mm | +1.0 mm |

The radius is the posterior median of `r50`. Comparisons and predictions use block fits only; a pooled fit is every scored trial of a policy and is given as a check.

**The plateau check** (protocol section 10): every fit is made a second time with the lapse values 0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, everything else the same. The first fit is the radius that is reported; the shift is the second fit's median minus the first's. The counts at d = 0 are the trials asked at a demonstration's own start among the trials of that fit.

No plateau is in doubt: no fit's two medians differ by more than 5 mm.

## Distances and the scale correction

**Every distance in this report and in `results.json` is divided by 1.0303.** The 4 scale run(s) read a true length as longer by factors from 1.0270 to 1.0330 (mean 1.0303); they differ by 0.60 percentage points. That is within the one percentage point of protocol section 7, so the correction applies.

Divided: every distance between two measured positions (a trial's distance to its demonstration, a placement error, a by-eye error, the spacing and scatter of the demonstrations). Not divided: positions (`x_mm`, `y_mm`), the distances the planner asked for (`level_mm`), and the scale check itself, which is given as read.

## The instrument (blocks V0 and V1)

**Scale check:** over 20 steps of 10 mm in 4 run(s), a measured step was off by 0.30 mm on average and by 0.33 mm at worst, as read. It passes the protocol's line (mean at most 1.5 mm, no step off by more than 3 mm).

| Run | Steps | Measured span | Factor | Mean step error | Worst step | Off the line |
|---|---|---|---|---|---|---|
| `data/instrument/ruler_x_20261006_081000.csv` | 5 | 51.50 of 50 mm | 1.0300 | 0.30 mm | 0.30 mm | 0.00 mm |
| `data/instrument/ruler_x_corner_20261006_081200.csv` | 5 | 51.65 of 50 mm | 1.0330 | 0.33 mm | 0.33 mm | 0.00 mm |
| `data/instrument/ruler_y_20261006_081100.csv` | 5 | 51.35 of 50 mm | 1.0270 | 0.27 mm | 0.27 mm | 0.00 mm |
| `data/instrument/ruler_y_corner_20261006_081300.csv` | 5 | 51.55 of 50 mm | 1.0310 | 0.31 mm | 0.31 mm | 0.00 mm |

**Repeatability:** 10 placements in 1 set(s) (`data/instrument/repeat_20261006_082500.csv`) lay at most 0.23 mm from the mean of their set, as read.

**Resets by eye:** 20 resets by 1 person(s) (op1) landed a median of 4.53 mm from where the cube was for the reference picture; 95th percentile 11.41 mm (interpolated between ranks), largest 11.52 mm. From `data/instrument/byeye_op1_20261007_160000.csv`.

Not in any committed record: the disagreement of each session's camera solve with the stored one (the tools print it).

## The two datasets

| Set | Trained on | Start to target, median | Largest | Layout as measured | Recording |
|---|---|---|---|---|---|
| clustered | 55 of 55 | 0.81 mm | 1.37 mm | scatter about a point at most 1.46 mm | 7 episodes recorded again (8 extra takes); 3 earlier lines recorded over; days 2026-10-06 |
| spread | 55 of 55 | 0.93 mm | 1.55 mm | closest two starts 16.65 mm apart | 7 episodes recorded again (8 extra takes); 0 earlier lines recorded over; days 2026-10-06 |

Both sets together: a start lay a median of 0.86 mm from its planned target, at most 1.55 mm (from the audits).

"Recorded again" counts kept episodes that had at least one earlier take in the same sitting. "Recorded over" counts lines of the placement log whose episode number was saved again later: a smoke test or a restart. No record says which of those were practice episodes.

Not in any committed record: torn frames and frame pacing of the new datasets (the checks print them).

## Trials

| Block | Condition | Planned | Scored | Void |
|---|---|---|---|---|
| T1b | replay | 22 | 22 | 2 |
| T1a | act_clustered | 34 | 34 | 3 |
| T2 | act_clustered and act_spread | 52 | 52 | 5 |
| T3 | act_spread | 24 | 24 | 3 |
| all |  | 132 | 132 | 13 |

**Void trials**, in the order of the logs. A void trial is not analysed and was asked again.

| Trial | Block | Condition | Reason, as typed |
|---|---|---|---|
| T1b-004 | T1b | replay | the handoff gate refused the start pose |
| T1a-015 | T1a | act_clustered | bumped the table |
| T1a-022 | T1a | act_clustered | ended by a key 0.4 s after it began (under 1 s) |
| T1a-029 | T1a | act_clustered | camera dropped 50% of frames_in #2 |
| T1b-020 | T1b | replay | Bumped the table. |
| T2-005 | T2 | act_spread | ended by a key 0.7 s after it began (under 1 s) |
| T2-016 | T2 | act_clustered | the handoff gate refused the start pose |
| T2-027 | T2 | act_clustered | bumped the table |
| T2-038 | T2 | act_spread | ended by a key 0.4 s after it began (under 1 s) |
| T2-049 | T2 | act_spread | camera dropped 50% of frames_in #2 |
| T3-003 | T3 | act_spread | Bumped the table. |
| T3-014 | T3 | act_spread | ended by a key 0.7 s after it began (under 1 s) |
| T3-025 | T3 | act_spread | the handoff gate refused the start pose |

By reason: bumped the table (4); ended by a key less than 1 s after it began (4); the handoff gate refused the start pose (3); camera dropped 50% of frames_in #2 (2).

## Clustered policy (`act_clustered`)

60 scored trials and 5 void, from the counted logs. In block T1a: 6 of 6 succeeded at d = 0, 0 of 4 at the far anchors. Over all its scored trials, the largest distance with a success is 25.2 mm and the smallest with a failure 17.7 mm.

| Distance (closed below, open above) | Successes | Wilson 95 % interval |
|---|---|---|
| 0 to 10 mm | 9 of 9 | 70 % to 100 % |
| 10 to 20 mm | 14 of 17 | 59 % to 94 % |
| 20 to 30 mm | 4 of 21 | 8 % to 40 % |
| 30 to 40 mm | 0 of 10 | 0 % to 28 % |
| 40 to 50 mm | 0 of 3 | 0 % to 56 % |

**Block fit** (T1a; 18 successes in 34 trials): radius 21.0 mm, 95 % interval 17.0 mm to 26.0 mm. Bounded by the data: yes, on both sides. Width 2.97 mm (median), lapse 0.035 and floor 0.045 (posterior means). Its trials: 6 of 6 at d = 0; farthest success 25.2 mm, nearest failure 17.7 mm. With lapse values up to 0.5: radius 21.5 mm (17.0 mm to 26.5 mm; yes, on both sides), a shift of +0.5 mm; lapse 0.054 (posterior mean).

**Pooled fit** (T1a, T2; 27 successes in 60 trials): radius 21.5 mm, 95 % interval 18.5 mm to 24.0 mm. Bounded by the data: yes, on both sides. Width 2.61 mm (median), lapse 0.029 and floor 0.030 (posterior means). Its trials: 6 of 6 at d = 0; farthest success 25.2 mm, nearest failure 17.7 mm. With lapse values up to 0.5: radius 21.5 mm (18.5 mm to 24.0 mm; yes, on both sides), a shift of 0.0 mm; lapse 0.035 (posterior mean).

## Replay (`replay`)

22 scored trials and 2 void, from the counted logs. In block T1b: 4 of 4 succeeded at d = 0. Over all its scored trials, the largest distance with a success is 13.3 mm and the smallest with a failure 10.1 mm.

| Distance (closed below, open above) | Successes | Wilson 95 % interval |
|---|---|---|
| 0 to 10 mm | 10 of 10 | 72 % to 100 % |
| 10 to 20 mm | 2 of 10 | 6 % to 51 % |
| 20 to 30 mm | 0 of 2 | 0 % to 66 % |

**Block fit** (T1b; 12 successes in 22 trials): radius 12.0 mm, 95 % interval 9.0 mm to 14.5 mm. Bounded by the data: yes, on both sides. Width 1.22 mm (median), lapse 0.037 and floor 0.042 (posterior means). Its trials: 4 of 4 at d = 0; farthest success 13.3 mm, nearest failure 10.1 mm. With lapse values up to 0.5: radius 12.0 mm (9.0 mm to 15.0 mm; yes, on both sides), a shift of 0.0 mm; lapse 0.054 (posterior mean).

Demonstrations replayed: episodes 0, 5, 15, 17, 18, 23, 25, 28, 35, 48, 51.

## Spread policy (`act_spread`)

50 scored trials and 6 void, from the counted logs. In block T3: 4 of 4 succeeded at d = 0. Over all its scored trials, the largest distance with a success is 25.0 mm and the smallest with a failure 2.8 mm.

| Distance (closed below, open above) | Successes | Wilson 95 % interval |
|---|---|---|
| 0 to 10 mm | 16 of 19 | 62 % to 94 % |
| 10 to 20 mm | 12 of 19 | 41 % to 81 % |
| 20 to 30 mm | 2 of 10 | 6 % to 51 % |
| 30 to 40 mm | 0 of 2 | 0 % to 66 % |

**Block fit** (T3; 14 successes in 24 trials): radius 20.0 mm, 95 % interval 11.0 mm to 31.0 mm. Bounded by the data: yes, on both sides. Width 6.37 mm (median), lapse 0.045 and floor 0.050 (posterior means). Its trials: 4 of 4 at d = 0; farthest success 25.0 mm, nearest failure 8.6 mm. With lapse values up to 0.5: radius 21.0 mm (12.0 mm to 51.0 mm; open above: report "at least"), a shift of +1.0 mm; lapse 0.116 (posterior mean).

**Pooled fit** (T2, T3; 30 successes in 50 trials): radius 18.0 mm, 95 % interval 13.5 mm to 25.0 mm. Bounded by the data: yes, on both sides. Width 5.61 mm (median), lapse 0.053 and floor 0.048 (posterior means). Its trials: 4 of 4 at d = 0; farthest success 25.0 mm, nearest failure 2.8 mm. With lapse values up to 0.5: radius 19.0 mm (13.5 mm to 26.5 mm; yes, on both sides), a shift of +1.0 mm; lapse 0.099 (posterior mean).

## Comparison of radii

**The clustered policy's radius minus the replay's:** 9.0 mm (95 % interval 4.0 mm to 14.5 mm). Posterior probability that the first is larger: 0.9981. Verdict by the rule of prediction 2: **larger**.

**The spread policy's radius minus the clustered policy's:** -1.5 mm (95 % interval -11.0 mm to 10.5 mm). Posterior probability that the first is larger: 0.3845. Verdict by the rule of prediction 2: **not told apart**.

From the two block fits, taken as independent, with every pair of grid values weighted exactly. "Larger" is a probability of at least 0.95, "smaller" at most 0.05. On the grid two radii can be equal; such a tie counts half for each side.

## The workspace test

26 placements were tested with both policies (block T2). Both succeeded at 8, neither at 9, only the spread policy at 8, only the clustered policy at 1. Exact McNemar test over all pairs together: p = 0.0391.

| Policy | Successes | Wilson 95 % interval | Median d | Largest d | Lookup prediction (expected, 95 % interval of the count) |
|---|---|---|---|---|---|
| the clustered policy | 9 of 26 | 19 % to 54 % | 26.1 mm | 49.5 mm | 9.55 (6 to 15): matches |
| the spread policy | 16 of 26 | 43 % to 78 % | 11.3 mm | 36.7 mm | 20.40 (15 to 24): matches |

The lookup prediction is the clustered policy's block fit, which has not seen these trials, at each trial's measured distance to its own policy's nearest demonstration. The interval is that of the predicted count: it carries the uncertainty of the fit and the chance in the trials. An observed count inside it, ends included, matches.

By region. A pair's region is the grid cell of its measured position (the mean of its two placements). The lookup prediction's verdict is made over all pairs; the expected count and the interval of the predicted count are given for each group besides.

| Region | Pairs | Policy | Successes | Wilson 95 % interval | Lookup expects |
|---|---|---|---|---|---|
| trained cells | 13 | the clustered policy | 9 of 13 | 42 % to 87 % | 8.00 (5 to 11) |
| trained cells | 13 | the spread policy | 11 of 13 | 58 % to 96 % | 12.09 (9 to 13) |
| held-out cells | 13 | the clustered policy | 0 of 13 | 0 % to 23 % | 1.55 (0 to 5) |
| held-out cells | 13 | the spread policy | 5 of 13 | 18 % to 64 % | 8.31 (5 to 11) |

By training run: every pair was run with one pair of runs, `act-tolrad-clustered-v1` for the clustered policy and `act-tolrad-spread-v1` for the spread policy (from the records' `checkpoint`).

## Reset sensitivity (a calculation, not a measurement)

The clustered policy's block fit, at a demonstration's own start, with a reset that is off by each of the 132 placement errors of this study's scored trials: 95 % success. With each of the 20 by-eye errors of block V1 instead: 94 %.

## The predictions of protocol section 9

1. **The curve.** Predicted: at least 5 successes in 6 at d = 0, at most 1 in 4 at the far anchors, and a radius between 5 and 40 mm. Found: 6 of 6 at d = 0, 0 of 4 at the far anchors, radius 21.0 mm (block fit, posterior median). **held**.
2. **Hardware.** Predicted: the clustered policy's radius is larger than the replay's, by a posterior probability of at least 0.95. Found: the difference is 9.0 mm (4.0 mm to 14.5 mm), and the probability that the policy's radius is larger is 0.9981: larger. **held**.
3. **Layout.** Predicted: at the placements of the workspace test the spread policy succeeds more often than the clustered one, by the exact paired test over all pairs: p below 0.05 with the spread policy ahead. Found: the spread policy succeeded at 16 of 26 placements and the clustered policy at 9 (in trained cells 11 against 9 of 13; in held-out cells 5 against 0 of 13); exact McNemar p = 0.0391. **held**.
4. **Lookup.** Predicted: each policy's count in the workspace test lies inside the 95 % interval of its lookup prediction. Found: the clustered policy 9 against 9.55 (6 to 15): matches; the spread policy 16 against 20.40 (15 to 24): matches. **held**.
5. **Layout and radius.** No prediction was made; reported as an estimate. Found: the spread policy's radius minus the clustered policy's is -1.5 mm (-11.0 mm to 10.5 mm); probability that it is larger 0.3845.

## Descriptive only

| Condition | Successes inside and outside the hull of the demonstrations | Failure labels (count, range of d) | Median duration |
|---|---|---|---|
| the clustered policy | 15 of 34 inside, 12 of 26 outside | grasp 13 (17.7 mm to 39.6 mm); no_recovery 5 (25.8 mm to 49.5 mm); perception 9 (18.8 mm to 42.3 mm); transport 6 (21.6 mm to 39.6 mm) | 20 s |
| the replay | does not apply | grasp 7 (10.1 mm to 24.5 mm); no_recovery 2 (15.7 mm to 18.7 mm); transport 1 (14.7 mm to 14.7 mm) | 19 s |
| the spread policy | 26 of 39 inside, 4 of 11 outside | grasp 12 (2.8 mm to 34.1 mm); no_recovery 2 (17.8 mm to 36.7 mm); perception 2 (5.9 mm to 26.3 mm); transport 4 (18.9 mm to 28.4 mm) | 20 s |

A duration runs from a record's `ts_start` to its `ts_end`: from just before the handoff gate to after the videos were closed, in whole seconds. It is longer than the attempt. The control loop's own duration is `loop_s`, and the rate made from it is under "How the trials were run".

**Success by the cell of the placement, at each distance.** A trial counts as in a trained cell or in a held-out cell by the grid cell its measured position falls in. Beyond the largest gap of a layout every placement is in a held-out cell, so on the far side of a curve distance and a cell without demonstrations cannot be told apart (protocol section 10).

Clustered policy (`act_clustered`), successes of trials with a Wilson 95 % interval.

| Distance | In trained cells | In held-out cells |
|---|---|---|
| 0 to 10 mm | 9 of 9 (70 % to 100 %) | no trial |
| 10 to 20 mm | 14 of 17 (59 % to 94 %) | no trial |
| 20 to 30 mm | 4 of 15 (11 % to 52 %) | 0 of 6 (0 % to 39 %) |
| 30 to 40 mm | no trial | 0 of 10 (0 % to 28 %) |
| 40 to 50 mm | no trial | 0 of 3 (0 % to 56 %) |
| all distances | 27 of 41 | 0 of 19 |

Replay (`replay`), successes of trials with a Wilson 95 % interval.

| Distance | In trained cells | In held-out cells |
|---|---|---|
| 0 to 10 mm | 10 of 10 (72 % to 100 %) | no trial |
| 10 to 20 mm | 2 of 10 (6 % to 51 %) | no trial |
| 20 to 30 mm | 0 of 2 (0 % to 66 %) | no trial |
| all distances | 12 of 22 | no trial |

Spread policy (`act_spread`), successes of trials with a Wilson 95 % interval.

| Distance | In trained cells | In held-out cells |
|---|---|---|
| 0 to 10 mm | 16 of 19 (62 % to 94 %) | no trial |
| 10 to 20 mm | 3 of 3 (44 % to 100 %) | 9 of 16 (33 % to 77 %) |
| 20 to 30 mm | no trial | 2 of 10 (6 % to 51 %) |
| 30 to 40 mm | no trial | 0 of 2 (0 % to 66 %) |
| all distances | 19 of 22 | 11 of 28 |

## The second scoring from video

29 of the 132 scored trials were scored again from their videos alone, with neither the condition nor the distance shown to the scorer: one trial in 5 of each condition of each block, rounded up, drawn with seed 20261006 from `data/trials/main.jsonl` after its last trial (protocol section 8). From `data/second_scoring/main/agreement.json`, written by `scripts/second_scoring.py compare`. The first score, given at the rig, is the one the analysis uses whatever the second says.

- **Agreement:** the two scores agree in 27 of the 28 trials the second scorer could tell: 96 % (Wilson 95 % interval 82 % to 99 %). They differ in 1, and in 1 the second scorer answered "cannot tell".
- **Who scored:** the first score was given at the rig by op1, the second from the video by sc2 (role codes, from the trial records and from the scores).

First score against second score, in trials:

| First score (at the rig) | Second: success | Second: failure | Second: cannot tell |
|---|---|---|---|
| success | 17 | 0 | 1 |
| failure | 1 | 10 | 0 |

By kind of trial:

| Kind of trial | Scored again | Agree | Differ | Cannot tell | Wilson 95 % interval of the share that agree |
|---|---|---|---|---|---|
| policy | 24 | 23 of 24 | 1 | 0 | 80 % to 99 % |
| replay | 5 | 4 of 4 | 0 | 1 | 51 % to 100 % |

- T1a-017: failure at the rig, success from the video.
- T1b-001: success at the rig; the second scorer could not tell.

## The scene at each trial's start

The basket stands on tape and the arm can carry it off; the same catch can move the wrist camera in its mount. So the first and the last frames of every trial's two videos were read for the basket's place and for the wrist camera's, each against its reference picture, and a trial that began with the basket more than 2 mm or the wrist camera more than 2 px from its place is named here (protocol section 8). From `data/audit/scene_main.json`, written by `scripts/rig/scene_audit.py`; a trial's kind and outcome are those of its record. An offset is the larger of across and down, as the audit reads it: millimetres on the floor for the basket, pixels of the wrist camera's picture for that camera.

- **Trials read:** 132 of the 132 scored trials are in the audit. The basket's place at the start was read in 131 of them and the wrist camera's in 131. **Videos whose first frames did not say:** the front camera's of T1a-010; the wrist camera's of T1a-011.
- **The basket was moved in 1 trial(s)** (it stood more than 5 mm from where the trial began, at the trial's last frames or, where the arm hid it then, when the next trial of the session began: the audit's rule): T1a-007 (the clustered policy, success, 31.3 mm). A trial in which the arm moves the basket stands as scored (protocol section 8). The end of 2 trial(s) says nothing of the basket (its last frames did not show the basket, and the audit found no later trial of the session to judge it by): T1b-006, T1a-010.
- **Named for the basket:** 22 trial(s) began with it more than 2 mm from its place: T1a-005, T1b-005, T1a-012, T1b-008, T1a-020, T1b-013, T1a-028, T1a-032, T1a-035, T1b-024, T2-011, T2-017, T2-023, T2-029, T2-035, T2-041, T2-047, T2-053, T3-002, T3-008, T3-020, T3-026.
- **Named for the wrist camera:** 1 trial(s) began with it more than 2 px from its place: T1b-001.
- **Largest offsets at a start:** the basket 2.4 mm, the wrist camera 2.4 px.
- **Successes, as plain counts and with no test:** 11 of the 23 trials named for the basket or for the wrist camera succeeded, and 57 of the 107 trials named for neither succeeded. Besides, 1 of the 2 trials whose start could not be read succeeded.

| The basket at a trial's start, from its place | Trials |
|---|---|
| up to 1 mm | 52 |
| over 1 mm, up to 2 mm | 57 |
| over 2 mm, up to 3 mm | 22 |
| over 3 mm | 0 |
| all trials whose start was read | 131 |

One pixel of the front camera is about 1.08 mm at the basket (`MM_PER_PX` in `src/tolerance_radius/rig/basket_place.py`: the two floor tags are 526.9 mm and 486 px apart), so the 2 mm line lies 1.9 pixels from the basket's place, and an offset of 2 whole pixels, about 2.2 mm, is the first pixel step past it. The trials named for the basket began 2.1 mm to 2.4 mm from it: at most 0.4 pixels past the line.

## How the trials were run

- **Resets:** over the 132 scored trials the accepted placement lay a median of 1.05 mm from its target, at most 1.55 mm (placed to target, from the records).
- **Pace:** a median of 2.08 minutes from one record's start to the next one's in the same log (144 intervals; void trials included, no pause left out).
- **Days:** 2026-10-07, 2026-10-08. **People** (operator and scorer codes): op1.
- **Execution settings of the clustered policy** (the same in every scored trial, and every setting of the protocol's `[execution]` section with the protocol's value): `clamp_deg` = 15.0, `control_hz` = 30, `handoff_tolerance_deg` = 8.0, `max_handoff_retries` = 3, `min_attempt_s` = 1.0, `replay_handoff_tolerance_deg` = 5.0, `stall_fraction` = 0.9, `stall_progress_deg` = 1.5, `stall_steps` = 30, `temporal_ensemble_coeff` = 0.01, `time_limit_s` = 45.0, `view_crept_limit_mm` = 1.0. Belonging to the single trial and not held to anything: `device` mps; `handoff_tolerance_used_deg` 8.0.
- **Execution settings of the replay** (the same in every scored trial, and every setting of the protocol's `[execution]` section with the protocol's value): `clamp_deg` = 15.0, `control_hz` = 30, `handoff_tolerance_deg` = 8.0, `max_handoff_retries` = 3, `min_attempt_s` = 1.0, `replay_handoff_tolerance_deg` = 5.0, `stall_fraction` = 0.9, `stall_progress_deg` = 1.5, `stall_steps` = 30, `temporal_ensemble_coeff` = 0.01, `time_limit_s` = 45.0, `view_crept_limit_mm` = 1.0. Belonging to the single trial and not held to anything: `device` mps; `handoff_tolerance_used_deg` 5.0.
- **Execution settings of the spread policy** (the same in every scored trial, and every setting of the protocol's `[execution]` section with the protocol's value): `clamp_deg` = 15.0, `control_hz` = 30, `handoff_tolerance_deg` = 8.0, `max_handoff_retries` = 3, `min_attempt_s` = 1.0, `replay_handoff_tolerance_deg` = 5.0, `stall_fraction` = 0.9, `stall_progress_deg` = 1.5, `stall_steps` = 30, `temporal_ensemble_coeff` = 0.01, `time_limit_s` = 45.0, `view_crept_limit_mm` = 1.0. Belonging to the single trial and not held to anything: `device` mps; `handoff_tolerance_used_deg` 8.0.
- **Control rate**, asked and achieved, from the records (`control_hz`, `steps_sent`, `loop_s`). Achieved is the rate between the first step sent and the last: the steps sent after the first, over the time from the first step to the last (the record's `loop_s`, taken on the trial runner's monotonic clock and kept to a millisecond). The handoff gate and the policy's first inference come before the first step and are outside it, and so is everything after the last step. The loop sends its second step as soon as it is predicted, without waiting out a period, so a trial's figure can read high by up to one interval of its attempt: a third of a percent for 300 steps.
  - policy trials: asked 30 Hz; achieved a median of 29.7 Hz (29.6 to 29.8 Hz over 109 trials); 1 scored trial(s) have no rate (fewer than two steps were sent, or the record carries no `loop_s`).
  - replay trials: asked 30 Hz; achieved a median of 29.7 Hz (29.6 to 29.8 Hz over 22 trials).
  - all scored trials: asked 30 Hz; achieved a median of 29.7 Hz (29.6 to 29.8 Hz over 131 trials); 1 scored trial(s) have no rate (fewer than two steps were sent, or the record carries no `loop_s`).
- **The camera's creep before an attempt:** 132 of the 132 scored trials record how far the front camera's picture stood from where its session solved the conversion, read just before the attempt (`execution.view_crept_mm`, from the records): a median of 0.52 mm, at most 0.79 mm. These are as the sessions read them, not divided by the scale factor. None lies beyond the protocol's limit of 1 mm (`execution.view_crept_limit_mm`), beyond which a session makes no attempt.
- **Start pose of the arm** when the attempt began, over the 132 scored trials that record it, in the records' own units: `elbow_flex.pos` 94.5 to 97.5 (median 95.9); `gripper.pos` 0.5 to 3.5 (median 1.8); `shoulder_lift.pos` -101.5 to -98.5 (median -100.2); `shoulder_pan.pos` -4.5 to -1.5 (median -3.0); `wrist_flex.pos` 73.5 to 76.5 (median 75.0); `wrist_roll.pos` -1.5 to 1.5 (median -0.1).
- **Checkpoints** (from the records): the clustered policy `act-tolrad-clustered-v1@080000`; the spread policy `act-tolrad-spread-v1@080000`. Training step: 80,000; every scored policy trial names the same one.
- **`versions.lerobot`:** `lerobot 0.6.1 at 6adf515116`.
- **`versions.lerobot101_commit`:** `02b3ced82f`.
- **`versions.project_commit`:** `0123abc456`.
- **`versions.protocol_status`:** `frozen`.
- **`versions.protocol_version`:** `1.0`.
- **Deviations:** the decision log marks none.

## What the paper quotes of the simulations of the design (simulated, not measured)

Each threshold block as written. A cell is the error of the estimated radius (RMSE) / the mean width of its 95 % interval, in millimetres, for the middle one of the simulated truths with that width (the simulation report's Table 1). The last column is the share of simulated blocks whose interval is open below, over the truths whose curve is already falling at 0 mm (radius under two widths).

| Block | Trials | Cliff (width 1 mm) | Middling (width 3 mm) | Slope (width 10 mm) | Open below when falling at 0 mm |
|---|---|---|---|---|---|
| T1b (the replay) | 22 | 1.35 / 8.02 | 2.15 / 11.94 | 4.95 / 25.66 | 33 % |
| T1a (the clustered policy) | 34 | 1.05 / 6.54 | 1.82 / 9.82 | 4.65 / 24.18 | 33 % |
| T3 (the spread policy) | 24 | 1.25 / 7.53 | 2.05 / 11.45 | 4.85 / 25.16 | 33 % |

Over the blocks together, the interval is open below in 33 % of such simulated blocks.

Fixed before any data: block T1a as written, against a curve with radius 20 mm, width 3 mm and lapse 0; 300 simulated blocks (`analysis/simulation/procedure.csv`, `analysis/simulation/paired_power.csv`, `analysis/simulation/procedure.md`, `analysis/simulation/comparison_rule.csv`, `analysis/simulation/comparison_rule.md`, made from protocol file `647d0a7f49ef`).

- As written: bias -0.2 mm, root-mean-square error 1.63 mm, mean interval width 7.80 mm, the interval held the truth in 97 % of blocks.
- The same trials at fixed, equally spaced distances: error 2.32 mm, interval width 9.01 mm.
- Paired test, 26 pairs, success rates 0.4 against 0.9: the exact McNemar test tells them apart 88 % of the time (computed exactly).

The comparison rule of prediction 2, simulated on the clustered policy's block and the replay's as written (`analysis/simulation/comparison_rule.csv`, made from protocol file `647d0a7f49ef`): the share of 300 simulated pairs of blocks with each verdict, both curves of one width, the replay's radius 15 mm.

| Width of both curves | True difference (policy minus replay) | larger | not told apart | smaller |
|---|---|---|---|---|
| 1 mm | 0 mm | 3 % | 95 % | 2 % |
| 1 mm | 3 mm | 60 % | 40 % | 0 % |
| 1 mm | 5 mm | 98 % | 2 % | 0 % |
| 1 mm | 8 mm | 100 % | 0 % | 0 % |
| 1 mm | 10 mm | 100 % | 0 % | 0 % |
| 1 mm | 15 mm | 100 % | 0 % | 0 % |
| 3 mm | 0 mm | 3 % | 95 % | 2 % |
| 3 mm | 3 mm | 22 % | 78 % | 0 % |
| 3 mm | 5 mm | 35 % | 65 % | 0 % |
| 3 mm | 8 mm | 54 % | 46 % | 0 % |
| 3 mm | 10 mm | 66 % | 34 % | 0 % |
| 3 mm | 15 mm | 98 % | 2 % | 0 % |
| 6 mm | 0 mm | 3 % | 95 % | 2 % |
| 6 mm | 3 mm | 13 % | 88 % | 0 % |
| 6 mm | 5 mm | 19 % | 81 % | 0 % |
| 6 mm | 8 mm | 28 % | 72 % | 0 % |
| 6 mm | 10 mm | 35 % | 65 % | 0 % |
| 6 mm | 15 mm | 51 % | 50 % | 0 % |

## What the analysis could not compute

- **The largest disagreement of a session's camera solve with the stored one:** the tools print it and no committed record holds it.
- **Torn frames and mispaced episodes of the new datasets:** the dataset checks print their findings and no committed record holds them.
- **The number of practice episodes:** no record says which episodes were practice; the placement logs only show lines that were later recorded over.

## The paper's number macros

213 of 217 have a value; the others print `??` in the paper. A value is written as the paper prints it; the source is the comment beside the macro in `paper/generated/numbers.tex`.

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
| `\simPairedPower` | `88` | analysis/simulation/paired_power.csv, the case fixed before the data (computed exactly, not measured) |
| `\simWidthCliff` | `1.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simWidthMiddling` | `3.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simWidthSlope` | `10.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseReplayCliff` | `1.4` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthReplayCliff` | `8.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseReplayMiddling` | `2.2` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthReplayMiddling` | `11.9` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseReplaySlope` | `5.0` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthReplaySlope` | `25.7` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simOpenBelowReplay` | `33` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simRmseClusteredCliff` | `1.1` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthClusteredCliff` | `6.5` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseClusteredMiddling` | `1.8` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthClusteredMiddling` | `9.8` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseClusteredSlope` | `4.7` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthClusteredSlope` | `24.2` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simOpenBelowClustered` | `33` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simRmseSpreadCliff` | `1.3` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthSpreadCliff` | `7.5` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseSpreadMiddling` | `2.1` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthSpreadMiddling` | `11.4` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simRmseSpreadSlope` | `4.9` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simIntervalWidthSpreadSlope` | `25.2` | analysis/simulation/procedure.csv: the block as written, the middle one of the truths with that width (simulated, not measured) |
| `\simOpenBelowSpread` | `33` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simOpenBelow` | `33` | analysis/simulation/procedure.csv: the block as written, mean over the truths with a radius under two widths (simulated, not measured) |
| `\simRuleReplayRadius` | `15.0` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRulePairs` | `300` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffEqual` | `3` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffThree` | `60` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffFive` | `98` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffEight` | `100` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffTen` | `100` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthOneDiffFifteen` | `100` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleSmallerWidthOneDiffEqual` | `2` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffEqual` | `3` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffThree` | `22` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffFive` | `35` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffEight` | `54` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffTen` | `66` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthThreeDiffFifteen` | `98` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleSmallerWidthThreeDiffEqual` | `2` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffEqual` | `3` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffThree` | `13` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffFive` | `19` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffEight` | `28` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffTen` | `35` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleLargerWidthSixDiffFifteen` | `51` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\simRuleSmallerWidthSixDiffEqual` | `2` | analysis/simulation/comparison_rule.csv: pairs of the clustered policy's block and the replay's as written (simulated, not measured) |
| `\nScaleRuns` | `4` | data/instrument/ruler_*.csv, as the camera reads |
| `\nScaleSteps` | `20` | data/instrument/ruler_*.csv, as the camera reads |
| `\instrumentMeanAbsError` | `0.3` | data/instrument/ruler_*.csv, as the camera reads |
| `\instrumentMaxStepError` | `0.3` | data/instrument/ruler_*.csv, as the camera reads |
| `\scaleFactor` | `1.030` | data/instrument/ruler_*.csv, as the camera reads: mean of the runs |
| `\scaleFactorMin` | `1.027` | data/instrument/ruler_*.csv, as the camera reads: smallest of the runs |
| `\scaleFactorMax` | `1.033` | data/instrument/ruler_*.csv, as the camera reads: largest of the runs |
| `\nRepeatability` | `10` | data/instrument/repeat_*.csv |
| `\repeatabilityScatter` | `0.2` | data/instrument/repeat_*.csv, as the camera reads |
| `\nByEye` | `20` | data/instrument/byeye_*.csv |
| `\nOperatorsByEye` | `1` | data/instrument/byeye_*.csv: the labels |
| `\byEyeErrorMedian` | `4.5` | data/instrument/byeye_*.csv |
| `\byEyeErrorNinetyFive` | `11.4` | data/instrument/byeye_*.csv: 95th percentile, interpolated between ranks |
| `\byEyeErrorMax` | `11.5` | data/instrument/byeye_*.csv |
| `\nDemosTrainedClustered` | `55` | data/audit/clustered.json |
| `\nDemosTrainedSpread` | `55` | data/audit/spread.json |
| `\nRerecordedClustered` | `7` | data/recording/clustered_placements.jsonl: kept episodes with at least one earlier take |
| `\nRerecordedSpread` | `7` | data/recording/spread_placements.jsonl: kept episodes with at least one earlier take |
| `\demoPlacementErrorMedian` | `0.9` | data/audit/clustered.json and data/audit/spread.json: measured start to planned target |
| `\demoPlacementErrorMax` | `1.6` | data/audit/clustered.json and data/audit/spread.json: measured start to planned target |
| `\clusteredScatterMax` | `1.5` | data/audit/clustered.json: trained starts about the mean of their point |
| `\spreadSpacingMin` | `16.6` | data/audit/spread.json: trained starts |
| `\nRecordingDays` | `1` | data/recording/*_placements.jsonl: dates of the kept episodes |
| `\nTrialsReplay` | `22` | the counted trial logs, replay, scored, every block |
| `\nVoidReplay` | `2` | the counted trial logs, replay, void, every block |
| `\successZeroReplay` | `4` | the counted trial logs, replay, block T1b, anchors at d = 0 |
| `\nZeroReplay` | `4` | the counted trial logs, replay, block T1b, anchors at d = 0 |
| `\radiusReplay` | `12.0` | block fit of replay: posterior median of r50 |
| `\radiusReplayLow` | `9.0` | block fit of replay: lower end of the interval |
| `\radiusReplayHigh` | `14.5` | block fit of replay: upper end of the interval |
| `\maxSuccessDistReplay` | `13.3` | the counted trial logs, replay, scored, block T1b alone |
| `\minFailureDistReplay` | `10.1` | the counted trial logs, replay, scored, block T1b alone |
| `\radiusReplayWide` | `12.0` | block fit of replay made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusReplayWideLow` | `9.0` | block fit of replay made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusReplayWideHigh` | `15.0` | block fit of replay made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftReplay` | `0.0` | block fit of replay: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nReplayEpisodes` | `11` | the counted trial logs, replay, scored: distinct replayed episodes |
| `\nTrialsClustered` | `60` | the counted trial logs, act_clustered, scored, every block |
| `\nVoidClustered` | `5` | the counted trial logs, act_clustered, void, every block |
| `\successZeroClustered` | `6` | the counted trial logs, act_clustered, block T1a, anchors at d = 0 |
| `\nZeroClustered` | `6` | the counted trial logs, act_clustered, block T1a, anchors at d = 0 |
| `\radiusClustered` | `21.5` | pooled fit of act_clustered: posterior median of r50 |
| `\radiusClusteredLow` | `18.5` | pooled fit of act_clustered: lower end of the interval |
| `\radiusClusteredHigh` | `24.0` | pooled fit of act_clustered: upper end of the interval |
| `\maxSuccessDistClustered` | `25.2` | the counted trial logs, act_clustered, scored, block T1a alone |
| `\minFailureDistClustered` | `17.7` | the counted trial logs, act_clustered, scored, block T1a alone |
| `\radiusClusteredWide` | `21.5` | pooled fit of act_clustered made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusClusteredWideLow` | `18.5` | pooled fit of act_clustered made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusClusteredWideHigh` | `24.0` | pooled fit of act_clustered made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftClustered` | `0.0` | pooled fit of act_clustered: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nTrialsClusteredThreshold` | `34` | the counted trial logs, block T1a, scored |
| `\nVoidClusteredThreshold` | `3` | the counted trial logs, block T1a, void |
| `\successFarClustered` | `0` | the counted trial logs, block T1a, far anchors |
| `\nFarClustered` | `4` | the counted trial logs, block T1a, far anchors |
| `\radiusClusteredThreshold` | `21.0` | block fit of act_clustered: posterior median of r50 |
| `\radiusClusteredThresholdLow` | `17.0` | block fit of act_clustered: lower end of the interval |
| `\radiusClusteredThresholdHigh` | `26.0` | block fit of act_clustered: upper end of the interval |
| `\radiusClusteredThresholdWide` | `21.5` | block fit of act_clustered made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusClusteredThresholdWideLow` | `17.0` | block fit of act_clustered made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusClusteredThresholdWideHigh` | `26.5` | block fit of act_clustered made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftClusteredThreshold` | `0.5` | block fit of act_clustered: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nTrialsSpread` | `50` | the counted trial logs, act_spread, scored, every block |
| `\nVoidSpread` | `6` | the counted trial logs, act_spread, void, every block |
| `\successZeroSpread` | `4` | the counted trial logs, act_spread, block T3, anchors at d = 0 |
| `\nZeroSpread` | `4` | the counted trial logs, act_spread, block T3, anchors at d = 0 |
| `\radiusSpread` | `18.0` | pooled fit of act_spread: posterior median of r50 |
| `\radiusSpreadLow` | `13.5` | pooled fit of act_spread: lower end of the interval |
| `\radiusSpreadHigh` | `25.0` | pooled fit of act_spread: upper end of the interval |
| `\maxSuccessDistSpread` | `25.0` | the counted trial logs, act_spread, scored, block T3 alone |
| `\minFailureDistSpread` | `8.6` | the counted trial logs, act_spread, scored, block T3 alone |
| `\radiusSpreadWide` | `19.0` | pooled fit of act_spread made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusSpreadWideLow` | `13.5` | pooled fit of act_spread made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusSpreadWideHigh` | `26.5` | pooled fit of act_spread made again with the protocol's longer list of lapse values: upper end of the interval |
| `\plateauShiftSpread` | `1.0` | pooled fit of act_spread: its posterior median with the protocol's longer list of lapse values, minus its posterior median as reported |
| `\nTrialsSpreadThreshold` | `24` | the counted trial logs, block T3, scored |
| `\nVoidSpreadThreshold` | `3` | the counted trial logs, block T3, void |
| `\radiusSpreadThreshold` | `20.0` | block fit of act_spread: posterior median of r50 |
| `\radiusSpreadThresholdLow` | `11.0` | block fit of act_spread: lower end of the interval |
| `\radiusSpreadThresholdHigh` | `31.0` | block fit of act_spread: upper end of the interval |
| `\radiusSpreadThresholdWide` | `21.0` | block fit of act_spread made again with the protocol's longer list of lapse values: posterior median of r50 |
| `\radiusSpreadThresholdWideLow` | `12.0` | block fit of act_spread made again with the protocol's longer list of lapse values: lower end of the interval |
| `\radiusSpreadThresholdWideHigh` | `51.0` | block fit of act_spread made again with the protocol's longer list of lapse values: upper end of the interval |
| `\nPairs` | `26` | the counted trial logs, block T2: placements with a scored trial of both policies |
| `\nTrialsWorkspace` | `52` | the counted trial logs, block T2, scored |
| `\nVoidWorkspace` | `5` | the counted trial logs, block T2, void |
| `\pairsBoth` | `8` | the paired table of block T2 |
| `\pairsNeither` | `9` | the paired table of block T2 |
| `\pairsOnlySpread` | `8` | the paired table of block T2 |
| `\pairsOnlyClustered` | `1` | the paired table of block T2 |
| `\pairsDiscordant` | `9` | the paired table of block T2 |
| `\mcnemarP` | `0.04` | exact McNemar test on the discordant pairs of block T2, all pairs together |
| `\workspaceSuccessClustered` | `9` | the pairs of block T2, act_clustered |
| `\workspaceClusteredLow` | `19` | the pairs of block T2, act_clustered: Wilson interval |
| `\workspaceClusteredHigh` | `54` | the pairs of block T2, act_clustered: Wilson interval |
| `\workspaceDistClusteredMedian` | `26.1` | the pairs of block T2, act_clustered: measured d |
| `\workspaceDistClusteredMax` | `49.5` | the pairs of block T2, act_clustered: measured d |
| `\lookupPredictedClustered` | `9.6` | block fit of act_clustered at the distances of the pairs of block T2, act_clustered |
| `\lookupClusteredLow` | `6` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_clustered |
| `\lookupClusteredHigh` | `15` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_clustered |
| `\workspaceSuccessSpread` | `16` | the pairs of block T2, act_spread |
| `\workspaceSpreadLow` | `43` | the pairs of block T2, act_spread: Wilson interval |
| `\workspaceSpreadHigh` | `78` | the pairs of block T2, act_spread: Wilson interval |
| `\workspaceDistSpreadMedian` | `11.3` | the pairs of block T2, act_spread: measured d |
| `\workspaceDistSpreadMax` | `36.7` | the pairs of block T2, act_spread: measured d |
| `\lookupPredictedSpread` | `20.4` | block fit of act_clustered at the distances of the pairs of block T2, act_spread |
| `\lookupSpreadLow` | `15` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_spread |
| `\lookupSpreadHigh` | `24` | predicted count's interval, block fit of act_clustered, the pairs of block T2, act_spread |
| `\nPairsTrained` | `13` | the pairs of block T2 whose measured position lies in trained cells |
| `\workspaceSuccessClusteredTrained` | `9` | the pairs of block T2 whose measured position lies in trained cells, act_clustered |
| `\lookupPredictedClusteredTrained` | `8.0` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in trained cells, act_clustered |
| `\workspaceSuccessSpreadTrained` | `11` | the pairs of block T2 whose measured position lies in trained cells, act_spread |
| `\lookupPredictedSpreadTrained` | `12.1` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in trained cells, act_spread |
| `\nPairsHeldOut` | `13` | the pairs of block T2 whose measured position lies in held-out cells |
| `\workspaceSuccessClusteredHeldOut` | `0` | the pairs of block T2 whose measured position lies in held-out cells, act_clustered |
| `\lookupPredictedClusteredHeldOut` | `1.6` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in held-out cells, act_clustered |
| `\workspaceSuccessSpreadHeldOut` | `5` | the pairs of block T2 whose measured position lies in held-out cells, act_spread |
| `\lookupPredictedSpreadHeldOut` | `8.3` | block fit of act_clustered at the distances of the pairs of block T2 whose measured position lies in held-out cells, act_spread |
| `\radiusDiffClusteredReplay` | `9.0` | block fits, act_clustered minus replay, taken as independent: median of the difference |
| `\radiusDiffClusteredReplayLow` | `4.0` | block fits, act_clustered minus replay, taken as independent: lower end of the interval |
| `\radiusDiffClusteredReplayHigh` | `14.5` | block fits, act_clustered minus replay, taken as independent: upper end of the interval |
| `\probClusteredExceedsReplay` | `>0.99` | block fits of act_clustered and replay, taken as independent; a tie on the grid counts half |
| `\radiusDiffSpreadClustered` | `\ensuremath{-}1.5` | block fits, act_spread minus act_clustered, taken as independent: median of the difference |
| `\radiusDiffSpreadClusteredLow` | `\ensuremath{-}11.0` | block fits, act_spread minus act_clustered, taken as independent: lower end of the interval |
| `\radiusDiffSpreadClusteredHigh` | `10.5` | block fits, act_spread minus act_clustered, taken as independent: upper end of the interval |
| `\probSpreadExceedsClustered` | `0.38` | block fits of act_spread and act_clustered, taken as independent; a tie on the grid counts half |
| `\resetSensitivityPlaced` | `95` | block fit of act_clustered at the placement errors of the scored trials (a calculation) |
| `\resetSensitivityByEye` | `94` | block fit of act_clustered at the by-eye errors of data/instrument/byeye_*.csv (a calculation) |
| `\nTrialsBasketNamed` | `22` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials whose first frames show the basket beyond the audit's limit from its place |
| `\nTrialsWristNamed` | `1` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials whose first frames show the wrist camera beyond the audit's limit from its place |
| `\nTrialsBasketMoved` | `1` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials that the audit marks as having moved the basket |
| `\basketStartMax` | `2.4` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials: the basket at the first frames against its reference, the larger of across and down |
| `\wristStartMax` | `2.4` | data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py from the trials' videos; scored trials: the fixed jaw in the wrist camera's first frames against its reference, the larger of across and down |
| `\nSecondScored` | `29` | data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare: the trials scored again |
| `\nSecondScoredAgree` | `27` | data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare: the trials whose second score is the first score |
| `\nSecondScoredCannotTell` | `1` | data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare: the trials the second scorer could not tell |
| `\nTrialsTotal` | `132` | the counted trial logs, scored |
| `\nVoidTotal` | `13` | the counted trial logs, void |
| `\voidReasons` | `bumped the table (4), ended by a key less than 1 s after it began (4), the handoff gate refused the start pose (3), camera dropped 50\% of frames\_in \#2 (2)` | the counted trial logs: void_reason, as typed |
| `\resetErrorMedian` | `1.1` | the counted trial logs, scored: placed to target |
| `\resetErrorMax` | `1.6` | the counted trial logs, scored: placed to target |
| `\viewCreptMedian` | `0.52` | the counted trial logs, scored: execution.view_crept_mm, as the session read it before the attempt |
| `\viewCreptMax` | `0.79` | the counted trial logs, scored: execution.view_crept_mm, as the session read it before the attempt |
| `\trialMinutesMedian` | `2.1` | the counted trial logs: one record's ts_start to the next one's in the same log |
| `\nTrialDays` | `2` | the counted trial logs: dates of ts_start |
| `\nOperators` | `1` | the counted trial logs: operator and scorer codes |
| `\nDeviations` | `0` | docs/decisions.md: rows marked DEVIATION |
| `\evalCheckpoint` | `80,000` | the counted trial logs: checkpoint of the policy trials |
| `\execClamp` | `15` | the counted trial logs: execution.clamp_deg |
| `\execEnsembleCoeff` | `0.01` | the counted trial logs: execution.temporal_ensemble_coeff of the policy trials |
| `\execControlRate` | `30` | the counted trial logs: execution.control_hz |
| `\execControlRateAchieved` | `29.7` | the counted trial logs, scored policy trials: the steps sent after the first over execution.loop_s, the time from the first step sent to the last; median over the trials |
| `\execTimeLimit` | `45` | the counted trial logs: execution.time_limit_s |
| `\lerobotCommit` | `6adf515116` | the counted trial logs: versions.lerobot |
| `\gridDriftMax` | ?? | cannot be produced: the tools print the disagreement of a session's camera solve and no committed record holds it |
| `\nPracticeEpisodes` | ?? | cannot be produced: no record says which episodes were practice |
| `\nTornFramesNew` | ?? | cannot be produced: the torn-frame scan prints its findings and no committed record holds them |
| `\nMispacedEpisodesNew` | ?? | cannot be produced: the pacing check prints its findings and no committed record holds them |
