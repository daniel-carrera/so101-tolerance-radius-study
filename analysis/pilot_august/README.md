# Pilot: where the cube started in the August trials

A re-analysis of the 31 evaluation trials of the August policy (act-cube-v1, on the earlier SO-101 project). The cube's
start position was not logged in August. Here it is measured from each trial's video, and each trial gets its distance
`d` to the nearest demonstration. The run recorded here is that of **2026-10-05, 16:25 PDT**. Nothing at the rig was
touched: the script only reads files.

Every number below says where it comes from: *measured* (by `recover_positions.py` in that run), *read* (from a file,
which is named) or *inferred* (an estimate or a judgement, not a measurement).

## The answer

- **All 31 start positions were recovered** *(measured)*: 30 from the first frame of the trial's video, and 1 (trial 1,
  which was not filmed) from the still its rollout saved 14 s before the attempt began.
- **The instrument check passed** *(measured)*. The nine "ghost" trials sit 0.39 to 2.67 px (0.80 to 3.03 mm once
  converted) from the training episode they re-created; the live tolerance was 4 px. All 55 training episodes,
  re-measured from the local dataset, give the positions of `placements.json` exactly.
- **The trials form two tight groups of distance, with nothing between them** *(measured)*:

  | Cube placed in | Trials | Successes | `d` to the nearest trained demonstration |
  |---|---|---|---|
  | a trained cell | 19 | 19 | 0.6 to 5.6 mm |
  | a held-out cell | 12 | 1 | 31.0 to 47.1 mm |

- **Largest `d` with a success: 40.0 mm** (trial 28, held-out cell B2). The operator's note for that trial says the arm
  aimed at the next cell and caught the cube by its edge *(read)*. Largest `d` with a success in a trained cell:
  5.6 mm. **Smallest `d` with a failure: 31.0 mm** (trial 21, held-out cell D2). *(measured)*
- **The 50 % point was not observed.** No trial lies between 5.6 and 31.0 mm. If success only falls with distance, the
  50 % point is somewhere in that stretch, and these data cannot say where. The protocol's model puts it at 26 mm
  (95 % interval 16 to 34 mm), but its lower end rests on trial 28 alone: without that trial it is 19.5 mm (7.5 to
  30.5 mm). *(inferred)*
- A single distance is good to about 2 mm, and to 4 mm at worst *(inferred from the measured noise, below)*. That does
  not touch the conclusion: the two groups are 25 mm apart.

## Run it again

From the repository's top folder, with the rig's interpreter (the `lerobot` conda environment; `RIG_PY` in the
Makefile). It takes a few seconds:

```bash
"$HOME/miniconda3/envs/lerobot/bin/python" analysis/pilot_august/recover_positions.py
```

It rewrites `trials_with_distance.csv` and the figure in this folder and prints the full report that the numbers here
are taken from. The earlier project is looked for beside this repository (`../lerobot-101`), or where
`LEROBOT101_ROOT` points, or at the path given as the first argument.

Three more commands, none of which changes this folder:

```bash
"$HOME/miniconda3/envs/lerobot/bin/python" analysis/pilot_august/recover_positions.py --self-test
"$HOME/miniconda3/envs/lerobot/bin/python" analysis/pilot_august/recover_positions.py --out "$TMPDIR/pilot_august_check" --contact-sheet "$TMPDIR/pilot_august_check/detections.png"
"$HOME/miniconda3/envs/lerobot/bin/python" analysis/pilot_august/recover_positions.py --out "$TMPDIR/pilot_august_check" --figure-distance B
```

- `--self-test` checks the statistics helpers against independent calculations and stops.
- `--contact-sheet` also writes a picture of every detection, enlarged, for a person to look at.
- `--figure-distance B` draws the figure with the cross-check distances (see "Two conversions") under the name
  `success_vs_distance_onemap`, so it never replaces the primary figure.

A few lines of the form `objc[...]: Class AVFFrameReceiver is implemented in both ...` appear on the error stream:
two libraries each carry a copy of the same video code. They are harmless here.

What the run read *(measured by the script; the commit is read from the earlier project's `.git` files)*:

| | |
|---|---|
| Earlier project | commit `02b3ced82f98487c4a5eb0c89b22ae6efb5474e7` |
| `eval/trials/act-cube-v1.jsonl` | SHA-256 begins `fd2fbd319fd7` |
| `eval/placements.json` | `0cef3c1cd535` |
| `eval/table_geometry.json` | `943f6511eba2` |
| `scripts/data/placement.py` | `20e9c1e4ab20` |
| `scripts/camera/tag_check.py` | `9c2c86c3bd71` |
| Interpreter | Python 3.12.13, NumPy 2.2.6, OpenCV 4.13.0; the dataset frames were decoded by LeRobot 0.6.0 |

The script opens no camera and no serial port, writes nothing into the earlier project (not even compiled Python
files), and downloads nothing (the Hugging Face libraries are forced offline).

## What was done

1. **Trial lines and videos.** The 31 lines of `eval/trials/act-cube-v1.jsonl` are numbered 1 to 31 in file order
   ("trial 28" is line 28). Each line is matched to its front video (next section).
2. **The detector.** The cube is found by the August detector itself, `detect_object` of the earlier project's
   `scripts/data/placement.py`, imported and not rewritten. It looks for a saturated orange blob that differs from an
   *empty-scene picture*. The empty-scene picture is the one that built `placements.json`: the per-pixel median of the
   55 training episodes' first frames, computed again from the copy of the session dataset on this machine.
3. **The frame.** Frame 0 of the video: the picture the policy was given before its first action, with the arm at its
   start pose and the cube untouched. The next nine frames are measured too, only to check that nothing moves.
4. **Pixels to table millimetres**, as for the demonstrations: the two AprilTags are detected in the same frame
   (`tag_check.detect`), the map from pixels to the table (a *homography*) is solved from their corners
   (`homography_from_tag_geometry`, a *tag solve*), and `assign_cells` applies it. `assign_cells` would fall back to
   the session's *consensus map* (the one tag solve closest to the median of all of them) for a frame with a hidden
   tag, and label it. That never happened: both tags are visible in all 31 pictures *(measured)*.
5. **Distance.** From the trial's position to each demonstration's position in `placements.json`, in table
   millimetres. Reported twice: to the nearest of the **49 trained** demonstrations, and to the nearest of **all 55**.
   The policy was trained with the last 6 episodes (49 to 54) held out for its evaluation loss *(read:
   `docs/run-journal.md` of the earlier project)*, so episodes 0 to 48 are the trained ones.

### Which video belongs to which trial line, and how sure that is

As sure as the logs themselves.

- 30 of the 31 lines name their own front video *(read)*: the rollout appended the line together with the paths of the
  files it had just closed. Every one of the 30 videos is named by exactly one line *(measured)*.
- Three independent checks agree *(measured)*. The rollout's own log for the same start time has an "attempt scored"
  line with the trial's timestamp, to the second, and the trial's score: 31 of 31. Where the log gives a step count
  (the 19 filmed attempts the operator ended), it equals the video's number of frames: 19 of 19. Each line was written
  after its own session started and before the next one started.
- **Trial 1 was not filmed** *(measured)*: its line names no video and the first recording starts about an hour
  later. Its session was found through the log (session `20260819_121218`, scored at the trial's timestamp, with its
  score). That session's start-of-session still, `logs/rollout_20260819_121218_front.jpg`, was written at 12:12:30
  *(read: the file's time)* and the attempt began at 12:12:44 *(read: the session log)*. The position of trial 1 comes
  from that still, and is marked as such in the table, the CSV and the figure.
- The two earlier lines in `act-cube-v1-preconfig.jsonl` are not part of the 31 and are not analysed: the earlier
  project set them aside because they ran with a different execution setting *(read)*.

### Two conversions

- **A, as in August (the primary one).** Each picture is converted with its own tag solve, exactly as each
  demonstration was. Distances are to the millimetre positions stored in `placements.json`.
- **B, one map (a cross-check).** The pixel positions of trials and demonstrations are all converted with one map: the
  training session's consensus map, the one `placements.json` itself used for the 10 episodes with a hidden tag. This
  is sound because the camera view was the same in both sessions: the mean position of each tag corner differs by
  0.91 px at most between them *(measured)*. B removes the frame-to-frame noise of the tag solve, which A carries
  twice (once in the trial, once in the demonstration). B is also how this project's own instrument converts: one
  stored map *(read: `docs/decisions.md`, 2026-10-05)*.

A is reported first because it is the instrument of record and what was asked for. B is probably the closer of the two
to the true distances *(inferred)*. No conclusion differs between them. The script's report calls them method A and
method B.

## The instrument check

### 1. Ghost trials: the recovered start against the training episode it re-created

In nine trials the operator re-created the placement of training episode 0, 5 or 7 with the live "ghost" overlay,
sliding the cube until the readout showed 4 px or less *(read: `docs/procedures/act-cube-v1-eval.md`)*. The readout
used this same detector. The targets in `eval/ghosts/ghosts.json` are the `placements.json` pixels of those episodes
*(measured: largest difference 0.0000 px)*.

*(All measured. "Error" is the distance between the trial's recovered start and the episode's recorded start.)*

| trial | scene | source | error px | error mm (A, as in August) | error mm (B, one map) | blob turned by (deg) |
|---|---|---|---|---|---|---|
| 1 | GHOST_EP00 | session-start still | 1.21 | 1.49 | 1.54 | 0.0 |
| 4 | GHOST_EP05 | video frame | 2.67 | 3.03 | 3.11 | 4.8 |
| 7 | GHOST_EP07 | video frame | 1.45 | 1.43 | 1.55 | 4.8 |
| 13 | GHOST_EP00 | video frame | 0.39 | 1.22 | 0.44 | 7.1 |
| 16 | GHOST_EP05 | video frame | 1.58 | 2.77 | 1.93 | -3.4 |
| 19 | GHOST_EP07 | video frame | 1.36 | 1.71 | 1.43 | 10.2 |
| 25 | GHOST_EP00 | video frame | 1.32 | 2.54 | 1.67 | 12.1 |
| 26 | GHOST_EP05 | video frame | 1.78 | 2.53 | 2.21 | -10.5 |
| 27 | GHOST_EP07 | video frame | 1.24 | 0.80 | 1.31 | -3.4 |

All nine are within the 4 px the readout allowed: 0.39 to 2.67 px, 0.80 to 3.03 mm with conversion A (mean 1.95 mm),
0.44 to 3.11 mm with B (mean 1.69 mm). As a vector the mean error is (+0.13, +0.59) px, so there is no common shift
between the trial recordings and the training frames.

The last column is a rough sign of how the cube was turned, not a measurement of it: the angle of the smallest
rectangle around the blob, against the same angle in the training episode. It moves in steps of several degrees. It
differs by 0.0 to 12.1 degrees (median 4.8). The ghost overlay matched position and outline, not rotation *(read:
the August procedure says so itself)*.

What this check does and does not show. The operator placed the cube with this same detector, in pixels. Agreement
shows that the recovery from recorded video gives what the live readout showed on the day. It does not test the
millimetre scale: that rests on `eval/table_geometry.json` (tag corners from a grid-line fit with a 0.33 mm mean
residual, cross-checked with a tape to 2 mm; *read*), and nothing in the videos tests it independently.

### 2. The training episodes, measured again

Frame 0 of all 55 training episodes was read again from the session dataset on this machine, through the earlier
project's own loader, and passed through the same functions *(measured)*:

- largest difference from `placements.json`: 0.0000 px and 0.0000 mm;
- same cell 55 of 55; same homography label 55 of 55;
- the training session's consensus map reproduces the 10 "session-consensus" rows to 0.0000 mm.

So the detector, the conversion and the decoded frames give today exactly what they gave on 18 August. The frames
were decoded by the LeRobot now in the rig's environment, 0.6.0 *(measured)*, which is not the build the environment
imported in August (0.6.1, from a source checkout; *read*: `docs/journal.md` of this project). The positions are
identical all the same.

This is a test that the instrument has not changed. It is not a test of its accuracy. Nor is the comparison bound to
give zero: the detector's answer does move when its inputs do, for example by up to 0.18 px when the trial pictures
are measured against the other empty-scene picture (check 4).

### 3. Two pictures of each placement

For the 30 filmed trials there is a second picture of the same placement: the still the rollout saved when its session
opened. The cube's position in the still and in the first video frame is at most 0.30 px apart (median 0.14 px)
*(measured)*. So in no filmed trial was the cube moved between the still and the attempt, and the video's compression
does not shift the detection. This is the ground for taking trial 1's position from its still.

### 4. The rest

*(All measured, except the last line.)*

- The cube is detected in 31 of 31 pictures, with exactly one cube-like blob in each.
- Blob area 1172 to 1344 px; the training episodes' is 1173 to 1392 px *(read: `placements.json`)*.
- The recovered cell equals the logged cell in 31 of 31 trials.
- Over the first 10 frames of a video the position moves by at most 0.34 px (median 0.14 px).
- With the other empty-scene picture (the median of the trials' own first frames) the position changes by at most
  0.18 px.
- Every one of the 31 detections was looked at on the contact sheet (`--contact-sheet`): the mark sits on the cube in
  all of them *(looked at, not measured)*.

### 5. How noisy the conversion is

*(All measured.)*

- **Image scale.** At the cubes' positions one pixel is 1.04 to 1.13 mm across the image and 1.11 to 1.32 mm down it.
  The live readout's 4 px is therefore about 4.2 to 5.3 mm, not the "about 2 mm" the August procedure states *(read)*.
- **Tag solve, trial session.** The 31 solves, applied to the trials' cube pixels, differ from their median by 0.36 mm
  on average, 1.09 mm at the 95th percentile, 1.72 mm at most. Two pictures of the same placement (still and video),
  each with its own solve, differ by up to 1.64 mm while the pixels differ by 0.30 px at most.
- **Tag solve, training session.** The 45 solves of the training episodes, applied to the demonstrations' cube
  pixels, differ from their median by 1.40 mm on average, 2.74 mm at the 95th percentile, 3.77 mm at most. The 45
  positions in `placements.json` that were converted with their own solve carry this.
- **Between the sessions.** The two sessions' consensus maps differ by 1.56 mm on average at the cubes' positions
  (2.16 mm at most). The data cannot tell whether that is a real sub-pixel change of the view (which a per-session
  map corrects) or noise in the two maps.
- **A against B.** `d` to the nearest trained demonstration differs by +0.48 mm on average (B minus A), from -1.27 to
  +3.96 mm. The largest differences are for the trials in E3. Their nearest demonstration, episode 39, is stored
  2.31 mm from where the one map puts it; for the other three held-out cells that figure is 1.41 mm (episode 11),
  0.76 mm (episode 33) and 0.00 mm (episode 24, itself converted with the consensus map).

From these: a single distance is good to about 2 mm, and to 4 mm at worst *(inferred)*.

## Results

### Every trial

*(Measured, except scene, cell, kind and outcome, which are read from the trial log. `x`, `y`: table millimetres,
conversion A. `d`: distance to the nearest demonstration in millimetres. "ep 33 (B1)" is training episode 33, which is
in cell B1.)*

| trial | scene | cell | kind | outcome | x | y | nearest of 49 | d (49 trained) | nearest of 55 | d (all 55) | d (49), one map (B) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GHOST_EP00 | B1 | id | success | 73.1 | 28.5 | ep 0 (B1) | 1.5 | ep 0 (B1) | 1.5 | 1.5 | session-start still |
| 2 | ID_B1 | B1 | id | success | 72.9 | 31.9 | ep 11 (B1) | 2.0 | ep 11 (B1) | 2.0 | 1.3 | video frame |
| 3 | OOD_B2 | B2 | ood | failure (grasp, subgoal 1) | 75.4 | 77.5 | ep 33 (B1) | 38.1 | ep 33 (B1) | 38.1 | 39.7 | video frame |
| 4 | GHOST_EP05 | C2 | id | success | 117.4 | 80.5 | ep 5 (C2) | 3.0 | ep 5 (C2) | 3.0 | 3.1 | video frame |
| 5 | ID_C2 | C2 | id | success | 122.5 | 80.7 | ep 5 (C2) | 2.4 | ep 5 (C2) | 2.4 | 2.0 | video frame |
| 6 | OOD_A1 | A1 | ood | failure (perception, subgoal 1) | 27.2 | 29.8 | ep 11 (B1) | 45.3 | ep 11 (B1) | 45.3 | 44.9 | video frame |
| 7 | GHOST_EP07 | A3 | id | success | 26.2 | 130.0 | ep 7 (A3) | 1.4 | ep 7 (A3) | 1.4 | 1.5 | video frame |
| 8 | ID_E2 | E2 | id | success | 221.2 | 79.5 | ep 6 (E2) | 2.3 | ep 6 (E2) | 2.3 | 1.1 | video frame |
| 9 | OOD_D2 | D2 | ood | failure (perception, subgoal 1) | 171.6 | 78.8 | ep 24 (D1) | 35.4 | ep 24 (D1) | 35.4 | 36.8 | video frame |
| 10 | ID_A3 | A3 | id | success | 24.8 | 127.5 | ep 7 (A3) | 2.1 | ep 7 (A3) | 2.1 | 2.8 | video frame |
| 11 | OOD_E3 | E3 | ood | failure (perception, subgoal 1) | 223.7 | 124.2 | ep 39 (E2) | 33.4 | ep 39 (E2) | 33.4 | 37.3 | video frame |
| 12 | ID_D3 | D3 | id | success | 172.8 | 123.4 | ep 10 (D3) | 5.6 | ep 10 (D3) | 5.6 | 4.4 | video frame |
| 13 | GHOST_EP00 | B1 | id | success | 72.8 | 28.0 | ep 0 (B1) | 1.2 | ep 0 (B1) | 1.2 | 0.4 | video frame |
| 14 | ID_B1 | B1 | id | success | 73.4 | 27.7 | ep 0 (B1) | 0.6 | ep 0 (B1) | 0.6 | 0.1 | video frame |
| 15 | OOD_B2 | B2 | ood | failure (perception, subgoal 1) | 75.4 | 78.0 | ep 33 (B1) | 38.6 | ep 33 (B1) | 38.6 | 40.5 | video frame |
| 16 | GHOST_EP05 | C2 | id | success | 119.9 | 82.4 | ep 16 (C2) | 1.3 | ep 16 (C2) | 1.3 | 1.6 | video frame |
| 17 | ID_C2 | C2 | id | success | 121.4 | 77.1 | ep 5 (C2) | 2.8 | ep 49 (C2) | 1.1 | 3.4 | video frame |
| 18 | OOD_A1 | A1 | ood | failure (perception, subgoal 1) | 25.6 | 27.8 | ep 11 (B1) | 47.1 | ep 11 (B1) | 47.1 | 46.9 | video frame |
| 19 | GHOST_EP07 | A3 | id | success | 26.4 | 130.4 | ep 18 (A3) | 1.4 | ep 18 (A3) | 1.4 | 1.2 | video frame |
| 20 | ID_E2 | E2 | id | success | 221.6 | 81.3 | ep 6 (E2) | 3.0 | ep 6 (E2) | 3.0 | 1.8 | video frame |
| 21 | OOD_D2 | D2 | ood | failure (perception, subgoal 1) | 175.4 | 74.5 | ep 24 (D1) | 31.0 | ep 24 (D1) | 31.0 | 32.5 | video frame |
| 22 | ID_A3 | A3 | id | success | 25.9 | 128.3 | ep 7 (A3) | 1.6 | ep 7 (A3) | 1.6 | 2.2 | video frame |
| 23 | OOD_E3 | E3 | ood | failure (perception, subgoal 1) | 219.6 | 128.0 | ep 39 (E2) | 36.4 | ep 39 (E2) | 36.4 | 39.7 | video frame |
| 24 | ID_D3 | D3 | id | success | 176.5 | 126.6 | ep 21 (D3) | 5.6 | ep 21 (D3) | 5.6 | 4.9 | video frame |
| 25 | GHOST_EP00 | B1 | id | success | 73.6 | 29.7 | ep 0 (B1) | 2.5 | ep 0 (B1) | 2.5 | 1.7 | video frame |
| 26 | GHOST_EP05 | C2 | id | success | 118.4 | 81.2 | ep 5 (C2) | 2.5 | ep 5 (C2) | 2.5 | 2.2 | video frame |
| 27 | GHOST_EP07 | A3 | id | success | 24.5 | 128.9 | ep 7 (A3) | 0.8 | ep 7 (A3) | 0.8 | 1.3 | video frame |
| 28 | OOD_B2 | B2 | ood | success | 72.6 | 79.4 | ep 33 (B1) | 40.0 | ep 33 (B1) | 40.0 | 41.5 | video frame |
| 29 | OOD_A1 | A1 | ood | failure (perception, subgoal 1) | 28.3 | 30.0 | ep 11 (B1) | 44.2 | ep 11 (B1) | 44.2 | 44.1 | video frame |
| 30 | OOD_D2 | D2 | ood | failure (perception, subgoal 1) | 173.8 | 77.8 | ep 24 (D1) | 34.3 | ep 24 (D1) | 34.3 | 36.2 | video frame |
| 31 | OOD_E3 | E3 | ood | failure (perception, subgoal 1) | 223.3 | 124.6 | ep 39 (E2) | 33.6 | ep 39 (E2) | 33.6 | 37.3 | video frame |

### By kind of trial

*(Measured.)*

| kind | trials | successes | d, method A | d, method B |
|---|---|---|---|---|
| trained cell, a demonstration's placement re-created | 9 | 9 | 0.80 to 3.03 | 0.44 to 3.11 |
| trained cell, placed freely in the cell | 10 | 10 | 0.63 to 5.63 | 0.11 to 4.89 |
| held-out cell, interior (B2, D2) | 6 | 1 | 31.01 to 40.01 | 32.46 to 41.50 |
| held-out cell, corner (A1, E3) | 6 | 0 | 33.37 to 47.14 | 37.33 to 46.89 |

- In the trained cells the cube was "placed freely in the cell" *(read: the August procedure)*, but in practice every
  such placement landed within 5.6 mm of a demonstration: 17 of the 19 trained-cell trials are within 3.0 mm, and the
  two at 5.6 mm are both in D3 *(measured)*. The August trials did not test the inside of a trained cell away from its
  demonstrations.
- For comparison, among the 49 trained demonstrations the nearest other one is 2.6 mm away (median; 0.4 to 8.2 mm),
  and the trained demonstrations of one cell lie within 4.0 to 16.0 mm of each other (largest pair per cell; 3.3 to
  13.7 mm with conversion B) *(measured from `placements.json`)*.
- Trained demonstrations within 10 mm of the cube: 2 to 5 for every trained-cell trial, none for any held-out-cell
  trial *(measured)*.
- For each held-out cell, all three trials have the same nearest trained demonstration: episode 11 (B1) for A1,
  episode 33 (B1) for B2, episode 24 (D1) for D2, episode 39 (E2) for E3 *(measured)*. Each is the demonstration of
  the neighbouring cell that lies farthest toward the held-out one *(read from the positions in `placements.json`)*.

### Success against distance, in 10 mm bins

Bins of 10 mm as the protocol defines them (section 10), with Wilson 95 % intervals. A *Wilson interval* is a range
for a success rate that stays sensible for small counts and for rates of 0 or 1.

Conversion A, nearest of the 49 trained demonstrations (the figure) *(measured)*:

| d (mm) | trials | successes | rate | Wilson 95 % |
|---|---|---|---|---|
| 0 to 10 | 19 | 19 | 100 % | 83.2 to 100.0 % |
| 10 to 20 | 0 | | no trials | |
| 20 to 30 | 0 | | no trials | |
| 30 to 40 | 8 | 0 | 0 % | 0.0 to 32.4 % |
| 40 to 50 | 4 | 1 | 25 % | 4.6 to 69.9 % |

Conversion B, the cross-check *(measured)*:

| d (mm) | trials | successes | rate | Wilson 95 % |
|---|---|---|---|---|
| 0 to 10 | 19 | 19 | 100 % | 83.2 to 100.0 % |
| 10 to 20 | 0 | | no trials | |
| 20 to 30 | 0 | | no trials | |
| 30 to 40 | 7 | 0 | 0 % | 0.0 to 35.4 % |
| 40 to 50 | 5 | 1 | 20 % | 3.6 to 62.4 % |

- Pooled *(measured)*: at or below 5.6 mm, 19 of 19 (Wilson 83.2 to 100 %); at or above 31.0 mm, 1 of 12 (Wilson 1.5
  to 35.4 %). With the 30 filmed trials only: 18 of 18 (82.4 to 100 %) and 1 of 12.
- The split of the far group into 30 to 40 mm and 40 to 50 mm should not be read: trial 28, the one success, sits at
  40.01 mm, a hundredth of a millimetre above a bin edge, and three more trials are within 2 mm of an edge. The
  pooled far group is the robust figure.
- Against the nearest of **all 55** demonstrations the bins and both extremes are the same *(measured)*. Only trial
  17 changes: 1.1 mm (to episode 49, which was held out of training) instead of 2.8 mm.
- These counts are the August journal's "ID 19/19" and "OOD 1/12" *(read: `docs/run-journal.md` of the earlier
  project)*, now with a distance attached. Distance separates the trials exactly as the cell did.

### The extremes

*(Measured.)*

| | Conversion A | Conversion B |
|---|---|---|
| Largest `d` with a success | 40.01 mm (trial 28, B2) | 41.50 mm (trial 28) |
| Largest `d` with a success in a trained cell | 5.63 mm (trial 12, D3) | 4.89 mm (trial 24, D3) |
| Smallest `d` with a failure | 31.01 mm (trial 21, D2) | 32.46 mm (trial 21) |
| Stretch with no trial | 5.63 to 31.01 mm | 4.89 to 32.46 mm |

In all 11 failures the operator's note says the gripper closed beside the cube: on a neighbouring cell, or a few
centimetres off. For trial 28, the success, it says the arm tried to grasp the cell above and the cube happened to be
in the gripper's way, caught just by its edge *(read: the `notes` of the trial log)*.

### Where the 50 % point can be

- **It was not observed.** No trial lies between 5.6 and 31.0 mm *(measured)*.
- **From the counts alone** *(inferred)*: if success only falls as distance grows, the 50 % point lies in that stretch.
  Below it the rate is high (19 of 19, lower Wilson bound 83 %), though only two trials are above 3.0 mm. Above it
  the rate is low (1 of 12, upper Wilson bound 35 %). Where in the 25 mm between, the data do not say.
- **With the protocol's model** *(inferred; `config/protocol.toml` draft 0.1: a logistic curve with a lapse, the
  posterior on the protocol's grid)*: `r50` has median 26.0 mm, 95 % interval 16.0 to 34.0 mm (B: 27.5 mm, 17.0 to
  36.0 mm). The lower end is the work of one trial. The model has no floor, so the success at 40 mm has to come from
  the tail of the curve, which rules out a sharp edge at a short distance. Without trial 28: median 19.5 mm, interval
  7.5 to 30.5 mm (B: 20.0 mm, 7.0 to 32.0 mm), which is little more than the empty stretch itself.
- **A fair statement:** somewhere between about 6 mm and about 31 to 34 mm. Narrowing the lower end to 16 mm means
  trusting one grasp that the operator described as accidental.

## Where the 55 demonstrations started, within their cells

*(measured on 2026-10-05 by `demo_scatter.py`, from the first frame of every episode of the dataset on this machine;
positions in `demo_starts.csv`)*

The dataset was recorded with a grid drawn over the live picture and no measurement: five demonstrations in each of 11
cells, each "at the cell's centre". Two things follow from measuring them.

- **A common offset.** The measured position is on average (-2.5, +8.1) mm from the cell's centre: 8.5 mm. This is the
  detector's known offset, seen for the first time as a number: it reports the centre of the cube's visible outline,
  the operator centred the footprint. It cancels in every distance between two measured positions, and it means the
  cube does not stand where a measured coordinate would suggest (`docs/runbook/01-camera-and-instrument.md`, step 6).
  It is an estimate of the offset only as far as the operator centred the footprint without a bias of their own, which
  nothing here can check.
- **"The same place" was about a centimetre wide.** With one conversion solved from all 55 frames, a demonstration
  lies 3.1 mm from the mean start of its cell on average and 7.9 mm at most, and the two farthest apart in a cell are
  between 4.8 mm (cell A3) and 13.7 mm (cell D1) apart. With August's own conversion, one tag solve per picture, the
  same figures are 3.3 mm, 8.2 mm, and 5.6 to 16.0 mm: the solve's own wander adds a little.

Nobody knew either number in August. The policy's 19 successes in trained cells were at 0.6 to 5.6 mm from the
nearest demonstration because the demonstrations of a cell were themselves spread over a centimetre: a trial placed
"at the centre" had five slightly different neighbours to be near. The experiment that follows places every clustered
demonstration within 2 mm of its target, so its cluster is tighter than August's, and its radius need not be the same.

## What these data cannot show

- **Anything between 5.6 and 31.0 mm.** Where success falls, and whether it falls as a cliff or a slope, was not
  tested. A curve drawn through that stretch would be an assumption, which is why the figure has none.
- **Distance apart from the cell.** Every near trial was in a trained cell and every far trial in a held-out cell.
  The data cannot separate "far from a demonstration" from "in a cell with no demonstrations", from the direction of
  the offset (each held-out cell probes one direction from one neighbour), or from how many demonstrations are close
  by (2 to 5 within 10 mm for the near trials, none for the far ones).
- **A distance measure other than the nearest one.** For the far trials `d` hangs on a single demonstration each. A
  measure that used several neighbours would place them farther out. None was computed.
- **31 independent tests.** The 31 trials are 12 scenes (3 ghost placements, 5 trained cells, 4 held-out cells), each
  run two or three times. The Wilson intervals treat every trial as independent, so they are narrower than the
  evidence warrants.
- **Anything beyond 47.1 mm**, or about another policy, checkpoint, object, lighting or camera mount. One session,
  one checkpoint.
- **The true size of a millimetre.** The detector reports the centre of the cube's visible outline (its top and the
  sides the camera sees), mapped through the table plane, not the centre of its footprint. The same is true of every
  demonstration, so the offset largely cancels in a distance, but distances read a few percent long because that
  point sits above the table: about 3 % by this project's estimate from geometry *(read: `docs/journal.md`; an
  estimate, not a measurement)*. The map from two tags also ignores the lens's distortion *(inferred: no test of it
  here)*. The protocol's check against a printed scale (block V0) measures the net effect.
- **The cube's rotation.** It was not measured. The rough sign of it in the ghost table differs by up to 12 degrees
  between a ghost trial and its training episode *(measured, coarsely)*, and in the pictures the cube of trial 25 looks
  turned against that of training episode 0 *(a judgement by eye)*. It may matter for a grasp, and it moves the
  outline's centre a little.
- **Trial 1's position as surely as the others'.** It rests on a still taken 14 s before the attempt. The 30 filmed
  trials show the cube never moved in that interval; for trial 1 this is assumed *(inferred)*. It is a success at
  1.5 mm and changes no result.

## For the experiment that starts tomorrow

*(All inferred from the August data. The new policy, the new demonstrations and the new camera mount are not
August's, so these are a starting belief, not a prediction.)*

- **Starting levels.** The protocol's draft has the clustered policy's staircases start at 10 mm (going up) and 40 mm
  (going down) *(read: `config/protocol.toml`, block T1a)*. August is consistent with both and gives no reason to
  move them: 10 mm is just beyond the largest distance at which August saw only successes (5.6 mm), and 40 mm is
  inside the group where August saw 1 success in 12 (31.0 to 47.1 mm). A down staircase started at 30 mm would begin
  at the edge of that group and spend one trial fewer in it. August says nothing about the replay block or the spread
  policy.
- **The model has no floor.** One lucky grasp far out moved the lower end of the `r50` interval from 7.5 to 16 mm
  here. If that happens again, the model-free bins and a fit without the outlier are worth reporting beside it; a
  small floor term in the model would be the other way.
- **Tag-solve noise is real, and the new instrument already avoids it.** One frame's solve was up to 1.7 mm (trial
  frames) and 3.8 mm (dataset frames) from its session's median *(measured here)*. That bears out the decision to use
  one stored conversion solved from many frames *(read: `docs/decisions.md`)*. Conversion B is its August
  equivalent.
- **Pixels and millimetres.** 2 mm, the protocol's placement tolerance, is about 1.5 to 1.9 px at the August scale.
  The August readout's "4 px" was 4.2 to 5.3 mm.
- **What "clustered" was in August.** The demonstrations of one cell lay within 4 to 16 mm of each other (3 to 14 mm
  with conversion B), not on one point. The new clustered layout is tighter: five episodes on the cell's centre, each
  set to within 2 mm *(read: `protocol/protocol.md`, section 4)*. So a trial at a given `d` inside a cell is not quite the
  same test as in August.

## Files in this folder

| File | What it is |
|---|---|
| `recover_positions.py` | The whole analysis. Documented at its top; re-runnable with the command above |
| `trials_with_distance.csv` | One row per trial line (31 rows) |
| `success_vs_distance.png`, `.pdf` | The figure (distances of conversion A) |
| `demo_scatter.py` | Where the 55 demonstrations started within their cells; re-runnable with the rig's interpreter |
| `demo_starts.csv` | One row per demonstration (55 rows): its measured start under both conversions, its offset from the cell's centre, its distance from the mean start of its cell |
| `README.md` | This record |

### The figure

Suggested caption: *Outcome of each of the 31 trials of the August policy against the distance `d` from the cube's
start position to the nearest of the 49 trained demonstrations. Each marker is one trial, drawn above the axis's 1
(success) or below its 0 (failure) and spread vertically only so that markers do not hide one another. Black: success
rate in 10 mm bins with Wilson 95 % intervals. No trial fell between 5.6 and 31.0 mm.*

The figure draws no fitted curve, on purpose. The two colours were checked as a pair for colour-blind readers, and
marker shape repeats what colour says.

### Columns of `trials_with_distance.csv`

Empty cells mean "does not apply". Lengths in millimetres are in the table frame of the earlier project
(`eval/table_geometry.json`).

| Column | Meaning | Source |
|---|---|---|
| `trial` | Line number of the trial in `act-cube-v1.jsonl`, from 1 | read |
| `ts`, `scene_id`, `cell_logged`, `kind`, `axis`, `subgoal`, `failure` | As logged in August | read |
| `success` | 1 when `subgoal` is 5, else 0 | read |
| `position_source` | `video frame`, or `session-start still` for the trial that was not filmed | measured |
| `source_file`, `frame_used` | The video or still the position comes from, and the frame's index | measured |
| `mapping_evidence` | What ties the line to that file | measured |
| `px_x`, `px_y`, `area_px`, `n_candidates` | The detected blob: centre and area in pixels, and how many cube-like blobs there were | measured |
| `blob_angle_deg` | Angle of the smallest rectangle around the blob, as the detector reports it: a rough sign of how the cube is turned | measured |
| `homography` | `episode`: converted with the tags of its own picture. `session-consensus`: with the session's map (never needed here) | measured |
| `x_mm`, `y_mm` | Start position, conversion A | measured |
| `cell_recovered`, `cell_matches_logged` | The cell that position falls in, and whether it is the logged one | measured |
| `nearest_trained49_ep`, `nearest_trained49_cell`, `d_nearest_trained49_mm` | Nearest of the 49 trained demonstrations (episodes 0 to 48), its cell, the distance: **`d`** | measured |
| `nearest_all55_ep`, `nearest_all55_cell`, `d_nearest_all55_mm` | The same among all 55 | measured |
| `n_trained49_within_10mm` | Trained demonstrations within 10 mm of the start | measured |
| `x_mm_onemap`, `y_mm_onemap`, `nearest_trained49_ep_onemap`, `d_nearest_trained49_mm_onemap`, `d_nearest_all55_mm_onemap` | The same with conversion B. Its coordinates are not comparable with `x_mm`, `y_mm`; only its distances are meant to be used | measured |
| `max_move_first_frames_px`, `max_move_first_frames_mm` | Largest change of the measured position over the first 10 frames (the millimetre figure with each frame's own tag solve) | measured |
| `still_vs_video_px`, `still_vs_video_mm` | Start-of-session still against the video frame | measured |
| `ghost_episode`, `ghost_error_px`, `ghost_error_mm`, `ghost_error_mm_onemap` | For ghost trials: the episode re-created, and the distance to its recorded start | measured |
| `ghost_turn_deg` | For ghost trials: `blob_angle_deg` minus the episode's, taken within 45 degrees either way | measured |
| `flags` | Anything unusual about the row | measured |

## Where this departs from the brief

- One trial has no video. Its position comes from a still, validated on the other 30, and is marked everywhere.
- The script's default for the earlier project is the folder beside this repository, not an absolute path, so that no
  machine path is written into the repository. On this machine it is the same folder.
- Added, beyond what was asked: conversion B, the protocol's model fitted to the August trials, the rough sign of the
  cube's rotation, and the options `--self-test`, `--contact-sheet` and `--figure-distance`.
- The script was kept within 120 columns and checked for syntax, but not run through `ruff`: it is not installed.

## Terms used here that are not yet in `protocol/glossary.md`

| Term | Meaning |
|---|---|
| **Consensus map** | Of all the tag solves of one session, the one closest to their median; the earlier project's fallback when a tag is hidden |
| **Empty-scene picture** | The scene with no cube, made as the per-pixel median of many frames in which the cube is in different places |
| **Ghost trial** | A trial whose cube was placed on the recorded start of a chosen training episode, guided by that episode's cube drawn over the live view |
| **Homography** | The map from image pixels to points on the table plane |
| **Start-of-session still** | The picture a rollout saves from the front camera when its session opens, before the arm is positioned |
| **Tag solve** | The homography computed from the two AprilTags' corners as seen in one picture |
| **Wilson interval** | A range for a success rate from a count, reliable for small counts and for rates of 0 or 1 |
