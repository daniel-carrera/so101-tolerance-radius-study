# Protocol

> **Status: version 1.0, frozen on 2026-10-08 at 12:48 PDT, before the first counted trial.** `version = "1.0"` and
> `status = "frozen"` in `config/protocol.toml`, the git tag `protocol-v1`, and an entry in `docs/decisions.md`. From
> here on any change is a deviation: logged with its reason, listed in section 13 and reported in the paper.
> Parameters live in `config/protocol.toml`; this document explains them. Where the two disagree, the file wins and
> this document is wrong.
> The items that draft 0.3 marked **[open]** were settled at the freeze, each from a measurement or a decision that
> the text now states; the items that draft 0.2 marked **[to confirm]** were settled on 2026-10-05
> (`docs/decisions.md`).
>
> **Changed since draft 0.3** (section 14 has the reasons, and what changed since the earlier drafts):
>
> - A session makes no attempt once the front camera's picture has crept more than 1.0 mm from where it solved its
>   conversion, and every record carries the reading before its attempt (was open: "re-solved before every
>   placement").
> - Distances are left as the camera reads them: the four runs of the scale check differ by more than the rule's
>   one percentage point, and their factors, 1.0123 to 1.0248, are reported as the instrument's accuracy.
> - The spread layout's outermost starts fit; the basket's size is the vendor's listing; the printed scale was not
>   checked with calipers.
> - The replay's handoff gate stays at 5 degrees; the achieved control rate of the replay pilot, 29.2 Hz, is
>   stated.
> - The pilots were run on Wednesday 2026-10-07 and the counted trials are run on Thursday 2026-10-08, a day later
>   than planned; the stop lines of section 12 are those of that day.
> - The basket stands on tape and has no stop: its place, and the wrist camera's place in its mount, are read from
>   the cameras' pictures before and after each block and, afterwards, for every trial. A trial in which the arm
>   catches the basket stands as scored.

## 1. The question

On a low-cost arm, how far from a demonstration's start state does an imitation policy still succeed, how much of that
distance is the hardware's own doing, and what does the answer imply for two choices that sit beneath the policy and are
rarely reported: **where an evaluation puts the object at reset**, and **how a fixed budget of demonstrations is laid
out**?

The venue and its criteria are in `docs/venue.md`. The policy (ACT) is the fixed probe, not the subject.

## 2. Terms

`protocol/glossary.md` has every term of the project. The ones this document leans on:

- **Start position**: where the front camera measures the cube to be when an episode or a trial begins, in the table
  frame, in millimetres (section 7 says exactly what point of the cube that is).
- **Distance to the nearest demonstration** (`d`): from a trial's measured start position to the nearest measured start
  position among the demonstrations the tested policy was trained on. For a replay: to the start of the one
  demonstration that is played back.
- **Tolerance radius** (`r50`): the distance `d` at which success is halfway between its plateau and its floor. It is a
  property of the whole system (arm, gripper, cameras, data, policy, execution settings), reported for this rig.
- **Plateau, lapse, floor, width**: the four other things the success curve has (section 10). The *plateau* is the
  success rate at a demonstration's own start; the *lapse* is what the plateau lacks to 1; the *floor* is the chance of
  a lucky success far from every demonstration; the *width* is the scale of the fall between them.
- **Replay**: playing back one demonstration's recorded actions, with no policy and no camera in the loop.
- **Clustered** and **spread**: the two demonstration layouts (section 4).
- **Largest gap** of a layout: the farthest any point of the trained cells is from its nearest demonstration.
- **Lookup prediction**: what success would be if it depended on nothing but the distance to the policy's nearest
  demonstration, through one curve: the clustered policy's fitted curve, read at each placement's distance.
- **Anchor**: a trial at a fixed distance. A *zero anchor* is at a demonstration's own start; a *far anchor* is at the
  block's far distance.
- **Pilot** and **counted trial**: a pilot trial tries the procedure and is never analysed; a counted trial is run under
  the frozen protocol and is.
- **Void**: a trial that could not test the policy. It stays in the log and is asked again.

## 3. The rig (the substrate, stated)

One SO-101 follower with its leader (Feetech STS3215 servos), a front camera and a wrist camera, driven through LeRobot
with the safety layer of the earlier project.

- **The light box** is the one the VLA-REPLICA benchmark specifies: a 32 inch (81 cm) photography box, the arm clamped
  at its open side with the base against the front pipe, the front camera on the top rear pipe looking toward the arm.
  The table of the August evaluation was taken down; this is a new scene, and nothing measured on the old one is
  carried over except the arm.
- **The task:** *pick up the orange cube and place it in the basket*. The cube is painted wood, 30 mm on a side; the
  edges of one of its faces are rounded and the others are sharp. **That face is always down**, so the cube the
  cameras see is the same whichever of its four sides is turned to them. The basket is a woven plastic
  storage basket (rim 252 × 192 mm, base 227 × 170 mm, 105 mm tall: the vendor's listing, not measured at the rig,
  since its size enters no measurement), in the front-left corner of the box, held on the floor by double-sided
  tape and clear of both pipes (drafts up to 2026-10-08 said "with its base against two pipes", which was the plan
  and not the rig). Its place is the one the front camera's kept frames show: the same to within 0.2 px in every
  kept start frame of 2026-10-07 and 2026-10-08, and put back to within 2 mm with `scripts/rig/measure.py basket`
  when the arm has moved it. A cube that comes to rest inside it is a success. **The basket stays on its tape as
  it is:** in three of the nine trials of the go or no-go the wrist camera's mount caught the basket's rim and
  carried the basket up to 5 cm, and a firmer hold would put that strain on the camera's mount (the operator, at
  the freeze).
- **The wrist camera's place in its mount** is read from its own picture: the gripper's fixed jaw is rigid with
  that camera, so the jaw's place in the picture is the camera's. In both training sets and until the first catch
  it stood in one place (its lower hole's centre within half a pixel); the first catch moved it 3.3 px across and
  0.8 px down; the operator brought it back to within a pixel and tightened the mount before the ninth trial, and
  it held through the third catch (`scripts/rig/measure.py wristplace`). The reading itself moves by about a
  pixel with the arm's pose, so it is taken with the arm at rest.
- **The grid** of 5 × 3 cells of 50 mm is virtual: nothing is drawn on the floor. It is tied to the floor by two
  AprilTags, whose places in the grid's frame were measured once against a printed sheet (section 7).

The appendix of the paper carries a table of the rig's settings in the workshop's categories (hardware, actuation,
controller gains and rates, action space, observations, demonstrations, reset distribution). Its entries are the macros
of `paper/numbers_by_hand.tex`, each copied from the place its line names: the rig, the dataset's metadata, the
training configuration (`training/settings.md`), the trial records. The entries that only the rig can give (camera
models and mounting, the servos' variant, the gripper, the basket's place) are collected with
`docs/runbook/04-rig-facts.md` before the paper is final: they describe the rig and change no trial. The servo gains
are LeRobot's defaults, which its follower code writes at every connection (*read*).

## 4. Demonstrations: one budget, two layouts

| | Clustered | Spread |
|---|---|---|
| Episodes | 55 | 55 |
| Per trained cell | 5, all at the cell's centre | 5, at distinct positions |
| Positions | 11 | 55 |
| Closest two planned positions | the same point | 17.2 mm apart |
| Largest gap | 35.4 mm | 17.2 mm |
| Held-out cells | A1, B2, D2, E3 | the same |

- **The clustered layout is the documented habit.** LeRobot's guide says: "We suggest recording at least 50 episodes,
  with 10 episodes per location." Ours has the same structure with other numbers: 5 episodes at each of 11 locations,
  the layout of the August dataset. For the same budget that is more locations than the guide's five, which makes the
  clustered layout harder to beat in the workspace comparison, not easier. The paper says "repeated episodes at a few
  locations, as the guide suggests" and gives both pairs of numbers; it does not say the guide's recipe was followed
  to the letter.
- Both layouts are generated by `scripts/make_layouts.py` from a seed and committed before recording
  (`config/layouts/`, written under draft 0.1; nothing they depend on has changed since). The spread generator is asked
  for at least 12 mm between positions and achieves 17.2 mm.
- **Same operator, same day, same task text, same episode length, same cube orientation** (square to the grid, the
  face with rounded edges down) and the
  same demonstration style: same start pose, same approach side, same grasp.
- Each start position is set with the target overlay to within 2 mm, **and the operator confirms it with a key, hands
  out of the picture, before the arm is enabled**. It is measured again from the episode's own first frame (the
  start-state audit). Distances always use the measured positions.
- **Recording order.** Rounds of 11 episodes, one per trained cell, in shuffled order. Four sittings alternate the sets
  (clustered rounds 1 to 3, spread 1 to 3, spread 4 to 5, clustered 4 to 5), so that a slow drift over the day (fatigue,
  servo temperature, camera exposure) falls equally on both.
- A practice block of about 10 episodes is recorded first and discarded. It also rehearses the upload and the training
  notebook (`training/README.md`, step 0.6).
- An episode is re-recorded when the operator fumbles, the cube was outside tolerance, or a check fails. Re-records are
  counted and reported.
- **Dataset checks before training:** episode and frame counts, both camera streams decoded, frame pacing, torn-frame
  scan, and the start-state audit (every episode in its planned cell and within 3 mm of its target).
- **This clustered set is tighter than August's.** August's five demonstrations of a cell were placed by a grid drawn
  over the picture and never measured: measured now, they lie 3.1 mm from their cell's mean start on average, and the
  two farthest apart in a cell are 4.8 to 13.7 mm apart (`analysis/pilot_august/`). Here every start is within 2 mm of
  its target. The August result is therefore repeated on a cleaner cluster, not on the same one, and the radius found
  here need not be the one August's policy had.
- The spread layout's outermost starts are 5 mm inside the grid's outline, and the cube's footprint is not
  centred on its measured position: in August's view the two were about 8 mm apart. The cube can therefore reach about
  18 mm outside the outline on one side. **They fit** (the recording of 2026-10-07): all 55 spread starts were placed
  and recorded, the one nearest the outline 5.0 mm inside it (episode 51) and seven more within 10 mm of it, each
  accepted within the 2 mm of a placement and found by the audit within 1.84 mm of its target. The margin was not
  widened.

## 5. Training

Two ACT policies, one per dataset, **identical settings**, trained in parallel on rented GPUs. The procedure and every
setting, with where each value was read, are in `training/` (`README.md`, `settings.md`).

- **All 55 episodes of each set are trained on** (`dataset.eval_split = 0.0`, `eval_steps = 0`). August held the last 6
  out for an evaluation loss and trained on 49; that loss is given up, because a held-out episode would be a
  demonstration the policy never saw and would have to be left out of every distance.
- **Step budget: 80,000 for both, fixed now**, with a checkpoint every 10,000. Both policies are evaluated at step
  80,000. No checkpoint is chosen by looking at rollouts or at a loss curve.
- **Fallback, declared in advance:** if either run cannot reach 80,000 steps before the trials must start, both policies
  are evaluated at their largest common saved step, and not below 40,000.
- **Identical is checked, not assumed:** `training/compare_train_configs.py` compares the two saved training
  configurations after the first checkpoint and names every difference other than the dataset and the run's name.
- One training run per dataset, both with the same seed. Run-to-run variation is not measured, and the paper says so:
  a difference between the two policies is then not yet a difference between the two layouts.
- **A second run of each layout with another seed was considered and is left out of this submission:** it would put
  four policies at the rig, an arm of the experiment that nothing has tried, on the day before the trials. It is the
  first extension for a longer version, and the records are ready for it: each trial's record names its run and
  checkpoint, and the analysis counts the workspace test by pair of runs.

## 6. Execution at evaluation

The same for every policy trial, and as in August. They are parameters of the protocol (`[execution]` in the protocol
file): the trial runner takes them from there, refuses a counted session in which one differs, and writes them into
each trial's record with the arm's start pose (the pose at the look of the handoff gate that passed) and the
camera's creep before the attempt, so the paper reports what ran and not what was meant to run:

| Setting | Value |
|---|---|
| Control rate | 30 Hz requested. Each record carries the time from the first step sent to the last; the achieved rate is steps over that time, and the paper says so if its median is more than 2 % under the rate requested. It is: the replay pilot of 2026-10-07 ran at a median of 29.2 Hz, 2.6 % under (the earlier project's attempt loop sleeps what is left of a step with the system's plain sleep, which returns about a millisecond late; the demonstrations were recorded by LeRobot's own loop at 29.99 Hz) |
| Action smoothing | temporal ensembling, coefficient 0.01, a new prediction at every step |
| Per-step clamp | a joint's command may be at most 15 degrees from its present position |
| Handoff gate | the first command must be within 8 degrees of the arm's pose on every joint (5 degrees for a replay) |
| Time limit | 45 seconds per attempt. A cube not released in the basket by then is a failure. (30 until 2026-10-06: a demonstration is recorded for 35 seconds, not 25, and an attempt must not be shorter than what it was shown) |
| Shortest attempt | an attempt that a key ends within 1 second of its start is void: a slip of the key, not a trial |
| Stall rule | the earlier project's: an attempt ends when the arm is held back against its commands for about a second |
| Start pose | set by the operator with the leader arm before each trial; there is no stored pose |
| Camera creep | no attempt while the front camera's picture reads more than 1.0 mm from where the session solved its conversion (section 7): the trial goes back to its placement for a new go, and at a second such reading the session ends and is started again, with the cube off the table. A reading that cannot be taken lets no attempt go either. Each record carries the reading of the look that let its attempt go |

**Replay trials** send the recorded actions of one demonstration through the same loop, clamp, gates and time limit,
with no policy and no camera in the loop.

- **Which demonstrations.** One demonstration of the clustered set from each of the 11 trained cells
  (`replay_demos = 11`), so that the replay is tested over the same places on the table as the policy. The
  planner's seed draws which of a cell's five demonstrations it is, before any trial. The block's 22 trials replay
  each of the 11 exactly twice, in two different parts of the block: the 4 trials at `d = 0` are 4 different
  demonstrations, and each staircase of 9 trials replays 9 different ones. So 7 of the 11 are not tested at their
  own start in this block. Which demonstration a trial replays is fixed by the seed, not by any outcome.
  **Tuesday's pilot replays each of the 11 once at its own start** (`config/pilot.toml`), and the paper gives how
  many of them succeeded there. None is replaced for failing: a demonstration that does not replay from its own
  start is part of what the hardware does.
- **Where the arm starts.** The handoff gate lets a replay begin only when every joint is within 5 degrees of that
  demonstration's first recorded action, so a replay starts from nearly the pose the demonstration started from. A
  policy trial has the looser 8 degrees: the policy sees the arm and may start from any nearby pose. In the replay
  pilot the gate refused the start in 9 of 22 trials, eleven times of twelve for the wrist's turn: the demonstrations'
  first poses differ by up to 14 units in that joint (the recorder lets it rest up to 12 degrees from a session's
  start pose), and a reset leaves it wherever the leader's wrist was left. The correction shows where the joint is
  and where it is wanted while the arm is moved; the 5 degrees stay. A trial's record holds the pose at the look
  that passed (the pilot's records hold, for the eight trials that were corrected, the pose that had been refused:
  found on the night of 2026-10-07 and put right before the counted trials).

## 7. Measuring a start position (the instrument)

The front camera detects the cube and converts its position to table millimetres through the two tags. The same
detector measures demonstrations, targets and trial placements, so a constant bias cancels in every distance.

- **The table is registered once** (`docs/runbook/00-table-layout.md`). A printed sheet with the grid's outline and six
  small tags at known places is seated where the grid should be. The sheet's tags give the conversion from pixels to
  the grid's millimetres (the table frame); the two floor tags' corners are put through it and stored
  (`config/table_geometry.json`); the sheet is taken away. The sheet's own scale is measured with calipers on the
  grid's outline, both ways, and the registration takes the sheet as it was printed.
- **The frame is fitted through the tags' outlines, not through the detector's corners.** At this view (0.8 to
  1.0 px per mm) the tag detector reports a tag's corners about 0.6 px inside its outline, and a corner of a small
  tag moved by 1.7 px when the picture changed by one grey level (*measured* 2026-10-05). Each side of a sheet tag is
  therefore measured from the picture as a line through sub-pixel edge positions. On synthetic pictures at the rig's
  view, with the truth known, the frame fitted that way is within 0.25 mm of the truth everywhere on the grid. The
  floor tags are measured by their outlines too, at the registration and in every session after it.
- **What the registration reports, and the paper with it:** how far the sheet's tags are from one flat plane seen
  through an ideal lens; how well each sheet tag is predicted by the other five when it is left out of the fit; each
  sheet tag's own size read through the frame against its printed size; the floor tags' size against calipers. Each
  has a value beyond which the registration says so and is still stored, and one beyond which it is refused
  (runbook 0, step 4, has the table). All six sheet tags are needed.
- **Where those limits come from, said plainly.** The first registration at the rig was worked through from two
  saved frames (`analysis/instrument/first_sitting.md`). The sheet's printed grid lines, which no fit uses, lay
  0.35 mm (root mean square) and up to 1.2 mm from the frame, while the sheet's fit read 0.35 mm on average and
  0.79 at worst: that sheet was not flat. The limits were written after those figures were known. They say so by
  their form: the note begins where a flat sheet ends on synthetic pictures (0.3 and 0.6 mm), and the refusal where
  the frame can no longer be within about 1.5 mm (0.5 and 1.0 mm), against a cube accepted within 2 mm.
- **What it cannot test.** While the camera stays put, solving the conversion again from the two floor tags gives
  the sheet's frame back exactly, whatever the floor tags' true size or flatness: their stored corners were made
  through that frame. An earlier draft called such a solve "the instrument's accuracy over the grid"; it could only
  repeat the sheet's fit, and is gone. A sheet lying tilted as one rigid plane also fits itself, and so does a sheet
  whose size was typed wrongly: only the floor tags' size against calipers can show either. The check that does not
  rest on the sheet is the cube moved along a printed scale (block V0, below).
- **The frame is the floor tags', and every session solves it afresh.** The camera of this rig does not keep its
  place: its pipe is carried by the light box's walls, and within an hour of the calibration its picture stood
  0.04, 0.28, 1.25 and 2.2 mm from where the calibration saw it, the far pipe, the arm and the basket moving with
  the tags (*measured*, 2026-10-06, `analysis/instrument/picture_motion.md` and the sessions' own lines). The tags
  are taped to the floor. Until that afternoon every measurement went through the one conversion the calibration
  stored, and a session only confirmed that a fresh solve lay within 1.5 mm of it: read that way, every position
  would have carried the camera's movement. Now a session solves the conversion from the two floor tags' outlines
  as it finds them (20 frames) and measures through that. It holds the stored calibration against it: the
  camera's movement since is said beyond 1.5 mm and the session refused beyond 5 mm, where the picture is no
  longer the one the demonstrations were recorded in; the tags' movement against each other since the registration
  is said beyond 1.5 mm and refused beyond 3 mm. **What this leaves open:** the camera moving within a session,
  and what the policy sees, which moves with the camera and is reported for every session. For the demonstrations
  the first is closed after the fact: the start-state audit measures every start through the floor tags as that
  episode's own first frame shows them, and writes down how far the camera's picture stood from the calibration in
  every episode. On synthetic pictures of the rig's view the starts so measured lie within 0.55 mm of their targets
  whether the camera stood still, moved by part of a pixel or jumped by three (the detector's own share is 0.5 mm);
  through one stored conversion they lie 1.0 to 3.7 mm off once it has moved (*measured*, tests). For the trials
  the placement is measured live through the conversion its session solved at its start, and there is no later
  frame of a trial to measure it again from. The camera goes on moving as it warms: on 2026-10-07 its picture crept
  a millimetre in the first eleven minutes of a session begun after it had been off, and 0.3 mm in seven minutes of
  a session begun warm; over two recording sessions begun cold the live readings came to stand 1.2 and 1.7 mm from
  the audit's. **So before every attempt a session measures how far the picture has crept since its conversion was
  solved, writes the figure into the trial's record, and beyond 1.0 mm (`view_crept_limit_mm`) makes no attempt:**
  the trial goes back to its placement for a new go, as after any look in which something stood in the picture,
  and at a second reading over the limit the session ends, is started again, solves the conversion afresh, and
  goes on with the same trial. A reading that cannot be taken lets no attempt go either. A trial's recorded start
  is therefore within about a millimetre of where a conversion solved at that moment would put it, besides the
  detector's own 0.5 mm. About, because the reading itself follows what stands
  in the picture: with the cube beside floor tag 0 it stood 0.13 to 0.36 mm above the readings before and after
  it in the replay pilot, and a shadow's edge along a tag's side reads 0.2 to 0.9 mm on drawn tags. For the same
  reason every session is started with the cube off the table: its conversion is solved from its first frames.
- **The calibration is the reference.** It is made straight after the registration, with the sheet off the floor.
  It solves the conversion from the two floor tags' outlines averaged over about 90 frames, keeps the frame it saw
  and the picture's levels, and stores them (`data/instrument/table_view.json`). Before it stores, it compares
  where each floor tag now is in the picture with where it was at the registration. Tags that moved the same way mean the
  camera was nudged, which is said and followed. Tags that did not move alike are what a camera that turned shows,
  and also what a tag moved on the floor shows: **the two tags cannot tell these apart.** Their movement against
  each other, read on the floor through the registered frame, is noted beyond half a pixel and stored; beyond
  3 mm the calibration is refused until the rest of the picture has said which it was
  (`analysis/instrument/picture_motion.py`: the pipe, the arm and the basket move with the tags when it was the
  camera). At the rig's first calibration the tags read 1.2 mm against each other; the rest of the picture had
  turned with them by 0.10 degrees, and 0.3 mm was left unexplained (*measured*,
  `analysis/instrument/picture_motion.md`). The limit was 1.5 mm for two hours, set after that figure was known;
  the camera's own movement has since read 1.0 mm by itself, and it is 3 mm, with every reading beyond 1.5 mm
  settled from the pictures.
- **The two floor tags are a watchman, not a ruler.** One pixel on one corner of a floor tag moves the far side of
  the grid by up to 2.5 mm (*measured* on the first sitting's geometry). Measured by the detector's own corners,
  which sit a few tenths of a pixel off by an amount that depends on where the pixels fall, a picture shifted by a
  tenth of a pixel put a solve from the two tags up to 1.2 mm from the frame; measured by their outlines, 0.5 mm at
  most, and a shift of half a pixel reads as the 0.6 to 0.7 mm it is (*measured* on synthetic pictures of the rig's
  view, 2026-10-06).
- **What the fit's own error can and cannot catch.** The tag fit's residual rises only when a tag has moved by
  centimetres: in a synthetic scene, a tag moved by 8 mm changed it from 0.3 to 0.6 mm while a true 50 mm read 50.7.
  A small slip of a tag is therefore caught by the agreement check above and by the scale check below, not by the
  residual.
- **What "position" means.** The detector reports the centre of the cube's visible outline (top and lit sides) mapped
  through the table plane, not the centre of its footprint. That point is offset from the footprint by several
  millimetres, and the offset changes slowly across the table, so distances are expected to read a few percent long.
  The scale check below measures that factor. **If its four runs agree within one percentage point, every distance in
  the analysis is divided by the measured factor, and the paper says so; if they do not agree, distances are left as
  read and the spread of the factor is reported as the instrument's accuracy.** The check was run on 2026-10-06, in
  four runs (a fifth, abandoned after a mark of the ruler was skipped, is kept under `data/instrument/not_used/`).
  The rule is read as the analysis computes it: each run by its two ends, over its whole length. So read, the four
  runs take a true length for one longer by factors of 1.0123 to 1.0248, 1.25 percentage points apart: **they do
  not agree within one point, distances are left as the camera reads them, and the paper gives the factor's range
  as the instrument's accuracy** (25 mm as read are 24.4 to 24.7 mm true).
- **One kind of millimetre.** The planner, the placement ring, the trial log and, with distances left as read, the
  analysis and the paper all work in millimetres as the camera reads them: a far anchor "at 40 mm" is 40 mm as read,
  which is 39.0 to 39.5 mm true by the factors above. The paper says once that its distances read 1.2 to 2.5 % long.
  Prediction 1's bounds, 5 and 40 mm, are compared with the radius as read.

Before any trial, the instrument is checked against a physical scale (block V0):

1. **Displacement accuracy.** Move the cube along a printed millimetre scale in 10 mm steps over 50 mm, in x and in y,
   at two places on the grid. Compare measured with true displacement. Pass line: mean absolute error at most 1.5 mm,
   no single step off by more than 3 mm. Reported besides: the measured length of the 50 mm run, as a percentage.
   The printed scale's own 100 mm were not checked with calipers. The tags and the registration sheet from the same
   printer were (2026-10-05): right along the page and 0.4 % short across it, which is what the scale can be off by.
2. **Repeatability.** Push the cube ten times against a fixed stop (a ruler or an angle taped to the floor), lifting
   it away each time, and report the scatter of the measured positions. A stop, not the overlay: the overlay is
   steered by the same camera that then measures, so its scatter would say nothing about the camera.
3. **Camera drift.** The tag baseline check passes before and after every sitting and every block.

If the pass line is missed, the cause is found before trials begin; if it cannot be fixed, the measured error is
reported as the instrument's accuracy and the step sizes are widened to match.

**Reset by eye (block V1, camera only).** To show what a reset without a placement aid does, the cube is placed 20
times from a reference picture shown beside the live view, with no overlay and no readout, and the error of each
placement is measured. This yields a distribution of by-eye reset errors for the paper's sensitivity calculation. It is
one or two people on one rig, and is reported as an illustration.

- **Order.** All 20 reference pictures are taken first, the cube placed on each target with the ring. The 20 by-eye
  placements follow in a shuffled order. A cube put back seconds after it sat on the ring would be placed from the
  memory of the hand, which an evaluation spread over days does not have.

- It is the practice of an evaluation that resets "to the same place" from a picture or from memory. It is **not** the
  practice of an evaluation that overlays a reference image on the live view: that sits between by-eye and this
  study's measured reset. The earlier project's own overlay resets landed 0.8 to 3.0 mm from their targets (9 resets,
  measured afterwards); the paper gives that figure beside the by-eye one and does not claim to have measured anyone
  else's procedure.

## 8. Evaluation trials

A trial, as the trial runner conducts it (`docs/runbook/03-trials.md`):

1. The planner names a target. The people at the rig see a running number and a code, never the condition.
2. The cube is placed on the target with the overlay, within 2 mm. **The operator confirms with a key, hands out, and
   only then can the arm move.** The position at that moment is the trial's measured start.
3. The condition's policy or replay runs through the gated loop of section 6.
4. The outcome is scored on the August rubric: a graded scale from 0 to 5, success only at 5 (the cube released in the
   basket), with a failure label otherwise.
5. One line is appended to the log. Lines are never edited or removed.

**The independent variable** is `d`, the measured distance to the nearest trained demonstration of the tested
condition. A requested distance `L` is turned into a target by drawing uniformly among the points of the test region
whose nearest-demonstration distance is `L` (within 0.5 mm). For a replay, the target is drawn uniformly in direction
on the circle of radius `L` around the replayed demonstration's start. The test region is the grid less a 5 mm margin.

**The adaptive rule.** One-up-one-down staircases: after a success the next distance is larger, after a failure smaller,
in steps of 10 mm until the first reversal and 5 mm after it. Two staircases per condition, one starting near and one
far, interleaved in a seeded order with the anchors, so that which part comes next cannot be told from what came
before. The staircases only choose where to test; the estimate comes from a model fitted to the trials (section 10).

### Trial budget, in priority order

| Block | Condition | Trials | What it gives |
|---|---|---|---|
| **T1a** | Clustered policy | 6 at `d = 0`, 2 staircases × 12 (from 10 and from 40 mm, up to 50), 4 at 40 mm: **34** | The tolerance radius of the documented habit; a repeat of August's result on new data |
| **T1b** | Replay | 4 at `d = 0`, 2 staircases × 9 (from 5 and from 25 mm, up to 50), each of 11 demonstrations twice: **22** | The hardware's own tolerance |
| **T2** | Both policies | 26 placements, 13 drawn uniformly over the trained cells and 13 over the held-out cells, each tested with both: **52** | The two layouts compared at the same placements; the test of the lookup prediction |
| **T3** | Spread policy | 4 at `d = 0`, 2 staircases × 10 (from 5 and from 25 mm, up to 50): **24** | Whether the radius depends on the layout |

Total 132 counted trials. T1a and T1b alone make a paper; T2 adds the controlled comparison; T3 is run if time allows,
and is the first block to be dropped if Wednesday runs short.

- **Why the far anchors are at 40 mm.** At 50 mm nearly every placeable point lies in the corner cells A1 and E3; at
  40 mm the interior held-out cells B2 and D2 take their share. August's held-out trials started 31 to 47 mm away.
- **Why the staircases start where they do.** August's trials (`analysis/pilot_august/`): 19 of 19 succeeded within
  5.6 mm of a trained demonstration, 1 of 12 between 31 and 47 mm, and nothing was tested in between.
- **The spread policy's staircases leave the trained cells early.** No point of a trained cell is farther than 17.2 mm
  from a spread demonstration, so a T3 trial beyond that distance is necessarily in a held-out cell. The paper says so
  where it reports T3.

### What this budget can and cannot show (simulated before any trial)

`analysis/simulation/procedure.md` plays each block against success curves assumed true and fits it as the analysis
will. Every figure below is simulated, none is a measurement.

| Block | Trials | Cliff (width 1 mm) | Width 3 mm | Slope (width 10 mm) |
|---|---|---|---|---|
| T1a | 34 | 0.8 / 4.2 | 1.6 / 8.1 | 4.2 / 18.4 |
| T1b | 22 | 1.0 / 6.3 | 2.1 / 12.6 | 5.2 / 23.6 |
| T3 | 24 | 0.9 / 5.3 | 1.9 / 11.3 | 4.8 / 21.7 |

Each cell: the root-mean-square error of the estimated radius / the mean width of its 95 % interval, in millimetres,
for the middle one of the simulated truths with that width (the report's Table 1; radii from 5 to 40 mm, lapse 0 and
0.05, 300 simulated blocks each). How sharp the real curve is decides which column applies, and that is not known
before the trials.

- **The intervals hold what they promise:** they contained the true radius in 97 % of the simulated blocks on average,
  and in 91 % for the worst truth.
- **A radius that is small against the width is reported as "at most".** Where the curve is already falling at 0 mm,
  the interval is open below in about half of the simulated blocks (50 to 60 %).
- **What the comparison of two radii can tell apart** (*simulated*, `analysis/simulation/comparison_rule.md`: the two
  blocks as planned, a replay radius of 15 mm, 300 pairs of blocks per case). The rule of prediction 2 says "larger" in
  at least four pairs out of five once the policy's radius is 5 mm beyond the replay's if both curves are cliffs
  (width 1 mm), 10 mm beyond at a width of 3 mm, and 15 mm beyond at 6 mm. At a width of 3 mm a true difference of
  5 mm is called "larger" in 38 % of pairs. At equal radii it gives a wrong verdict in 2 to 3 % of pairs either way.
  So this comparison settles differences of a centimetre, not of a few millimetres. (An estimate that stood here in
  an earlier state of this draft, 6 mm at a width of 3 mm from a normal approximation, was too hopeful.)
- **Staircases are kept.** The same number of trials at fixed, equally spaced distances does worse: the error is 38 %
  larger, and the interval is open on a side in 29 % of blocks against 16 %.
- **The number of trials is the parameter that moves the result**, not the step: a final step of 2.5 mm in place of
  5 mm changes nothing that matters. If Wednesday has room for more, the trials go to T1b or T3 first, and that is
  decided before the freeze, not during the day.
- **The paired test detects large differences only.** With 26 pairs the exact test reaches p < 0.05 at least 80 % of
  the time when the two success rates are about 0.30 against 0.75, 0.40 against 0.85, or 0.50 against 0.90
  (*computed*, `simulate.rate_needed`: an exact sum that takes each policy's rate as the same at every placement and
  the two outcomes at a placement as independent; with the 20 pairs of draft 0.2 the second rates were 0.80, 0.90
  and 0.95). T2's placements are not alike, half of them lying in cells with demonstrations and half in cells
  without, so the calculation that follows the design is the one under the lookup model, in the next item. If the
  two policies turn out closer than that, the paper reports the counts and their intervals and says that the test
  could not have decided.
- **Four far anchors do not measure the floor.** They guard the radius against one lucky success (section 10); how
  often luck strikes is reported as a count, `k of 4`, and nothing more.
- **Why half of the workspace test's placements are in held-out cells** (*calculated* under the lookup model,
  `analysis/simulation/workspace_designs.md`; not a measurement). Eleven clusters on a 50 mm grid cover their own cells
  once the radius nears 25 mm. Draft 0.2 drew 20 placements, all over the trained cells: the paired test would then
  reach p < 0.05 in 81 % of cases at a radius of 15 mm, 70 % at 20 mm, 22 % at 25 mm and 1 % at 30 mm (width 3 mm). It
  was blind to a large radius. With 26 placements, 13 drawn over the trained cells and 13 over the held-out cells, it
  does so in 76, 89, 88 and 76 % of cases at the same four radii, for 12 more trials. The price is at the small end:
  40 % against 59 % at a radius of 10 mm. And if the curve is a slope (width 6 mm) no design that was compared
  reaches 80 %: this one 62 % at its best.
- **What the workspace test now covers of T3.** The 26 placements that the planner's seed draws lie 2.8 to 39.0 mm
  from the nearest planned spread demonstration and 7.5 to 52.6 mm from the nearest clustered one (*computed* from
  `planner.paired_placements` and the planned layouts; the starts as measured will lie within 3 mm of the planned
  ones, section 4, and these figures will shift by as much). For the spread policy that is most of the range T3's
  staircases were for, which is why T3 is the block to drop first.

### Pilots, and when trials start to count

- **Pilot trials** try the procedure: the replay pilot on Tuesday afternoon and the nine exact repeats of the go/no-go
  on Tuesday night. They run with `--pilot`, under `config/pilot.toml`, into log files named `pilot_*`, and are never
  analysed with the counted trials. Their results may change this draft; that is what they are for.
- **Counted trials** are refused by the trial runner until the protocol's status is `frozen`, and a log file never
  holds trials of two protocol versions.
- **Tuesday's replay trials are pilots only.** The replay baseline that enters the paper is run on Wednesday, in one
  session with the clustered policy's block.
- **As run, a day later than planned.** The replay pilot was run on Wednesday 2026-10-07: 21 scored trials and one
  void. Ten of the eleven demonstrations that the counted block replays were replayed at their own start, and all
  ten succeeded; the eleventh attempt was ended by a key after 5.7 seconds and says nothing of its demonstration.
  At 5 and 10 mm from a demonstration 4 of 4 succeeded, at 15 mm 2 of 4, at 20 and 25 mm 0 of 2. The counted
  replay block's staircases start at 5 and at 25 mm as planned. The go or no-go of the clustered policy, nine
  trials at exact demonstration starts, was run on Thursday 2026-10-08 before the freeze: 8 of 9 succeeded (the pass
  line is 8 of 9). The one failure closed the gripper beside the cube and ran to the time limit. The counted
  trials are run on the same day.

### Order, blinding, voids

- **T1a and T1b run interleaved in one session**: their 56 trials are shuffled into one seeded order, so the policy
  and the replay meet the same hour, the same temperature of the servos and the same state of the cameras. Then T2,
  then T3. Within a block the order is seeded and fixed by the planner, which is a pure function of the log: a restart
  resumes exactly where the log ends.
- **In T2** the two trials of a pair run back to back, and each policy goes first in half of the pairs, in a seeded
  order. The 26 placements are drawn by a seed before any trial and tested in one shuffled order, so that a
  placement's place in the order says nothing about whether it lies in a trained or a held-out cell.
- **Blinding.** A trial is announced by a running number and a code drawn without replacement from one shuffled list,
  assigned by the trial's number alone. Neither says which policy runs, which block the trial belongs to, or whether
  it is a replay. With one person at the rig the scorer is also the operator. A replay can often be told from a policy
  by the way the arm moves; the score is whether the cube was released in the basket, which leaves little to judgement,
  and the paper states this limit.
- **A second scoring from video.** Every attempt is recorded. A fifth of the counted trials is scored again from
  the video alone by a person who is shown neither the condition nor the distance (`docs/runbook/05-second-scoring.md`).
  The fifth is one scored trial in five of each condition of each block, rounded up (29 of the 132 planned), so that
  every condition is there; it is drawn after the last trial, by a seed and the hash of the trial log itself, so
  that nobody can know before the trials which of them will be looked at again. The two trials of a workspace pair
  are not both drawn. If no second person is available, the operator scores them again with the trials' identities
  hidden.
  The score given at the rig stays the trial's outcome, whatever the second one says. The paper reports in how many
  trials the two scores agree, and says who gave each.
- A trial is **void** when it could not test the policy: the placement was later found outside tolerance, a camera or
  motor fault ended the attempt, a person interfered, **the handoff gate refused three times** (the arm was never
  handed over), or **a key ended the attempt within a second of its start**. A void trial stays in the log with its
  reason and its place in the plan is asked again with the same target, under a new number and a new code.
- A stop caused by the policy's own motion after the handoff (the clamp, the stall rule) is a **failure**, not a void.
- **A trial in which the arm catches or moves the basket stands as scored**: it is the policy's own motion, and in
  the go or no-go every catch came after the cube had been released. Before the next trial the session is ended at
  the reset, the basket is put back and the wrist camera looked at (`scripts/rig/measure.py basket` and
  `wristplace`, as before and after each block), and the session is started again. Afterwards the first frame of
  every counted trial's two videos is read for both, and a trial that began with the basket more than 2 mm or the
  wrist camera more than 2 px from its place is named in the report.
- The tag baseline is checked at the start and end of each block. A failed check stops the block.

## 9. What is predicted before the data

Stated now so that they cannot be fitted afterwards.

1. **The curve.** The clustered policy succeeds at least 5 times in 6 at `d = 0` and at most once in 4 at 40 mm, and its
   tolerance radius lies between 5 and 40 mm. (A repeat of August on a tighter cluster, with the edge located.)
2. **Hardware.** The policy's radius is larger than the replay's: the cameras buy some correction. **The rule:** from
   the two block fits, taken as independent, "larger" means a posterior probability of at least 0.95 that the policy's
   `r50` exceeds the replay's; "smaller" means at most 0.05; anything between is reported as "not told apart by these
   trials". The difference and its 95 % interval are reported in every case. If the verdict is "smaller", the policy
   does worse than its own demonstrations played back blind, and that is the finding; if it is "not told apart", the
   paper says that these trials could not separate what the policy adds from what the hardware gives.
3. **Layout.** At the placements of the workspace test, the spread policy succeeds more often than the clustered one.
   **The rule:** the exact paired test gives p below 0.05 with the spread policy ahead (more successes over the
   complete pairs). With p below 0.05 and the clustered policy ahead, the prediction failed and the paper says so. In
   every other case the paper reports the counts and says the test did not decide.
4. **Lookup.** Each policy's success in T2 matches the lookup prediction computed from the clustered policy's block
   fit and each placement's distance to that policy's own nearest demonstration. Falling short of it means that
   spreading thins the policy (repetition at a point matters); exceeding it where the prediction is low means the
   policy interpolates between demonstrations. "Matches" means the observed count lies inside the prediction's 95 %
   interval (section 10).
5. **Layout and radius** (T3): no prediction; reported as an estimate.

## 10. Analysis, fixed in advance

- **Model.** `P(success at d) = floor + (1 − lapse − floor) / (1 + exp((d − r50) / width))`. The tolerance radius is
  `r50`: the distance at which success is halfway between the plateau `1 − lapse` and the floor.
- **Why a floor.** Far from every demonstration a trial can still succeed by luck: in August one did, an edge grasp at
  40 mm. With the floor fixed at zero that one trial moved the fitted radius by 6 mm. In simulation, leaving the floor
  out when the truth has one (0.08) makes the error 22 % larger; having it in when the truth has none makes the
  interval 3 % wider.
- **Estimate.** A posterior on a grid: flat prior on `r50` from 0 to 80 mm in steps of 0.5 mm; flat prior on the
  logarithm of `width` from 0.5 to 20 mm (30 values); lapse in {0, 0.02, 0.05, 0.10} and floor in {0, 0.05, 0.10},
  equally weighted. Reported: the posterior median of `r50` and its 95 % interval.
- **The plateau is checked, not assumed.** The lapse values above stop at 0.10: they fit a condition that nearly always
  succeeds at a demonstration's own start, which is what the go/no-go asks of the clustered policy. A replay, or a
  policy taught from starts that are never repeated, may have a lower plateau, and the model would then explain its
  failures near a demonstration by a smaller radius. In simulation (`analysis/simulation/lapse_values.md`) a true
  plateau of 0.80 pulls the radius inward by 3 mm, 0.65 by 8 mm, 0.50 by 13 mm, and at 0.65 the interval holds the
  truth in 75 % of blocks, not 95. So every fit is made a second time with lapse values up to 0.50 and **both are
  reported**, with the counts at `d = 0`. The first is the radius the paper reports; if the two medians differ by more
  than 5 mm, the paper says that this condition's plateau is in doubt and its radius is not established. Whatever rests
  on such a radius (a comparison, the lookup prediction, the reset sensitivity, prediction 1) is still computed by its
  rule and is reported with that caveat beside it, never without. The longer
  list is not the main fit because, when the plateau is high, it makes every interval about two thirds wider.
- **When the data do not bound the radius.** An interval is **open on a side** when the posterior at that end of the
  grid is at least half of what it is at the interval's own bound on that side. The paper then reports "at most" or "at
  least" and not a radius. This happens when the curve is already falling at 0 mm, or still on its plateau at the
  largest distance tested.
- **Two fits per policy, and which one is used for what.**
  - The **block fit** uses the condition's own threshold block only: T1a for the clustered policy, T1b for the replay,
    T3 for the spread policy. It is the fit the simulation describes. **Every comparison and every prediction uses
    block fits:** the comparison of the policy with the replay (two blocks of the same session), and the lookup
    prediction, which must not have seen the T2 trials it predicts.
  - The **pooled fit** adds the policy's T2 trials at their measured distances. It is reported beside the block fit as
    a check that the two agree. If T3 is not run, the spread policy has a pooled fit only, from its 26 T2 trials:
    placements fixed in advance, not chosen by a staircase, half of them in held-out cells. Whether they bound its
    radius is decided by the rule for an open interval, as for every fit, and the paper says that this radius is a
    pooled fit.
- **Without a model.** Success counts in 10 mm bins of `d` with Wilson 95 % intervals; the largest `d` with a success
  and the smallest with a failure. Bins are closed below and open above, and a trial is binned by its measured `d`.
- **Comparison of radii.** The posterior probability that one condition's `r50` exceeds another's, from independent
  posteriors, and the difference with its interval (the rule of prediction 2).
- **T2.** The 2 × 2 table of paired outcomes; the exact McNemar test; each policy's success with a Wilson interval.
  All 26 placements make one test. The same counts are also given for the trained cells and for the held-out cells
  apart, a pair counting by the cell its cube was measured in: these are descriptive and have no test of their own.
- **Lookup prediction.** For each T2 placement and each policy, the predicted success probability is the clustered
  policy's block fit evaluated at that placement's measured distance to the policy's own nearest demonstration,
  averaged over the posterior. Reported for each policy: the expected number of successes, the 95 % interval of the
  predicted count (it carries both the uncertainty of the fit and the chance in 26 trials), the observed count, and
  whether it lies inside.
- **Reset sensitivity.** The success a paper would report at "trained positions" if its resets carried the by-eye error
  distribution of block V1, computed from the clustered policy's block fit, beside the same calculation with this
  study's own placement errors. Labelled as a calculation, not a measurement.
- **Descriptive only:** success by placement inside or outside the convex hull of the demonstrations; success by
  whether the placement lies in a trained cell or a held-out cell, at each distance (beyond about 35 mm from a
  clustered demonstration every placeable point is in a held-out cell, so on the far side of the curve distance and
  "a cell with no demonstrations" cannot be told apart, and the paper says so); failure labels by distance;
  durations; the achieved control rate.
- **No trial is excluded except void trials and pilot trials.** No outcome is changed after the fact: the second
  scoring from video (section 8) is reported beside the scores given at the rig and replaces none of them. Counts of
  void trials and their reasons are reported.
- Every number in the paper is written by `scripts/make_report.py` from the logs into `paper/generated/`.

## 11. What this design cannot show

One task, one object, one arm, one gripper, one camera placement, one policy family, one training run per dataset, one
or two operators, about 130 trials.

- The radius is averaged over directions and over the table, and is specific to this rig. Run-to-run and arm-to-arm
  variation are not measured.
- Distances come from one camera and one detector. A constant offset cancels; the scale factor is measured (section 7);
  what is left is the instrument's accuracy, and the paper gives it in millimetres.
- The scorer is not blind to a replay's manner of moving, and with one person is also the operator.
- The paired test tells large differences only (section 8). The lookup comparison has 26 placements per policy.
- The by-eye reset measurement is an illustration from one or two people.
- The spread layout's result holds for 55 demonstrations on this workspace and says nothing certain about larger
  budgets, other tasks, or policies that are pretrained.
- The clustered layout has 5 episodes at each of 11 locations, not the guide's 10 at each of 5.

## 12. Stop lines

| When | Pass line | If missed |
|---|---|---|
| Before recording | Rig check passes; camera baseline registered; instrument check V0 within its pass line | Fix first; if not fixed by Tuesday noon, drop the spread set and keep T1 only |
| Tuesday night | Both datasets recorded and audited; a clustered checkpoint succeeds at least 8 of 9 exact repeats | Stop: no submission to this deadline; continue toward the later venue |
| Thursday 15:00 (a day later than planned, as everything since the recording) | T1a and T1b complete | Skip T3; finish T2 only if its 26 pairs fit before 19:00 |
| Thursday night | At least T1a and T1b complete | Write with what is complete; say what was planned and not run |

## 13. Deviations

Each deviation from version 1.0 is recorded in `docs/decisions.md`, marked **DEVIATION**, and listed here. Three, all
of 2026-10-08. The first two were found after the trials by a review of the paper against this document, and neither
changes what a trial measured; the third was known that day and is counted as a deviation since the blind reads of
the paper:

1. **The repeatability check (section 7, block V0, item 2) was run after the last counted trial**, not before any
   trial: 17:45 to 17:48 on 2026-10-08.
2. **The tag check at the end of blocks 1 and 2 (sections 7 and 8) was not run as a step of its own.** The next
   session's start made the same check about 47 and 39 minutes after each block's last trial, and both passed.
3. **Block T3 (the spread policy's threshold block) was not run.** The stop line that allows leaving it out
   (section 12) was not the reason: T1a and T1b were complete at 14:54, before 15:00. The operator decided it at about
   17:07, before any result had been computed by kind of trial.

## 14. Changes since earlier drafts

### Since draft 0.3 (settled at the freeze; `docs/decisions.md`)

| Change | Why |
|---|---|
| A session makes no attempt once the front camera's picture has crept more than 1.0 mm from where it solved its conversion; `view_crept_limit_mm` is an execution setting, and each record carries the reading before its attempt (was: "re-solved before every placement", marked open) | The camera moves as it warms, by a millimetre in the first minutes after it has been off, and a trial's start is what its session measured: over two recording sessions the live readings drifted 1.2 and 1.7 mm from the audit's |
| Distances are left as the camera reads them; the four scale runs' factors, 1.0123 to 1.0248, are the instrument's accuracy (the rule was open until the check was run) | By each run's two ends over its whole length the four differ by 1.25 percentage points, more than the rule's one |
| The spread layout's outermost starts fit, the margin stays 5 mm | All 55 starts were recorded |
| The basket's size is the vendor's listing; the printed scale's 100 mm were not checked with calipers | Neither enters a measurement of a trial; the same printer's other sheets were measured |
| The replay's handoff gate stays at 5 degrees for every joint | With the correction's readout a refused start costs ten to thirty-five seconds and no trial; a replay begun from another wrist turn than its demonstration's would not be that demonstration |
| The achieved control rate is stated: 29.2 Hz in the replay pilot | The rule of section 6: more than 2 % under the 30 Hz asked |
| The pilots were run a day later than planned, and the counted trials on Thursday; the stop lines of section 12 name Thursday | The sets were recorded on Tuesday night and Wednesday, the policies trained on Wednesday |
| The basket stands on double-sided tape, clear of both pipes, and stays so; its place and the wrist camera's place in its mount are read from the cameras' pictures before and after each block and for every trial afterwards; a trial in which the arm catches the basket stands as scored | The text had the basket against two pipes, which was the plan and not the rig. In three of the nine trials of the go or no-go the wrist camera's mount caught the basket, carried it up to 5 cm and, the first time, moved that camera 3.3 px in its mount. A firmer hold on the basket would strain the camera's mount instead (the operator) |

### Since draft 0.2 (decided on 2026-10-05; `docs/decisions.md`)

| Change | Why |
|---|---|
| T2 has 26 placements, 13 drawn over the trained cells and 13 over the held-out cells (was 20, all over the trained cells); 132 counted trials in all (was 120) | With every placement in a trained cell the paired test could hardly show a difference once the radius is 25 mm or more (section 8) |
| The replay block plays back one demonstration from each of the 11 trained cells, each twice (was 4 demonstrations, five or six times each) | The replay is tested over the same places on the table as the policy, in the same 22 trials |
| All 55 episodes trained on for 80,000 steps; the replay baseline of the paper run on Wednesday, interleaved with T1a | Confirmed as drafted |
| A second training run per layout with another seed is left out | Four policies at the rig would be an untried arm of the experiment on the day before the trials; first extension for a longer version |
| A random fifth of the counted trials is scored a second time from video, blind to condition and distance; the score given at the rig stays the outcome | The scorer at the rig is also the operator |
| What the paired test detects, restated for 26 pairs | 0.30 against 0.75, 0.40 against 0.85, 0.50 against 0.90 (with 20 pairs: 0.80, 0.90, 0.95) |
| A spread policy without T3 is no longer said to have a radius that cannot be bounded from above | Its T2 trials now reach held-out cells, up to 39 mm from its nearest planned demonstration |
| T2's counts are also given for trained and held-out cells apart, as description | The placements are now of two kinds; the test stays one test over all of them |

### Since draft 0.1

| Change | Why |
|---|---|
| The model has a floor (`analysis.floor_values`) | One lucky success far out moved August's fit by 6 mm; simulation: 3 % wider intervals when not needed, 22 % less error when needed |
| A rule for an interval that is open on a side | "Reaches the edge of the grid" was not a rule; the analysis code and this text now state the same one |
| Replay staircases may go to 50 mm (was 40) | The simulation of draft 0.1: with 40, the interval was open above in 29 % of blocks when the radius is near 40 mm, against 5 % with 50 |
| Far anchors at 40 mm (was 50) | At 50 mm nearly every placeable point is in a corner cell |
| The replay baseline is interleaved with the clustered policy's block; Tuesday's replay is a pilot | A baseline run on another day would differ from the policy by the day as well |
| Block fit and pooled fit named, and the block fit used for every comparison | Draft 0.1 fitted the clustered curve twice without saying which was used where |
| A rule for "larger" in prediction 2 | It had none |
| Pilots, the frozen rule, the operator's confirmation, the refused handoff as a void, the blind label | The tools now enforce them; the text follows the tools |
| Which demonstrations are replayed and from what pose | It was not stated |
| All 55 episodes trained on; the four training settings that differ from August | Checked against the training code (`training/settings.md`) |
| The guide's sentence quoted, and the clustered layout's numbers set beside it | The layout is the guide's structure, not its numbers |
| By-eye resets distinguished from overlay resets | They are different practices; only the first is measured here |
| What the budget can show, from simulation | So that the paper cannot claim more than the design could have found |
| Every fit is repeated with lapse values up to 0.50 and both are reported | The protocol's list stops at 0.10; in simulation a plateau of 0.65 then pulls the radius inward by 8 mm |
| Execution settings are read from the protocol file by the trial runner; an attempt cut short by a key is void; a record carries the arm's start pose | A setting that lives in a command line's default drifted once already; a doubled key press must not become a failure; the policy takes the joints as input |
| The rig is the light box of the VLA-REPLICA setup, the target a basket; the table is registered against a printed sheet | The table of August was taken down on 2026-10-05; its tag geometry describes a table that no longer exists |
| The clustered set said to be tighter than August's; the footprint's offset added to the clearance check | Measured on 2026-10-05 from the August dataset's first frames |
