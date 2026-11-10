# Glossary

Terms as this project uses them. Add a term here when it is first used.

| Term | Meaning |
|---|---|
| **ACT** | Action Chunking with Transformers: the imitation-learning policy used here as the fixed probe. At rollout it is deterministic: it outputs one action sequence for an observation |
| **Achieved control rate** | The rate at which the control loop really sent its steps in a trial: the steps sent after the first, over the time from the first step sent to the last (`loop_s` in the trial's record). The requested rate is the protocol's `control_hz`; the handoff gate and the first inference are outside the measurement |
| **Agreement (of the two scorings)** | The share of second-scoring items in which the second score is the first score, among the items the second scorer could tell. Reported with the counts it is made from and a 95 % interval |
| **Anchor** | A trial at a fixed distance, not chosen by a staircase. A *zero anchor* is at a demonstration's own measured start; a *far anchor* is at the block's far distance (40 mm) |
| **Audit (start-state audit)** | Measuring the cube's start position in every recorded episode from that episode's own first frame |
| **Baseline (camera baseline)** | The stored pixel positions of the AprilTag corners for a camera mount. A later check against it shows whether the camera or a tag has moved |
| **Blind code** | A label shown to the person scoring a trial in place of the condition's name |
| **Block** | A planned group of trials with one purpose (T1a, T1b, T2, T3) |
| **Block fit** | The success curve fitted to one condition's own threshold block only (T1a, T1b or T3). Every comparison and every prediction uses block fits |
| **By-eye reset** | Putting the cube back from a reference picture, with no overlay and no readout. Block V1 measures how far off such resets land |
| **Calibration (of the joints)** | LeRobot's record of where each joint's zero and two ends lie, in the servo's own counts. It is kept in a file and in the servos, and a joint value in degrees means a pose only together with it |
| **Cannot tell** | The second scorer's answer when a video plays and does not show whether the cube was released in the basket. Not a void, and neither an agreement nor a disagreement: counted on its own |
| **Clamp (per-step clamp)** | The limit on how far a joint's command may be from the joint's present position at one control step: 15 degrees here |
| **Clustered layout** | 5 demonstrations at each of 11 points: repeated episodes at a few locations, the structure LeRobot's guide suggests (its numbers are 10 at each location) |
| **Comparison rule** | The rule of prediction 2 for two radii: from the two block fits, "larger" at a posterior probability of at least 0.95 that the first exceeds the second, "smaller" at most 0.05, "not told apart" between. `analysis/simulation/comparison_rule.md` has what it can tell apart |
| **Condition** | What produces the motion in a trial: the clustered policy, the spread policy, or a replay |
| **Counted trial** | A trial run under the frozen protocol, which enters the analysis. The opposite of a pilot trial |
| **Creep (of the camera's picture)** | How far the front camera's picture stands from where its session solved the conversion from pixels to millimetres. A session reads it just before each attempt, writes it into the trial's record (`view_crept_mm`) and makes no attempt beyond the protocol's limit of 1.0 mm. The report gives its median and its largest value over the scored trials, as read |
| **Demonstration** | One teleoperated episode of the task, recorded for training |
| **Digest** | A SHA-256 hash: of one file's bytes, or of several files' hashes taken together. The second scoring uses two. The *logs' digest* names the draw's random streams beside the seed, so the draw is not known before the last trial is logged. An item's *videos' digest* is recorded at the draw and again with the score, so a score given on other videos is refused |
| **Distance to the nearest demonstration (`d`)** | From a trial's measured start position to the nearest measured start position among the demonstrations the tested policy was trained on |
| **First score, second score** | The first score of a trial is the one given at the rig when it was run, and the one the analysis uses. The second score is given later from the video alone, for the trials the second scoring draws |
| **Floor** | The chance of a success far from every demonstration: the level the success curve falls to. Fitted, not assumed to be zero |
| **Frozen (protocol)** | The protocol's state once nothing in it may change without being logged as a deviation: set before the first counted trial, with a git tag |
| **Gains (position-loop gains, P, I, D)** | Three numbers inside each servo that set how it pushes toward a commanded position: in proportion to the error (P), to the error added up over time (I), and to how fast the error changes (D). LeRobot writes P = 16, I = 0, D = 32 into every follower servo each time it connects |
| **Go/no-go** | Tuesday night's test of whether there is a paper to this deadline: the clustered policy at nine exact demonstration starts, pass line 8 of 9 |
| **Group (of a draw)** | The scored trials of one condition in one block: T1a's clustered policy, T2's spread policy, and so on. The second scoring draws one trial in five from each group, rounded up |
| **Handoff gate** | The earlier project's check before a policy or a replay gets the motors: its first command must be close to where the arm is (8 degrees per joint for a policy, 5 for a replay) |
| **Held-out cell** | A grid cell in which no demonstration starts (A1, B2, D2, E3) |
| **Horn (servo horn)** | The toothed disc on a servo's shaft that the next part of the arm screws to |
| **Item** | One drawn trial as the second scorer meets it: a neutral code (`item-07`) and copies of its videos under that code. Without the draw's key, the code and the item's place in the list say nothing about the trial |
| **Key (of a draw)** | The file that says which trial each item is (`key/key.csv`). Nobody who scores opens it before every item has a score |
| **Lapse** | The share of failures that have nothing to do with distance; it lowers the plateau of the success curve |
| **Largest gap** | Of a demonstration layout: the farthest any point of the trained cells is from its nearest demonstration (clustered 35.4 mm, spread 17.2 mm) |
| **Lead** | What an older record suggests a fact may be. It says what to look for at the source and is never entered as a value. The by-hand numbers file calls it a hint |
| **Lookup prediction** | The success expected if a policy did no more than work within its tolerance radius of the nearest demonstration |
| **Named (trial)** | A trial that began with the basket more than 2 mm, or the wrist camera more than 2 px, from its place, as the scene audit read it from the trial's first frames. The protocol has such trials named in the report; they stand as scored |
| **Nominal** | As asked for, or as printed, as opposed to measured: a tag printed at a nominal 60 mm, whatever a ruler then reads; a frame whose nominal time stamp is 1/30 s after the one before, whenever it was really captured |
| **Open on a side (interval)** | An interval the data do not close on that side: the posterior at that end of the grid is at least half of what it is at the interval's bound. Reported as "at most" or "at least" |
| **Outline (of a tag)** | The tag's four sides as measured from the picture: each a straight line through sub-pixel edge positions. The table registration fits the table frame through outlines, because the tag detector's own corners sit about half a pixel inside them at the rig's view |
| **Part (of a block)** | One of the pieces a threshold block is made of: its zero anchors, its far anchors, or one of its staircases |
| **Pilot trial** | A trial that tries the procedure before the protocol is frozen. Logged to a file named `pilot_*` and never analysed with the counted trials |
| **Planner** | The function that names the next trial from the protocol and the log |
| **Plateau** | The success rate at a demonstration's own start: 1 minus the lapse |
| **Plateau check** | Every fit made a second time with a longer list of lapse values (up to 0.50), reported beside the first. If the two radii differ by more than 5 mm, the condition's plateau is in doubt and its radius is not established |
| **Pooled fit** | The success curve fitted to all of a policy's counted trials: its threshold block and its trials of the workspace test. Reported beside the block fit as a check |
| **Psychometric function** | A smooth curve of success probability against a stimulus level, here distance; the term comes from perception research |
| **Region (of a placement)** | Whether the cube was in a trained cell or in a held-out cell: by the grid cell of its measured position, not by where the plan asked for it |
| **Register (servo register)** | A numbered cell of a servo's memory that holds one setting or one reading: a gain, the goal position, the present position. Programs read and write registers over the arm's serial line |
| **Registration sheet** | A printed Letter page with the grid's outline and six 36 mm tags at known places, seated once where the grid should be; the two floor tags are measured against it and it is taken away |
| **Rehearsal (of the second scoring)** | A draw made from pilot trials to try the tool before the counted trials exist. Its record and its comparison are marked as a rehearsal, and it is never reported |
| **Repeated frame** | A recorded frame whose picture is the frame before's again although the arm was moving: what a frame the camera did not deliver looks like in the recording. `scripts/rig/torn_frames.py` counts them among the frames the video codes by themselves; a measurement, not a verdict |
| **Replay** | Playing back one demonstration's recorded actions with no policy in the loop |
| **Reset** | Putting the scene in its starting state before a trial |
| **Reset distribution** | Where resets fall, and how accurately, across the trials of an evaluation |
| **Rest picture (of the wrist camera)** | What the wrist camera sees with the arm at rest in its start pose: the white floor, and its own two jaws at the bottom. The one picture of that camera in which its tone state can be read |
| **Role code** | What stands for a person in every record: `op1` for the operator, `sc1` or `sc2` for a scorer. Never a name or initials: the review is double blind |
| **Round** | In recording: 11 episodes, one per trained cell, in shuffled order. Each dataset is 5 rounds |
| **Scale factor** | The measured length of a true 50 mm displacement divided by 50: how long the instrument reads distances (expected a few percent long) |
| **Scene audit** | Reading, from the first and last frames of every trial's two videos, where the basket stood against its reference picture and where the wrist camera stood in its mount (`scripts/rig/scene_audit.py`, which writes `data/audit/scene_<log>.json`). It says which trials began with either out of place, and in which the arm moved the basket |
| **Second scoring** | Scoring a part of the counted trials again from their videos alone, by a person who is shown neither the condition nor the distance. One trial in five of each group is drawn after the last trial, by a seed fixed beforehand and the logs' own digest; the paper reports in how many the two scores agree. The first score stays the trial's outcome |
| **Servo variant** | Which version of a servo model a servo is: its supply voltage and its gear ratio. It is printed on the label; the software sees only the model |
| **Set aside (a recorded set)** | To rename a recorded dataset's folder and its placement log together, so that the set can be recorded anew. Nothing is deleted (`scripts/rig/set_aside.py`) |
| **Shift (of a plateau check)** | The radius of a fit made again with the longer list of lapse values, minus the radius that is reported. A shift larger in size than 5 mm puts the plateau in doubt |
| **Single-frame wander** | How far the table frame solved from one frame's floor tags lies from the registered frame, over the grid: what the detector's noise is worth |
| **Sitting** | One uninterrupted stretch of recording for one dataset. Four sittings alternate the two datasets |
| **Slip of the key** | An attempt that a key ended within a second of its start. It tested nothing: the trial is void and asked again |
| **Slot** | One place a block has for a scored trial. A void trial fills no slot: the slot is asked again with the same target |
| **Spread layout** | 55 demonstrations at 55 distinct, evenly scattered positions |
| **Staircase** | An adaptive rule: after a success the next trial is harder, after a failure easier, so trials gather near the 50 % point |
| **Stall rule** | The earlier project's rule that ends an attempt when the arm is blocked: a joint far behind its command while nothing has moved for about a second |
| **Start position** | Where the front camera measures the cube to be at the start of an episode or trial, in table millimetres (the centre of its visible outline, a few millimetres from the centre of its footprint; the same for every measurement) |
| **Stored conversion (table view)** | The one tie between the front camera's pixels and table millimetres, solved from many frames and kept in `data/instrument/table_view.json`. Every measurement goes through it |
| **Substrate** | The workshop's word for everything beneath the policy: body, actuators, controllers, action and observation spaces, data pipeline, evaluation protocol |
| **Table frame** | The coordinate frame on the table, in millimetres, with its origin at the grid's corner, x along columns A to E and y along rows 1 to 3; fixed to the table by two AprilTags |
| **Temporal ensembling** | ACT's way of smoothing actions at rollout: a new action sequence is predicted at every step and the overlapping predictions for the present step are averaged, with weights set by a coefficient (0.01 here) |
| **Test region** | Where a trial's cube may be placed: the grid less a 5 mm margin |
| **Threshold block** | A block that locates one condition's tolerance radius: zero anchors, staircases, and in T1a far anchors |
| **Tolerance radius (`r50`)** | The distance to the nearest demonstration at which success is halfway between its plateau and its floor |
| **Tone state (of a camera's picture)** | Which of two ways a camera's stream is handed to the programs on this computer: as the camera gives it (the usual state), or with its tone range stretched, 16 to 0 and 235 to 255 (whites up, blacks down, the contrast higher). Seen on both cameras on 2026-10-06, at an opening or after a USB transfer error; a policy trained in one sees another picture in the other |
| **Torn frame** | A video frame whose picture is damaged in a band of its rows while the other rows are right. On this rig's cameras the band runs from some row to the bottom of the picture or lies at its top, starts and ends on the 8-row grid of the camera's own JPEG compression, and is shifted sideways, has a colour cast or is too bright; often the same damaged picture then comes once to four times more. The video file is valid, so only a look at the pixels finds it (`scripts/rig/torn_frames.py`) |
| **Training run (run)** | One training of a policy on one dataset with one seed. A trial's record names the run and the training step of the checkpoint it used (`run@step`) |
| **Transfer error (USB)** | A piece of a camera's video that did not arrive whole over its USB link, as the system's log records it. Each one is a frame the system drops; a link in good order has none (`scripts/rig/usb_errors.py`) |
| **Unclosed set** | A recorded dataset that a session left without closing (the terminal closed, the process killed): its own count says more episodes than its tables hold. The recorder refuses to append to it (`does not add up`) |
| **Vendor default** | The value a setting has when a part leaves its maker, before any software changes it |
| **View (of a trial)** | One camera's video of a trial: `front`, from the fixed camera, or `wrist`, from the camera on the gripper. The second scoring shows both side by side |
| **Void trial** | A trial that could not test the policy (placement out of tolerance, a camera or motor fault, interference, the handoff gate refusing three times). Kept in the log with its reason, and asked again |
| **Watched again** | How many times the second scorer had an item's whole video shown again after it had ended, in the sitting its score was recorded in. Called so because "replay" is the name of a condition here |
| **Width** | The scale of the fitted success curve: a small width is a cliff, a large one a slope |
| **Workspace test** | Block T2: 26 placements, 13 drawn uniformly over the trained cells and 13 over the held-out cells, each tested with both policies |
