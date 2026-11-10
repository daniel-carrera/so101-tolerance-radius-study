# How far from its demonstrations does an imitation policy still work?

An evaluation study on a low-cost SO-101 robot arm, run in October 2026. Author: Daniel Carrera.

The task is pick-and-place: the arm picks up a 30 mm cube from a table and drops it in a basket. The policy is ACT,
trained with [LeRobot](https://github.com/huggingface/lerobot). The question is how success changes as the cube
starts farther from where it started in the demonstrations, and how the arm itself behaves when the demonstrations
are simply played back with no policy.

The paper is in this repository: [paper/paper.pdf](paper/paper.pdf), *Measuring the Reset Distribution: A Tolerance
Radius, Referenced to Open-Loop Replay*. It was submitted to the CoRL 2026 workshop *Everything Beneath the Policy*
and was not accepted. The PDF is the submitted version, which is anonymous because the review was double blind.

The [trial map](https://daniel-carrera.github.io/so101-tolerance-radius-study/) shows every counted trial on the table
grid, with its outcome.

## The study

The table carries a grid of 15 cells, 5 by 3, each 50 mm square. Two AprilTags on the table fix its frame, and a front
camera measures where the cube is in millimetres. No demonstration starts in four of the cells (A1, B2, D2 and E3), so
the policy can be tested in places it was never shown.

Two sets of 55 demonstrations were recorded by teleoperation with a leader arm. In the **clustered** set, 5
demonstrations start at the centre of each of 11 cells. In the **spread** set, all 55 start at different positions.
One ACT policy was trained on each set with identical settings, and both were evaluated at the same training step.

The evaluation protocol was frozen before the first counted trial. The trials were run on 8 October 2026. Every reset
put the cube within 2 mm of its planned position, as measured by the camera. Each trial was announced by a code
rather than its condition, and the operator scored it at the rig. A replay can often be told from a policy by the way
the arm moves, so that blinding is partial.

| Condition | Counted trials |
|---|---|
| Open-loop replay of a recorded demonstration, no policy | 22 |
| Policy trained on the clustered set | 60 |
| Policy trained on the spread set | 26 |
| **Total** | **108** (none void) |

24 of the 108 trials, drawn at random, were scored a second time from video, with the condition and the distance
hidden. The two scores agreed in all 24. The same person gave both scores.

## What was found

| Distance from the nearest demonstration | Clustered policy | Replay |
|---|---|---|
| Within 20 mm | 17 of 21 | 13 of 18 |
| Beyond 20 mm | 20 of 39 | 0 of 4 |

- **The replay works only close to where it was recorded.** Its tolerance radius, the distance at which success falls
  halfway, is 16.5 mm (95 % interval 13.0 to 21.5 mm).
- **The replay is reliable at a demonstration's own start.** In the counted trials it succeeded in 4 of 4 there. In
  the replay pilot the day before, it succeeded in 10 of 11; the eleventh attempt was cut short by an accidental key
  press.
- **The policy succeeds farther out than the replay, but distance is not the whole story.** Inside the convex hull of
  its demonstrations the clustered policy succeeded in 28 of 36 trials, and outside it in 9 of 24. Its fitted radius
  is 39.0 mm, but a robustness check fixed in advance moved one of its two fits past the set limit. So the paper
  reports the policy's radius as not established.
- **The two layouts were not told apart.** At the same 26 test placements, the spread policy succeeded at 22 and the
  clustered policy at 16. The exact McNemar test gives p = 0.11, which does not establish a difference.
- **Resets by eye are not good enough for this kind of test.** Placed by eye from a reference picture, the cube landed
  a median of 15.9 mm from its target over 20 tries. That is close to the replay's whole tolerance radius.

Every figure above comes from the trial logs in `data/trials/` and the analysis in `analysis/report/report.md`.

## The safety layer at the arm

The arm only moves through a set of gates around LeRobot's control loop. A preflight check runs before torque is
enabled and refuses an arm whose stored goal position is stale. An alignment gate runs before every control loop,
so the follower is never commanded to jump to a far-away pose. A handoff gate lets a policy take over only when its
first command is within 8 degrees of the arm's pose on every joint, or 5 degrees for a replay. Every step is clamped,
so no joint is commanded more than 15 degrees from where it is. A stall rule ends an attempt when the arm is held back
against its commands for about a second, and every attempt has a 45-second limit. When a session ends, the motors keep
their torque and hold the present position, so the arm does not drop. In this repository the trial runner is
[src/tolerance_radius/rig/trial_session.py](src/tolerance_radius/rig/trial_session.py). Every tool that can command a
motor has a dry-run mode, and the tests run each one against a fake arm.

The gates themselves live in a separate project that wraps LeRobot's own functions. That project is not published
here. This repository pins the commit it was tested against, and LeRobot 0.6.1 at commit `6adf51511b`. LeRobot itself
is not copied or changed here.

## How the data were collected

- **Table.** The table's frame was registered from the two AprilTags before each session. The camera's conversion to
  millimetres was checked against a printed scale.
- **Demonstrations.** For each episode the recorder showed where the cube should go. The operator placed it until the
  camera read it within tolerance, then drove the follower with the leader arm. The planned positions are in
  `config/layouts/`, and the recorder's placement logs are in `data/recording/`.
- **Training.** Both policies were trained on Colab with the notebook in `training/`, for 80,000 steps each.
  `training/settings.md` lists every setting and where it was read.
- **Trials.** A planner chose each trial's placement, partly by adaptive staircases on the distance. The operator
  placed the cube with the camera's guidance, the arm ran through the gates, and the score went into an append-only
  log, one JSON line per trial.

## Reproducing the analysis

The analysis needs Python 3.12 and NumPy. The figures need Matplotlib, and the tests also use SciPy and pytest.

```bash
python3.12 -m venv .venv
.venv/bin/pip install "numpy>=2.2" scipy matplotlib pytest
.venv/bin/python -m pytest
.venv/bin/python scripts/make_report.py
.venv/bin/python scripts/make_figures.py
```

`make_report.py` reads the trial logs, the start-state audits and the instrument checks, and writes the report, the
results file and the number macros that the paper prints. Run from this repository, it reproduces the committed report
with one exception. The count of deviations from the protocol is read from a decision log that is not published, so
that count prints as unknown. The paper's appendix lists the three deviations. The tests need no hardware and no
network.

The full video recordings, the two datasets and the trained checkpoints are not in this repository because of their
size. They are available on request.

## Layout

| Path | What it holds |
|---|---|
| `paper/` | The paper: [paper.pdf](paper/paper.pdf), its LaTeX sources, figures, and the number macros the analysis writes |
| `protocol/` | The evaluation protocol, the data contracts for every record, and a glossary |
| `config/` | The protocol's parameters, the table's geometry and the two demonstration layouts |
| `src/tolerance_radius/` | The package: geometry, layouts, the trial planner, the statistics, the report and the figures |
| `src/tolerance_radius/rig/` | The tools that run at the arm: camera measurement, recording, trial sessions, audits |
| `scripts/` | Thin command-line entry points; `scripts/rig/` holds the ones used at the arm |
| `tests/` | Hardware-free tests, including full trial sessions on a rig made of fakes |
| `training/` | The Colab training notebook, its tests, and the training settings |
| `analysis/` | The procedure's simulations, the analysis of an earlier evaluation in August, and the counted report |
| `data/` | The trial logs, the start-state audits, the placement logs, the instrument checks and the second scoring |
| `docs/` | The trial-map page and its data and clips, served by GitHub Pages |

Comments in the code and in the paper's sources sometimes refer to a lab journal, a decision log and runbooks for the
rig. Those notes are not part of this repository.

## Trial map

The trial map is an interactive page that draws every counted trial where the cube started on the grid. It also shows
where the demonstrations started and whether each trial succeeded. It is at
<https://daniel-carrera.github.io/so101-tolerance-radius-study/>, served by GitHub Pages from `docs/`. A trial opens
straight from the address, for example
<https://daniel-carrera.github.io/so101-tolerance-radius-study/#T1a-025>. The page also works offline: open
`docs/index.html` in a browser.

The page reads `docs/data/trials.json`. [scripts/export_data.py](scripts/export_data.py) writes that file from the
trial log, the demonstration audits, the protocol, the table's tag positions and each session's camera view
(`data/camera_views.json`). Every trial has its front-camera clip in `docs/assets/trials/`, and the page can draw the
grid over it. `docs/assets/` also holds a clip of the reset overlay, and photos and a clip of the setup. These clips
are the only videos published; the full recordings, with the wrist camera, are not.

## License

MIT, in [LICENSE](LICENSE). The paper's PDF is the submitted version.
