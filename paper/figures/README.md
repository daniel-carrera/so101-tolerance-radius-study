# Figure plan

Four figures in the main text, each drawn from data by `scripts/make_figures.py` into this folder under the file
name given here. Figure 1 is drawn; the other three wait for their data. Until a file exists, `main.tex` prints a
framed box of the planned height in its place (`\figureorbox`), so the page budget already counts the figures.

```bash
.venv/bin/python scripts/make_figures.py                 # every figure whose input is there
.venv/bin/python scripts/make_figures.py --only 3 4 --results analysis/report-draft/results.json
.venv/bin/python scripts/make_figures.py --only 2 --planned --png "$TMPDIR/figures"
```

- A figure whose input is missing is skipped, with a line that names the file it waits for. That is not an error.
- **A draft never comes here.** When the results file says `draft`, Figures 3 and 4 carry the word DRAFT and go to
  `analysis/report-draft/figures/`; naming a folder under `paper/` for them is refused.
- `--planned` also draws Figure 2 from the planned positions, with PLANNED across it, into `analysis/layouts/`.
- `--png DIR` also saves each figure as a picture at 300 dots per inch, for looking at. Look at every figure after
  it is drawn: the tests cannot see a label that sits on a mark.
- The code is `src/tolerance_radius/figures/`; a test keeps the file of Figure 1 in this folder equal to what the
  code draws from its data today, so a change of style means drawing it again.

## Rules for every figure

- **Drawn by a script from the data files named below**, never by hand, and never from numbers copied out of the
  paper. A number that appears both in a figure and in the text (a radius, a count) must come from the same analysis
  run that wrote `paper/generated/numbers.tex`.
- **One column wide: 88 mm.** Height as given (the box in the paper has the same height). Vector PDF. Text no smaller
  than 7 pt at that size. No title inside the figure: the caption is in the `.tex` file.
- **File names** use letters, digits and hyphens only. `check_paper.py` refuses anything else.
- **Readable without colour.** Two colours at most for the two conditions, checked as a pair for colour-blind
  readers, and a marker shape or line style that repeats what the colour says.
- **Clean.** No file path, user name, hub name, window title or date in the picture or in the file's metadata.
  `check_paper.py` searches the bytes of every file in this folder for the forbidden strings; that catches plain-text
  metadata, not text inside compressed streams, so look at the file's properties as well. The figures are saved
  with an empty information dictionary (no creator, no producer, no date), their fonts embedded as TrueType, and are
  refused if a text or the bytes hold a forbidden string, this machine's user name or its home directory; the tests
  also open the compressed streams and search those.
- **Axes in plain words with units**: "Distance to the nearest demonstration, d (mm)", not a variable name.
- **Say what is not there.** An untested range is shaded and labelled. No curve is drawn through data that cannot
  support one.

## Figure 1: the earlier evaluation, by distance

| | |
|---|---|
| File | `figures/fig1-earlier-evaluation.pdf`, 88 mm × 42 mm |
| Where | Introduction (`sections/intro.tex`, label `fig:earlier`) |
| Shows | Each of the 31 trials of the earlier evaluation as a mark: outcome (success above, failure below) against the distance from the cube's start position to the nearest demonstration the policy was trained on. Trials in cells with demonstrations and in held-out cells have different markers; the one trial whose position comes from a still has an open marker. The stretch with no trial, 5.6 to 31.0 mm, is shaded and labelled. Success rates of the two groups with Wilson intervals. **No fitted curve.** |
| Data | `analysis/pilot_august/trials_with_distance.csv`: `d_nearest_trained49_mm` (x), `success` (y), `kind` and `scene_id` (marker), `position_source` (open marker) |
| Exists already | `analysis/pilot_august/success_vs_distance.pdf`, drawn by `analysis/pilot_august/recover_positions.py`. It is wider than a column and has a six-entry legend: redraw it at column width with two groups rather than four. It bins by 10 mm; the one success at 40.01 mm sits a hundredth of a millimetre above a bin edge, so show the pooled far group (1 of 12), not two bins |
| Status | **Drawn** (`scripts/make_figures.py --only 1`), and committed. The marks of trials that fall together are piled up and never moved sideways; the legend sits in the rate band, clear of the shaded stretch |
| The one sentence | Every trial of the earlier evaluation started within 5.6 mm of a trained demonstration or at least 31 mm from one, so its two success rates say nothing about the distances in between |

## Figure 2: the two layouts on the grid

| | |
|---|---|
| File | `figures/fig2-layouts.pdf`, 88 mm × 32 mm |
| Where | Method (`sections/method.tex`, label `fig:layouts`) |
| Shows | The 5 × 3 grid twice, side by side: on the left the measured start positions of the clustered set, on the right those of the spread set. Held-out cells hatched. Cell names along the edges. A scale bar in millimetres. The cube's footprint drawn once to scale, so the reader sees how 12 mm of spacing compares with a 30 mm cube |
| Data | `data/audit/clustered.json` and `data/audit/spread.json` (each demonstration's `measured` position, `cell` and `trained` flag); the grid from `config/protocol.toml`, `[table]`. Planned positions (`config/layouts/`) are not drawn: the figure shows what was recorded |
| Status | Waits for the recording and the start-state audit. Proven on synthetic audits. A picture from the planned positions, marked PLANNED, is in `analysis/layouts/fig2-layouts-planned.pdf` (`--planned`): it is not the paper's |
| The one sentence | Both datasets put five demonstrations in each of the same eleven cells; they differ only in whether the five start at one point or at five |

If the four pages run over, this is the first figure to move to the appendix: Figure 4 also shows both sets of
demonstrations.

## Figure 3: the response curves of the clustered policy and of the replay

| | |
|---|---|
| File | `figures/fig3-response-curves.pdf`, 88 mm × 46 mm |
| Where | Results (`sections/results.tex`, label `fig:curves`) |
| Shows | Success against `d` for two conditions on one axis, **each from its block fit, with the trials of its threshold block and no others** (T1a for the clustered policy, T1b for the replay: what the captions say). For each: every such trial as a mark at its measured `d` (successes above, failures below, spread vertically only so that marks do not hide one another); success rates in 10 mm bins with Wilson 95 % intervals, each drawn at the mean distance of its bin's trials; the posterior-mean curve of the block fit with its 95 % band, from the condition's nearest trial to its farthest; in a row above the axis, the 95 % interval of the tolerance radius as a bar with the posterior median marked on it, and a vertical line from that mark up to the curve. Staircase trials and anchor trials are not told apart |
| Data | `analysis/report/results.json` and nothing else from the trials (`protocol/data-contracts.md`): `conditions.act_clustered` and `conditions.replay`, their `trials` (`block`, `d_mm`, `success`) and `block_fit` (the radius, its interval, `open_below`, `open_above`, the `curve`); `distance_scale.applied` for the axis label. The radii are those of `\radiusClusteredThreshold` and `\radiusReplay`. The rates are counted by the figure from the trials it draws; the same counting must give the file's own `bins` for all of a condition's trials, or the figure is refused |
| Status | Waits for blocks T1a and T1b. Proven on synthetic results (`tests/fixtures/results/`) |
| The one sentence | The distance at which the policy stops working and the distance at which a blind replay stops working, each with its interval, on one axis: the gap between them is what the policy adds to the hardware, or fails to add |

If an interval is open on a side (`open_below`, `open_above`), the bar runs off that end of the axis in an arrow,
and no median is marked: the trials do not locate the radius.
If distances are divided by the scale factor (the method's IF block), the axis shows the corrected distances and its
label says "corrected for scale"; otherwise it says "as read".
A stretch that neither condition tested is shaded and labelled "not tested": between trials when longer than 15 mm,
and beyond the farthest trial when the axis runs on for more than 5 mm (it does only when an end of an interval lies
out there). No curve is drawn across a stretch of more than 15 mm without a trial of its own condition.
A pooled-fit version (the clustered policy's pooled fit with all its trials) can be drawn by the code
(`response_curves.figure(results, fits="pooled")`); the command does not, because the paper's figure is the block fit.

## Figure 4: the workspace test and the lookup prediction

| | |
|---|---|
| File | `figures/fig4-workspace.pdf`, 88 mm × 40 mm |
| Where | Results (`sections/results.tex`, label `fig:workspace`) |
| Shows | The grid twice, one panel per policy. In each panel: that policy's demonstrations; a disc of radius r50 around each (the clustered policy's radius, from its block fit, which has not seen the workspace trials), drawn as one tinted area with one outline; the shared placements, however many there are and in whichever cells (13 in trained and 13 in held-out cells as the protocol stands since draft 0.3; a file with 20 in trained cells, the design before it, is drawn as well), filled for a success and open for a failure. Below each panel, the observed successes and failures and the lookup prediction with its 95 % predictive interval; when the placements lie in two regions, a line for each region as well (observed, the expected count, and its interval when the results give one). A last line says what the marks and the discs are. The cells are not named: the paper never speaks of a cell by its name, and the room pays for that line |
| Data | `analysis/report/results.json` and nothing else from the trials: `workspace` (`pairs`, `success`, `lookup`), `demonstrations`, `conditions.act_clustered.block_fit` (the radius of `\radiusClusteredThreshold`), `distance_scale`; the grid from `config/protocol.toml`. Used when the file has them, done without when it does not: a pair's `region`, `workspace.by_region` (`k`, `n`, `lookup_expected`, `lookup_lo`, `lookup_hi`), each policy's own position in a pair, `distance_scale.divisor` |
| Status | Waits for block T2, and for T1a. Proven on synthetic results |
| The one sentence | At the same placements, each policy's successes are set against discs of one radius around its own demonstrations: where the lookup prediction holds and where it does not |

The discs are a picture of the prediction, not the prediction: that is the fitted curve at each placement's distance.
The caption says so.

If the clustered policy's interval is open on a side, the paper reports no radius, and the discs follow it: they have
the radius of the end the trials do close (the smallest radius the trials allow when the interval is open above, the
largest when it is open below), a dashed outline, and the last line says which. Open on both sides: no disc.
Positions are as the instrument reads them while distances may be corrected for scale, so a disc is drawn with the
radius times the scale factor: a placement is then inside a disc exactly when its distance is within the radius.
If the counts by region do not fit (more than two regions), the totals stand alone and the command says so.

## Not planned for the main text

Candidates for the appendix if they help a reader check the work: the staircases' requested distance against trial
number, by condition; the distribution of by-eye reset errors (block V1); the scale check's measured against true
displacement (block V0).

## Figure 5 (appendix): the placement overlay

`fig5-overlay-example.png` is not drawn by `make_figures.py`. It is a picture: `scripts/make_overlay_example.py` renders
the session's own overlay (`placement_phase.render`) on the first frame of a counted trial's front video, once against a
target chosen for the illustration (25 mm from the cube) and once against the trial's planned target. It needs the
rig's interpreter (OpenCV), the trial's video and the frame its session started on, so it runs only where the trials
were run. The picture shows the floor, the tags, the arm and the basket, and no person, screen or room.
