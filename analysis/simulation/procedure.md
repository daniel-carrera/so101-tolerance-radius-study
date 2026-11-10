# Procedure simulation: how well a block of trials estimates the tolerance radius

**Every number in this report is simulated, or computed exactly from stated assumptions. Nothing here is a
measurement.**

Written by `python -m tolerance_radius.simulate`. Seed 20261005; 300 simulated blocks for every row. Protocol file
`protocol.toml`: version 1.0, status frozen, SHA-256 `78fd2bfa905d`. The rows behind the tables are in `procedure.csv`
(sections 1 to 5), `floor_check.csv` (section 6) and `paired_power.csv` (section 7).

## The question

Before any trial is run: with the trial budget of the protocol, how well is the tolerance radius `r50` estimated, and
are the trials placed the right way? Where the budget cannot pin the radius down, this report says so with the
numbers, so that the protocol can be changed before it is frozen.

## What was simulated

A block of trials is played against a success curve that is assumed true. Each requested distance comes from the
protocol's own staircase rule (`staircase.next_level`). The cube lands at the requested distance plus a placement
error (normal, standard deviation 1 mm, and not below 0). The trial succeeds with the probability the assumed curve
gives at the distance achieved. The block is then fitted as the analysis will fit the real trials
(`psychometric.fit_threshold`, on the achieved distances), and the fit is compared with the truth.

- **The fit.** A posterior on the protocol's grid: `r50` from 0 to 80 mm in steps of 0.5 mm; `width` at 30 values from
  0.5 to 20 mm, evenly spaced in its logarithm; lapse values 0, 0.02, 0.05 and 0.1, equally weighted; floor values 0,
  0.05 and 0.1, equally weighted. Reported: the posterior median of `r50` and its 95 % interval.
- **The truths.** Every combination of `r50` = 5, 10, 15, 20, 30 and 40 mm, width = 1, 3, 6 and 10 mm and lapse = 0
  and 0.05: 48 curves. A width of 1 mm is a cliff: success goes from 88 % to 12 % of its plateau within 4 mm. A width
  of 10 mm is a slope: the same fall takes 40 mm.

**The blocks**, as the protocol file states them (read from the file):

| Block | Trials | At 0 mm | Staircases | Far anchors | Largest distance (mm) |
|---|---|---|---|---|---|
| T1b | 22 | 4 | up from 5 mm (9 trials); down from 25 mm (9 trials) | none | 50 |
| T1a | 34 | 6 | up from 10 mm (12 trials); down from 40 mm (12 trials) | 4 at 40 mm | 50 |
| T3 | 24 | 4 | up from 5 mm (10 trials); down from 25 mm (10 trials) | none | 50 |

**The designs** compared for each block:

- **as written**: the block and the staircase steps of the protocol file (10 mm until a staircase first reverses, then
  5 mm);
- **fixed distances**: the same number of trials at fixed distances, equally spaced from 0 to the block's largest
  distance, one trial at each (the method of constant stimuli);
- **final step 2.5 mm**: as written, with a final step of 2.5 mm;
- **staircases 4 longer** and **4 shorter**: as written, with every staircase 4 trials longer or shorter, so 8 trials
  more or fewer in a block of two staircases;

**How exact the simulation is.** With 300 simulated blocks in a row, a share near 95 % is known to about 1.3 points
(one standard error), a share near 50 % to about 2.9 points, and an RMSE to about 4 % of its value. Numbers that pool
several truths are steadier. A difference smaller than that between two single rows is noise.

## Terms

- **Tolerance radius (`r50`)**: the distance to the nearest demonstration at which success is halfway between its
  plateau and its floor (half the plateau when there is no floor).
- **Width**: the scale of the curve's fall. Success goes from 88 % to 12 % of the way down over four widths.
- **Lapse**: the share of failures that have nothing to do with distance. It lowers the plateau.
- **Floor**: the chance of a lucky success far from every demonstration. It is what is left of the curve far out.
- **Posterior median, 95 % interval**: the estimate and its range: the values of `r50` that leave half, and 2.5 % on
  each side, of the probability the trials give to each value.
- **Bias**: the mean of (estimate - true radius) over the simulated blocks.
- **RMSE**: root-mean-square error: the square root of the mean of (estimate - true radius) squared. It is the typical
  distance of the estimate from the truth, in millimetres.
- **Contains the truth**: the share of simulated blocks whose 95 % interval holds the true radius. Statistics calls
  this the coverage of the interval; it has nothing to do with coverage of the workspace.
- **Open on a side**: the interval reaches an end of the `r50` grid (0 mm below, the grid's largest value above): the
  trials do not bound the radius on that side. The rule: the posterior at the grid's end value is at least half of
  what it is at the interval's own bound on that side.
- **Anchor**: a trial at a fixed distance, not chosen by a staircase: at 0 mm, or far out.
- **Reversal**: a staircase trial whose outcome differs from the one before, so that the staircase turns.
- **Method of constant stimuli**: testing at distances fixed in advance, whatever the outcomes.
- **Replication**: one simulated block. A seed makes the random numbers, and so the whole report, repeatable.
- **Discordant pair**: a placement at which one policy succeeded and the other failed.
- **Exact McNemar test**: the test of a paired comparison: are the discordant pairs split more unevenly than a fair
  coin would split them?
- **Power**: the probability that a test detects a difference that is really there.
- **False-alarm rate**: the probability that a test reports a difference that is not there.

## 1. The design as written

Table 1 pools, for each block and each width of the true curve, the truths with that width (every radius and lapse
rate). In each cell the first number is the middle one of those truths and the number in brackets is the worst.

**Table 1.** The design as written, by block and by width of the true curve. Simulated.

| Block | Trials | Width (mm) | RMSE (mm) | Interval width (mm) | Contains the truth (%) | Open on a side (%) |
|---|---|---|---|---|---|---|
| T1b | 22 | 1 | 1.0 (1.9) | 6.3 (11.2) | 99 (97) | 0 (9) |
| T1b | 22 | 3 | 2.1 (2.7) | 12.6 (18.1) | 97 (96) | 1 (56) |
| T1b | 22 | 6 | 3.6 (4.4) | 18.6 (25.4) | 97 (94) | 8 (87) |
| T1b | 22 | 10 | 5.2 (6.5) | 23.6 (30.1) | 96 (93) | 23 (89) |
| T1a | 34 | 1 | 0.8 (1.0) | 4.2 (6.4) | 99 (98) | 0 (0) |
| T1a | 34 | 3 | 1.6 (2.0) | 8.1 (12.4) | 97 (96) | 0 (41) |
| T1a | 34 | 6 | 2.8 (3.4) | 13.3 (18.3) | 96 (93) | 1 (79) |
| T1a | 34 | 10 | 4.2 (5.4) | 18.4 (22.3) | 95 (91) | 14 (90) |
| T3 | 24 | 1 | 0.9 (1.1) | 5.3 (9.5) | 99 (98) | 0 (6) |
| T3 | 24 | 3 | 1.9 (2.4) | 11.3 (17.2) | 98 (97) | 1 (53) |
| T3 | 24 | 6 | 3.4 (4.3) | 17.3 (23.7) | 97 (94) | 5 (85) |
| T3 | 24 | 10 | 4.8 (6.0) | 21.7 (28.0) | 94 (92) | 17 (89) |

Tables 2 to 5 give the same quantities, and the bias, for each true radius, averaged over the lapse rates.

**Table 2.** Bias / RMSE of the estimate, in mm, by true radius (rows) and width of the true curve (columns).
Simulated.

| Block | True `r50` (mm) | width 1 | width 3 | width 6 | width 10 |
|---|---|---|---|---|---|
| T1b | 5 | 0.0 / 0.9 | 0.0 / 1.7 | 0.0 / 2.4 | +0.8 / 3.4 |
| T1b | 10 | -0.2 / 1.0 | -0.2 / 1.9 | -0.2 / 3.4 | 0.0 / 4.4 |
| T1b | 15 | -0.1 / 1.0 | -0.1 / 2.0 | -0.2 / 3.4 | -0.5 / 5.1 |
| T1b | 20 | -0.1 / 1.0 | -0.2 / 2.1 | -0.3 / 3.7 | -0.1 / 5.3 |
| T1b | 30 | -0.1 / 1.0 | -0.1 / 2.2 | +0.2 / 4.1 | -0.1 / 5.9 |
| T1b | 40 | -0.1 / 1.5 | -0.1 / 2.4 | +0.4 / 4.3 | -0.6 / 6.2 |
| T1a | 5 | 0.0 / 0.8 | 0.0 / 1.6 | -0.1 / 2.3 | +0.4 / 3.1 |
| T1a | 10 | -0.1 / 0.8 | -0.2 / 1.6 | -0.4 / 2.8 | -0.7 / 4.0 |
| T1a | 15 | -0.1 / 0.9 | -0.2 / 1.7 | -0.3 / 3.0 | -0.8 / 4.4 |
| T1a | 20 | -0.1 / 0.9 | -0.4 / 1.7 | -0.4 / 2.9 | -0.8 / 4.5 |
| T1a | 30 | -0.2 / 0.9 | -0.4 / 1.8 | -0.8 / 3.1 | -1.0 / 4.9 |
| T1a | 40 | -0.1 / 0.8 | -0.2 / 1.6 | -0.2 / 2.9 | -0.4 / 4.5 |
| T3 | 5 | 0.0 / 0.8 | -0.1 / 1.6 | 0.0 / 2.5 | +0.8 / 3.4 |
| T3 | 10 | -0.1 / 0.9 | -0.2 / 1.9 | -0.4 / 3.0 | 0.0 / 4.2 |
| T3 | 15 | -0.1 / 0.9 | -0.2 / 1.9 | -0.1 / 3.4 | -0.5 / 4.8 |
| T3 | 20 | -0.1 / 1.0 | -0.3 / 2.0 | -0.1 / 3.5 | -0.9 / 5.1 |
| T3 | 30 | -0.2 / 1.0 | -0.1 / 2.1 | -0.2 / 3.6 | -0.2 / 5.5 |
| T3 | 40 | -0.1 / 1.0 | 0.0 / 2.2 | +0.2 / 4.0 | -0.3 / 5.7 |

**Table 3.** Mean width of the 95 % interval, in mm, by true radius (rows) and width of the true curve (columns).
Simulated.

| Block | True `r50` (mm) | width 1 | width 3 | width 6 | width 10 |
|---|---|---|---|---|---|
| T1b | 5 | 5.3 | 9.8 | 13.4 | 15.4 |
| T1b | 10 | 5.7 | 11.3 | 16.4 | 19.7 |
| T1b | 15 | 6.5 | 11.6 | 18.4 | 22.4 |
| T1b | 20 | 6.5 | 13.1 | 19.3 | 24.8 |
| T1b | 30 | 7.7 | 14.9 | 21.9 | 26.3 |
| T1b | 40 | 8.8 | 16.5 | 24.0 | 28.9 |
| T1a | 5 | 3.8 | 7.9 | 10.5 | 12.8 |
| T1a | 10 | 4.5 | 7.9 | 12.8 | 15.9 |
| T1a | 15 | 4.4 | 8.3 | 12.8 | 17.9 |
| T1a | 20 | 4.7 | 8.4 | 13.2 | 19.2 |
| T1a | 30 | 4.8 | 8.4 | 14.5 | 20.0 |
| T1a | 40 | 5.3 | 11.1 | 17.1 | 21.7 |
| T3 | 5 | 4.6 | 9.4 | 12.7 | 14.7 |
| T3 | 10 | 5.0 | 10.3 | 15.6 | 19.1 |
| T3 | 15 | 5.9 | 10.7 | 17.0 | 21.2 |
| T3 | 20 | 5.6 | 11.8 | 18.3 | 22.4 |
| T3 | 30 | 6.0 | 13.2 | 19.4 | 25.9 |
| T3 | 40 | 7.8 | 15.3 | 22.2 | 27.3 |

**Table 4.** Share of blocks whose interval is open below / above, in %, by true radius (rows) and width of the true
curve (columns). Simulated.

| Block | True `r50` (mm) | width 1 | width 3 | width 6 | width 10 |
|---|---|---|---|---|---|
| T1b | 5 | 5 / 0 | 49 / 0 | 85 / 0 | 89 / 0 |
| T1b | 10 | 2 / 0 | 9 / 0 | 40 / 0 | 61 / 0 |
| T1b | 15 | 0 / 0 | 1 / 0 | 12 / 0 | 33 / 0 |
| T1b | 20 | 0 / 0 | 0 / 0 | 2 / 0 | 11 / 1 |
| T1b | 30 | 0 / 0 | 0 / 0 | 0 / 3 | 0 / 5 |
| T1b | 40 | 0 / 1 | 0 / 1 | 0 / 7 | 0 / 15 |
| T1a | 5 | 0 / 0 | 34 / 0 | 74 / 0 | 86 / 0 |
| T1a | 10 | 0 / 0 | 1 / 0 | 27 / 0 | 60 / 0 |
| T1a | 15 | 0 / 0 | 0 / 0 | 4 / 0 | 22 / 0 |
| T1a | 20 | 0 / 0 | 0 / 0 | 0 / 0 | 9 / 0 |
| T1a | 30 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| T1a | 40 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 1 |
| T3 | 5 | 3 / 0 | 45 / 0 | 82 / 0 | 86 / 0 |
| T3 | 10 | 0 / 0 | 6 / 0 | 37 / 0 | 62 / 0 |
| T3 | 15 | 0 / 0 | 1 / 0 | 9 / 0 | 28 / 0 |
| T3 | 20 | 0 / 0 | 0 / 0 | 2 / 0 | 10 / 0 |
| T3 | 30 | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 3 |
| T3 | 40 | 0 / 0 | 0 / 1 | 0 / 4 | 0 / 9 |

**Table 5.** Share of blocks whose interval contains the true radius, in %, by true radius (rows) and width of the
true curve (columns). Simulated.

| Block | True `r50` (mm) | width 1 | width 3 | width 6 | width 10 |
|---|---|---|---|---|---|
| T1b | 5 | 98 | 98 | 98 | 98 |
| T1b | 10 | 99 | 98 | 96 | 97 |
| T1b | 15 | 99 | 98 | 97 | 95 |
| T1b | 20 | 99 | 97 | 96 | 96 |
| T1b | 30 | 99 | 97 | 96 | 96 |
| T1b | 40 | 99 | 98 | 96 | 94 |
| T1a | 5 | 99 | 97 | 97 | 96 |
| T1a | 10 | 98 | 98 | 96 | 95 |
| T1a | 15 | 98 | 97 | 96 | 93 |
| T1a | 20 | 99 | 97 | 96 | 96 |
| T1a | 30 | 98 | 97 | 96 | 93 |
| T1a | 40 | 98 | 97 | 97 | 94 |
| T3 | 5 | 99 | 98 | 98 | 97 |
| T3 | 10 | 98 | 98 | 96 | 96 |
| T3 | 15 | 99 | 97 | 96 | 95 |
| T3 | 20 | 99 | 99 | 96 | 94 |
| T3 | 30 | 99 | 98 | 95 | 95 |
| T3 | 40 | 99 | 98 | 96 | 93 |

## 2. Where the trials cannot pin the radius down

A radius is hard to bound on a side when the fall of the curve does not lie inside the distances that are tested.
Table 6 sorts the truths by where their fall lies. Under way at 0 mm: `r50` is less than two widths, so success at 0
mm is already below 88 % of its plateau and the trials cannot show where the plateau ends. Running past the largest
distance: `r50` is within two widths of the largest distance the block tests, so success there is still above 12 % of
the plateau.

**Table 6.** The design as written, by where the fall of the true curve lies. Means over the truths of each kind.
Simulated.

| Block | The fall of the true curve is | Truths | RMSE (mm) | Interval width (mm) | Open below (%) | Open above (%) | Contains the truth (%) |
|---|---|---|---|---|---|---|---|
| T1b | inside the tested range | 32 | 2.5 | 13.7 | 3 | 1 | 97 |
| T1b | already under way at 0 mm (`r50` under two widths) | 12 | 3.4 | 16.2 | 60 | 0 | 97 |
| T1b | running past the largest distance (`r50` within two widths of it) | 4 | 5.2 | 26.5 | 0 | 11 | 95 |
| T1a | inside the tested range | 32 | 2.0 | 9.5 | 1 | 0 | 97 |
| T1a | already under way at 0 mm (`r50` under two widths) | 12 | 3.0 | 12.9 | 50 | 0 | 96 |
| T1a | running past the largest distance (`r50` within two widths of it) | 4 | 3.7 | 19.4 | 0 | 0 | 96 |
| T3 | inside the tested range | 32 | 2.3 | 12.4 | 2 | 0 | 97 |
| T3 | already under way at 0 mm (`r50` under two widths) | 12 | 3.2 | 15.4 | 57 | 0 | 97 |
| T3 | running past the largest distance (`r50` within two widths of it) | 4 | 4.8 | 24.8 | 0 | 6 | 95 |

## 3. Other ways to place the same trials

Two designs with the same number of trials as the design as written. In Table 7 the RMSE and the interval width are
given relative to the design as written: 1.00 is equal, 1.20 is 20 % larger. The ratios are averaged over the truths
on a logarithmic scale, so that no single truth dominates.

**Table 7.** Fixed distances and a finer final step, relative to the design as written. Simulated.

| Block | Design | Trials | RMSE, all truths | RMSE, width 1 | RMSE, width 3 | RMSE, width 6 | RMSE, width 10 | Interval width, all truths | Open on a side (%), design : as written | Contains the truth (%), design : as written |
|---|---|---|---|---|---|---|---|---|---|---|
| T1b | fixed distances | 22 | 1.37 | 1.66 | 1.44 | 1.29 | 1.14 | 1.10 | 31 : 18 | 96 : 97 |
| T1b | final step 2.5 mm | 22 | 1.02 | 0.94 | 1.04 | 1.04 | 1.06 | 1.04 | 19 : 18 | 97 : 97 |
| T1a | fixed distances | 34 | 1.37 | 1.66 | 1.45 | 1.31 | 1.11 | 1.20 | 24 : 13 | 96 : 97 |
| T1a | final step 2.5 mm | 34 | 0.99 | 0.92 | 1.01 | 1.05 | 1.01 | 1.01 | 13 : 13 | 96 : 97 |
| T3 | fixed distances | 24 | 1.40 | 1.75 | 1.49 | 1.29 | 1.15 | 1.14 | 30 : 16 | 96 : 97 |
| T3 | final step 2.5 mm | 24 | 1.04 | 0.98 | 1.09 | 1.07 | 1.04 | 1.05 | 18 : 16 | 97 : 97 |

## 4. What trials buy

Every staircase made 4 trials shorter, and 4 trials longer. Table 8 reads like Table 7.

**Table 8.** Shorter and longer staircases, relative to the design as written. Simulated.

| Block | Design | Trials | RMSE, all truths | RMSE, width 1 | RMSE, width 3 | RMSE, width 6 | RMSE, width 10 | Interval width, all truths | Open on a side (%), design : as written | Contains the truth (%), design : as written |
|---|---|---|---|---|---|---|---|---|---|---|
| T1b | staircases 4 shorter | 14 | 1.51 | 1.58 | 1.59 | 1.48 | 1.38 | 1.65 | 32 : 18 | 98 : 97 |
| T1b | staircases 4 longer | 30 | 0.80 | 0.76 | 0.79 | 0.82 | 0.84 | 0.76 | 14 : 18 | 97 : 97 |
| T1a | staircases 4 shorter | 26 | 1.28 | 1.39 | 1.29 | 1.26 | 1.17 | 1.28 | 16 : 13 | 96 : 97 |
| T1a | staircases 4 longer | 42 | 0.85 | 0.83 | 0.84 | 0.88 | 0.86 | 0.84 | 12 : 13 | 97 : 97 |
| T3 | staircases 4 shorter | 16 | 1.39 | 1.49 | 1.41 | 1.32 | 1.33 | 1.52 | 26 : 16 | 97 : 97 |
| T3 | staircases 4 longer | 32 | 0.82 | 0.79 | 0.81 | 0.82 | 0.85 | 0.78 | 14 : 16 | 97 : 97 |

## 5. A larger range for a block that stops short

Every block of the protocol goes out to the same largest distance: nothing to compare.

## 6. A floor in the model

The curves of sections 1 to 5 end at 0 far from the demonstrations. If a policy now and then succeeds far out by luck,
a model without a floor can explain that success only by a wider or a farther curve, and the radius moves. This check
plays truths with a floor and the same truths without one, and fits every simulated block twice: without a floor in
the model, and with floor values 0, 0.05 and 0.1 (equally weighted). The two fits see the same trials.

- **Truths:** floor = 0.08 and 0; `r50` = 15 and 30 mm; width = 1, 3 and 6 mm; lapse = 0.05.
- **Blocks:** T1a as written; T3 as written; T3 with 4 far anchors added; T3 with 4 staircase trials added. The far
  anchors added to a block that has none stand at its largest distance; the staircase trials added instead give the
  far anchors a fair comparison at an equal number of trials. T1b has no far anchors either and was not run in this
  check.

**Table 10.** The two fits of the same simulated blocks, by block and by true floor. Means over the 6 truths of each
floor. The last column is the posterior mean of the floor, averaged over the blocks (its prior mean is 0.050).
Simulated.

| Block | Trials | True floor | Fit | Bias (mm) | RMSE (mm) | Interval width (mm) | Contains the truth (%) | Floor estimate |
|---|---|---|---|---|---|---|---|---|
| T1a, as written | 34 | 0.08 | without a floor | +1.4 | 2.9 | 12.7 | 96 | - |
| T1a, as written | 34 | 0.08 | with floor values | +0.3 | 2.4 | 11.5 | 97 | 0.053 |
| T1a, as written | 34 | 0 | without a floor | 0.0 | 1.9 | 9.2 | 96 | - |
| T1a, as written | 34 | 0 | with floor values | -0.6 | 2.0 | 9.4 | 96 | 0.043 |
| T3, as written | 24 | 0.08 | without a floor | +1.4 | 3.1 | 15.0 | 95 | - |
| T3, as written | 24 | 0.08 | with floor values | +0.6 | 2.7 | 14.9 | 97 | 0.051 |
| T3, as written | 24 | 0 | without a floor | +0.1 | 2.2 | 12.9 | 97 | - |
| T3, as written | 24 | 0 | with floor values | -0.4 | 2.3 | 13.1 | 97 | 0.048 |
| T3, with 4 far anchors added | 28 | 0.08 | without a floor | +1.9 | 3.7 | 15.7 | 96 | - |
| T3, with 4 far anchors added | 28 | 0.08 | with floor values | +0.4 | 2.6 | 13.6 | 97 | 0.053 |
| T3, with 4 far anchors added | 28 | 0 | without a floor | 0.0 | 2.1 | 10.8 | 96 | - |
| T3, with 4 far anchors added | 28 | 0 | with floor values | -0.6 | 2.2 | 11.1 | 96 | 0.043 |
| T3, with 4 staircase trials added | 28 | 0.08 | without a floor | +1.3 | 2.8 | 13.0 | 96 | - |
| T3, with 4 staircase trials added | 28 | 0.08 | with floor values | +0.5 | 2.4 | 12.9 | 98 | 0.051 |
| T3, with 4 staircase trials added | 28 | 0 | without a floor | +0.2 | 2.0 | 11.1 | 98 | - |
| T3, with 4 staircase trials added | 28 | 0 | with floor values | -0.4 | 2.0 | 11.2 | 97 | 0.048 |

**Table 11.** The blocks as written, by width of the true curve: fit without a floor → fit with floor values.
Simulated.

| Block | Width (mm) | True floor 0.08: RMSE (mm) | True floor 0.08: bias (mm) | True floor 0.08: interval (mm) | True floor 0: RMSE (mm) | True floor 0: bias (mm) | True floor 0: interval (mm) |
|---|---|---|---|---|---|---|---|
| T1a | 1 | 1.9 → 1.2 | +1.0 → +0.1 | 9.2 → 7.1 | 0.8 → 0.9 | -0.1 → -0.2 | 4.7 → 5.0 |
| T1a | 3 | 2.8 → 2.2 | +1.5 → +0.3 | 12.5 → 11.3 | 1.7 → 1.8 | -0.1 → -0.5 | 8.9 → 9.1 |
| T1a | 6 | 4.0 → 3.7 | +1.8 → +0.4 | 16.4 → 16.0 | 3.1 → 3.3 | +0.1 → -0.9 | 14.0 → 14.2 |
| T3 | 1 | 2.0 → 1.6 | +0.8 → +0.3 | 9.3 → 9.1 | 1.0 → 1.1 | -0.1 → -0.3 | 6.6 → 7.0 |
| T3 | 3 | 2.8 → 2.5 | +1.4 → +0.6 | 15.0 → 14.9 | 1.9 → 1.9 | +0.2 → -0.4 | 12.9 → 13.0 |
| T3 | 6 | 4.6 → 4.1 | +2.1 → +1.1 | 20.6 → 20.7 | 3.7 → 3.8 | +0.3 → -0.6 | 19.2 → 19.3 |

## 7. The paired comparison

The workspace test places the cube at 26 positions, runs both policies at each, and compares them with the exact
McNemar test. Table 12 gives the probability that the test reaches p < 0.05 for assumed success rates of the two
policies, with outcomes independent of each other. These are exact sums over every possible table of outcomes, not
simulations. On the diagonal the two policies are equal: the entry is the test's false-alarm rate.

**Table 12.** Probability (%) that the exact McNemar test gives p < 0.05 with 26 pairs. Rows: the success rate of one
policy. Columns: that of the other. Computed.

|  | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 7 | 29 | 60 | 84 | 96 | 99 | 100 | 100 |
| 0.2 | 7 | 2 | 7 | 23 | 50 | 77 | 94 | 99 | 100 |
| 0.3 | 29 | 7 | 2 | 7 | 21 | 47 | 75 | 94 | 99 |
| 0.4 | 60 | 23 | 7 | 2 | 6 | 21 | 47 | 77 | 96 |
| 0.5 | 84 | 50 | 21 | 6 | 3 | 6 | 21 | 50 | 84 |
| 0.6 | 96 | 77 | 47 | 21 | 6 | 2 | 7 | 23 | 60 |
| 0.7 | 99 | 94 | 75 | 47 | 21 | 7 | 2 | 7 | 29 |
| 0.8 | 100 | 99 | 94 | 77 | 50 | 23 | 7 | 2 | 7 |
| 0.9 | 100 | 100 | 99 | 96 | 84 | 60 | 29 | 7 | 1 |

**Table 13.** The smallest success rate (in steps of 0.05) that the test tells from the lower one at least 80 % of the
time, by number of pairs. Computed.

| Lower rate | 22 pairs | 26 pairs | 30 pairs |
|---|---|---|---|
| 0.10 | 0.55 | 0.50 | 0.45 |
| 0.20 | 0.70 | 0.65 | 0.60 |
| 0.30 | 0.80 | 0.75 | 0.70 |
| 0.40 | 0.85 | 0.85 | 0.80 |
| 0.50 | 0.95 | 0.90 | 0.90 |
| 0.60 | 1.00 | 0.95 | 0.95 |
| 0.70 | not reached | 1.00 | 1.00 |

## Reading

- **T1b as written (22 trials).** A cliff (width 1 mm) is typically located with an RMSE of 1.0 mm and a 95 % interval
  6.3 mm wide. A slope of width 10 mm is located with an RMSE of 5.2 mm and an interval 23.6 mm wide.
- **T1a as written (34 trials).** A cliff (width 1 mm) is typically located with an RMSE of 0.8 mm and a 95 % interval
  4.2 mm wide. A slope of width 10 mm is located with an RMSE of 4.2 mm and an interval 18.4 mm wide.
- **T3 as written (24 trials).** A cliff (width 1 mm) is typically located with an RMSE of 0.9 mm and a 95 % interval
  5.3 mm wide. A slope of width 10 mm is located with an RMSE of 4.8 mm and an interval 21.7 mm wide.
- **Which way the estimate leans.** The mean error of the estimate (its bias) is -0.1 mm at width 1 mm and -0.3 mm at
  width 10 mm, averaged over the blocks, radii and lapse rates. The largest in any row is -1.9 mm (T1a, `r50` 30 mm,
  width 10 mm, lapse 0.05). A positive bias is an estimate beyond the true radius.
- **The interval holds what it promises.** Over the 144 rows of the design as written, the 95 % interval contained the
  true radius in 97 % of the simulated blocks on average and in 91 % in the lowest row (T1a, `r50` 15 mm, width 10 mm,
  lapse 0.05). No row is below 95 % by more than the simulation's own noise (3.8 points, three standard errors of one
  row).
- **What the trials cannot bound.** Where the fall of the true curve lies inside the tested range, the interval is
  open on a side in 2 % of the simulated blocks. Where the curve is already falling at 0 mm (`r50` under two widths),
  it is open below in 55 %: the estimate then reads as "at most" and not as a radius, although its interval still
  holds the truth (96 %). Where the fall runs past the largest distance tested, it is open above in 6 % and 23.5 mm
  wide on average. Table 6 has this block by block.
- **Staircases against fixed distances, same number of trials.** With fixed distances the RMSE is 38 % larger (from 13
  % larger at width 10 mm to 69 % larger at width 1 mm) and the interval is 15 % wider; the interval is open on a side
  in 29 % of blocks against 16 %.
- **A final step of 2.5 mm in place of 5 mm.** The RMSE is 2 % larger (from 5 % smaller at width 1 mm to 5 % larger at
  width 6 mm) and the interval is 3 % wider.
- **What trials buy.** T1b: 8 trials fewer make the RMSE 51 % larger and the interval 65 % wider; 8 more make the RMSE
  20 % smaller and the interval 24 % narrower. T1a: 8 trials fewer make the RMSE 28 % larger and the interval 28 %
  wider; 8 more make the RMSE 15 % smaller and the interval 16 % narrower. T3: 8 trials fewer make the RMSE 39 %
  larger and the interval 52 % wider; 8 more make the RMSE 18 % smaller and the interval 22 % narrower.
- **A floor that the model ignores.** With a true floor of 0.08, the fit without a floor is off by +1.4 mm on average,
  against +0.5 mm for the fit with floor values, and its RMSE is 22 % larger. Both intervals hold the truth about as
  often (96 % and 97 %): the fit without a floor pays with a radius pushed outward and a wider interval.
- **A floor in the model when there is none in the truth.** The interval is 3 % wider, the RMSE is 5 % larger, the
  estimate is off by -0.5 mm on average (against +0.1 mm without a floor in the model), and the interval holds the
  truth in 96 % of blocks (against 97 %).
- **The size of the floor cannot be read from 4 far anchors.** In T1a the posterior mean of the floor is 0.053 when
  the true floor is 0.08 and 0.043 when it is 0; its prior mean is 0.050. The two differ by less than half the true
  floor. That is arithmetic more than chance: 4 far anchors that all fail multiply the weight of each floor value by
  0.81 for a floor of 0.05 and 0.66 for a floor of 0.1, against 1 for no floor. The floor values make the radius
  robust to a lucky success; they do not measure how often luck strikes.
- **Far anchors for a block that has none (T3).** With floor values in the model, 4 far anchors give an RMSE 10 %
  larger and an interval 5 % wider than the same 4 trials spent on the staircases. With the far anchors the posterior
  mean of the floor is 0.053 when the true floor is 0.08 and 0.043 when it is 0 (without them: 0.051 and 0.048): they
  do not make the floor readable either. Without floor values in the model and with a floor in the truth, the far
  anchors make the RMSE 22 % larger than in the same block without them: a far anchor is where a lucky success does
  most harm to a model that cannot explain it.
- **The paired test with 26 pairs detects large differences only.** The success rate the other policy needs for the
  test to reach p < 0.05 at least 80 % of the time: from 0.30, 0.75; from 0.40, 0.85; from 0.50, 0.90. For the same
  lower rates, with 22 pairs: 0.80, 0.85, 0.95; with 30 pairs: 0.70, 0.80, 0.90. The false-alarm rate is at most 2.5
  %.

## Recommendation

**In short: the simulation supports the design as written; no parameter of the protocol file needs to change.** The
reasons follow.

The items below are written from the numbers above by fixed rules. A design counts as better or worse than the design
as written when its RMSE, pooled over the truths, differs by more than 5 %. A floor in the model counts as cheap when
it widens the interval by at most 10 % where the truth has no floor.

1. **Keep the staircases** (`staircase.rule = "1up1down"`). With the same trials at fixed, equally spaced distances
   the RMSE is 38 % larger and the interval 15 % wider, and the interval is open on a side in 29 % of blocks against
   16 %.
2. **Keep `staircase.step_final_mm = 5`.** With a final step of 2.5 mm the RMSE is 2 % larger and the interval 3 %
   wider: nothing to gain. `staircase.step_initial_mm` and the starting levels were not varied.
3. **Do not shorten a staircase, and if trials can be added, add them to T1b or T3 first** (`blocks.T1b.staircases`
   from 9 and 9 trials to 13 and 13; `blocks.T3.staircases` from 10 and 10 trials to 14 and 14). 4 trials more per
   staircase change the interval's width by: T1b: 24 % narrower; T1a: 16 % narrower; T3: 22 % narrower. 4 fewer: T1b:
   65 % wider; T1a: 28 % wider; T3: 52 % wider. The number of trials is the one parameter here that moves the result;
   whether the session has room for more is not something this simulation can weigh.
4. **Keep `analysis.floor_values = [0.00, 0.05, 0.10]`.** With a true floor of 0.08, leaving the floor out of the
   model makes the RMSE 22 % larger and shifts the estimate by +1.0 mm. With no floor in the truth, having it in the
   model makes the interval 3 % wider and the RMSE 5 % larger, and shifts the estimate by -0.6 mm. The floor is cheap
   insurance against one lucky success; it is not an estimate of how often luck strikes (see the reading). For a
   condition in which a lucky success far out cannot happen, it costs that little and buys nothing.
5. **Do not add far anchors to T1b and T3 for the estimate's sake.** In T3, 4 far anchors give an RMSE 10 % larger and
   an interval 5 % wider than 4 more staircase trials, and they do not make the floor readable. Spend spare trials on
   the staircases. Far anchors remain what they were meant to be: a direct count of successes far out.
6. **Do not run far anchors with a model that has no floor.** With a floor in the truth and none in the model, far
   anchors make the RMSE 22 % larger in T3. If `floor_values` is not adopted, the far anchors of the blocks that have
   them should be reported as counts only and left out of the fit.
7. **Keep the paired test at 26 pairs and write down now what it can show.** It tells 0.40 from 0.85 and 0.50 from
   0.90 at least 80 % of the time, and smaller differences mostly not. With 30 pairs it tells 0.40 from 0.80 and 0.50
   from 0.90: at most one step of 0.05 better. If the two policies turn out closer than that, the paper reports the
   counts and their intervals and says that the test could not have decided.
8. **Write the rule for an open interval into `protocol/protocol.md` before the freeze.** An equal-tailed interval never
   ends exactly on the last value of the grid, so "the interval reaches the edge" needs a rule. The one used here and
   in the analysis code (`psychometric.EDGE_RATIO = 0.5`): the interval is open on a side when the posterior at that
   end of the grid is at least half of what it is at the interval's own bound on that side.

## What this simulation does not test

- **The shape of the curve.** Every truth is a curve of the model's own family. A real curve that is lopsided, or
  whose plateau wears away with distance, is not covered.
- **Independence.** Each simulated trial depends on its distance only. Nothing drifts during a block, no trial affects
  the next, and neither the direction from the demonstration nor the place on the table matters. Under these
  assumptions the order of the trials within a block is irrelevant, so the seeded interleaving of the protocol was not
  simulated.
- **The placement error.** It is normal with a standard deviation of 1 mm along the distance and is cut off at 0. A
  real placement at 0 mm misses by a small distance in some direction and never lands exactly on 0.
- **Trials from other blocks.** Each block is fitted alone here. An analysis that pools a condition's trials from
  several blocks (the paired test also yields trials at measured distances) will have narrower intervals than these.
- **Truths outside the sweep.** Radii from 5 to 40 mm, widths from 1 to 10 mm, lapse rates up to 0.05. The staircases'
  starting levels and the initial step were not varied.
- **Other floors.** Only a model without a floor and one with the floor values 0, 0.05 and 0.1 were compared, against
  truths with a floor of 0.08 and 0, at 12 truths in all.
- **The paired test.** Its table assumes that the two policies' outcomes at a placement are independent. Placements
  that are hard for both policies would change the numbers.
- **Trials that do not count.** Void trials, scoring mistakes and placements outside tolerance are not simulated.
