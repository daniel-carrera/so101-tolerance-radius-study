# The model's lapse values: do they reach far enough?

**Every number in this report is simulated. Nothing here is a measurement.**

Written by `python -m tolerance_radius.lapse_check`. Seed 20261005; 200 simulated blocks for every
row of `lapse_values.csv`. Protocol file SHA-256 `78fd2bfa905d`.

## The question

The plateau of the success curve, the success rate at a demonstration's own start, is `1 - lapse`. The
protocol's model weighs the lapse values 0, 0.02, 0.05, 0.1. If a condition's
true plateau is lower than that list allows (a replay that does not always repeat, a policy taught from
starts that are never repeated), what happens to the estimate of its tolerance radius? And what does a longer
list
(0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5) cost when the plateau is high after all?

Every block of the protocol was played against curves with a radius of 10, 20, 30 mm, a width of
1, 3, 6 mm and no floor, at each plateau below, and fitted both ways on the same
simulated trials.

## All blocks together, by the true plateau

| True plateau | List | Bias (mm) | RMSE (mm) | Interval width (mm) | Contains the truth (%) | Lowest row (%) |
|---|---|---|---|---|---|---|
| 1.00 | as the protocol | +0.1 | 2.3 | 10.3 | 97 | 94 |
| 1.00 | wider list | +0.6 | 2.7 | 16.2 | 97 | 95 |
| 0.95 | as the protocol | -0.5 | 2.4 | 11.9 | 97 | 94 |
| 0.95 | wider list | +0.5 | 3.0 | 20.7 | 98 | 96 |
| 0.80 | as the protocol | -3.1 | 4.9 | 16.9 | 94 | 84 |
| 0.80 | wider list | -0.0 | 4.2 | 35.7 | 99 | 96 |
| 0.65 | as the protocol | -7.8 | 9.7 | 18.9 | 75 | 44 |
| 0.65 | wider list | -0.9 | 5.9 | 49.1 | 99 | 94 |
| 0.50 | as the protocol | -13.4 | 15.0 | 15.7 | 35 | 8 |
| 0.50 | wider list | -3.9 | 8.6 | 55.8 | 99 | 94 |

## By block

| Block | True plateau | Bias: protocol / wider (mm) | RMSE: protocol / wider (mm) | Contains the truth: protocol / wider (%) |
|---|---|---|---|---|
| T1b | 1.00 | +0.1 / +0.8 | 2.5 / 3.1 | 97 / 98 |
| T1b | 0.95 | -0.4 / +0.6 | 2.5 / 3.2 | 98 / 98 |
| T1b | 0.80 | -3.0 / +0.5 | 4.9 / 4.8 | 95 / 99 |
| T1b | 0.65 | -7.8 / -0.0 | 9.7 / 6.5 | 83 / 99 |
| T1b | 0.50 | -13.1 / -3.0 | 14.8 / 8.6 | 44 / 99 |
| T1a | 1.00 | -0.0 / +0.3 | 1.9 / 2.1 | 97 / 97 |
| T1a | 0.95 | -0.5 / +0.2 | 2.1 / 2.3 | 97 / 97 |
| T1a | 0.80 | -3.3 / -0.8 | 4.8 / 3.2 | 90 / 98 |
| T1a | 0.65 | -8.3 / -2.4 | 9.9 / 5.1 | 60 / 98 |
| T1a | 0.50 | -13.8 / -5.5 | 15.3 / 8.4 | 20 / 97 |
| T3 | 1.00 | +0.1 / +0.8 | 2.3 / 2.9 | 97 / 98 |
| T3 | 0.95 | -0.5 / +0.6 | 2.5 / 3.3 | 98 / 98 |
| T3 | 0.80 | -3.1 / +0.2 | 4.9 / 4.3 | 95 / 99 |
| T3 | 0.65 | -7.5 / -0.3 | 9.3 / 6.0 | 81 / 99 |
| T3 | 0.50 | -13.2 / -3.2 | 14.9 / 8.8 | 41 / 100 |

## Reading

- **With a high plateau** (0.95 or more) the longer list makes the interval +66 % wider on
  average: that is its whole cost.
- **With a low plateau** the protocol's list has no value that fits, so the model explains the failures at
  small distances by pulling the radius inward: the bias column shows by how much, and the interval then
  misses the truth more often than its level promises.
- The longer list removes most of that bias. It cannot remove all of it at a plateau of 0.5, where a
  one-up-one-down staircase itself drifts toward zero and half the trials near a demonstration fail.

## What this does not show

- Curves with a floor as well as a low plateau, or of another shape than the model's.
- How to tell a low plateau from a small radius with 4 to 6 trials at a demonstration's own start: the fit
  cannot, and the counts at 0 mm are reported beside every radius for that reason.
