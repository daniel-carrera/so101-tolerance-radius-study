# The workspace test: what each way of drawing its placements could tell apart

**Every number here is calculated from stated assumptions. Nothing here is a measurement.**

Written by `python -m tolerance_radius.workspace_power`. Seed 20261005; each figure is the mean over 200
draws of a design's placements. Planned layouts: `config/layouts/`, SHA-256 `b46d78074a47`. The rows are in
`workspace_designs.csv`.

## The question

Block T2 tests the two policies at the same placements and compares them with an exact paired test. Where
should the placements be drawn so that the test can show a difference, whatever the tolerance radius turns
out to be?

## The assumptions

- **The lookup model.** Each policy succeeds with the probability a success curve gives at the placement's
  distance to that policy's nearest planned demonstration: plateau 0.97, floor 0.03, and a radius and a width
  as in the tables. This is the hypothesis the workspace test exists to check, used here only to choose
  its design.
- The two policies' outcomes at one placement are independent, and both policies have the same curve.
- For one set of placements the test's power is computed exactly; the mean over the draws describes the
  design.

## How often the test would reach p < 0.05 (%)

**Width 1 mm.**

| Design | radius 10 mm | radius 15 mm | radius 20 mm | radius 25 mm | radius 30 mm | radius 35 mm |
|---|---|---|---|---|---|---|
| trained cells, 20 | 76 | 98 | 83 | 13 | 0 | 0 |
| whole grid, 20 | 54 | 92 | 84 | 51 | 20 | 4 |
| whole grid, 26 | 73 | 98 | 95 | 71 | 34 | 11 |
| whole grid, 30 | 80 | 99 | 97 | 79 | 41 | 13 |
| 10 trained + 10 held-out | 29 | 80 | 86 | 85 | 63 | 19 |
| 13 trained + 13 held-out | 43 | 92 | 96 | 94 | 82 | 40 |

**Width 3 mm.**

| Design | radius 10 mm | radius 15 mm | radius 20 mm | radius 25 mm | radius 30 mm | radius 35 mm |
|---|---|---|---|---|---|---|
| trained cells, 20 | 59 | 81 | 70 | 22 | 1 | 0 |
| whole grid, 20 | 43 | 71 | 73 | 51 | 22 | 6 |
| whole grid, 26 | 61 | 86 | 87 | 71 | 39 | 13 |
| whole grid, 30 | 69 | 91 | 92 | 78 | 46 | 17 |
| 10 trained + 10 held-out | 27 | 61 | 76 | 75 | 56 | 22 |
| 13 trained + 13 held-out | 40 | 76 | 89 | 88 | 76 | 44 |

**Width 6 mm.**

| Design | radius 10 mm | radius 15 mm | radius 20 mm | radius 25 mm | radius 30 mm | radius 35 mm |
|---|---|---|---|---|---|---|
| trained cells, 20 | 32 | 39 | 36 | 22 | 7 | 1 |
| whole grid, 20 | 27 | 38 | 41 | 35 | 22 | 10 |
| whole grid, 26 | 40 | 53 | 57 | 50 | 36 | 19 |
| whole grid, 30 | 46 | 60 | 64 | 57 | 42 | 24 |
| 10 trained + 10 held-out | 22 | 37 | 46 | 47 | 38 | 23 |
| 13 trained + 13 held-out | 32 | 49 | 60 | 62 | 54 | 39 |

## Expected successes of each policy (clustered / spread), width 3 mm

| Design | Pairs | radius 10 mm | radius 15 mm | radius 20 mm | radius 25 mm | radius 30 mm | radius 35 mm |
|---|---|---|---|---|---|---|---|
| trained cells, 20 | 20 | 3.9 / 11.7 | 7.1 / 16.7 | 11.2 / 18.8 | 15.1 / 19.3 | 17.8 / 19.4 | 19.0 / 19.4 |
| whole grid, 20 | 20 | 3.0 / 9.1 | 5.3 / 13.4 | 8.3 / 16.0 | 11.4 / 17.4 | 14.1 / 18.2 | 16.1 / 18.7 |
| whole grid, 26 | 26 | 3.9 / 12.0 | 6.9 / 17.6 | 10.9 / 20.9 | 15.0 / 22.7 | 18.5 / 23.8 | 21.0 / 24.4 |
| whole grid, 30 | 30 | 4.6 / 13.9 | 8.1 / 20.4 | 12.7 / 24.2 | 17.5 / 26.2 | 21.5 / 27.4 | 24.4 / 28.1 |
| 10 trained + 10 held-out | 20 | 2.4 / 7.1 | 4.0 / 11.0 | 6.1 / 14.1 | 8.6 / 16.2 | 11.3 / 17.5 | 14.0 / 18.3 |
| 13 trained + 13 held-out | 26 | 2.9 / 8.9 | 5.0 / 14.0 | 7.8 / 17.9 | 11.0 / 20.6 | 14.5 / 22.4 | 18.0 / 23.6 |

## Reading

- **Placements in the trained cells alone are blind to a large radius.** Eleven clusters on a 50 mm grid
  already cover their own cells once the radius approaches 25 mm, so both policies succeed there and the
  test has nothing to find.
- **Placements in the held-out cells alone are blind to a small radius**: neither policy reaches them.
- **Half in each keeps the test alive over the whole plausible range**, at the cost of being a little weaker
  than the best single region at either end.
- More pairs help every design; the tables show by how much.

## What this does not show

- Whether the lookup model is true. If the spread policy is weaker at each point because no start is
  repeated, or the clustered one reaches between its clusters, the real powers differ.
- Outcomes that depend on anything but distance: direction, place on the table, the day.
- Curves that differ between the two policies.
