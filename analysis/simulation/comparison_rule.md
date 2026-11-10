# The comparison rule: what two blocks of trials can tell apart

**Every number in this report is simulated. Nothing here is a measurement.**

Written by `python -m tolerance_radius.compare_power`. Seed 20261005; 300 simulated pairs of blocks for every row.
Protocol file `protocol.toml`: version 1.0, status frozen, SHA-256 `78fd2bfa905d`. The rows behind the tables are in
`comparison_rule.csv`.

## The question

Prediction 2 of the protocol is that the clustered policy's tolerance radius is larger than the replay's. The rule
that decides it (`protocol/protocol.md`, section 9): from the two block fits, taken as independent, "larger" means a
posterior probability of at least 0.95 that the policy's radius exceeds the replay's, "smaller" means at most 0.05,
and anything between is "not told apart". Before any trial is run: with the two blocks as the protocol plans them, how
large must a true difference be for the rule to say "larger", and how often does it give a verdict when the two radii
are equal?

## What was simulated, and what was assumed

- **The two blocks**, as the protocol file states them (read from the file). The policy's: T1a, 34 trials (6 at 0 mm,
  staircases from 10 mm (12 trials) and from 40 mm (12 trials) up to 50 mm, 4 trials at 40 mm). The replay's: T1b, 22
  trials (4 at 0 mm, staircases from 5 mm (9 trials) and from 25 mm (9 trials) up to 50 mm).
- **The curves assumed true.** The replay's radius is 15 mm in every case. The policy's radius is equal to it, or
  larger by 3, 5, 8, 10 and 15 mm. Both curves have the same width (1, 3 and 6 mm), a lapse of 0 and no floor. A width
  of 1 mm is a cliff: success goes from 88 % to 12 % of its plateau within 4 mm.
- **A trial**, as in the procedure simulation (`procedure.md`). The next distance of a staircase comes from the
  protocol's own rule. The cube lands at the requested distance plus a placement error (normal, standard deviation 1
  mm, and not below 0). The trial succeeds with the probability its curve gives at the distance achieved.
- **The two blocks are independent.** Each is played with its own random numbers. Nothing is shared between them, and
  nothing drifts. The rule assumes the same.
- **The fit and the rule are the analysis's own.** Each block is fitted by itself on the protocol's grid
  (`psychometric.fit_threshold`). The two fits are compared by `report.estimates.radius_difference`: every pair of
  grid values is weighted by the product of the two posteriors, a tie on the grid counts half for each side, and the
  difference gets its 95 % interval and the verdict.

**How exact the simulation is.** With 300 pairs in a row, a share near 50 % is known to about 2.9 points (one standard
error), and a share near 5 % or 95 % to about 1.3 points. A smaller difference between two rows is noise.

## 1. How often each verdict comes out

**Table 1.** Share of the simulated pairs of blocks with each verdict, in %, by the width of the two curves and by the
true difference (the policy's radius minus the replay's). Simulated.

| Width (mm) | True difference (mm) | larger | not told apart | smaller |
|---|---|---|---|---|
| 1 | 0 | 3 | 95 | 2 |
| 1 | 3 | 56 | 44 | 0 |
| 1 | 5 | 92 | 8 | 0 |
| 1 | 8 | 100 | 0 | 0 |
| 1 | 10 | 100 | 0 | 0 |
| 1 | 15 | 100 | 0 | 0 |
| 3 | 0 | 2 | 95 | 3 |
| 3 | 3 | 17 | 82 | 1 |
| 3 | 5 | 38 | 62 | 0 |
| 3 | 8 | 72 | 28 | 0 |
| 3 | 10 | 86 | 14 | 0 |
| 3 | 15 | 99 | 1 | 0 |
| 6 | 0 | 3 | 94 | 3 |
| 6 | 3 | 10 | 88 | 2 |
| 6 | 5 | 20 | 80 | 0 |
| 6 | 8 | 38 | 62 | 0 |
| 6 | 10 | 55 | 45 | 0 |
| 6 | 15 | 83 | 17 | 0 |

## 2. The difference that is reported beside the verdict

**Table 2.** The estimated difference (the posterior median of the policy's radius minus the replay's), the width of
its 95 % interval, and how often that interval holds the true difference. Means over the pairs. Simulated.

| Width (mm) | True difference (mm) | Estimated difference (mm) | Interval width (mm) | Contains the truth (%) |
|---|---|---|---|---|
| 1 | 0 | 0.0 | 7.0 | 99 |
| 1 | 3 | +3.0 | 7.0 | 100 |
| 1 | 5 | +4.9 | 7.3 | 99 |
| 1 | 8 | +8.0 | 6.8 | 100 |
| 1 | 10 | +10.0 | 6.9 | 99 |
| 1 | 15 | +15.0 | 7.1 | 100 |
| 3 | 0 | 0.0 | 13.4 | 99 |
| 3 | 3 | +2.9 | 13.8 | 98 |
| 3 | 5 | +4.7 | 13.7 | 98 |
| 3 | 8 | +7.5 | 13.9 | 98 |
| 3 | 10 | +9.6 | 13.8 | 99 |
| 3 | 15 | +14.7 | 13.9 | 99 |
| 6 | 0 | -0.1 | 22.2 | 97 |
| 6 | 3 | +2.5 | 21.9 | 95 |
| 6 | 5 | +5.0 | 22.7 | 95 |
| 6 | 8 | +7.6 | 22.3 | 95 |
| 6 | 10 | +9.8 | 21.6 | 97 |
| 6 | 15 | +14.6 | 22.6 | 96 |

## Reading

- **When the two radii are equal**, the right answer is "not told apart". The rule gave a verdict all the same: at
  width 1 mm, "larger" in 3 % of pairs and "smaller" in 2 %; at width 3 mm, "larger" in 2 % of pairs and "smaller" in
  3 %; at width 6 mm, "larger" in 3 % of pairs and "smaller" in 3 %. At most 6 % of pairs got a verdict for a
  difference that is not there.
- **Both curves of width 1 mm.** The rule says "larger" in 56 % of pairs at a true difference of 3 mm, 92 % at 5 mm,
  100 % at 8 mm, 100 % at 10 mm, 100 % at 15 mm. The smallest simulated difference from which it does so in at least
  80 % of pairs is 5 mm.
- **Both curves of width 3 mm.** The rule says "larger" in 17 % of pairs at a true difference of 3 mm, 38 % at 5 mm,
  72 % at 8 mm, 86 % at 10 mm, 99 % at 15 mm. The smallest simulated difference from which it does so in at least 80 %
  of pairs is 10 mm.
- **Both curves of width 6 mm.** The rule says "larger" in 10 % of pairs at a true difference of 3 mm, 20 % at 5 mm,
  38 % at 8 mm, 55 % at 10 mm, 83 % at 15 mm. The smallest simulated difference from which it does so in at least 80 %
  of pairs is 15 mm.
- **The wrong way round.** With the policy's radius truly the larger, the rule said "smaller" in at most 2 % of pairs.
- **The interval of the difference** held the true difference in 95 % of pairs in the lowest row and in 100 % in the
  highest.

## What this simulation does not show

- **Other radii.** The replay's radius is 15 mm in every case. A radius near 0 mm, or near the largest distance a
  block tests, is located less well (`procedure.md`, section 2), and the rule then tells less apart than here.
- **Curves that differ in more than their radius.** Both curves have one width, no lapse and no floor. A replay that
  is a cliff against a policy that is a slope was not simulated, nor was a plateau below 1 (`lapse_values.md` has what
  a low plateau does to one radius).
- **The shape of the curve.** Every truth is a curve of the model's own family.
- **What the two blocks share.** They are run interleaved in one session. Whatever the two share there (the hour, the
  servos' temperature, the state of the cameras) is not in this simulation.
- **Trials that do not count.** Void trials and scoring mistakes are not simulated.
- **The other comparison.** The spread policy's block against the clustered policy's is another pair of blocks, run on
  another day, and was not simulated.
