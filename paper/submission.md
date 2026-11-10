# Drafts for the OpenReview form

The form's fields and limits are in `docs/venue.md`, section 4. Each draft below sits in a fenced block under a line
`<!-- field: NAME max=N -->`; `python3 paper/check_paper.py` fails if a block reaches N characters. `[??]` stands
for a result that is not measured yet. Nothing here may identify the authors: the checker scans this file too.

**Before pasting anything into the form:** re-read the live workshop page (pages change), fill every `[??]` from
`paper/generated/numbers.tex`, and settle every `[ONE OF: ...]` with the same choice as the paper's IF block.

**Which radius goes into a `[??]`.** Every radius in these drafts is a *block fit*, as in the paper's abstract: the
clustered policy's is `\radiusClusteredThreshold` (with `...Low` and `...High`), the replay's is `\radiusReplay`. The
pooled fit (`\radiusClustered`) is a check that the paper reports beside it and does not belong in the form. If a
fit's interval is open on a side, the form says "at most" or "at least" and gives no radius, as the paper does.

**Submitted on 2026-10-09 at 17:14 PDT as number 59**, with the first title and the keywords, TL;DR, abstract and
lesson below, unchanged. The form asked two more things that are not drafted here: whether the work is under review
at another venue (answered "No") and a licence (CC BY 4.0). What follows is kept as it was submitted.

## Title: three candidates (each under 90 characters)

The first is the title in `main.tex`.

<!-- field: title-1 max=90 -->
```text
Measuring the Reset Distribution: A Tolerance Radius, Referenced to Open-Loop Replay
```

<!-- field: title-2 max=90 -->
```text
Where Did the Cube Start? Reset Distributions and Demonstration Layout Beneath the Policy
```

<!-- field: title-3 max=90 -->
```text
The Tolerance Radius: Reset Distributions and Demonstration Layout as Measured Variables
```

What each one promises. The first is the paper's title since 2026-10-08 (the author's decision after the blind reads:
the layout comparison did not decide, and the paper's weight is on the resets and the replay reference). The second
leads with the motivating example and uses the workshop's own phrase. The third was the working title until then: it
names the radius and both choices. None names a policy or claims a rate.

## Keywords

```text
evaluation protocol, reset distribution, demonstration layout, tolerance radius, imitation learning, open-loop replay, adaptive staircase, psychometric function, reporting checklist, low-cost manipulator, reproducibility
```

## TL;DR (under 250 characters)

<!-- field: tldr max=250 -->
```text
Two choices beneath a robot policy are rarely reported: where resets fall relative to demonstrations, and how a demonstration budget is laid out. We measure both with a tolerance radius referenced to open-loop replay, and propose checklist entries.
```

## Abstract (under 5,000 characters)

Written by `python3 paper/check_paper.py --abstract` from `sections/abstract.tex`: run it again after any change to
the abstract and paste the output here, so that the two never differ.

<!-- field: abstract max=5000 -->
```text
Two choices beneath a robot policy are rarely reported: where an evaluation places the object at reset relative to the demonstrations, and how a fixed budget of demonstrations is distributed. On a low-cost arm with one imitation-learning method (ACT), we place every reset within 2 mm of a target (median error 0.3 mm), log it, and report success against the distance to the nearest demonstration, summarised by a tolerance radius: the distance at which success is halfway between its levels near and far from the demonstrations. An open-loop replay of demonstrations has a radius of 16.5 mm (95 % interval 13.0 to 21.5); resets placed by eye from a reference image miss by a median of 15.9 mm. The policy succeeds at larger distances than the replay, but its success depends on location as well as distance (28 of 36 trials inside the convex hull of its demonstrations, 9 of 24 outside), and its fitted radius of 39.0 mm is not established under a pre-specified robustness check. A policy trained on 55 distinct start positions succeeded at 22 of 26 shared test placements, and one trained on 5 demonstrations at each of 11 locations at 16; the paired test does not establish a difference (p=0.11), and the spread layout's own radius was not measured. We propose five reporting-checklist entries.
```

## System-level lesson (optional, under 2,000 characters)

The form asks: "Which system-level choice—such as hardware, control, data, simulation, or inference—had a larger
effect on your policy's training or performance than you initially expected? Please describe what you learned and the
lesson you would like to share with the community."

The draft answers with the reset distribution. Its first three paragraphs rest on the earlier evaluation
(`analysis/pilot_august/trials_with_distance.csv`); the sentence "On this rig" has the two block fits of the counted
report (`\radiusClusteredThreshold`, `\radiusReplay` in `paper/generated/numbers.tex`, 2026-10-08) and the caveat the
plateau check puts on the first. The demonstration layout is left out on purpose: its test did not decide, and the
form asks for one choice.

<!-- field: lesson max=2000 -->
```text
The choice that mattered more than we expected was one we had not treated as a choice: where the evaluation puts the object at reset, relative to where it was in the demonstrations.

In an earlier evaluation of the same pick-and-place task on a low-cost arm, we scored 31 trials of an imitation policy and reported two success rates: one for cells that held demonstrations, one for held-out cells. We had not logged where the cube started. Measured later from the trial recordings, every trial in a cell with demonstrations had started 0.6 to 5.6 mm from the nearest demonstration the policy was trained on (19 of 19 succeeded), and every trial in a held-out cell 31 to 47 mm away (1 of 12 succeeded). Nothing in between had been tested. Both rates were fixed by where we had put the cube. Neither could say where between 6 and 31 mm the policy stops working, or whether resets a centimetre off would have changed the first rate.

What we learned: a success rate describes the reset distribution as much as the policy, and the rate alone does not say which. We treat the reset as a measured variable now. A camera overlay places each reset within 2 mm of a target, and the measured position is logged. Success is reported against the distance to the nearest demonstration. An adaptive staircase puts the trials where the outcome is uncertain, and the result is a threshold with an interval, the tolerance radius, compared with the same radius for an open-loop replay, a baseline with no policy. On this rig: 16.5 mm for the replay; the policy succeeded farther out, but its fitted 39.0 mm radius is not established.

For the community: report a map of the demonstrations' start states; each evaluation reset's measured position and its distance to the nearest demonstration; how resets were placed, and how accurately; and the distances that were never tested. A camera already in the scene and two printed tags suffice. Where trials allow, add the tolerance radius and its replay reference.
```

## The two authorisations on the form

"We authorize the sharing of all author emails with Program Chairs" and "We authorize the release of our submission
and author names to the public in the event of acceptance" are the authors' own decisions. They are not drafted here.

## Anonymity checklist, to run before the upload

Tick each line on the day, on the files that are actually uploaded.

1. **The private list is filled.** Outside the repository, or in `paper/forbidden.local.txt` (never committed, never
   uploaded): every author's name and initials, usernames, e-mail names, the institution and the lab, hub and
   repository namespaces, the names of the earlier project's repository, datasets and policies, the exact task
   sentence stored in the datasets, and the room or building if a photograph could show it.
2. **`python3 paper/check_paper.py --final --forbidden <the private list>` passes.** It reads every file under
   `paper/`, the comments and the figures' bytes included.
3. **The PDF's first page** says "Anonymous Authors" and no affiliation. There is no section of thanks, no funding
   line, no footnote with a contact.
4. **The PDF's properties** (open them in a PDF reader): no author, no title that differs from the paper's, no
   path to a source file, no producer string that names a person or an account.
5. **The text.** The earlier work appears only as "our earlier evaluation of the same task". No repository, dataset
   or policy of ours is named. The task is described in other words than the sentence stored in the datasets. Nothing
   says "our lab", "our previous paper", or where the rig stands.
6. **The citations.** Work by the authors, if any is cited, is cited in the third person. `lerobot2026wilson` stays
   out unless it is confirmed that no author wrote it. The two SO-101 repositories that are cited belong to others.
7. **The figures.** Each one is drawn by a script from the data. None shows a window title, a file path, a user
   name, a hub name or a clock. A photograph of the rig, if one is used, shows no face, badge, screen, poster or
   recognisable room, and its metadata is removed.
8. **Links and releases.** The paper has no link to code or data unless the link is anonymous. Whatever is released
   for review is checked on its own: dataset metadata (the task sentence, the hub name), trial logs (role codes only,
   paths relative to the repository), and commit ids of our own repositories, which can be searched for.
9. **The form.** Title, keywords, TL;DR, abstract and lesson carry no name or place. Every author has an OpenReview
   profile already, and the conflicts listed there are complete.
10. **The upload itself.** One PDF, under 50 MB, the appendix inside it. The file's name is neutral. The Overleaf
    project is not shared by a public link, and the private list was not uploaded to it.
