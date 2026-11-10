# Instrument runs that are kept and not used

A file here was logged at the rig and is left out of every analysis, for the reason given. Nothing is deleted. The
analysis reads `data/instrument/*.csv` and does not look in this folder.

| File | What it is | Why it is not used | Decided |
|---|---|---|---|
| `ruler_x_20261006_134725.csv` | A scale run along x at y = 22 mm, nine readings from x = -21 to 71 mm | One mark of the ruler was skipped between the fifth and the sixth reading (a step of 20.2 mm for 10), and the session was stopped by the camera changing its picture format before the run was finished. It was run again along x at 14:28 and at 14:35 the same day | `docs/journal.md`, 2026-10-06 ("A second run, along x at y = 22 mm, is not used"); moved here on 2026-10-07, when the analysis's first run on real records read it as a fifth scale run |
