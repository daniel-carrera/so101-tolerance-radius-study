# The first sitting's registration, worked through from its two saved frames

Written by `analysis/instrument/first_sitting.py` from `view_raw_20261005_220946.png`, `view_raw_20261005_220950.png` (in `data/instrument/`). Every figure here is
*measured* by that script.

## The registration's own figures

- The sheet's tags fit one flat plane to 0.35 mm on average, 0.79 mm at worst.
- A sheet tag left out is predicted by the others to 0.49 mm on average, 1.30 mm at worst.
- The detector's corners sit 0.61 px inside the tags' outlines.
- NOTE: floor tag 0 reads 60.4 x 62.4 mm by its outline (across x up the tag) against 59.7 x 60 stated, +1.2 % and +4.0 %: the sheet and this tag do not lie in one plane, or a size is wrong. This is the only figure here that can see a sheet lying tilted, so do not pass over it: tape the sheet flat along all four edges, press this tag flat and look along the floor at it, check the two sizes typed, and register again. If it stays with both plainly flat, say so: the cause is then the floor or the lens, and the cube moved along a printed scale (runbook 1, step 4) is what shows whether the grid is right.
- NOTE: the sheet's tags fit one flat plane to 0.35 mm on average and 0.79 mm at worst; a flat sheet gives about 0.2 and 0.45 at this view (noted beyond 0.3 and 0.6). The sheet is probably not quite flat: on 2026-10-05 a sheet that read 0.35 and 0.79 had its printed lines up to 1.2 mm from the frame. To do: tape the sheet flat along all four edges, see that no tag is in glare or partly covered, and register again.
- NOTE: a sheet tag left out of the fit is predicted by the others to 1.30 mm at worst (noted beyond 1.0): the frame is less sure toward the grid's corners than a flat sheet makes it.
- NOTE: single sheet tags read off their printed size: tag 12 -2.2 % up; tag 13 -2.1 % up (noted beyond 2 %). A tag that reads short one way lies on a slope: the sheet is not flat there.

## 1. The sheet's printed grid lines, which no fit uses

| Frame fitted through | Scans | Lines from the frame, rms (mm) | Lines of constant x | Lines of constant y | Largest mean over 10 mm of a line (mm) |
|---|---|---|---|---|---|
| the tags' outlines (what the registration uses) | 1074 | 0.35 | 0.15 | 0.47 | 1.19 |
| the detector's corners | 1074 | 0.36 | 0.19 | 0.47 | 0.98 |

The largest is on the line y = 150 mm, 20 to 30 mm along it (outline frame). Lines of constant y test the frame along the camera's line of sight, where a sheet that is not flat shows: a millimetre of height moves what the camera sees by most of a millimetre that way and hardly at all across.

## 2. The frame from four of the six sheet tags

Leaving out two tags, the frame moves by up to 1.89 mm somewhere on the grid (without tags [12, 15]); over all 15 choices of two the median is 0.81 mm.

## 3. One pixel on one corner of a floor tag

Solved from the two floor tags' corners, the frame moves by up to 2.51 mm somewhere on the grid when one corner is moved by one pixel (tag 0, corner 1, along x); the median over the 16 coordinates is 0.85 mm.
