# Has the camera moved, or a floor tag? The pictures of 2026-10-06 held against each other

Made by `analysis/instrument/picture_motion.py` from pictures kept in `data/instrument/`; every figure below is
*measured* by the commands shown. The script's own text says how it works.

The motion fitted to the picture is a shift, a turn and a change of size. The script's first version, of 13:00 that
day, fitted a shift and a turn only; the journal's entries up to 14:25 quote its figures, and `--no-scale` gives
them again. With the size, what is left unexplained changes by up to 0.2 px either way (0.27 to 0.18 px in the
first comparison, 0.16 to 0.34 px in the third): that remainder is the measure's own limit, about 0.3 px.

## What happened

- 12:25:41: the table was registered with the printed sheet on the floor (three frames kept).
- The sheet was peeled off, the camera baseline was stored and checked.
- About 12:30: the calibration was refused. Since the registration floor tag 0 had moved by (-1.12, +0.13) px in the
  picture and floor tag 1 by (-0.51, -0.66) px (*read* from the calibration's output; outlines averaged over 90
  frames): 1.0 px apart. The rule of that morning took shifts that differ for a floor tag moved on the floor.
- 12:31:19 and 12:31:30: two view checks, whose marked pictures were kept.
- 13:16:59: the calibration was stored, under a rule that notes tags that have not moved alike.
- 14:12 to 14:28: view checks and a logging session, each keeping its frame as it came.

## 1. The refused calibration: the rest of the picture against the tags

    analysis/instrument/picture_motion.py --marked \
        --after data/instrument/view_check_20261006_123119.png data/instrument/view_check_20261006_123130.png \
        --tag-shift 0:-1.12,0.13 --tag-shift 1:-0.51,-0.66

```
the table as registered 2026-10-06T12:25:41-07:00
before: table_registration_20261006_122541_raw1.png, table_registration_20261006_122541_raw2.png, table_registration_20261006_122541_raw3.png

after: view_check_20261006_123119.png
  45 patches with texture both ways, 45 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (-0.86, -0.35) px at its centre, turned by -0.098 degrees (with any one patch left out: -0.100 to -0.096), and grown by +0.070 % (+0.067 to +0.074)
  the patches lie 0.09 px (rms) and up to 0.19 px from that one motion
  floor tag 0 at (72, 171): moved by (-1.12, +0.13) px; the rest of the picture moved by (-1.15, +0.03) px there; left over (+0.03, +0.10) px
  floor tag 1 at (515, 372): moved by (-0.51, -0.66) px; the rest of the picture moved by (-0.50, -0.59) px there; left over (-0.01, -0.07) px
  the two tags as a pair turned by -0.114 degrees and drew +0.047 % apart; the rest of the picture turned by -0.098 degrees and grew by +0.070 %
  of the 1.00 px by which the two tags' shifts differ, the picture's own motion leaves 0.18 px unexplained (about 0.2 mm on the floor)

after: view_check_20261006_123130.png
  45 patches with texture both ways, 44 of them moving as one (1 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (-0.86, -0.35) px at its centre, turned by -0.098 degrees (with any one patch left out: -0.101 to -0.096), and grown by +0.068 % (+0.065 to +0.073)
  the patches lie 0.09 px (rms) and up to 0.20 px from that one motion
  floor tag 0 at (72, 171): moved by (-1.12, +0.13) px; the rest of the picture moved by (-1.15, +0.03) px there; left over (+0.03, +0.10) px
  floor tag 1 at (515, 372): moved by (-0.51, -0.66) px; the rest of the picture moved by (-0.50, -0.59) px there; left over (-0.01, -0.07) px
  the two tags as a pair turned by -0.114 degrees and drew +0.047 % apart; the rest of the picture turned by -0.098 degrees and grew by +0.068 %
  of the 1.00 px by which the two tags' shifts differ, the picture's own motion leaves 0.17 px unexplained (about 0.2 mm on the floor)
```

The 45 patches show the far pipe, the arm and the basket (44 of them in the top quarter of the picture, one at
the basket's foot; the floor sheet's far edge under the pipe is in a few). They moved as one: a shift of 0.9 px, a
turn of 0.10 degrees and a picture 0.07 % larger. The two floor tags as a pair turned by 0.11 degrees. Each tag
moved with the rest of the picture to within 0.10 px, and 0.18 px of what the two differ by is left unexplained.

## 2. A control: two pictures with nothing done between them

A view check's frame of 12:24:33 against a registration frame of 12:25:41, the sheet on the floor in both; the tags'
shifts measured between the two pictures by their outlines.

    analysis/instrument/picture_motion.py --before data/instrument/view_raw_20261006_122433.png \
        --after data/instrument/table_registration_20261006_122541_raw2.png

```
the table as registered 2026-10-06T12:25:41-07:00
before: view_raw_20261006_122433.png

after: table_registration_20261006_122541_raw2.png
  45 patches with texture both ways, 45 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (-0.01, -0.02) px at its centre, turned by -0.000 degrees (with any one patch left out: -0.000 to +0.000), and grown by +0.000 % (-0.000 to +0.000)
  the patches lie 0.01 px (rms) and up to 0.02 px from that one motion
  floor tag 0 at (72, 171): moved by (-0.02, -0.01) px; the rest of the picture moved by (-0.01, -0.01) px there; left over (-0.00, +0.01) px
  floor tag 1 at (515, 372): moved by (-0.01, -0.01) px; the rest of the picture moved by (-0.01, -0.02) px there; left over (+0.01, +0.00) px
  the two tags as a pair turned by -0.001 degrees and drew +0.002 % apart; the rest of the picture turned by -0.000 degrees and grew by +0.000 %
  of the 0.01 px by which the two tags' shifts differ, the picture's own motion leaves 0.01 px unexplained (about 0.0 mm on the floor)
```

## 3. The calibration as it was stored, 13:16:59

It kept the frame it saw, unmarked, so the tags' shifts are measured from that frame by their outlines, and the
stored conversion can be held against the registered frame carried along with the picture's own motion.

    analysis/instrument/picture_motion.py

```
the table as registered 2026-10-06T12:25:41-07:00
before: table_registration_20261006_122541_raw1.png, table_registration_20261006_122541_raw2.png, table_registration_20261006_122541_raw3.png

after: calibration_20261006_131657.png
  45 patches with texture both ways, 45 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (-0.47, -1.09) px at its centre, turned by -0.090 degrees (with any one patch left out: -0.091 to -0.089), and grown by +0.042 % (+0.040 to +0.045)
  the patches lie 0.08 px (rms) and up to 0.15 px from that one motion
  the calibration stored from this frame (2026-10-06T13:16:59-07:00) lies 0.29 mm on average and 0.56 mm at worst from the registered frame carried along with that motion, over the grid; a distance of up to 100 mm differs by 0.16 mm at most (0.20 %)
  floor tag 0 at (72, 171): moved by (-0.57, -0.77) px; the rest of the picture moved by (-0.68, -0.73) px there; left over (+0.11, -0.04) px
  floor tag 1 at (515, 372): moved by (-0.31, -1.61) px; the rest of the picture moved by (-0.18, -1.34) px there; left over (-0.13, -0.27) px
  the two tags as a pair turned by -0.103 degrees and drew -0.024 % apart; the rest of the picture turned by -0.090 degrees and grew by +0.042 %
  of the 0.88 px by which the two tags' shifts differ, the picture's own motion leaves 0.34 px unexplained (about 0.3 mm on the floor)
```

- The picture stood 1.2 px from where it was at the registration, turned by 0.09 degrees and 0.04 % larger; the
  floor tags lie 0.12 and 0.30 px from that motion.
- **A second opinion on the stored conversion.** The calibration's conversion is solved from the two floor tags'
  outlines. The registered frame carried along with the picture's motion never looks at the tags after the
  registration. Over the grid the two lie 0.29 mm apart on average and 0.56 mm at worst, and a distance of up to
  100 mm differs by 0.16 mm at most. That is as good as this comparison is: the patches themselves lie 0.08 px
  (rms) from the one motion fitted to them, about 0.1 mm.
- **The camera had moved a second time between the refused calibration and this one.** The 12:31:30 check's picture
  against this calibration's frame (the tags' shifts given as the difference of the two calibrations' readings):

      analysis/instrument/picture_motion.py --marked --before data/instrument/view_check_20261006_123130.png \
          --after data/instrument/calibration_20261006_131657.png --tag-shift 0:0.553,-0.898 --tag-shift 1:0.199,-0.952

```
the table as registered 2026-10-06T12:25:41-07:00
before: view_check_20261006_123130.png

after: calibration_20261006_131657.png
  46 patches with texture both ways, 46 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (+0.35, -0.75) px at its centre, turned by +0.018 degrees (with any one patch left out: +0.017 to +0.019), and grown by -0.028 % (-0.030 to -0.024)
  the patches lie 0.06 px (rms) and up to 0.19 px from that one motion
  floor tag 0 at (72, 171): moved by (+0.55, -0.90) px; the rest of the picture moved by (+0.44, -0.80) px there; left over (+0.11, -0.09) px
  floor tag 1 at (515, 372): moved by (+0.20, -0.95) px; the rest of the picture moved by (+0.26, -0.72) px there; left over (-0.06, -0.23) px
  the two tags as a pair turned by +0.011 degrees and drew -0.071 % apart; the rest of the picture turned by +0.018 degrees and grew by -0.028 %
  of the 0.36 px by which the two tags' shifts differ, the picture's own motion leaves 0.22 px unexplained (about 0.2 mm on the floor)
```

  A shift of 0.8 px, almost no turn, a picture 0.03 % smaller. The camera baseline of the arm tooling had been
  stored at 12:29:32, before this second movement; it was stored again at 13:33:11.

## 4. After the calibration: the camera keeps moving, the tags keep their place

Four frames kept by view checks and a logging session, against the calibration's own frame; the tags' shifts
measured between the pictures by their outlines.

    analysis/instrument/picture_motion.py --before data/instrument/calibration_20261006_131657.png \
        --after data/instrument/view_raw_20261006_131810.png data/instrument/view_raw_20261006_141218.png \
                data/instrument/view_raw_20261006_142511.png data/instrument/session_20261006_142829.png

```
the table as registered 2026-10-06T12:25:41-07:00
before: calibration_20261006_131657.png

after: view_raw_20261006_131810.png
  46 patches with texture both ways, 46 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (-0.04, -0.00) px at its centre, turned by +0.000 degrees (with any one patch left out: +0.000 to +0.001), and grown by +0.005 % (+0.005 to +0.005)
  the patches lie 0.01 px (rms) and up to 0.03 px from that one motion
  floor tag 0 at (72, 171): moved by (-0.03, +0.00) px; the rest of the picture moved by (-0.05, -0.01) px there; left over (+0.02, +0.01) px
  floor tag 1 at (515, 372): moved by (-0.02, -0.00) px; the rest of the picture moved by (-0.03, +0.01) px there; left over (+0.01, -0.01) px
  the two tags as a pair turned by -0.001 degrees and drew +0.002 % apart; the rest of the picture turned by +0.000 degrees and grew by +0.005 %
  of the 0.02 px by which the two tags' shifts differ, the picture's own motion leaves 0.02 px unexplained (about 0.0 mm on the floor)

after: view_raw_20261006_141218.png
  46 patches with texture both ways, 46 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (-0.04, +1.17) px at its centre, turned by +0.013 degrees (with any one patch left out: +0.011 to +0.014), and grown by +0.069 % (+0.065 to +0.071)
  the patches lie 0.10 px (rms) and up to 0.23 px from that one motion
  floor tag 0 at (72, 171): moved by (-0.22, +1.16) px; the rest of the picture moved by (-0.20, +1.07) px there; left over (-0.02, +0.10) px
  floor tag 1 at (515, 372): moved by (+0.18, +1.56) px; the rest of the picture moved by (+0.06, +1.30) px there; left over (+0.12, +0.26) px
  the two tags as a pair turned by +0.023 degrees and drew +0.108 % apart; the rest of the picture turned by +0.013 degrees and grew by +0.069 %
  of the 0.56 px by which the two tags' shifts differ, the picture's own motion leaves 0.21 px unexplained (about 0.2 mm on the floor)

after: view_raw_20261006_142511.png
  46 patches with texture both ways, 46 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (+0.05, +1.14) px at its centre, turned by +0.018 degrees (with any one patch left out: +0.017 to +0.020), and grown by +0.038 % (+0.034 to +0.041)
  the patches lie 0.11 px (rms) and up to 0.26 px from that one motion
  floor tag 0 at (72, 171): moved by (-0.03, +1.08) px; the rest of the picture moved by (-0.02, +1.04) px there; left over (-0.01, +0.04) px
  floor tag 1 at (515, 372): moved by (+0.23, +1.48) px; the rest of the picture moved by (+0.08, +1.26) px there; left over (+0.15, +0.23) px
  the two tags as a pair turned by +0.031 degrees and drew +0.083 % apart; the rest of the picture turned by +0.018 degrees and grew by +0.038 %
  of the 0.48 px by which the two tags' shifts differ, the picture's own motion leaves 0.24 px unexplained (about 0.2 mm on the floor)

after: session_20261006_142829.png
  46 patches with texture both ways, 46 of them moving as one (0 left out); they reach from x = 96 to 600 and y = 24 to 216
  the picture has moved by (+0.03, +0.98) px at its centre, turned by +0.008 degrees (with any one patch left out: +0.007 to +0.009), and grown by +0.046 % (+0.043 to +0.050)
  the patches lie 0.10 px (rms) and up to 0.26 px from that one motion
  floor tag 0 at (72, 171): moved by (-0.07, +0.95) px; the rest of the picture moved by (-0.08, +0.91) px there; left over (+0.00, +0.04) px
  floor tag 1 at (515, 372): moved by (+0.19, +1.22) px; the rest of the picture moved by (+0.10, +1.06) px there; left over (+0.09, +0.15) px
  the two tags as a pair turned by +0.017 degrees and drew +0.072 % apart; the rest of the picture turned by +0.008 degrees and grew by +0.046 %
  of the 0.38 px by which the two tags' shifts differ, the picture's own motion leaves 0.14 px unexplained (about 0.1 mm on the floor)
```

| Frame | The picture since the calibration | The two tags against each other, unexplained |
|---|---|---|
| 13:18:10 | 0.04 px, no turn, 0.005 % larger | 0.02 px |
| 14:12:18 | 1.17 px, 0.013 degrees, 0.069 % larger | 0.21 px |
| 14:25:11 | 1.14 px, 0.018 degrees, 0.038 % larger | 0.24 px |
| 14:28:29 | 0.98 px, 0.008 degrees, 0.046 % larger | 0.14 px |

## What this supports, and what it does not

- **The camera moves and the floor tags do not.** In every comparison the pipe, the arm and the basket moved in the
  picture the way the two floor tags did: by a pixel or so, with a turn of up to 0.1 degrees and a change of size
  of up to 0.07 %. A floor tag moved by hand takes none of them with it.
- **No sign that a floor tag moved against the other.** What the picture's own motion leaves unexplained of the
  tags' shifts is 0.14 to 0.34 px (0.1 to 0.3 mm on the floor) in every comparison in which the picture had moved. The patches
  themselves lie 0.06 to 0.11 px (rms) and up to 0.26 px from the motion fitted to them, so a tag that moved by
  0.3 mm or less would not be seen here.
- **The other reading the pictures allow** is that the floor sheet slid as a whole with everything that stands on
  it. That carries both tags alike and leaves the frame, which is the tags', as it was.
- **Why something is left over at all** (*not established*): the pipe, the arm and the basket stand nearer the
  camera than the floor, so a camera that moved shifts them a little differently; a shift, a turn and a size are
  not all that a camera's movement does to a picture; and nearly all the patches are in the top quarter of the
  picture, so what the fit says of the picture's lower half is carried there by the turn and the size.
- **Why the camera moves:** the operator reported at 13:15 that closing a zipper of the light box had tightened it
  and that its back wall had moved the camera. *Not established:* what moved it between 13:18 and 14:12, and
  whether it has come to rest.
- **What was done with it:** the registration of 12:25:41 was kept; the calibration notes tags that have not moved
  alike instead of refusing them; and since 14:25 every session takes the table frame from the floor tags as it
  finds them (`docs/decisions.md`, the rows of 2026-10-06).
