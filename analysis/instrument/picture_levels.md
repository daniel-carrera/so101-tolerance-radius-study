# What the box's windows and the room's light do to the picture, 2026-10-06

Made by `analysis/instrument/picture_levels.py` from frames kept by view checks; *measured* by the command shown.
What each frame's state was is the operator's report of that afternoon, except where it says otherwise.

| Frame | The box and the room |
|---|---|
| 15:00:28 | big flap closed and unzipped, both side windows open; all blinds closed, the room's light off |
| 15:00:05 | the same box; the room's light on, blinds closed |
| 14:59:27 | the same box, the opposite window just opened; the room as it had been all afternoon |
| 14:48:23 | one side window open (*inferred* from what the operator wrote at 14:50); the basket off its seat |
| 13:16:57 | the calibration's own frame: the box zipped (*inferred*: the operator had just closed a zipper), windows not yet in use |

    analysis/instrument/picture_levels.py data/instrument/view_raw_20261006_150028.png \
        data/instrument/view_raw_20261006_150005.png data/instrument/view_raw_20261006_145927.png \
        data/instrument/view_raw_20261006_144823.png data/instrument/calibration_20261006_131657.png

```
grey level and red over blue of the bare floor, place by place

  view_raw_20261006_150028.png
    grid, A's third        grey  200.1   red over blue 0.923
    grid, middle third     grey  200.6   red over blue 0.923
    grid, E's third        grey  200.6   red over blue 0.923
    in front of the grid   grey  202.6   red over blue 0.919
    beyond column E        grey  199.3   red over blue 0.922
  view_raw_20261006_150005.png
    grid, A's third        grey  200.1   red over blue 0.933
    grid, middle third     grey  201.1   red over blue 0.932
    grid, E's third        grey  201.3   red over blue 0.932
    in front of the grid   grey  203.7   red over blue 0.925
    beyond column E        grey  201.1   red over blue 0.932
  view_raw_20261006_145927.png
    grid, A's third        grey  200.1   red over blue 0.923
    grid, middle third     grey  201.1   red over blue 0.923
    grid, E's third        grey  200.6   red over blue 0.923
    in front of the grid   grey  202.6   red over blue 0.919
    beyond column E        grey  200.3   red over blue 0.922
  view_raw_20261006_144823.png
    grid, A's third        grey  201.3   red over blue 0.932
    grid, middle third     grey  202.3   red over blue 0.925
    grid, E's third        grey  202.3   red over blue 0.925
    in front of the grid   grey  206.3   red over blue 0.919
    beyond column E        grey  201.3   red over blue 0.925
  calibration_20261006_131657.png
    grid, A's third        grey  201.3   red over blue 0.924
    grid, middle third     grey  202.3   red over blue 0.923
    grid, E's third        grey  202.3   red over blue 0.919
    in front of the grid   grey  205.2   red over blue 0.912
    beyond column E        grey  201.6   red over blue 0.919

against the first frame (view_raw_20261006_150028.png): grey levels, and red over blue in thousandths

                                      150005            145927            144823            131657
    grid, A's third            +0.0   +10        +0.0    +0        +1.2    +9        +1.2    +0   
    grid, middle third         +0.5    +9        +0.5    +0        +1.7    +2        +1.7    +0   
    grid, E's third            +0.7    +9        +0.0    +0        +1.7    +1        +1.7    -4   
    in front of the grid       +1.1    +6        +0.0    +0        +3.7    +0        +2.6    -7   
    beyond column E            +1.8   +10        +1.1    +0        +2.0    +3        +2.3    -3
```

## What it says

- **The room's light hardly reaches the picture.** With both windows open, the room's light on against a dark room:
  0.0 to 0.7 grey levels on the grid (of 201), 1.1 in front of it and 1.8 beyond column E, and the floor 0.006 to
  0.010 warmer in red over blue (about 1 %). The room as it was at 14:59 reads as the dark room does.
- **Opening the box darkens it a little.** From the zipped box of the calibration to both windows open: 1.2 to 1.7
  grey levels less on the grid (under 1 %), 2.6 in front of it. An open window is a hole in a white wall that sent
  the lamps' light back.
- **The grid is lit evenly in every state:** its three thirds lie within 1.2 grey levels of each other.
- *Not looked at:* the wrist camera, which may see out through an open window; daylight with the blinds open on a
  bright day (the afternoon's "room as it had been" is one such state, and reads as the dark room).
