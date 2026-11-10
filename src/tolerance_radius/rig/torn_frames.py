"""Torn camera frames in a recorded dataset: a picture that is damaged in a band of its rows, for one frame or a few.

A USB camera that loses part of a transfer can still hand the recorder a full-sized picture. The recorder writes it
into the video like any other, the video is valid, and nothing that reads the files' structure can see it: the damage
is in the pixels. What it looks like was *seen* on this rig's front camera (2026-10-06 and 07: 278 frames of the
August dataset and 4 of a set recorded on 2026-10-06): from some row down to the bottom of the picture, or in a band
at its top, the picture is shifted sideways, or has a colour cast, or is too bright, and the other rows are right. The
damage starts and ends on the 8-row grid of the camera's own JPEG compression. In 31 cases the camera then gave the
same damaged picture once to four times more before a good one came. A stale buffer (rows that still hold an earlier
frame) was not seen and is looked for all the same.

The rule (`scan_pictures`) holds every frame against the two frames before it and the two after:

* a pixel is *still* where those four agree (within `STILL_RANGE` levels in brightness and in both colour planes);
* a still pixel is *broken* when the frame lies more than `OUTSIDE` levels outside everything the four show at that
  pixel or right next to it. A picture that shakes by a pixel or two, a moving edge, an exposure step or a change of
  zoom break nothing: either the four do not agree, or the frame lies between them.

A frame is called torn when its broken pixels cover a quarter of the picture's width in at least 8 rows running (a
*band*: wrong colour or brightness), or when at least 1000 of them lie in 16 rows where they outnumber the pixels
that moved (*displaced*: the picture's own things standing somewhere else), or when it shows, where the frames around
agree on something else, what an earlier frame showed there (*stale*). A picture that comes again unchanged is judged
as the frame it repeats, and each picture is also held against the pictures beyond a run of up to three damaged ones
(more is asked for then): damaged frames next to each other would otherwise hide each other.

*Measured* 2026-10-07 on every frame of three sets (August's 79,254; the 38,234 of the set of 2026-10-06; the 7,110
of the set put aside that day), each called frame looked at with the frames next to it:

* August: 278 called, all of the front camera, in 48 of 53 episodes, and all torn: 179 single frames and 99 in 30
  runs of two to six. The earlier project's tool names 171 frames of this set: 168 of these, and 3 whole frames that
  are not called here.
* The set put aside: 4 called, all torn (a band at the top of the front camera's picture, twice in two frames one
  after the other). The set of 2026-10-06: none called.
* Whole frames looked at and not called: the 93 the earlier tool calls corrupt in the set of 2026-10-06, the 56
  frames of the two sets that come nearest to the rule's numbers, and the frame in which the front camera went back
  to its own zoom with its exposure overshooting. The nearest whole frame has 0.13 of the width broken in its best
  four rows running (0.08 in the set of 2026-10-06), where a band needs 0.25.
* Tears put into four takes of the set of 2026-10-06, the takes written again with the recorder's video settings
  (with `GUARD` at 2; at 1.5 the rule can only call more). Of single frames damaged the way the camera does it (a
  part shifted, a colour cast, a brighter or darker part; from a row down, at the top, in the last 8 rows): front
  camera 229 of the 229 that changed 1000 pixels or more (27 more changed nothing: plain floor shifted), wrist
  camera 233 of 256. Of the same band in two or three frames running: 42 of 46 and 30 of 46. Of a stale part from
  two to four frames before: 1 of 72 and 40 of 72 (a slow arm's stale part is never outside the frames around).
  The 16 takes written again untouched: none called. Called without being touched: 7 frames, each the frame after
  a damaged one.

What it does not find: a band whose only fault is a brightness that is off by `OUTSIDE` levels or less (*found*
2026-10-07, at the end of the work, by a separate look at the steps of brightness from frame to frame in August's
front camera: 9 frames in 4 places, the picture from some row down 5 to 8 levels brighter or darker for one to four
frames; none of them is called. That look finds no such step in the front camera of the set of 2026-10-06 or of the
set put aside, and says nothing of the wrist camera, whose picture moves. A smaller `OUTSIDE` has not been tried
against the sets); a band thinner than 8 rows; damage that changes fewer than 1000 pixels, or that lies where the
picture is moving in the frames around (in the wrist camera's picture on its way, only the plain parts can show it);
rows that repeat the frame just before; more than three different damaged pictures one after the other; the first or
last frame of a video file when something moves next to it (the script names such a frame). What it may name without
a camera's fault: the frame after a torn one, which the video codes as a change of the torn frame and which keeps a
trace of it; a frame next to a damaged one; a whole picture that is brighter or darker for a single frame.

The rule itself is NumPy on arrays (`half_picture`, `still_and_broken`, `judge`, `stale_rows`, `scan_pictures`) and
is tested on synthetic pictures. `scan_dataset` reads a dataset's video files with PyAV; `save_pictures` writes a
flagged frame with its neighbours for a person to look at.

`repeats` and `stood` are measurements beside the rule: frames whose picture is the one before again while the arm
was moving, which is how a frame the camera did not deliver shows in the recording, and pictures that stood for
three frames or more.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: The frames around a frame agree at a pixel when their values lie within this many levels (of 256), in each plane.
STILL_RANGE = 4
#: A still pixel is broken when the frame lies more than this many levels outside what the frames around show there.
OUTSIDE = 8
#: "There" is the pixel and its neighbours this far away, in the half-size picture (so 2 pixels of the camera's).
#: Without it the rail at the top of the front camera's picture, which shakes by half a pixel when the arm moves,
#: broke a whole row of blocks (*measured* 2026-10-06, episode 24 frame 620).
REACH = 1
#: A band: rows in which broken pixels cover at least this share of the picture's width ...
BAND_SHARE = 0.25
#: ... for at least this many rows of the half-size picture running: 8 of the camera's, one row of its JPEG blocks.
BAND_ROWS = 4
#: Displaced: a row counts when at least this many of its pixels are broken and they are at least as many as the
#: pixels of the row that are not still (something moving through a row breaks pixels there and is not damage) ...
QUIET_PIXELS = 4
#: ... and the frame is torn when such rows are at least this many and hold at least this many broken pixels, in the
#: half-size picture (1000 of the camera's). The rows keep out an edge that swings out for a frame and comes back.
DISPLACED_ROWS = 8
DISPLACED_PIXELS = 250
#: Stale: the frame shows what the frame this many before it showed, where the frame just before it and as many
#: frames after it agree on something else. The frame just before is not looked for: a part of the picture that
#: did not change for one frame is what a frame that was not delivered looks like, and what a video encoder does by
#: itself (`repeats` counts whole frames of that kind).
STALE_BACK = (2, 3, 4)
#: A row is stale when at least this many of its pixels show the earlier picture again, and at least this share of
#: the pixels where the earlier picture showed something that has gone since ...
STALE_ROW_PIXELS = 2
STALE_SHARE = 0.8
#: ... and the row as a whole is the earlier picture's row, at this share of its pixels: something that only stands
#: where it stood a few frames ago (a wall of holes passing by) has moved a little and differs at every edge.
STALE_ROW_SAME = 0.95
#: ... and the frame is torn when such rows are at least this many and hold at least this many pixels.
STALE_ROWS = 16
STALE_PIXELS = 150
#: Besides its neighbours, a picture is held against pictures further off: (a, b) is the picture `a` places before
#: it with the one before that, and the picture `b` places after it with the one after that (`references_for`).
#: They step over a run of two or three damaged pictures, and over the frame after a damaged one, which in the
#: video still shows a trace of it when it is coded as a change of that frame. Displaced things then need this many
#: broken pixels ...
FURTHER_PAIRS = ((1, 2), (2, 1), (1, 3), (2, 2), (3, 1))
FURTHER_DISPLACED_PIXELS = 1000
#: ... and whenever a picture is not simply one frame held against the four frames next to it (pictures further
#: off; a picture that came more than once, itself or one next to it), every row that counts must hold this many
#: times more broken pixels than pixels that are not still. Frames that are not next to each other can vouch for
#: each other by chance: holes of a wall passing under the wrist camera stand, a few frames on, where holes stood
#: before, and a picture between two such frames is outside both in a third of a row, where as many pixels again
#: are not still (*found* 2026-10-07 on synthetic takes with a camera that stalls: 1.0 times). The other way, a
#: run of three damaged frames of August has 1.5 to 2 times in 9 rows running, because a quarter of that set's
#: wooden floor is not still from the video's own noise. 1.5 lies between the two; with 1 the 125,000 recorded
#: frames give the same verdicts and that run, with 2 they give the same verdicts without it.
GUARD = 1.5
#: The farthest frame the rule looks at, counted from a picture's first frame back and from its last frame on.
SPAN = 12
#: The first and last frame of a run of frames have neighbours on one side only. They are judged when at least this
#: share of the picture is still in those: *measured* 2026-10-06, 0.992 to 0.997 at the starts and ends of the 31
#: takes of that day (the arm at rest), 0.987 at the middle one of August's 53.
AT_REST_SHARE = 0.98

#: `repeats`: the arm counts as moving into a frame when some joint's reading stepped by at least this, in degrees ...
MOVING_DEG = 0.5
#: ... and the picture as changing when at least this many pixels of the half-size picture moved (`moved_pixels`).
MOVING_PIXELS = 100
#: A frame is a repeat when at most this many moved.
REPEAT_PIXELS = 5


@dataclass(frozen=True, slots=True)
class Tear:
    """What the rule found in one frame."""

    #: "band" (wrong colour or brightness across the picture), "displaced" (things standing somewhere else) or
    #: "stale" (rows that show an earlier frame).
    kind: str
    #: The rows of the camera's picture that carry the damage: from this row ...
    first_row_px: int
    #: ... up to, not including, this one.
    last_row_px: int
    #: Damaged pixels counted, in the camera's pixels.
    pixels: int
    #: The frames it was held against, as steps from the frame itself: (-2, -1, 1, 2) for its neighbours.
    held_against: tuple[int, ...]

    def words(self) -> str:
        what = {"band": "wrong colour or brightness across the picture",
                "displaced": "things in the picture standing somewhere else",
                "stale": "rows that show an earlier frame"}[self.kind]
        return f"{what}, rows {self.first_row_px} to {self.last_row_px} ({self.pixels} pixels)"


@dataclass(frozen=True, slots=True)
class Finding:
    """A torn frame of a dataset."""

    episode: int
    camera: str
    #: The frame within its episode, from 0.
    frame: int
    time_s: float
    tear: Tear


@dataclass(slots=True)
class Report:
    """What a scan of a dataset found."""

    findings: list[Finding] = field(default_factory=list)
    #: Frames looked at, by camera.
    frames: dict[str, int] = field(default_factory=dict)
    episodes: int = 0
    fps: float = 30.0
    #: Frames the rule could not judge, as (camera, episode, frame): the first or last frame of a video file when
    #: something moves in the frames next to it.
    not_judged: list[tuple[str, int, int]] = field(default_factory=list)
    #: By camera and episode: for every frame, how many pixels moved from the frame before (`moved_pixels`) and
    #: whether the video codes the frame by itself. Filled when the scan is asked to measure repeats.
    moved: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    keyframes: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    #: ... and whether the frame is, value for value, the frame before again (what `stood` wants).
    same: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Frame:
    """A half-size picture with what a neighbour may show before the frame counts as broken there."""

    picture: np.ndarray
    #: The lowest value at the pixel or next to it, less `OUTSIDE`; the highest, plus `OUTSIDE`.
    low: np.ndarray
    high: np.ndarray


def half_picture(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The picture the rule works on, from a frame's three planes as a camera's video stores them: the brightness
    averaged two by two, and the two colour planes, which are stored at half size, as they are. Height/2 x width/2 x 3.
    """
    if (y.ndim != 2 or y.shape[0] % 2 or y.shape[1] % 2 or u.shape != (y.shape[0] // 2, y.shape[1] // 2)
            or v.shape != u.shape):
        raise ValueError(f"planes of a 4:2:0 picture are wanted, not {y.shape}, {u.shape}, {v.shape}")
    height, width = u.shape
    out = np.empty((height, width, 3), np.uint8)
    summed = y.reshape(height, 2, width, 2).sum(axis=(1, 3), dtype=np.uint16)
    out[..., 0] = (summed + 2) >> 2
    out[..., 1] = u
    out[..., 2] = v
    return out


def _prepare(picture: np.ndarray) -> _Frame:
    """The picture with the lowest and highest value at and next to every pixel, widened by `OUTSIDE`."""
    height, width = picture.shape[:2]
    rows = np.pad(picture, ((REACH, REACH), (0, 0), (0, 0)), mode="edge")
    low = np.minimum.reduce([rows[step:step + height] for step in range(2 * REACH + 1)])
    high = np.maximum.reduce([rows[step:step + height] for step in range(2 * REACH + 1)])
    low = np.pad(low, ((0, 0), (REACH, REACH), (0, 0)), mode="edge")
    high = np.pad(high, ((0, 0), (REACH, REACH), (0, 0)), mode="edge")
    low = np.minimum.reduce([low[:, step:step + width] for step in range(2 * REACH + 1)])
    high = np.maximum.reduce([high[:, step:step + width] for step in range(2 * REACH + 1)])
    low = np.where(low > OUTSIDE, low - OUTSIDE, 0).astype(np.uint8)
    high = np.where(high < 255 - OUTSIDE, high + OUTSIDE, 255).astype(np.uint8)
    return _Frame(picture, low, high)


def _outside(picture: np.ndarray, references: Sequence[_Frame]) -> np.ndarray:
    """Where the picture lies more than `OUTSIDE` levels outside what every reference shows at the pixel or next to
    it, in any plane."""
    low = np.minimum.reduce([reference.low for reference in references])
    high = np.maximum.reduce([reference.high for reference in references])
    return ((picture > high) | (picture < low)).any(axis=2)


def _agree(pictures: Sequence[np.ndarray]) -> np.ndarray:
    """Where the pictures lie within `STILL_RANGE` levels of each other in every plane."""
    return ((np.maximum.reduce(pictures) - np.minimum.reduce(pictures)) <= STILL_RANGE).all(axis=2)


def still_and_broken(picture: np.ndarray, references: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Two maps of a half-size picture held against the frames around it: where those frames agree with each other
    (*still*), and where they do and the picture shows something none of them shows there or next to it (*broken*).
    """
    prepared = [_prepare(reference) for reference in references]
    still = _agree([reference.picture for reference in prepared])
    return still, still & _outside(picture, prepared)


def moved_pixels(previous: np.ndarray, picture: np.ndarray) -> int:
    """How many pixels of a half-size picture show something the frame before does not show there or next to it.
    A picture that came twice has next to none, however the video coded the two; an arm on its way has hundreds."""
    return int(np.count_nonzero(_outside(picture, [_prepare(previous)])))


def _longest_run(marked: np.ndarray) -> tuple[int, int]:
    """Where the longest run of True begins, and how long it is; (0, 0) when there is none."""
    best_start = best = start = run = 0
    for place, value in enumerate(marked.tolist()):
        if value:
            if run == 0:
                start = place
            run += 1
            if run > best:
                best_start, best = start, run
        else:
            run = 0
    return best_start, best


def judge(still: np.ndarray, broken: np.ndarray, held_against: tuple[int, ...] = (-2, -1, 1, 2), *,
          displaced_pixels: int = DISPLACED_PIXELS, guarded: bool = False) -> Tear | None:
    """A band or displaced things, from the two maps of `still_and_broken`; None for a whole picture.

    `guarded` is for a picture that is not simply one frame held against the four frames next to it: every row
    that counts, of a band or of displaced things, must then hold `GUARD` times more broken pixels than pixels that
    are not still. `displaced_pixels` is how many broken pixels displaced things need.
    """
    width = broken.shape[1]
    per_row = np.count_nonzero(broken, axis=1)
    moving = width - np.count_nonzero(still, axis=1)
    wide = (per_row >= BAND_SHARE * width) & (per_row >= (GUARD if guarded else 0.0) * moving)
    _start, rows = _longest_run(wide)
    band = rows >= BAND_ROWS
    quiet = (per_row >= QUIET_PIXELS) & (per_row >= (GUARD if guarded else 1.0) * moving)
    displaced = int(np.count_nonzero(quiet)) >= DISPLACED_ROWS and int(per_row[quiet].sum()) >= displaced_pixels
    if not band and not displaced:
        return None
    marked = np.flatnonzero((wide & band) | (quiet & displaced))
    return Tear("band" if band else "displaced", 2 * int(marked[0]), 2 * int(marked[-1]) + 2,
                4 * int(per_row[marked].sum()), held_against)


def stale_rows(picture: np.ndarray, source: np.ndarray, references: Sequence[np.ndarray]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row by row of a half-size picture held against an earlier one (`source`): how many pixels there are where the
    earlier picture showed something that the frame before the picture and the frames after it (`references`) agree
    is not there; at how many of those the picture shows what the earlier one did; and at how many pixels of the
    whole row it does."""
    prepared = [_prepare(reference) for reference in references]
    gone = _agree([reference.picture for reference in prepared]) & _outside(source, prepared)
    like_source = _agree([picture, source])
    return (np.count_nonzero(gone, axis=1), np.count_nonzero(gone & like_source, axis=1),
            np.count_nonzero(like_source, axis=1))


def judge_stale(gone: np.ndarray, back_again: np.ndarray, same: np.ndarray, width: int,
                held_against: tuple[int, ...] = ()) -> Tear | None:
    """Rows that show an earlier frame, from the three counts of `stale_rows`; None when they are too few to say.

    A row counts when nearly all of what was gone is back in it and nearly the whole row is the earlier frame's. A
    wall of holes passing under the wrist camera brings a hole back to where one was a few frames earlier in many
    places, but never the row as it was: every edge in it has moved a little.
    """
    marked = np.flatnonzero((back_again >= STALE_ROW_PIXELS) & (back_again >= STALE_SHARE * gone)
                            & (same >= STALE_ROW_SAME * width))
    if len(marked) < STALE_ROWS or int(back_again[marked].sum()) < STALE_PIXELS:
        return None
    return Tear("stale", 2 * int(marked[0]), 2 * int(marked[-1]) + 2, 4 * int(back_again[marked].sum()), held_against)


def references_for(starts: Sequence[int], place: int, length: int, before: int, after: int) -> list[int]:
    """The four frames a picture is held against for the pair (before, after).

    `starts` holds the first frame of every picture of a run of `length` frames, in order (a picture that comes
    several times running is one picture); `place` says which of them is meant. The frames are the last frame of
    the picture `before` places back and of the one before that, and the first frame of the picture `after` places
    on and of the one after that: (1, 1) is the pictures next to it, (1, 2) steps over the picture after it,
    however often that one came. Two frames of one picture are taken only where no other picture is within `SPAN`
    frames (a scene at rest). At an end of the run the side that has frames gives all four; fewer than four when
    there are not that many within `SPAN`.
    """
    first = starts[place]
    last = (starts[place + 1] if place + 1 < len(starts) else length) - 1

    def side(pictures: list[int], frames: list[int], take: int) -> list[int]:
        """`take` frames from one side: a frame of each picture first, nearest first, then whatever frames follow."""
        reach = [j for j in pictures if first - SPAN <= j <= last + SPAN]
        return (reach + [j for j in frames if first - SPAN <= j <= last + SPAN and j not in reach])[:take]

    back, on = place - before, place + after
    earlier_pictures = [starts[q + 1] - 1 for q in range(back, -1, -1)] if back >= 0 else []
    earlier_frames = list(range(earlier_pictures[0], -1, -1)) if earlier_pictures else []
    later_pictures = [starts[q] for q in range(on, len(starts))]
    later_frames = list(range(later_pictures[0], length)) if later_pictures else []
    earlier, later = side(earlier_pictures, earlier_frames, 2), side(later_pictures, later_frames, 2)
    if len(earlier) < 2:
        later = side(later_pictures, later_frames, 4 - len(earlier))
    if len(later) < 2:
        earlier = side(earlier_pictures, earlier_frames, 4 - len(later))
    return sorted(earlier + later)


def _judge_picture(window: dict[int, _Frame], starts: Sequence[int], place: int, length: int
                   ) -> tuple[Tear | None, tuple[int, ...], bool]:
    """One picture of a run (`starts[place]` is its first frame): held against the frames next to it, then against
    frames further off, then against earlier frames. Returns the tear, if any, the frames it was held against, and
    whether the picture could be judged at all."""
    first = starts[place]
    last = (starts[place + 1] if place + 1 < len(starts) else length) - 1
    frame = window[first].picture
    width = frame.shape[1]
    # Where the picture is outside the frame just before it and the frame just after: worked out once, since most
    # pairs hold it against one of the two, and a picture that is outside neither is outside nothing.
    beside = {j: _outside(frame, [window[j]]) for j in (first - 1, last + 1) if 0 <= j < length}
    beside_count = {j: int(np.count_nonzero(mask)) for j, mask in beside.items()}
    judged = False
    for before, after in ((1, 1), *FURTHER_PAIRS):
        chosen = references_for(starts, place, length, before, after)
        if len(chosen) < 4:
            continue
        both_sides = chosen[0] < first and chosen[-1] > last
        judged = judged or both_sides
        # The plain case: one frame held against the four frames next to it, or the four on one side at an end of
        # the run. Anything else (a picture that came again, here or next to it; pictures further off) is guarded.
        plain = first == last and chosen in ([first - 2, first - 1, first + 1, first + 2],
                                             [first + 1, first + 2, first + 3, first + 4],
                                             [first - 4, first - 3, first - 2, first - 1])
        displaced_pixels = DISPLACED_PIXELS if (before, after) == (1, 1) else FURTHER_DISPLACED_PIXELS
        least = min(displaced_pixels, int(BAND_ROWS * BAND_SHARE * width))
        if both_sides and any(beside_count[j] < least for j in chosen if j in beside):
            continue
        outside = None
        for mask in (beside[j] for j in chosen if j in beside):
            outside = mask if outside is None else outside & mask
        others = [window[j] for j in chosen if j not in beside]
        if others:
            outside = _outside(frame, others) if outside is None else outside & _outside(frame, others)
        if both_sides and int(np.count_nonzero(outside)) < least:
            continue
        still = _agree([window[j].picture for j in chosen])
        if not both_sides:
            # The first or last frame of a run has frames on one side only. Something that moves on from where it
            # was in that frame would count as damage, so the frame is judged only when next to nothing moves there,
            # and only a band counts: an arm that leaves its place slowly is no band.
            if int(np.count_nonzero(still)) < AT_REST_SHARE * still.size:
                continue
            judged = True
            displaced_pixels = still.size + 1
        tear = judge(still, still & outside, displaced_pixels=displaced_pixels, guarded=not plain)
        if tear is not None:
            return tear, tuple(chosen), True
    if len(beside) == 2 and min(beside_count.values()) >= STALE_PIXELS:
        outside = beside[first - 1] & beside[last + 1]
        for back in STALE_BACK:
            # the frame before, and the pictures after: as many as the earlier frame lies back
            chosen = [first - 1, *starts[place + 1:place + 1 + back]]
            if first - back < 0 or len(chosen) < back + 1 or first - back < last - SPAN or chosen[-1] > last + SPAN:
                continue
            source = window[first - back].picture
            like_source = _agree([frame, source])
            if int(np.count_nonzero(like_source & outside)) < STALE_PIXELS:
                continue
            references = [window[j] for j in chosen]
            gone = _agree([reference.picture for reference in references]) & _outside(source, references)
            tear = judge_stale(np.count_nonzero(gone, axis=1), np.count_nonzero(gone & like_source, axis=1),
                               np.count_nonzero(like_source, axis=1), width)
            if tear is not None:
                return tear, (first - back,), True
    return None, (), judged


def scan_pictures(pictures: Iterable[np.ndarray], not_judged: list[int] | None = None) -> Iterator[tuple[int, Tear]]:
    """`(frame, tear)` for every torn frame of a run of frames, given its half-size pictures in the order they were
    recorded (`half_picture`): one take, or a video file that holds several takes one after the other. Where one take
    ends and the next begins nothing needs saying: the frames of the two agree where the scene is the same and the
    rule looks only there.

    Frames that show, value for value, the same picture one after the other are one picture: judged together,
    against the frames before the first of them and after the last, and stepped over as one. (*Seen* 2026-10-06:
    after a damaged frame the camera gave the same picture once to four times more, and a video codes a picture
    that comes again as itself, bit for bit. Held against each other, such frames would all pass.)

    The run's first and last frame have neighbours on one side only and are judged when nothing moves in those.
    A frame that could not be judged is added to `not_judged`. The pictures are taken one at a time and only those
    within `SPAN` frames of the picture in hand are kept.
    """
    window: dict[int, _Frame] = {}               # by frame; frames that show one picture share it
    starts: list[int] = []                       # the first frame of every picture still of use, in order
    seen = at = 0                                # frames taken; the place in `starts` of the next picture to judge

    def judge_next(length: int) -> Iterator[tuple[int, Tear]]:
        nonlocal at
        first = starts[at]
        last = (starts[at + 1] if at + 1 < len(starts) else length) - 1
        tear, against, judged = _judge_picture(window, starts, at, length)
        for frame in range(first, last + 1):
            if tear is not None:
                steps = tuple(j - frame for j in against)
                yield frame, Tear(tear.kind, tear.first_row_px, tear.last_row_px, tear.pixels, steps)
            elif not judged and not_judged is not None:
                not_judged.append(frame)
        at += 1
        keep_from = (starts[at] if at < len(starts) else length) - SPAN
        for old in [j for j in window if j < keep_from]:
            del window[old]
        while at > 4:                            # four pictures back is as far as any pair looks
            starts.pop(0)
            at -= 1

    for picture in pictures:
        if seen and np.array_equal(picture, window[seen - 1].picture):
            window[seen] = window[seen - 1]
        else:
            window[seen] = _prepare(picture)
            starts.append(seen)
        seen += 1
        # a picture is judged once every frame within SPAN of its last frame has come
        while at + 1 < len(starts) and starts[at + 1] - 1 + SPAN < seen:
            yield from judge_next(seen)
    while at < len(starts):
        yield from judge_next(seen)


def repeats(moved: np.ndarray, joint_step_deg: np.ndarray, keyframes: np.ndarray, *, moving_deg: float = MOVING_DEG,
            moving_pixels: int = MOVING_PIXELS, repeat_pixels: int = REPEAT_PIXELS) -> dict[str, Any]:
    """The frames of one take whose picture is the one before again although the arm was moving.

    `moved[i]` is how many pixels of frame i moved from frame i - 1 (`moved_pixels`), `joint_step_deg[i]` the largest
    step of a joint's reading from frame i - 1 to frame i, `keyframes[i]` whether the video codes frame i by itself.
    Only such frames are judged: a frame coded as a change of the one before is left unchanged by the encoder where
    little moved. A frame is judged when the arm stepped into it and into the frame before (a picture is a little
    older than the reading stored with it) and the picture was changing in the frames coded by themselves on either
    side of it, two before and two after. Returns how many frames were judged (`judged`) and the repeats among them
    (`repeats`, the frames' numbers).
    """
    moved, joint_step_deg, keyframes = np.asarray(moved), np.asarray(joint_step_deg), np.asarray(keyframes, bool)
    count = len(moved)
    judged = np.zeros(count, bool)
    if count >= 5:
        middle = slice(2, count - 2)
        judged[middle] = (keyframes[middle] & keyframes[:-4] & keyframes[4:]
                          & (joint_step_deg[middle] >= moving_deg) & (joint_step_deg[1:-3] >= moving_deg)
                          & (moved[:-4] >= moving_pixels) & (moved[4:] >= moving_pixels))
    return {"judged": int(np.count_nonzero(judged)),
            "repeats": [int(i) for i in np.flatnonzero(judged & (moved <= repeat_pixels))]}


def stood(same: np.ndarray, joint_step_deg: np.ndarray, *, frames: int = 3, moving_deg: float = MOVING_DEG
          ) -> list[tuple[int, int]]:
    """The pictures of one take that stood for `frames` frames or more while the arm was moving: `(first frame,
    frames)` of each. `same[i]` says that frame i is, value for value, frame i - 1 again; the arm counts as moving
    when a joint stepped by `moving_deg` into every frame of the stand after its first.

    Two frames alike say little: the video's encoder leaves a frame as the one before where little moved. A third
    is a frame the video codes by itself, and it comes out the same only when the camera gave the same picture.
    """
    same, joint_step_deg = np.asarray(same, bool), np.asarray(joint_step_deg)
    found = []
    first = 0
    for index in range(1, len(same) + 1):
        if index < len(same) and same[index]:
            continue
        count = index - first
        if count >= frames and bool((joint_step_deg[first + 1:index] >= moving_deg).all()):
            found.append((first, count))
        first = index
    return found


# ---- reading a dataset on disk (PyAV, pandas) ----


def _episode_table(root: Path) -> tuple[Any, float]:
    """The dataset's table of episodes and its frame rate."""
    import pandas as pd

    fps = float(json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"])
    # use_threads=False as in the audit's reader: with the reader's thread pool a process in the rig's environment
    # that has read parquet files failed to exit about half the time (measured 2026-10-05).
    tables = [pd.read_parquet(path, use_threads=False)
              for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))]
    return pd.concat(tables, ignore_index=True), fps


def cameras_of(root: str | Path) -> list[str]:
    """The cameras a dataset on disk has videos of, by their short names ("front", "wrist")."""
    prefix = "observation.images."
    return sorted(folder.name[len(prefix):] for folder in (Path(root) / "videos").glob(prefix + "*") if folder.is_dir())


def _video_files(root: Path, camera: str, episodes: Any, fps: float) -> list[tuple[Path, list[tuple[int, int, int]]]]:
    """Every video file of a camera with the takes it holds: `(episode, first frame in the file, frames)`, in order."""
    key = f"videos/observation.images.{camera}"
    files = []
    for (chunk, number), group in episodes.groupby([f"{key}/chunk_index", f"{key}/file_index"]):
        takes = sorted((int(round(float(row[f"{key}/from_timestamp"]) * fps)), int(row["episode_index"]),
                        int(row["length"])) for _, row in group.iterrows())
        path = (root / "videos" / f"observation.images.{camera}" / f"chunk-{int(chunk):03d}"
                / f"file-{int(number):03d}.mp4")
        files.append((path, [(episode, start, length) for start, episode, length in takes]))
    return files


def _planes(frame: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A decoded video frame's three planes: brightness at full size, the colour planes at half."""
    if frame.format.name not in ("yuv420p", "yuvj420p"):
        frame = frame.reformat(format="yuv420p")
    stacked = frame.to_ndarray()
    height, width = frame.height, frame.width
    quarter = height // 4
    return (stacked[:height], stacked[height:height + quarter].reshape(height // 2, width // 2),
            stacked[height + quarter:].reshape(height // 2, width // 2))


def scan_dataset(root: str | Path, cameras: Sequence[str] | None = None, *, measure_repeats: bool = False) -> Report:
    """Look at every frame of a dataset on disk; every video file is decoded once, from its first frame to its last.

    With `measure_repeats` the report also holds, for every frame, how many pixels moved from the frame before and
    whether the frame is coded by itself (what `repeats` wants).
    """
    import av

    root = Path(root)
    episodes, fps = _episode_table(root)
    report = Report(episodes=len(episodes), fps=fps)
    for camera in cameras if cameras is not None else cameras_of(root):
        report.frames[camera] = 0
        for path, takes in _video_files(root, camera, episodes, fps):
            total = max(start + length for _episode, start, length in takes)
            moved, keys, same = np.zeros(total, np.int32), np.zeros(total, bool), np.zeros(total, bool)

            def pictures(path: Path = path, total: int = total, moved: np.ndarray = moved, keys: np.ndarray = keys,
                         same: np.ndarray = same) -> Iterator[np.ndarray]:
                previous = None
                with av.open(str(path)) as container:
                    for index, frame in enumerate(itertools.islice(container.decode(video=0), total)):
                        picture = half_picture(*_planes(frame))
                        if measure_repeats:
                            keys[index] = bool(frame.key_frame)
                            if previous is not None:
                                same[index] = np.array_equal(previous, picture)
                                moved[index] = 0 if same[index] else moved_pixels(previous, picture)
                            previous = picture
                        report.frames[camera] += 1
                        yield picture

            def place(index: int, takes: list[tuple[int, int, int]] = takes) -> tuple[int, int] | None:
                """The take a frame of the file belongs to, and its number within it."""
                for episode, start, length in takes:
                    if start <= index < start + length:
                        return episode, index - start
                return None

            not_judged: list[int] = []
            for index, tear in scan_pictures(pictures(), not_judged):
                where = place(index)
                if where is not None:
                    report.findings.append(Finding(where[0], camera, where[1], where[1] / fps, tear))
            report.not_judged.extend((camera, *where) for where in map(place, not_judged) if where is not None)
            if measure_repeats:
                for episode, start, length in takes:
                    # a take's first frame is held against the take before it: that is no measure of anything
                    counts, alike = moved[start:start + length].copy(), same[start:start + length].copy()
                    counts[:1], alike[:1] = 0, False
                    report.moved.setdefault(camera, {})[episode] = counts
                    report.keyframes.setdefault(camera, {})[episode] = keys[start:start + length].copy()
                    report.same.setdefault(camera, {})[episode] = alike
    report.findings.sort(key=lambda finding: (finding.episode, finding.camera, finding.frame))
    return report


def joint_steps(root: str | Path) -> dict[int, np.ndarray]:
    """By episode: for every frame, the largest step of a joint's reading from the frame before, in degrees.

    The gripper's own joint is left out: jaws that close move nothing in the front camera's picture, and the arm
    would count as moving there while that camera has nothing new to show.
    """
    import pandas as pd

    root = Path(root)
    names = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8")).get("features", {}).get(
        "observation.state", {}).get("names") or []
    tables = [pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state"], use_threads=False)
              for path in sorted((root / "data").rglob("*.parquet"))]
    table = pd.concat(tables, ignore_index=True).sort_values(["episode_index", "frame_index"])
    steps = {}
    for episode, group in table.groupby("episode_index"):
        states = np.stack(group["observation.state"].to_numpy()).astype(float)
        if len(names) == states.shape[1]:
            states = states[:, [place for place, name in enumerate(names) if not str(name).startswith("gripper")]]
        step = np.zeros(len(states))
        step[1:] = np.abs(np.diff(states, axis=0)).max(axis=1)
        steps[int(episode)] = step
    return steps


def save_pictures(root: str | Path, findings: Sequence[Finding], folder: str | Path) -> list[Path]:
    """One picture per finding for a person to look at: the frame before, the frame, the frame after, side by side,
    with a red bar down both edges of the frame over the damaged rows. Returns the files written."""
    import av
    import cv2

    root, folder = Path(root), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    episodes, fps = _episode_table(root)
    written = []
    for camera in sorted({finding.camera for finding in findings}):
        for path, takes in _video_files(root, camera, episodes, fps):
            wanted: dict[int, list[tuple[Finding, int]]] = {}
            for finding in findings:
                for episode, start, length in takes:
                    if finding.camera == camera and finding.episode == episode:
                        for place, step in enumerate((-1, 0, 1)):
                            if 0 <= finding.frame + step < length:
                                wanted.setdefault(start + finding.frame + step, []).append((finding, place))
            if not wanted:
                continue
            tiles: dict[tuple[int, int], dict[int, np.ndarray]] = {}
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    for finding, place in wanted.get(index, ()):
                        tiles.setdefault((finding.episode, finding.frame), {})[place] = frame.to_ndarray(format="bgr24")
                    if index >= max(wanted):
                        break
            for finding in findings:
                got = tiles.get((finding.episode, finding.frame)) if finding.camera == camera else None
                if not got or 1 not in got:
                    continue
                middle = got[1].copy()
                middle[finding.tear.first_row_px:finding.tear.last_row_px, :4] = (0, 0, 255)
                middle[finding.tear.first_row_px:finding.tear.last_row_px, -4:] = (0, 0, 255)
                blank = np.zeros_like(middle)
                strip = np.hstack([got.get(0, blank), middle, got.get(2, blank)])
                label = np.zeros((22, strip.shape[1], 3), np.uint8)
                text = (f"episode {finding.episode}, {camera} camera, frame {finding.frame} ({finding.time_s:.3f} s): "
                        f"the frame before, the frame, the frame after. {finding.tear.words()}")
                cv2.putText(label, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                out = folder / f"episode-{finding.episode:03d}_{camera}_frame-{finding.frame:05d}.png"
                cv2.imwrite(str(out), np.vstack([label, strip]))
                written.append(out)
    return sorted(written)
