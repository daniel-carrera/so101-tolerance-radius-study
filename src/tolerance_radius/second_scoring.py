"""A second scoring of the counted trials, from video alone.

At the rig the person who scores an attempt is also the person who placed the cube, and a replay can often be told
from a policy by the way the arm moves. So a part of the counted trials is scored a second time by a person who sees
only the trial's video: not its condition, its block, its distance, its number or its first score
(`protocol/protocol.md`, section 8). The paper reports in how many of them the two scores agree.

Three steps, each a sub-command of `scripts/second_scoring.py` (`docs/runbook/05-second-scoring.md`):

* **the draw** (`read_population`, `draw_items`): which trials are scored again, each under a neutral item code, in a
  shuffled order;
* **the scoring**: one key per item, appended to a file at once;
* **the comparison**: the two scores side by side, after every item is scored.

Which trials are drawn
----------------------
The trials that can be drawn are the scored trials (not void) of the counted logs: logs that are not pilot logs,
every record run under a frozen protocol. They are split into groups, one for each condition a block tests:
T1a's clustered policy, T1b's replay, T2's clustered policy, T2's spread policy, T3's spread policy. **Each group
gives one trial in five, rounded up** (`quota`), so every condition of every block is there and the whole is a fifth
of the counted trials or slightly more: 29 of 132 for the plan of 2026-10-05 with 26 workspace pairs (34, 22, twice
26 and 24 trials give 7, 5, twice 6 and 5).

Within a group the trials are put in a random order, and the first ones of that order are taken. Two things can
make the draw pass a trial over, and in both the next trial of the same order takes its place:

* **its videos cannot be used**: a file is missing, does not decode, or holds fewer frames than the trial sent
  steps (the trial runner records a frame for every step, so such a video has lost some and may not show how the
  attempt ended). The trial is listed with the reason, and the list is kept with the draw and reported with the
  result. A group that cannot be filled this way stops the draw;
* **the other trial of its pair is already drawn.** The two trials of a workspace pair start from the same place, and
  a video shows where the cube starts: both in one set of items would tell the scorer that the two belong together
  and invite scoring one against the other. The groups are drawn in the protocol's order, so in T2 the clustered
  policy's trials are drawn first and a spread trial whose partner was drawn is passed over. With every pair
  complete and every video usable, every trial of a group still has the same chance of being drawn; what is
  excluded is only that both of a pair are. Such a trial is taken after all only when its group has nothing else
  left, and the draw's record then names the pair.

The drawn trials are then shuffled into the order they are shown in and numbered in that order (`item-01`,
`item-02`, ...).

**What the random orders depend on: the seed and the logs' own bytes.** Every stream is named by the seed and by
one SHA-256 of the logs (`Population.digest`). Two things follow, and both are wanted:

* nobody can work out which trials will be scored again before the last trial is logged, so nobody can score those
  with more care at the rig. With the seed alone the draw would be known in advance: the seed is public, and which
  places of a group are taken would depend on nothing but the group's size;
* nobody can choose the draw afterwards: the seed is fixed before the first trial and a log is never edited.

It also keeps an item's place in the list from giving its block away. Without the logs' hash in it, the order shown
would be the same shuffle for any 29 items, and the place of an item among them, counted in name order, says which
block it is from.

The same logs, the same seed and the same list of trials passed over for their videos always give the same items:
the comparison draws them again and refuses a key that differs.

What the scorer is given
------------------------
`make_draw` writes one folder (`protocol/data-contracts.md` has the formats):

    items.csv        an item code and, for each view, the name of a video under `videos/`: all the scoring reads
    videos/          a copy of each drawn trial's videos, named by item code and view
    key/key.csv      which trial each item is, with a digest of its copies: not for the scorer until every item
                     is scored
    key/draw.json    the draw's own record: seed, logs with their hashes, counts, trials passed over

**The videos are copied, not linked or named in place.** A trial's video is named after the trial
(`<time>_T1b-007_front.mp4`), and the trial's name carries its block, which for three blocks out of four is its
condition: the scorer's tool cannot be pointed at that file. A hard link would hide the name but keep the file's
date, which gives the order the trials were run in, and it cannot leave the disk it is on. A copy has a neutral name
and the date of the draw, and the folder can be handed to a second person whole, without `key/`. The copies are made
in the order the items are shown in. The names in `items.csv` are relative to the folder that holds it, so the
folder can be scored wherever it is put.

A video's own content cannot be hidden: it shows where the cube started, how the arm moved and how long the attempt
lasted.

The scoring
-----------
`score_items` shows each item that has no score yet, in the order of `items.csv`: the views side by side. An item
waits on its first picture until the scorer starts it, so that no video runs while the scorer is still reading or
looking for the window. It then runs from the first frame to the last, each picture staying up for the video's own
frame time (reading and drawing it come on top, so a video runs a little slower than life). The last picture then
stays, and only then do the score keys work: the rule is about where the cube is released, and an attempt can still
succeed in its last second. The scorer chooses success, failure or "cannot tell" and records the choice with the
Return key; until then the choice can be changed, so a slip of the key is not a score, and after it nothing can be.
Each score is appended to `scores.csv` and forced to disk before the next item is shown, so a sitting can stop at
any moment and the next one carries on at the first item without a score. With each score goes a digest of the video
files that were shown for it: the comparison holds it against the digest the draw recorded for that item's copies,
so scores that belong to another draw, or to copies that were swapped, are not taken for this one's.

What the three answers mean in the first scoring's terms: `success` is the first scoring's success (grade 5, "released
it in the basket"), `failure` is any of its failures (grades 0 to 4, whatever the label), and `cannot_tell` has no
counterpart there: the video does not show whether the cube was released in the basket. The second scoring has no
grades and no failure labels; it checks the one outcome the analysis uses.

Everything the scorer is shown or told comes from here: the item's code, its place in the list, the keys and the
rule. The window's title is one fixed text. The word "replay" is the name of a condition in this project, so the key
that shows a video again is called "watch again", and the count of it `watched_again`. It works once the video has
ended and counts whole showings, so a key held down counts once.

A key is read with its whole code (`Window.wait_key`, `_key`): OpenCV's usual `waitKey` hands back only the lowest
eight bits, which turns a Mac's right arrow and F10 into an Enter. Only the Return key records a choice.

The comparison
--------------
`compare` joins the scores, the key and the logs, and is refused until every item has a score: its output names
trials and their first scores, and a scorer who has items left must not see it. It also checks the draw before it
reports on it. The logs must be the very files the draw was made from (their hashes are in the draw's record). The
draw is then made again from them, and the key and the record must say what comes out. And each score must have been
given on the video files the draw copied for its item.

What is reported, for all items and for the policy trials and the replay trials apart: how many items; in how many
the two scores are the same; in how many they differ, each listed by its trial; in how many the scorer could not
tell; and the share that agree **among the items the scorer could tell**, with a Wilson 95 % interval. The four
counts of first score against second score are given too, because they show which way the disagreements run (a
scorer at the rig who knows the condition may be lenient or strict with it).

Cohen's kappa is left out on purpose. It corrects the share that agree for the agreement two scorers would reach by
chance, which matters when one outcome is rare; here most trials are placed by staircases, which gather them where
success and failure are about equally likely, and the four counts show any imbalance there is and let a reader
compute it. With some thirty items it would add a number whose interval says nothing the counts do not.

Standard library and NumPy only, so that it imports in every environment the project uses. OpenCV is needed to read
and show videos; it is imported inside the functions that use it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tolerance_radius import schema
from tolerance_radius.intervals import wilson_interval
from tolerance_radius.schema import TrialRecord
from tolerance_radius.seeds import derived_rng, shuffled

SCHEMA_VERSION = 1
#: One scored trial in this many is scored again, in every group.
ONE_IN = 5
#: A log whose file name starts with this is a pilot log (the trial runner's own rule).
PILOT_PREFIX = "pilot"
FROZEN = "frozen"
#: The name every random stream of the draw carries, so that the protocol's seed can be used for it.
STREAM = "second-scoring"
#: The camera views shown for an item, in the order they are laid side by side. The front camera is fixed and sees
#: the basket from above, so it shows where the cube ends up; the wrist camera rides on the gripper and shows the
#: grasp and the release from close by, and loses the cube once the arm moves on.
VIEWS = ("front", "wrist")
#: The files of a second scoring, inside its folder.
ITEMS_FILE = "items.csv"
SCORES_FILE = "scores.csv"
VIDEO_DIR = "videos"
KEY_DIR = "key"
KEY_FILE = "key.csv"
DRAW_FILE = "draw.json"
AGREEMENT_FILE = "agreement.json"
#: The level of the interval reported with the share of items that agree.
INTERVAL_LEVEL = 0.95
#: The two kinds of trial the agreement is also given for.
KINDS = ("policy", "replay")
_VIEW_NAME = re.compile(r"[a-z][a-z0-9]*")
#: What the second scorer can answer. The first two are the first scoring's two outcomes; the third is not a void.
SCORES = ("success", "failure", "cannot_tell")
SCORE_HEADER = ("item", "score", "scorer", "watched_again", "videos_sha256")
KEY_HEADER = ("item", "trial_id", "log", "videos_sha256")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: A person is named by a role code, never by a name or by initials: the review is double blind. The project has
#: two roles, the operator (`op1`) and a scorer (`sc1`, `sc2`), and nothing else is taken.
ROLE_CODE = re.compile(r"(?:op|sc)[0-9]{1,2}")
#: The scorer's keys.
KEY_OF_SCORE = {"success": "s", "failure": "f", "cannot_tell": "c"}
KEY_START = " "
KEY_AGAIN = "r"
KEY_STOP = "x"
#: A choice is recorded by the Return key and by no other. The keypad's Enter is left out on purpose: in a Mac's
#: window it is the same character as Ctrl+C (3), and a line feed (10) is Ctrl+J.
RETURN_CODE = 13
ESCAPE_CODE = 27
#: Some window systems put what was held with a key above the key's own sixteen bits. Shift, Caps Lock and Num Lock
#: change nothing here; a key pressed with Control, Alt or Command is no key at all.
_HARMLESS_FLAGS = 0x01 | 0x02 | 0x10
#: The scoring window's title: the same for every item.
WINDOW_TITLE = "second scoring"
#: How long the last picture waits for a key before it is drawn again, in milliseconds.
HOLD_MS = 50
#: How long a picture stays up when a video does not say its own rate: 30 pictures a second, the trials' control rate.
DEFAULT_FRAME_MS = 33
#: The scoring window is at least this wide, so that its two lines of text can be read under one view alone.
MIN_WINDOW_WIDTH_PX = 1200
#: What each view is, in the scorer's words.
VIEW_WORDS = {"front": "the fixed camera", "wrist": "the camera on the gripper"}
TASK = "pick up the orange cube and place it in the basket"
#: How many trial names a message lists before it says how many more there are.
_NAMES_SHOWN = 12


class SecondScoringError(Exception):
    """A step of the second scoring is refused. The message is one plain sentence: what was refused, and why."""


# --------------------------------------------------------------------------------------------------------------------
# The trials that can be drawn
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """A scored trial of a counted log: one that can be drawn."""

    #: The log it was read from, relative to the repository.
    log: str
    record: TrialRecord

    @property
    def trial_id(self) -> str:
        return self.record.planned.trial_id

    @property
    def block(self) -> str:
        return self.record.planned.block

    @property
    def condition(self) -> str:
        return self.record.planned.condition

    @property
    def pair_id(self) -> str | None:
        return self.record.planned.pair_id

    @property
    def kind(self) -> str:
        """`replay` or `policy`: what moved the arm, without saying which policy."""
        return "replay" if self.condition == "replay" else "policy"


@dataclass(frozen=True, slots=True)
class LogFacts:
    """A trial log as the draw found it."""

    file: str
    sha256: str
    n_records: int
    n_scored: int


@dataclass(frozen=True, slots=True)
class Population:
    """Everything a draw is made from: the logs as read, and their scored trials in the order written."""

    logs: tuple[LogFacts, ...]
    candidates: tuple[Candidate, ...]

    @property
    def digest(self) -> str:
        """One SHA-256 for all the logs as they are now: of their own SHA-256s, in name order.

        It changes with every line a log gains, so it is not known before the last trial is logged.
        """
        return hashlib.sha256("".join(facts.sha256 for facts in self.logs).encode("ascii")).hexdigest()


def is_pilot_log(path: str | Path) -> bool:
    """True for a log that holds pilot trials: its file name starts with `pilot`."""
    return Path(path).name.startswith(PILOT_PREFIX)


def read_population(logs: Sequence[str | Path], root: str | Path, *, rehearsal: bool = False) -> Population:
    """Read the trial logs and return their scored trials, the ones a draw is made from.

    The logs are read in name order, whatever order they are given in. Refused: a pilot log, a record that was not
    run under a frozen protocol, a trial name that occurs twice, a log that is missing or cannot be read, and logs
    without a scored trial.

    `rehearsal` turns the first two around, for trying the tool before the counted trials exist: pilot logs and
    records that were not run under a frozen protocol are taken, and a counted trial is refused, because a scorer who
    has met a counted trial in a rehearsal is no longer blind to it. A rehearsal is marked as one in the draw's
    record and in the comparison made from it.
    """
    root = Path(root)
    paths = sorted({Path(log) for log in logs}, key=lambda path: (_name(path, root), str(path)))
    if not paths:
        raise SecondScoringError("no trial log was given, so there is nothing to draw from")
    facts: list[LogFacts] = []
    candidates: list[Candidate] = []
    seen: dict[str, str] = {}
    for path in paths:
        name = _name(path, root)
        if is_pilot_log(path) and not rehearsal:
            raise SecondScoringError(
                f"{name} is a pilot log, and pilot trials are never scored again with the counted trials: "
                f"pass --rehearsal to try the tool on them"
            )
        if not path.is_file():
            raise SecondScoringError(f"the trial log {name} does not exist")
        try:
            records = schema.read_trials(path)
        except schema.SchemaError as error:
            raise SecondScoringError(f"the trial log {name} cannot be read: {_plain(error, path, name)}") from error
        unfrozen = [record for record in records if record.versions.get("protocol_status") != FROZEN]
        if unfrozen and not rehearsal:
            status = unfrozen[0].versions.get("protocol_status", "none")
            raise SecondScoringError(
                f"not run under a frozen protocol (status {status!r}), and so not counted: "
                f"{_listed([record.planned.trial_id for record in unfrozen])} ({name}); "
                f"pass --rehearsal to try the tool on such trials"
            )
        if rehearsal and not is_pilot_log(path) and len(unfrozen) < len(records):
            raise SecondScoringError(
                f"{name} holds counted trials, and a rehearsal is made only on trials that are not counted: a "
                f"scorer who has met a counted trial in a rehearsal is no longer blind to it"
            )
        for record in records:
            trial_id = record.planned.trial_id
            if trial_id in seen:
                raise SecondScoringError(
                    f"the trial name {trial_id} appears twice ({seen[trial_id]} and {name}): a trial is logged once"
                )
            seen[trial_id] = name
        scored = [Candidate(name, record) for record in records if record.scored]
        facts.append(LogFacts(name, hashlib.sha256(path.read_bytes()).hexdigest(), len(records), len(scored)))
        candidates.extend(scored)
    if not candidates:
        raise SecondScoringError(
            f"{' and '.join(fact.file for fact in facts)} hold no scored trial, so there is nothing to draw from"
        )
    return Population(tuple(facts), tuple(candidates))


# --------------------------------------------------------------------------------------------------------------------
# The draw
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Group:
    """The scored trials of one condition in one block, and how many of them were drawn."""

    block: str
    condition: str
    n_scored: int
    n_drawn: int


@dataclass(frozen=True, slots=True)
class PassedOver:
    """A trial that came up in the draw and was not taken because its videos cannot be used."""

    trial_id: str
    log: str
    reason: str


@dataclass(frozen=True, slots=True)
class DrawnItem:
    """One drawn trial under the neutral code the scorer knows it by."""

    code: str
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class Draw:
    """The outcome of a draw: the items in the order they are shown, and what the draw did to get them."""

    seed: int
    #: What names the random streams beside the seed: the digest of the logs the candidates were read from.
    salt: str
    one_in: int
    n_scored: int
    items: tuple[DrawnItem, ...]
    groups: tuple[Group, ...]
    passed_over: tuple[PassedOver, ...]
    #: Pairs of which both trials were drawn, because a group had nothing else left. Empty in an ordinary draw.
    whole_pairs: tuple[str, ...]


def quota(n_scored: int, one_in: int = ONE_IN) -> int:
    """How many trials a group of `n_scored` gives to the second scoring: one in `one_in`, rounded up.

    Rounded up so that a group with any trial gives at least one, and no group gives less than its fifth.
    """
    if not isinstance(n_scored, Integral) or isinstance(n_scored, bool) or n_scored < 0:
        raise ValueError(f"a group has a whole number of trials, got {n_scored!r}")
    if not isinstance(one_in, Integral) or isinstance(one_in, bool) or one_in < 1:
        raise ValueError(f"one trial in a whole number of trials is drawn, got {one_in!r}")
    return -(-int(n_scored) // int(one_in))


def draw_items(
    candidates: Sequence[Candidate],
    seed: int,
    salt: str,
    *,
    check: Callable[[Candidate], str | None] | None = None,
    one_in: int = ONE_IN,
) -> Draw:
    """Draw the trials that are scored again, and put them in the order they are shown in.

    The module's notes say exactly what is done. In short: the candidates are grouped by block and condition; each
    group, taken in the protocol's order, is shuffled and gives its first `quota` trials, passing over a trial whose
    videos cannot be used and a trial whose pair partner is already drawn; the drawn trials are shuffled again and
    numbered.

    `salt` names the random streams together with the seed. For a real draw it is the digest of the logs the
    candidates were read from (`Population.digest`), which is what keeps the draw from being known before the last
    trial. `check` is asked about a trial when it comes up, and only then: it returns None when the trial's videos
    can be used, or the reason why not in a few words. Without it every trial can be used. The result depends on the
    candidates, the seed, the salt, `one_in` and what `check` answers, and on nothing else: not on the order the
    candidates are given in.
    """
    if not isinstance(salt, str):
        raise TypeError(f"the salt is text, such as the digest of the logs, got {salt!r}")
    if not candidates:
        raise SecondScoringError("there is no scored trial to draw from")
    names = [candidate.trial_id for candidate in candidates]
    if len(set(names)) != len(names):
        twice = sorted({name for name in names if names.count(name) > 1})
        raise SecondScoringError(f"a trial is drawn from once, and these are listed twice: {_listed(twice)}")

    members: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in candidates:
        members.setdefault((candidate.block, candidate.condition), []).append(candidate)
    in_protocol_order = sorted(members, key=lambda key: (schema.BLOCKS.index(key[0]), schema.CONDITIONS.index(key[1])))

    taken: list[Candidate] = []
    groups: list[Group] = []
    passed_over: list[PassedOver] = []
    pairs_taken: set[tuple[str, str]] = set()
    whole_pairs: list[str] = []

    def usable(candidate: Candidate) -> bool:
        reason = check(candidate) if check is not None else None
        if reason is not None:
            passed_over.append(PassedOver(candidate.trial_id, candidate.log, reason))
        return reason is None

    for block, condition in in_protocol_order:
        group = sorted(members[(block, condition)], key=lambda candidate: candidate.trial_id)
        order = shuffled(group, derived_rng(seed, STREAM, salt, "draw", block, condition))
        wanted = quota(len(group), one_in)
        chosen: list[Candidate] = []
        partners: list[Candidate] = []
        for candidate in order:
            if len(chosen) == wanted:
                break
            if candidate.pair_id is not None and (block, candidate.pair_id) in pairs_taken:
                partners.append(candidate)
            elif usable(candidate):
                chosen.append(candidate)
        for candidate in partners:  # only when the group has nothing else left
            if len(chosen) == wanted:
                break
            if usable(candidate):
                chosen.append(candidate)
                whole_pairs.append(candidate.pair_id)
        if len(chosen) < wanted:
            raise SecondScoringError(_cannot_fill(block, condition, group, wanted, check))
        pairs_taken.update((block, candidate.pair_id) for candidate in chosen if candidate.pair_id is not None)
        taken.extend(chosen)
        groups.append(Group(block, condition, len(group), len(chosen)))

    in_name_order = sorted(taken, key=lambda candidate: candidate.trial_id)
    shown = shuffled(in_name_order, derived_rng(seed, STREAM, salt, "order"))
    digits = max(2, len(str(len(shown))))
    items = tuple(DrawnItem(f"item-{number:0{digits}d}", candidate) for number, candidate in enumerate(shown, start=1))
    return Draw(seed, salt, int(one_in), len(candidates), items, tuple(groups), tuple(passed_over),
                tuple(sorted(whole_pairs)))


def _cannot_fill(
    block: str,
    condition: str,
    group: Sequence[Candidate],
    wanted: int,
    check: Callable[[Candidate], str | None] | None,
) -> str:
    """Why a group could not give its trials: every trial of it whose videos cannot be used, in name order.

    All of them are named, not only the ones the draw came to, so that the message says nothing about the order the
    seed puts the group in.
    """
    without = [(candidate.trial_id, check(candidate) if check is not None else None) for candidate in group]
    without = [(trial_id, reason) for trial_id, reason in without if reason is not None]
    listed = ", ".join(f"{trial_id} ({reason})" for trial_id, reason in without[:_NAMES_SHOWN])
    if len(without) > _NAMES_SHOWN:
        listed += f" and {len(without) - _NAMES_SHOWN} more"
    return (
        f"block {block} ({condition}): {wanted} of its {len(group)} scored trials are to be drawn, and "
        f"{len(group) - len(without)} have videos that can be used; {len(without)} of them do not: {listed}"
    )


# --------------------------------------------------------------------------------------------------------------------
# The videos of a trial
# --------------------------------------------------------------------------------------------------------------------


class VideoError(Exception):
    """A file cannot be read as a video. The message completes the sentence "its front video ..."."""


class VideoCheck:
    """Whether a trial's videos can be used for the second scoring: what `draw_items` asks about each trial it meets.

    A trial can be used when its record names a video for every view, each name is a path inside the repository, the
    file is there, it decodes to at least one picture, and it holds no fewer frames than the trial sent steps.
    `probe` counts the frames of a video file that decode, or raises `VideoError`; the real one (`probe_video`) reads
    the file to its end with OpenCV.

    Why the last condition: the trial runner records one frame for every control step it sends (one more for each
    start the handoff gate refused), so a video with fewer frames than its trial's `steps_sent` has lost some, and
    nothing says where. If they were its last ones, the scorer would see a cube still in the gripper "at the end"
    and rightly call it a failure by the rule he was given. A record that does not say how many steps were sent is
    not held to this.
    """

    def __init__(self, root: str | Path, views: Sequence[str] = VIEWS,
                 probe: Callable[[Path], int] | None = None) -> None:
        views = tuple(views)
        if not views or len(set(views)) != len(views):
            raise SecondScoringError(f"the views to show are one or more camera names, each given once, got {views}")
        for view in views:
            if not _VIEW_NAME.fullmatch(view) or view == "item":
                raise SecondScoringError(f"a view is a camera's plain name, such as front or wrist, got {view!r}")
        self.views = views
        self._root = Path(root)
        self._probe = probe if probe is not None else probe_video

    def __call__(self, candidate: Candidate) -> str | None:
        """None when the trial's videos can be used, or the reason why not, in a few words."""
        steps = candidate.record.execution.get("steps_sent")
        for view in self.views:
            located = self._locate(candidate, view)
            if isinstance(located, str):
                return located
            try:
                frames = int(self._probe(located))
            except VideoError as error:
                return f"its {view} video {error}"
            if frames < 1:
                return f"its {view} video holds no picture"
            if isinstance(steps, int) and not isinstance(steps, bool) and frames < steps:
                return f"its {view} video holds {frames} frames and the trial sent {steps} steps"
        return None

    def sources(self, candidate: Candidate) -> dict[str, Path]:
        """Where a usable trial's videos are, by view, in the order shown."""
        return {view: self._root / candidate.record.media[view] for view in self.views}

    def _locate(self, candidate: Candidate, view: str) -> Path | str:
        """The file of one view of a trial, or the reason it cannot be used."""
        name = candidate.record.media.get(view)
        if not name:
            return f"its record names no {view} video"
        if Path(name).is_absolute():
            return f"its record names its {view} video by an absolute path"
        if ".." in Path(name).parts:  # judged by the name alone: a folder of recordings may be a link to another disk
            return f"its record names its {view} video outside the repository"
        path = self._root / name
        if not path.is_file():
            return f"its {view} video is not there"
        return path


def probe_video(path: str | Path) -> int:
    """How many frames of a video file decode, reading it to its end. Raises `VideoError` when it does not open.

    It is given a file, never a camera's number: nothing here can open a camera.
    """
    capture = _open_video(path)
    try:
        if not capture.isOpened():
            raise VideoError("does not open as a video")
        frames = 0
        while capture.read()[0]:
            frames += 1
        return frames
    finally:
        capture.release()


def _opencv() -> Any:
    """OpenCV, which only the rig's interpreter has. Imported here so that the module imports without it."""
    try:
        import cv2
    except ImportError as error:
        raise SecondScoringError(
            "the videos are read and shown with OpenCV, and this interpreter has none: run the command with the "
            "rig's interpreter (scripts/second_scoring.py starts with it)"
        ) from error
    return cv2


def _open_video(path: str | Path) -> Any:
    """An OpenCV reader on a video file. The file's reader is named outright where the build has it, so that a file
    that does not open is not handed on to the camera framework, which prints the file's name."""
    if not isinstance(path, (str, Path)):
        raise TypeError(f"a video is a file, got {path!r}")
    cv2 = _opencv()
    if cv2.CAP_FFMPEG in cv2.videoio_registry.getBackends():
        return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    return cv2.VideoCapture(str(path))


# --------------------------------------------------------------------------------------------------------------------
# Writing the draw
# --------------------------------------------------------------------------------------------------------------------


def make_draw(
    logs: Sequence[str | Path],
    seed: int,
    out_dir: str | Path,
    *,
    root: str | Path,
    views: Sequence[str] = VIEWS,
    rehearsal: bool = False,
    probe: Callable[[Path], int] | None = None,
    copy: Callable[[Path, Path], Any] = shutil.copyfile,
) -> dict[str, Any]:
    """Draw the items of a second scoring and write its folder. Returns the draw's record, as `key/draw.json` has it.

    `logs` are the counted trial logs, `root` the repository the logs' video names are relative to. The folder must
    not hold anything yet: a draw is never made into a folder that has scores, a draw, or anything else. Nothing is
    written before the draw itself has succeeded, and `items.csv` is written last: a folder that has one is complete,
    and one that was cut short while the videos were being copied has none.
    """
    root, out_dir = Path(root), Path(out_dir)
    out_name = _name(out_dir, root)
    if (out_dir / SCORES_FILE).exists():
        raise SecondScoringError(
            f"{out_name} already holds scores: a draw is never made again into a folder that is being scored"
        )
    if (out_dir / ITEMS_FILE).exists():
        raise SecondScoringError(f"{out_name} already holds a draw: name another folder for another draw")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise SecondScoringError(
            f"{out_name} is not empty: a draw is made into a new folder (a draw that was cut short leaves a folder "
            f"without {ITEMS_FILE}, to be removed by hand)"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SecondScoringError(f"the seed is a whole number, got {seed!r}")

    population = read_population(logs, root, rehearsal=rehearsal)
    check = VideoCheck(root, views, probe)
    draw = draw_items(population.candidates, seed, population.digest, check=check)
    record = _draw_record(draw, population, check.views, rehearsal)

    rows, key_rows = [], []
    try:
        (out_dir / VIDEO_DIR).mkdir(parents=True, exist_ok=True)
        (out_dir / KEY_DIR).mkdir(exist_ok=True)
        for item in draw.items:  # in the order shown: the copies' dates then say nothing about the order run
            copies = []
            for view, source in check.sources(item.candidate).items():
                copies.append(f"{VIDEO_DIR}/{item.code}_{view}{source.suffix.lower()}")
                copy(source, out_dir / copies[-1])
            rows.append([item.code, *copies])
            key_rows.append([item.code, item.candidate.trial_id, item.candidate.log,
                             videos_digest([out_dir / name for name in copies])])
    except OSError as error:
        # The system's own message names the trial's file beside the item's: it is not passed on.
        raise SecondScoringError(
            f"a video could not be copied into {out_name} ({error.strerror or type(error).__name__}): the draw was "
            f"cut short, and the folder is to be removed by hand before it is made again"
        ) from None
    _write_csv(out_dir / KEY_DIR / KEY_FILE, KEY_HEADER, key_rows)
    (out_dir / KEY_DIR / DRAW_FILE).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / ITEMS_FILE, ["item", *check.views], rows)
    return record


def _draw_record(draw: Draw, population: Population, views: Sequence[str], rehearsal: bool) -> dict[str, Any]:
    """A draw as `key/draw.json` records it. The comparison builds it again from the logs and expects the same."""
    return {
        "schema_version": SCHEMA_VERSION,
        "rehearsal": bool(rehearsal),
        "seed": draw.seed,
        "one_in": draw.one_in,
        "views": list(views),
        "logs": [asdict(facts) for facts in population.logs],
        "logs_digest": draw.salt,
        "n_scored": draw.n_scored,
        "n_drawn": len(draw.items),
        "groups": [asdict(group) for group in draw.groups],
        "passed_over": [asdict(passed) for passed in draw.passed_over],
        "whole_pairs": list(draw.whole_pairs),
    }


def videos_digest(files: Sequence[str | Path]) -> str:
    """One SHA-256 for an item's video files, in the order shown: of the files' own SHA-256s.

    The draw records it for the copies it made, and the scoring writes it with each score for the files it showed, so
    that the comparison can tell a score that was given on other videos.
    """
    digests = []
    for file in files:
        digest = hashlib.sha256()
        with open(file, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digests.append(digest.hexdigest())
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


def draw_lines(record: Mapping[str, Any], out_name: str) -> list[str]:
    """What the draw command prints: counts, and no trial's name, because the person who runs it may be the scorer."""
    logs = " and ".join(log["file"] for log in record["logs"])
    lines = []
    if record["rehearsal"]:
        lines.append("REHEARSAL: drawn from trials that are not counted. Nothing made from this draw is a result.")
    lines.append(
        f"drew {record['n_drawn']} of the {record['n_scored']} scored trials of {logs} (seed {record['seed']}): "
        f"one in {record['one_in']} of each condition of each block, rounded up"
    )
    for group in record["groups"]:
        lines.append(f"  {group['block']:<3}  {group['condition']:<13}  {group['n_drawn']} of {group['n_scored']}")
    lines.append(f"  shown for each item: {' and '.join(record['views'])}, copied under neutral names")
    passed, whole = len(record["passed_over"]), len(record["whole_pairs"])
    if passed:
        lines.append(
            f"  NOTE: {_count(passed, 'trial')} came up and {'was' if passed == 1 else 'were'} passed over because "
            f"{'its' if passed == 1 else 'their'} videos cannot be used; the next of the same order took "
            f"{'its' if passed == 1 else 'their'} place (listed in {KEY_DIR}/{DRAW_FILE})"
        )
    else:
        lines.append("  every drawn video could be read to its end, and none has fewer frames than its trial sent "
                     "steps")
    if whole:
        lines.append(
            f"  NOTE: both trials of {_count(whole, 'pair')} were drawn, because a group had no other trial left: "
            f"two items start from the same place (the pairs are named in {KEY_DIR}/{DRAW_FILE})"
        )
    lines.append(f"wrote {out_name}/{ITEMS_FILE}, {VIDEO_DIR}/, {KEY_DIR}/{KEY_FILE} and {KEY_DIR}/{DRAW_FILE}")
    lines.append(f"the scorer does not open {out_name}/{KEY_DIR}/ before every item is scored")
    lines.append(f"next: scripts/second_scoring.py score --items {out_name}/{ITEMS_FILE} --scorer <role code>")
    return lines


# --------------------------------------------------------------------------------------------------------------------
# The items and the scores, as files
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Item:
    """One thing to score: a neutral code and the video file of each view, in the order they are laid side by side."""

    code: str
    videos: tuple[tuple[str, Path], ...]


@dataclass(frozen=True, slots=True)
class Score:
    """One line of `scores.csv`: the second score of an item, who gave it, how often the video was started again,
    and the digest of the video files that were shown for it (`videos_digest`)."""

    item: str
    score: str
    scorer: str
    watched_again: int
    videos_sha256: str


def check_role_code(code: str) -> str:
    """The code, if it is a role code such as `sc2` or `op1`; refused otherwise, so that no name reaches a file."""
    if not isinstance(code, str) or not ROLE_CODE.fullmatch(code):
        raise SecondScoringError(
            f"a person is named by a role code, op or sc and a number such as op1 or sc2, never by a name or by "
            f"initials: got {code!r}"
        )
    return code


def read_items(path: str | Path) -> list[Item]:
    """The items of a second scoring, in the order they are shown. Video names are taken from the file's own folder."""
    path = Path(path)
    name = _short(path)
    if not path.is_file():
        raise SecondScoringError(f"there is no {name}: make the draw first (scripts/second_scoring.py draw)")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if len(rows) < 2:
        raise SecondScoringError(f"{name} holds no items")
    header = rows[0]
    views = header[1:]
    if header[0] != "item" or not views or len(set(views)) != len(views) or not all(
            _VIEW_NAME.fullmatch(view) and view != "item" for view in views):
        raise SecondScoringError(f"{name}: the first line must be item and then one camera name for each view")
    items: list[Item] = []
    seen: set[str] = set()
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header) or not row[0].strip():
            raise SecondScoringError(f"{name}, line {number}: an item is a code and one video for each view")
        code = row[0]
        if code in seen:
            raise SecondScoringError(f"{name}, line {number}: the item {code} is listed twice")
        seen.add(code)
        for video in row[1:]:
            if not video or Path(video).is_absolute() or ".." in Path(video).parts:
                raise SecondScoringError(
                    f"{name}, line {number}: a video is named by its place inside the folder that holds {path.name}"
                )
        items.append(Item(code, tuple((view, path.parent / video) for view, video in zip(views, row[1:]))))
    return items


def read_scores(path: str | Path) -> list[Score]:
    """The scores given so far, in the order given. No file yet means no score yet."""
    path = Path(path)
    name = _short(path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    if tuple(rows[0]) != SCORE_HEADER:
        raise SecondScoringError(f"{name}: the first line must be {','.join(SCORE_HEADER)}")
    scores = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(SCORE_HEADER):
            raise SecondScoringError(f"{name}, line {number}: a score has {len(SCORE_HEADER)} entries, got {len(row)}")
        item, score, scorer, again, digest = row
        if score not in SCORES:
            raise SecondScoringError(f"{name}, line {number}: the score is one of {', '.join(SCORES)}, got {score!r}")
        try:
            check_role_code(scorer)
        except SecondScoringError as error:
            raise SecondScoringError(f"{name}, line {number}: {error}") from error
        if not again.isdigit():
            raise SecondScoringError(f"{name}, line {number}: watched_again is a count, got {again!r}")
        if not _SHA256.fullmatch(digest):
            raise SecondScoringError(f"{name}, line {number}: videos_sha256 is a SHA-256 in small letters")
        scores.append(Score(item, score, scorer, int(again), digest))
    return scores


def append_score(path: str | Path, score: Score) -> None:
    """Append one score and force it to disk before returning, under the heading if the file is new."""
    path = Path(path)
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new:
            writer.writerow(SCORE_HEADER)
        writer.writerow([score.item, score.score, score.scorer, score.watched_again, score.videos_sha256])
        handle.flush()
        os.fsync(handle.fileno())


def unscored(items: Sequence[Item], scores: Sequence[Score], name: str = SCORES_FILE) -> list[Item]:
    """The items that have no score yet, in the order shown. A score for no item, or a second one, is refused."""
    codes = {item.code for item in items}
    done: set[str] = set()
    for score in scores:
        if score.item not in codes:
            raise SecondScoringError(f"{name} holds a score for {score.item}, which is not an item of this draw")
        if score.item in done:
            raise SecondScoringError(f"{name} holds a score for {score.item} twice: an item is scored once")
        done.add(score.item)
    return [item for item in items if item.code not in done]


# --------------------------------------------------------------------------------------------------------------------
# The scoring
# --------------------------------------------------------------------------------------------------------------------


class Player(Protocol):
    """Reads an item's videos. The real one is `VideoPlayer`; the tests have their own."""

    def frames(self, item: Item) -> Iterable[Any]:
        """The item's pictures from the first to the last, the views side by side: a fresh run at every call."""

    def frame_ms(self, item: Item) -> int:
        """How long one picture stays up, in milliseconds, for the video to run at its own rate."""


class Screen(Protocol):
    """Shows a picture with lines of text and waits for a key. The real one is `Window`; the tests have their own."""

    def show(self, picture: Any, lines: Sequence[str]) -> None: ...

    def wait_key(self, ms: int) -> int:
        """Wait up to `ms` milliseconds for a key: its whole code, or -1 when there was none (as OpenCV's
        `waitKeyEx`, not its `waitKey`)."""

    def close(self) -> None: ...


def instructions(items: Sequence[Item]) -> list[str]:
    """What the scorer is told at the start of every sitting: the rule, in the first scoring's words, and the keys."""
    views = [VIEW_WORDS.get(view, f"the {view} camera") for view, _ in items[0].videos]
    pictures = f"From left to right: {', '.join(views)}." if len(views) > 1 else f"The picture is {views[0]}."
    return [
        "",
        f"Second scoring from video. Each item is one attempt at the task: {TASK}.",
        f"{pictures} There is no sound.",
        "",
        "Click the video window so that it gets the keys. Each item waits on its first picture.",
        "  space   start the video, and watch it to its end. The last picture stays, and then:",
        "  s       success      the cube was released in the basket",
        "  f       failure      it was not: never picked up, dropped outside the basket, still in the gripper at "
        "the end",
        "  c       cannot tell  the video does not show whether the cube was released in the basket",
        "  Enter   (the Return key) records your choice. Until then you can choose again; after it the score cannot "
        "be changed.",
        "  r       watch the video again",
        "  x       stop for now, at any moment: the same command carries on at the first item without a score",
        "",
        "Score what the video shows: not what nearly happened, and not what you remember or expect.",
        "If a video does not play, stop and say so. Cannot tell is for a video that plays and does not show it.",
        "",
    ]


def score_items(
    items_path: str | Path,
    scorer: str,
    *,
    player: Player,
    screen: Screen,
    say: Callable[[str], None] = print,
) -> tuple[int, int]:
    """One sitting of the second scoring. Returns how many items were scored in it, and how many are still to score.

    Every item without a score is shown, in the order of the item list, and its score is appended to `scores.csv`
    beside the list as soon as it is recorded. The sitting ends when every item has a score or the scorer stops.
    `scorer` is the role code written with each score.
    """
    scorer = check_role_code(scorer)
    items_path = Path(items_path)
    items = read_items(items_path)
    scores_path = items_path.parent / SCORES_FILE
    todo = unscored(items, read_scores(scores_path))
    total = len(items)
    if not todo:
        say(f"all {total} items are scored: there is nothing left to show.")
        return 0, 0
    absent = [item.code for item in todo if not all(file.is_file() for _, file in item.videos)]
    if absent:
        raise SecondScoringError(f"a video of {_listed(absent)} is not in the folder: nothing was shown")
    for line in instructions(items):
        say(line)
    say(f"{total} items, {total - len(todo)} of them scored so far. The next is {todo[0].code}.")
    done = 0
    try:
        for item in todo:
            try:
                shown = videos_digest([file for _, file in item.videos])
            except OSError as error:
                raise SecondScoringError(
                    f"a video of {item.code} cannot be read ({error.strerror or type(error).__name__}): nothing was "
                    f"scored for it"
                ) from None
            score, again = _score_one(item, items.index(item) + 1, total, player, screen)
            if score is None:
                say(f"stopped at {item.code}, which has no score yet: {total - len(todo) + done} of {total} items "
                    f"are scored. The same command carries on from here.")
                return done, len(todo) - done
            if any(earlier.item == item.code for earlier in read_scores(scores_path)):
                raise SecondScoringError(
                    f"{item.code} was given a score by another sitting while this one was showing it: an item is "
                    f"scored once, so this sitting stops and its own score for {item.code} was not written"
                )
            append_score(scores_path, Score(item.code, score, scorer, again, shown))
            done += 1
            say(f"  {item.code}: {_spoken(score)}")
    finally:
        screen.close()
    say(f"all {total} items are scored. Say so to the person who made the draw; the key stays closed until then.")
    return done, 0


def _score_one(item: Item, position: int, total: int, player: Player, screen: Screen) -> tuple[str | None, int]:
    """Show one item until it is scored. Returns the score, or None when the scorer stops for now, and how many
    times the video was shown again.

    The item waits on its first picture until the scorer starts it. While the video runs only the stop key does
    anything; the score keys and "watch again" work once its last picture is up. So a key held down or pressed too
    soon is used up by the pictures that pass and never becomes a score, and "watch again" counts whole showings.
    """
    head = f"{item.code}     {position} of {total}"
    keys = "   ".join(f"{key}: {_spoken(score)}" for score, key in KEY_OF_SCORE.items())
    score_of_key = {key: score for score, key in KEY_OF_SCORE.items()}
    frame_ms = max(1, int(player.frame_ms(item)))
    again = 0
    waiting = True
    while True:
        last = None
        frames = player.frames(item)
        try:
            for picture in frames:
                last = picture
                while waiting:
                    screen.show(picture, [head, f"space: start the video        {KEY_STOP}: stop for now"])
                    key = _key(screen.wait_key(HOLD_MS))
                    if key == KEY_STOP:
                        return None, again
                    waiting = key != KEY_START
                screen.show(picture, [head, f"watch to the end        {KEY_STOP}: stop for now"])
                if _key(screen.wait_key(frame_ms)) == KEY_STOP:
                    return None, again
        finally:
            close = getattr(frames, "close", None)  # a real video's reader is given back at once
            if close is not None:
                close()
        if last is None:
            raise SecondScoringError(f"the video of {item.code} gave no picture, so nothing was scored for it")
        chosen: str | None = None
        while True:
            if chosen is None:
                ask = f"the video has ended.   {keys}        {KEY_AGAIN}: watch again     {KEY_STOP}: stop for now"
            else:
                ask = (f"{_spoken(chosen).upper()}?   Enter: record it     {' / '.join(score_of_key)}: choose again"
                       f"        {KEY_AGAIN}: watch again     {KEY_STOP}: stop for now")
            screen.show(last, [head, ask])
            key = _key(screen.wait_key(HOLD_MS))
            if key in score_of_key:
                chosen = score_of_key[key]
            elif key == "enter" and chosen is not None:
                return chosen, again
            elif key == KEY_AGAIN:
                again += 1
                break
            elif key == KEY_STOP:
                return None, again


class VideoPlayer:
    """Reads an item's videos with OpenCV, one frame of each view at a time, and lays the views side by side.

    The trial runner writes one frame of every camera per control step, so frame `n` of each view is the same
    moment. A view that ends before another keeps its last picture until the longest one ends.
    """

    def __init__(self) -> None:
        _opencv()  # refuse now, before anything is shown, where there is no OpenCV

    def frame_ms(self, item: Item) -> int:
        cv2 = _opencv()
        capture = _open_video(item.videos[0][1])
        try:
            rate = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
        finally:
            capture.release()
        return round(1000.0 / rate) if 1.0 <= rate <= 240.0 else DEFAULT_FRAME_MS

    def frames(self, item: Item) -> Iterator[np.ndarray]:
        captures = [_open_video(file) for _, file in item.videos]
        try:
            latest: list[np.ndarray | None] = [None] * len(captures)
            while True:
                fresh = False
                for index, capture in enumerate(captures):
                    read, frame = capture.read()
                    if read:
                        latest[index], fresh = frame, True
                if not fresh:
                    return
                if any(frame is None for frame in latest):
                    raise SecondScoringError(f"one of the videos of {item.code} gave no picture")
                yield side_by_side(latest)
        finally:
            for capture in captures:
                capture.release()


def side_by_side(pictures: Sequence[np.ndarray]) -> np.ndarray:
    """The pictures in one row, left to right, each brought to the height of the first."""
    height = pictures[0].shape[0]
    row = []
    for picture in pictures:
        if picture.shape[0] != height:
            width = max(1, round(picture.shape[1] * height / picture.shape[0]))
            picture = _opencv().resize(picture, (width, height))
        row.append(picture)
    return np.hstack(row)


def with_lines(picture: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """The picture with the lines of text on a dark strip under it: no text ever covers the video.

    A picture narrower than `MIN_WINDOW_WIDTH_PX` is centred on a canvas of that width, and a line too long for the
    canvas is drawn smaller.
    """
    cv2 = _opencv()
    height, width = picture.shape[:2]
    canvas_width = max(width, MIN_WINDOW_WIDTH_PX)
    line_px, font = 28, cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.zeros((height + 12 + line_px * len(lines), canvas_width, 3), np.uint8)
    left = (canvas_width - width) // 2
    canvas[:height, left:left + width] = picture
    for row, text in enumerate(lines):
        scale = 0.6
        (text_width, _), _ = cv2.getTextSize(text, font, scale, 1)
        if text_width > canvas_width - 24:
            scale *= (canvas_width - 24) / text_width
        cv2.putText(canvas, text, (12, height + 24 + line_px * row), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class Window:
    """The scoring window: OpenCV's, under one fixed title. It shows only what `score_items` hands it."""

    def __init__(self) -> None:
        _opencv()

    def show(self, picture: np.ndarray, lines: Sequence[str]) -> None:
        cv2 = _opencv()
        try:
            cv2.imshow(WINDOW_TITLE, with_lines(picture, lines))
        except cv2.error as error:
            raise SecondScoringError(
                "this OpenCV cannot open a window (it was built without one): run the command with the rig's "
                "interpreter"
            ) from error

    def wait_key(self, ms: int) -> int:
        """The whole code of the key pressed within `ms` milliseconds, or -1.

        OpenCV's `waitKeyEx`, never its `waitKey`, which hands back only the lowest eight bits of the code (see
        `_key` for what that would do).
        """
        return _opencv().waitKeyEx(max(1, int(ms)))

    def close(self) -> None:
        cv2 = _opencv()
        cv2.destroyAllWindows()
        cv2.waitKeyEx(1)  # the window system closes the window only when it is given a moment


def _key(code: int) -> str | None:
    """A key as the scoring names it: a small letter, a space, `enter` for Return, the stop key for Escape, or None.

    `code` is the key's whole code, as OpenCV's `waitKeyEx` gives it. It must never be cut down to its lowest eight
    bits, which is what OpenCV's `waitKey` does: on a Mac the right arrow is 0xF703 and F10 is 0xF70D, and cut down
    they read as 3 and 13, an Enter that would record a score nobody confirmed. An arrow, a function key, the
    keypad's Enter, a key pressed with Control, Alt or Command, and "no key" (-1) are all nothing.
    """
    if code is None or code < 0:
        return None
    flags, code = code >> 16, code & 0xFFFF
    if flags & ~_HARMLESS_FLAGS:
        return None
    if code == RETURN_CODE:
        return "enter"
    if code == ESCAPE_CODE:
        return KEY_STOP
    if 32 <= code < 127:
        return chr(code).lower()
    return None


def _spoken(score: str) -> str:
    """A score as the scorer reads it: `cannot tell` for `cannot_tell`."""
    return score.replace("_", " ")


# --------------------------------------------------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------------------------------------------------


def compare(scoring_dir: str | Path, logs: Sequence[str | Path], *, root: str | Path) -> dict[str, Any]:
    """The two scores of every item side by side. Returns the agreement, as `agreement.json` has it.

    `logs` are the trial logs the draw was made from. Refused: a folder without a draw; logs that are not the files
    the draw was made from; a key or a record of the draw that is not what the recorded seed draws from them; an item
    list that is not the key's; scores that do not cover every item once; and a score that was given on other video
    files than the copies the draw made for its item.
    """
    root, scoring_dir = Path(root), Path(scoring_dir)
    folder = _name(scoring_dir, root)
    record = _read_draw_record(scoring_dir, folder)

    made_from = [log["file"] for log in record["logs"]]
    given = sorted({_name(Path(log), root) for log in logs})
    if given != sorted(made_from):
        raise SecondScoringError(
            f"the draw was made from {' and '.join(made_from)}, and the comparison was given "
            f"{' and '.join(given) or 'no log'}: give it the same logs"
        )
    population = read_population(logs, root, rehearsal=bool(record["rehearsal"]))
    hashes = {log["file"]: log["sha256"] for log in record["logs"]}
    for facts in population.logs:
        if facts.sha256 != hashes[facts.file]:
            raise SecondScoringError(
                f"{facts.file} has changed since the draw (its SHA-256 is not the one the draw recorded): a draw "
                f"is made after the last trial, and trials logged since need a new draw"
            )

    # The draw is made again from the logs, the recorded seed and the recorded list of trials passed over.
    # Everything the key and the record say about the draw must come out the same.
    reasons = {passed["trial_id"]: passed["reason"] for passed in record["passed_over"]}
    key_name = f"{folder}/{KEY_DIR}/{KEY_FILE}"
    mismatch = SecondScoringError(
        f"{key_name} is not what seed {record['seed']} draws from {' and '.join(made_from)}: the key or the draw's "
        f"record was changed after the draw"
    )
    try:
        again = draw_items(population.candidates, record["seed"], population.digest, one_in=record["one_in"],
                           check=lambda candidate: reasons.get(candidate.trial_id))
    except SecondScoringError as error:
        raise mismatch from error
    key = _read_key(scoring_dir / KEY_DIR / KEY_FILE, key_name)
    if [row[:3] for row in key] != [(item.code, item.candidate.trial_id, item.candidate.log) for item in again.items]:
        raise mismatch
    if record != _draw_record(again, population, record["views"], record["rehearsal"]):
        raise SecondScoringError(
            f"{folder}/{KEY_DIR}/{DRAW_FILE} does not say what seed {record['seed']} draws from "
            f"{' and '.join(made_from)}: the draw's record was changed after the draw"
        )

    items = read_items(scoring_dir / ITEMS_FILE)
    if [item.code for item in items] != [item.code for item in again.items] or [
            view for view, _ in items[0].videos] != record["views"]:
        raise SecondScoringError(
            f"{folder}/{ITEMS_FILE} does not list the items of the key, in the key's order, with the draw's views"
        )
    scores = read_scores(scoring_dir / SCORES_FILE)
    left = unscored(items, scores, f"{folder}/{SCORES_FILE}")
    if left:
        raise SecondScoringError(
            f"{len(items) - len(left)} of {len(items)} items are scored: the comparison waits until every item is, "
            f"because it names trials and their first scores"
        )
    second_of = {score.item: score for score in scores}
    elsewhere = [code for code, _, _, digest in key if second_of[code].videos_sha256 != digest]
    if elsewhere:
        raise SecondScoringError(
            f"{folder}/{SCORES_FILE}: the score of {_listed(elsewhere)} was given on other video files than the "
            f"copies this draw made for it: the scores belong to another draw, or a copy was changed after the draw"
        )

    rows = [_Row(item.code, item.candidate, second_of[item.code]) for item in again.items]
    listed = sorted(rows, key=lambda row: row.candidate.trial_id)
    replayed = [row.second.watched_again for row in rows if row.second.watched_again > 0]
    return {
        "schema_version": SCHEMA_VERSION,
        "rehearsal": bool(record["rehearsal"]),
        "provenance": {
            "folder": folder,
            "logs": made_from,
            **{name: record[name] for name in ("seed", "one_in", "views", "logs_digest", "n_scored", "n_drawn",
                                               "groups", "passed_over", "whole_pairs")},
        },
        "scorers": {
            "first": sorted({row.candidate.record.scorer for row in rows} - {""}),
            "second": sorted({row.second.scorer for row in rows}),
            "n_same_role": sum(1 for row in rows if row.second.scorer == row.candidate.record.scorer),
        },
        "interval_level": INTERVAL_LEVEL,
        "all": _tally(rows),
        "by_kind": {kind: _tally([row for row in rows if row.candidate.kind == kind]) for kind in KINDS},
        "disagreements": [row.listed() for row in listed if row.verdict == "disagree"],
        "cannot_tell": [row.listed() for row in listed if row.verdict == "cannot_tell"],
        "watched_again": {"n_items": len(replayed), "total": sum(replayed)},
    }


def write_agreement(scoring_dir: str | Path, agreement: Mapping[str, Any]) -> Path:
    """Write the agreement beside the scores. The same agreement always gives the same bytes."""
    path = Path(scoring_dir) / AGREEMENT_FILE
    path.write_text(json.dumps(agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def agreement_lines(agreement: Mapping[str, Any]) -> list[str]:
    """What the comparison prints: the counts, and each trial whose two scores differ or whose video did not tell."""
    source, whole, scorers = agreement["provenance"], agreement["all"], agreement["scorers"]
    lines = []
    if agreement["rehearsal"]:
        lines.append("REHEARSAL: drawn from trials that are not counted. Nothing below is a result.")
    lines.append(
        f"second scoring of {source['n_drawn']} of the {source['n_scored']} scored trials of "
        f"{' and '.join(source['logs'])} (seed {source['seed']})"
    )
    lines.append(
        f"  first scored at the rig by {' and '.join(scorers['first']) or 'nobody named'}, scored again from video "
        f"by {' and '.join(scorers['second'])}"
    )
    if scorers["n_same_role"]:
        lines.append(
            f"  NOTE: in {scorers['n_same_role']} of {whole['n_items']} items both scores are by the same person, "
            f"who scored again with the trials' names hidden: say so wherever the agreement is reported"
        )
    if whole["n_compared"]:
        lines.append(
            f"  the two scores agree in {whole['n_agree']} of the {whole['n_compared']} items the second scorer "
            f"could tell: "
            f"{_percent(whole['agreement'])} ({_percent(agreement['interval_level'])} interval "
            f"{_percent(whole['wilson_lo'], sign=False)} to {_percent(whole['wilson_hi'])})"
        )
    else:
        lines.append(f"  the second scorer could tell none of the {whole['n_items']} items: there is no share to give")
    table = whole["table"]
    lines.append(
        f"  first score against second: success and success {table['success_success']}, success and failure "
        f"{table['success_failure']}, failure and success {table['failure_success']}, failure and failure "
        f"{table['failure_failure']}"
    )
    if agreement["disagreements"]:
        lines.append(f"  they differ in {whole['n_disagree']}:")
        lines.extend(f"    {_entry_line(entry)}, second {_spoken(entry['second'])}"
                     for entry in agreement["disagreements"])
    if agreement["cannot_tell"]:
        lines.append(f"  the second scorer could not tell in {whole['n_cannot_tell']}:")
        lines.extend(f"    {_entry_line(entry)}" for entry in agreement["cannot_tell"])
    lines.append("  by kind of trial:")
    for kind in KINDS:
        part = agreement["by_kind"][kind]
        text = f"    {kind}  {part['n_agree']} of {part['n_compared']} agree"
        if part["n_cannot_tell"]:
            text += f", {part['n_cannot_tell']} cannot tell"
        lines.append(text)
    if source["passed_over"]:
        lines.append(
            f"  passed over at the draw because their videos could not be used: "
            f"{', '.join(passed['trial_id'] + ' (' + passed['reason'] + ')' for passed in source['passed_over'])}"
        )
    if source["whole_pairs"]:
        lines.append(
            f"  both trials of a pair were drawn, because a group had no other trial left: "
            f"{', '.join(source['whole_pairs'])}"
        )
    return lines


@dataclass(frozen=True, slots=True)
class _Row:
    """One item with both its scores."""

    code: str
    candidate: Candidate
    second: Score

    @property
    def verdict(self) -> str:
        """`agree`, `disagree`, or `cannot_tell` when the second scorer could not tell."""
        if self.second.score == "cannot_tell":
            return "cannot_tell"
        return "agree" if self.second.score == self.candidate.record.outcome else "disagree"

    def listed(self) -> dict[str, Any]:
        """The item as the agreement lists it when its scores differ or its video did not tell."""
        record = self.candidate.record
        return {
            "trial_id": self.candidate.trial_id,
            "item": self.code,
            "block": self.candidate.block,
            "condition": self.candidate.condition,
            "kind": self.candidate.kind,
            "first": record.outcome,
            "first_subgoal": record.subgoal,
            "first_failure": record.failure,
            "second": self.second.score,
            "watched_again": self.second.watched_again,
        }


def _tally(rows: Sequence[_Row]) -> dict[str, Any]:
    """The counts of a set of items, and the share that agree among those the second scorer could tell."""
    verdicts = [row.verdict for row in rows]
    agree, disagree, untold = (verdicts.count(name) for name in ("agree", "disagree", "cannot_tell"))
    compared = agree + disagree
    low, high = wilson_interval(agree, compared, INTERVAL_LEVEL) if compared else (None, None)
    outcomes = ("success", "failure")
    return {
        "n_items": len(rows),
        "n_agree": agree,
        "n_disagree": disagree,
        "n_cannot_tell": untold,
        "n_compared": compared,
        "agreement": _rounded(agree / compared) if compared else None,
        "wilson_lo": _rounded(low),
        "wilson_hi": _rounded(high),
        # First score, then second score.
        "table": {
            f"{first}_{second}": sum(1 for row in rows
                                     if row.candidate.record.outcome == first and row.second.score == second)
            for first in outcomes for second in outcomes
        },
        "cannot_tell_first": {
            first: sum(1 for row in rows
                       if row.candidate.record.outcome == first and row.second.score == "cannot_tell")
            for first in outcomes
        },
    }


def _read_draw_record(scoring_dir: Path, folder: str) -> dict[str, Any]:
    path = scoring_dir / KEY_DIR / DRAW_FILE
    if not path.is_file():
        raise SecondScoringError(f"{folder} holds no draw (no {KEY_DIR}/{DRAW_FILE}): make the draw first")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SecondScoringError(f"{folder}/{KEY_DIR}/{DRAW_FILE} cannot be read: {error}") from error
    needed = ("rehearsal", "seed", "one_in", "views", "logs", "logs_digest", "n_scored", "n_drawn", "groups",
              "passed_over", "whole_pairs")
    whole = lambda value: isinstance(value, int) and not isinstance(value, bool)  # noqa: E731
    sound = (
        isinstance(record, dict)
        and record.get("schema_version") == SCHEMA_VERSION
        and all(name in record for name in needed)
        and whole(record["seed"])
        and whole(record["one_in"])
        and record["one_in"] >= 1
        and isinstance(record["logs"], list)
        and all(isinstance(log, dict) and {"file", "sha256"} <= set(log) for log in record["logs"])
        and isinstance(record["passed_over"], list)
        and all(isinstance(passed, dict) and {"trial_id", "reason"} <= set(passed) for passed in record["passed_over"])
    )
    if not sound:
        raise SecondScoringError(f"{folder}/{KEY_DIR}/{DRAW_FILE} is not the record of a draw made by this tool")
    return record


def _read_key(path: Path, name: str) -> list[tuple[str, str, str, str]]:
    """The key's lines: an item, the trial it is, that trial's log, and the digest of the item's video copies."""
    if not path.is_file():
        raise SecondScoringError(f"there is no {name}: the draw is not complete")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [tuple(row) for row in csv.reader(handle) if row]
    if not rows or rows[0] != KEY_HEADER or any(len(row) != len(KEY_HEADER) for row in rows) or not all(
            _SHA256.fullmatch(row[3]) for row in rows[1:]):
        raise SecondScoringError(f"{name}: the first line must be {','.join(KEY_HEADER)}, and every line after it "
                                 f"an item, its trial, the trial's log and a SHA-256")
    return rows[1:]


def _entry_line(entry: Mapping[str, Any]) -> str:
    """A listed item for the printed lines: the trial, its condition, and its first score with the grade."""
    first = entry["first"]
    if first == "failure":
        first += f" (grade {entry['first_subgoal']}, {entry['first_failure']})"
    return f"{entry['trial_id']}  {entry['condition']}  first {first}"


def _percent(share: float, *, sign: bool = True) -> str:
    return f"{100 * share:.0f}{' %' if sign else ''}"


def _rounded(value: float | None) -> float | None:
    """A number as the agreement file keeps it: to 9 significant digits, so that the bytes do not depend on the last
    digits of a sum."""
    return None if value is None else float(f"{value:.9g}")


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _short(path: Path) -> str:
    """A file by its folder's name and its own: enough to find it, and nothing of the machine it is on."""
    return f"{path.parent.name}/{path.name}" if path.parent.name else path.name



def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _count(number: int, noun: str) -> str:
    """`1 trial`, `2 trials`."""
    return f"{number} {noun}{'' if number == 1 else 's'}"



def _name(path: str | Path, root: str | Path) -> str:
    """A file as the outputs name it: relative to the root folder, or by its bare name when it lies elsewhere."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _plain(error: Exception, path: Path, name: str) -> str:
    """An error's text with the file's place on this machine replaced by its name."""
    return str(error).replace(str(path.resolve()), name).replace(str(path), name)


def _listed(names: Sequence[str]) -> str:
    """Names for a message: all of them when they are few, the first ones and a count otherwise."""
    if len(names) <= _NAMES_SHOWN:
        return ", ".join(names)
    return f"{', '.join(names[:_NAMES_SHOWN])} and {len(names) - _NAMES_SHOWN} more"
