"""Put a recorded set that a session left without closing back to its last sound state. Nothing is deleted.

    look at the folder -> judge what is sound -> say what would be done -> (with --apply) do it, and look again

A recording session that ends without closing the dataset (the terminal closed, the process killed) leaves the set's
folder in a state the recorder refuses to append to (`record_layout.resume_problem`). This module is the way back.

*Sound* means closed properly: an episode is sound when it has its row in a readable episode table, its rows in a
readable data file, and its span of every camera's video file. A session writes files of its own (a data file, an
episode-table file and a video file per camera, each numbered after the last session's), so what a dead session left
can be told from what the sessions before it closed, and moved out whole. Only `meta/info.json` and
`meta/stats.json` are shared: every save rewrites them, so they are written again for the sound episodes alone.

What a dead session leaves, *measured* 2026-10-06 with LeRobot's own writer (0.6.1 at the pinned commit), called as
the recorder calls it, on copies of that day's clustered set (27 episodes, 19,117 frames, in files 000 and 001):

    killed in its first take, nothing saved       images/<camera>/episode-000027/frame-*.png and nothing else: the
                                                  counts still agree, and the recorder goes by the pictures
    killed after one save                         info.json says 28 episodes; data/chunk-000/file-002.parquet has no
                                                  end; videos/<camera>/chunk-000/file-002.mp4 hold 60 frames each;
                                                  images/ holds an empty folder per camera; stats.json describes 28
    killed after two saves, a third take begun    the same for 29 episodes, with the third take's pictures in images/
    killed between a save and its placement line  the same, and the placement log is one line short of info.json
    killed after eleven saves                     also meta/episodes/chunk-000/file-002.parquet with no end: LeRobot
                                                  writes episode rows ten at a time, and none before the tenth
    killed while a save encodes its videos        info.json unchanged; data file-002 with no end; two tmp*/ folders
                                                  in the set's folder with that take's two videos; no file-002.mp4
    a new set's first session, killed after two   info.json says 2 episodes; no episode table at all; file-000 of
                                                  each kind; stats.json and tasks.parquet of episodes never closed

In each of them the files the earlier sessions closed were byte for byte what they had been.

One more thing leaves a set unsound, and it is not a session that died: a save that is cut short while it encodes
its videos (Ctrl-C, a fault in the encoder) in a session that then closes the set, as Ctrl-C and a fault do. The
take's rows are already in the session's data file, and nothing else of it is kept. *Measured* the same day: the
counts all agree, and the recorder sees it only by looking the set through with this module's rule
(`record_layout.deeper_look`); when the cut save is the session's first, a readable
data file holds 60 rows of an episode the table does not have; when the session had saved an episode before, the two
share a data file (120 rows in two row groups, for one episode in the table). Resumed as it was, the next session's
episode took the cut take's number: the set then loaded with 19,282 rows for 19,222 frames, 45 index values held
twice, and the new episode's 45 frames were read from the cut take's rows beside the new take's pictures. Such rows
are taken out by writing their file again without them (`keep_episodes`); the file as it was is kept.

The rule that says what is sound (`judge`) is a pure function of plain counts and lists of files (`Found`), so that
it is tested without LeRobot. Reading a folder into those counts (`look`) needs the rig's interpreter (pyarrow for
the tables, PyAV for the videos); the statistics need LeRobot's own aggregation (`sound_stats`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tolerance_radius.rig import record_layout

#: The layout this module knows: LeRobot's v3.0 (chunked files, one episode table, video files shared by episodes).
LAYOUT = "v3.0"
#: Where LeRobot keeps a set's episode table, and the three files of `meta/` that are not that table.
TABLE_FOLDER = "meta/episodes"
INFO, STATS, TASKS = "meta/info.json", "meta/stats.json", "meta/tasks.parquet"
#: What a writer that died leaves at the top of the set's folder: the pictures of a take that was never saved, and
#: the folders in which a save encodes its videos before it puts them in place (`tempfile.mkdtemp`: tmp + 8 letters).
PICTURES = "images"
ENCODING_PREFIX = "tmp"
#: The folder a repair moves things into is the set's own name with this and a time after it.
SIDE_LABEL = "unclosed"
#: A folder that was written into less than this long ago is watched until that long has passed, and must not
#: change meanwhile. A killed session leaves its two video encoders running, and they finish their videos within
#: seconds (*measured* 2026-10-06, a take of 60 frames; they then stay, idle, holding nothing of the set).
QUIET_S = 5.0

#: The programs that record into a set or run trials, by the name of the script the interpreter was started on ...
SESSION_SCRIPTS = ("record_layout.py", "run_trials.py", "record_session.py", "rollout_session.py")
#: ... by module, when started with `-m` ...
SESSION_MODULES = ("tolerance_radius.rig.record_layout", "tolerance_radius.rig.trial_session")
#: ... and LeRobot's own commands, which this project never runs on the arm but which write the same folders.
SESSION_COMMANDS = ("lerobot-record", "lerobot-rollout", "lerobot-teleoperate", "lerobot-replay")


# --------------------------------------------------------------------------------------------------------------------
# What a folder holds, as plain values
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Segment:
    """Where one camera's pictures of one episode lie: a video file, and a span of it in seconds."""

    camera: str
    file: str
    from_s: float
    to_s: float


@dataclass(frozen=True, slots=True)
class EpisodeRow:
    """One row of the episode table: an episode that a session closed, and where its rows and pictures are.

    Files are named by their path inside the set's folder (`data/chunk-000/file-001.parquet`)."""

    episode_index: int
    #: How many frames the episode has (the table's `length`).
    frames: int
    #: The set-wide number of its first frame, and of the frame after its last (`dataset_from_index`, `_to_index`).
    first_frame: int
    end_frame: int
    #: The episode-table file the row was read from, and the data file that holds the episode's rows.
    table_file: str
    data_file: str
    segments: tuple[Segment, ...] = ()
    tasks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Found:
    """What a set's folder and its placement log hold, as plain counts and lists of files. `judge` reads nothing
    else. A file that is there and cannot be read (a table whose writer was never closed has no end) is listed with
    None in place of its count."""

    #: What `meta/info.json` says: `codebase_version`, `total_episodes`, `total_frames`, `total_tasks`, `splits`,
    #: `fps`. None when the file cannot be read.
    info: Mapping[str, Any] | None
    #: The set's cameras: its features stored as video.
    cameras: tuple[str, ...] = ()
    #: The rows of every readable episode-table file, in the files' order.
    rows: tuple[EpisodeRow, ...] = ()
    #: Every episode-table file with its rows.
    table_files: Mapping[str, int | None] = field(default_factory=dict)
    #: Every data file with the rows it holds of each episode.
    data_files: Mapping[str, Mapping[int, int] | None] = field(default_factory=dict)
    #: Every video file with its frames.
    video_files: Mapping[str, int | None] = field(default_factory=dict)
    #: How many frames `meta/stats.json` describes; None when there is no such file or it cannot be read.
    stats_frames: int | None = None
    has_stats: bool = False
    has_tasks: bool = False
    #: Entries at the top of the folder that a writer leaves when it dies (`images`, `tmp*`), and everything else in
    #: the folder that is none of the above (left alone, and said).
    leftovers: tuple[str, ...] = ()
    strangers: tuple[str, ...] = ()
    #: The `episode_index` of every line of the placement log, in the log's order; None for a line that is not one.
    log_lines: tuple[int | None, ...] = ()


@dataclass(frozen=True, slots=True)
class Verdict:
    """What `judge` makes of a `Found`: either why this tool will not touch the set, or its last sound state and what
    stands between the folder and that state."""

    #: Why nothing can be done here; None when the sound state is known.
    refusal: str | None = None
    sound_episodes: int = 0
    sound_frames: int = 0
    #: Episodes and frames `meta/info.json` counts beyond the sound ones: saved by a session that never closed.
    unclosed_episodes: int = 0
    unclosed_frames: int = 0
    #: Files and folders of the session that died, by their path in the set's folder: all of them are moved out.
    move_out: tuple[str, ...] = ()
    #: Data files that hold sound episodes and also rows of a take that was never saved, each with the episodes it
    #: is to hold: they are written again with those episodes' rows only.
    rewrite_data: tuple[tuple[str, tuple[int, ...]], ...] = ()
    #: What `meta/info.json` must say; None when it says so already.
    info_wanted: Mapping[str, Any] | None = None
    #: True when `meta/stats.json` must be written again from the sound episodes' own statistics.
    rewrite_stats: bool = False
    #: How many lines the placement log keeps, and how many are cut from its end.
    log_keep: int = 0
    log_cut: int = 0
    #: Things seen that are left as they are, for the person to read.
    notes: tuple[str, ...] = ()

    @property
    def sound_already(self) -> bool:
        """True when the set adds up as it is and there is nothing to repair."""
        return (self.refusal is None and not self.move_out and not self.rewrite_data and self.info_wanted is None
                and not self.rewrite_stats and self.log_cut == 0)


# --------------------------------------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------------------------------------


def _run(numbers: Sequence[Any], limit: int = 6) -> str:
    """A list of numbers for a sentence: `27, 28` or `27 to 37`."""
    numbers = list(numbers)
    if len(numbers) > 2 and all(isinstance(n, int) for n in numbers) and numbers == list(
            range(numbers[0], numbers[0] + len(numbers))):
        return f"{numbers[0]} to {numbers[-1]}"
    return ", ".join(str(n) for n in numbers[:limit]) + (", ..." if len(numbers) > limit else "")


def judge(found: Found) -> Verdict:  # noqa: C901, PLR0911, PLR0912, PLR0915 -- one rule, read from top to bottom
    """Say what of a set is sound and what a session that died left in it, from plain counts and lists of files.

    The sound episodes are the rows of the readable episode table, and only if they hold together: numbered from 0
    without a gap, their frames one unbroken run, each with its rows in a readable data file and its span in every
    camera's video file. LeRobot writes an episode's row last, when the set is closed, so a row that is there
    belongs to a session that closed. If the rows do not hold together, something other than a session that died
    has happened to the set, and the answer is a refusal.

    Everything else in the folder that a writer makes belongs to the session that died: table files that cannot be
    read, data and video files no sound episode points to, the pictures of an unsaved take, the folders a save
    encodes in. They must all come after the sound files (a session numbers its files after the last one's).

    One thing is not a whole file: rows of a take that was never saved, in a data file that sound episodes are in
    too (a save cut short while it encoded, in a session that went on to close the set). Those are the rows of any
    episode the table does not give to that file; the file is to be written again without them.
    """
    info = found.info
    if info is None or info.get("total_episodes") is None or info.get("total_frames") is None:
        return Verdict(refusal="its meta/info.json cannot be read, so there is no telling what the set should be")
    if info.get("codebase_version") != LAYOUT:
        return Verdict(refusal=f"it is written in the layout {info.get('codebase_version')!r}, and this tool knows "
                               f"only {LAYOUT}")
    fps = info.get("fps")
    if not fps or fps <= 0:
        return Verdict(refusal="its meta/info.json gives no frame rate")

    rows = found.rows
    sound = len(rows)
    indices = [row.episode_index for row in rows]
    if indices != list(range(sound)):
        return Verdict(refusal=f"its episode table holds the episodes {_run(indices, limit=40)}, which is not an "
                               "unbroken run from 0: a session recorded on top of an unclosed one leaves this, and "
                               "moving files aside does not mend it")
    frames = 0
    for row in rows:
        if row.first_frame != frames or row.end_frame - row.first_frame != row.frames or row.frames <= 0:
            return Verdict(refusal=f"episode {row.episode_index} is given the frames {row.first_frame} to "
                                   f"{row.end_frame} with a length of {row.frames}, after {frames} frames before "
                                   "it: the episode table's frame numbers are not one unbroken run")
        frames = row.end_frame

    notes: list[str] = []
    # Each sound episode's rows: in a data file that can be read, as many as its length.
    for row in rows:
        if row.data_file not in found.data_files:
            return Verdict(refusal=f"episode {row.episode_index} is in the episode table and its data file "
                                   f"{row.data_file} is not there")
        held = found.data_files[row.data_file]
        if held is None:
            return Verdict(refusal=f"episode {row.episode_index} is in the episode table and its data file "
                                   f"{row.data_file} cannot be read")
        if held.get(row.episode_index, 0) != row.frames:
            return Verdict(refusal=f"episode {row.episode_index} has {row.frames} frames in the episode table and "
                                   f"{held.get(row.episode_index, 0)} rows in {row.data_file}")
    # ... and what else such a file holds are rows of takes that were never saved: a save writes the take's rows
    # first, and one that is cut short while it encodes leaves them behind with no row in the table.
    pointed_data = {row.data_file for row in rows}
    rewrite: list[tuple[str, tuple[int, ...]]] = []
    for name in sorted(pointed_data):
        mine = tuple(row.episode_index for row in rows if row.data_file == name)
        extra = sorted(set(found.data_files[name]) - set(mine))
        if extra and extra[0] < mine[-1]:
            return Verdict(refusal=f"{name} holds rows of episode {_run(extra)}, which the episode table does not "
                                   f"give to it, before those of its own episode {mine[-1]}: a take that was never "
                                   "saved comes after the episodes its session saved")
        if extra:
            rewrite.append((name, mine))
            notes.append(f"{name} holds {sum(found.data_files[name][episode] for episode in extra)} rows of episode "
                         f"{_run(extra)} beside its sound episode{'' if len(mine) == 1 else 's'} {_run(mine)}: rows "
                         "of a take that was never saved (a save cut short, by Ctrl-C or a fault, while it encoded "
                         "its videos)")

    # Each sound episode's pictures: a span of a video file that can be read, at least as long as the episode.
    pointed_video: set[str] = set()
    for row in rows:
        spans = {segment.camera: segment for segment in row.segments}
        for camera in found.cameras:
            segment = spans.get(camera)
            if segment is None:
                return Verdict(refusal=f"episode {row.episode_index} has no video span for the camera {camera}")
            in_file = found.video_files.get(segment.file)
            if in_file is None:
                missing = "is not there" if segment.file not in found.video_files else "cannot be read"
                return Verdict(refusal=f"episode {row.episode_index} is in the episode table and its video file "
                                       f"{segment.file} {missing}")
            span_frames = round((segment.to_s - segment.from_s) * fps)
            if segment.from_s < -1e-6 or span_frames < row.frames or segment.to_s * fps > in_file + 0.5:
                return Verdict(refusal=f"episode {row.episode_index} has {row.frames} frames, and its span of "
                                       f"{segment.file} ({segment.from_s:.3f} to {segment.to_s:.3f} s, {span_frames} "
                                       f"frames) does not hold them in a file of {in_file} frames")
            if span_frames > row.frames:
                notes.append(f"episode {row.episode_index}: its span of {segment.file} holds {span_frames} frames "
                             f"for {row.frames} rows (pictures of a take that was never saved, left in "
                             f"{PICTURES}/ when a later take was saved over them?). Left as it is")
            pointed_video.add(segment.file)
    for name in sorted(pointed_video):
        spanned = max(segment.to_s for row in rows for segment in row.segments if segment.file == name) * fps
        if found.video_files[name] - spanned > 0.5:
            notes.append(f"{name} holds {found.video_files[name]} frames, {found.video_files[name] - round(spanned)} "
                         "more than its episodes' spans (the video of a take that was never saved). Left as it is: "
                         "no episode reads them")

    said_episodes, said_frames = int(info["total_episodes"]), int(info["total_frames"])
    if said_episodes < sound:
        return Verdict(refusal=f"its meta/info.json counts {said_episodes} episodes and its episode table holds "
                               f"{sound}: a session that died leaves the count ahead of the table, never behind")
    if (said_episodes == sound) != (said_frames == frames) or said_frames < frames:
        return Verdict(refusal=f"its meta/info.json counts {said_episodes} episodes and {said_frames} frames, and "
                               f"its episode table holds {sound} episodes and {frames} frames: the two counts do "
                               "not go together")

    # What the session that died left. Its files come after the sound ones, or the picture is not understood.
    move: list[str] = []
    dead_tables = sorted(name for name, held in found.table_files.items() if held is None)
    sound_tables = sorted(name for name, held in found.table_files.items() if held is not None)
    empty_tables = [name for name in sound_tables if name not in {row.table_file for row in rows}]
    if empty_tables:
        return Verdict(refusal=f"the episode-table file {empty_tables[0]} can be read and holds no episode")
    dead_data = sorted(name for name in found.data_files if name not in pointed_data)
    dead_video = sorted(name for name in found.video_files if name not in pointed_video)
    for kind, dead, kept in (("episode-table", dead_tables, sound_tables), ("data", dead_data, sorted(pointed_data))):
        early = [name for name in dead if kept and name < kept[-1]]
        if early:
            return Verdict(refusal=f"the {kind} file {early[0]} belongs to no sound episode and comes before "
                                   f"{kept[-1]}, which does: a session that died leaves its files after the sound "
                                   "ones")
    for name in dead_video:
        folder = name.rsplit("/", 2)[0] if name.count("/") >= 2 else ""
        later = sorted(kept for kept in pointed_video if kept.startswith(folder + "/") and kept > name)
        if later:
            return Verdict(refusal=f"the video file {name} belongs to no sound episode and comes before {later[-1]},"
                                   " which does: a session that died leaves its files after the sound ones")
    move += dead_data + dead_tables + dead_video + sorted(found.leftovers)
    if sound == 0:
        # A set none of whose episodes was closed is put back to the empty set its first session created.
        move += [name for name, there in ((STATS, found.has_stats), (TASKS, found.has_tasks)) if there]
    for name in found.strangers:
        notes.append(f"{name} is not a file this tool knows. Left as it is")

    # The placement log: one line per sound episode, in order; what follows belongs to the session that died.
    lines = found.log_lines
    if len(lines) < sound:
        short = sound - len(lines)
        return Verdict(refusal=f"the set holds {sound} sound episodes and the placement log has {len(lines)} lines: "
                               f"{short} sound episode{' has' if short == 1 else 's have'} no placement line. This "
                               "tool never writes a placement line (a measured position cannot be made up); the "
                               "session's own log under data/logs has the line 'episode N placed at ...'. Is this "
                               "the log of this set, and the checkout it was recorded from?")
    if list(lines[:sound]) != list(range(sound)):
        first_wrong = next(i for i in range(sound) if lines[i] != i)
        return Verdict(refusal=f"line {first_wrong + 1} of the placement log is not episode {first_wrong}'s: the "
                               "log's lines are not this set's sound episodes in order")
    cut = list(lines[sound:])
    if cut != list(range(sound, sound + len(cut))):
        return Verdict(refusal=f"the placement log's {len(cut)} lines after line {sound} are not the episodes "
                               f"{sound} onwards in order (they say {_run(['?' if c is None else c for c in cut])}): "
                               "they are not the lines of one session that died")

    wanted = {"total_episodes": sound, "total_frames": frames,
              "total_tasks": info.get("total_tasks") if sound else 0,
              "splits": {"train": f"0:{sound}"} if sound else {}}
    info_wanted = None if all(info.get(key) == value for key, value in wanted.items()) else wanted
    rewrite_stats = sound > 0 and (said_episodes != sound or not found.has_stats or found.stats_frames != frames)
    return Verdict(sound_episodes=sound, sound_frames=frames, unclosed_episodes=said_episodes - sound,
                   unclosed_frames=said_frames - frames, move_out=tuple(move), rewrite_data=tuple(rewrite),
                   info_wanted=info_wanted, rewrite_stats=rewrite_stats, log_keep=sound, log_cut=len(cut),
                   notes=tuple(notes))


# --------------------------------------------------------------------------------------------------------------------
# Reading a folder (the rig's interpreter: pyarrow and PyAV)
# --------------------------------------------------------------------------------------------------------------------


def log_episodes(path: Path) -> tuple[int | None, ...]:
    """The `episode_index` of every line of a placement log that is not empty, in order; None for a line that
    cannot be read as one. Empty when there is no such file. The same lines `record_layout.placement_lines` counts
    (bytes that are not text do not stop the count there, and do not here)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    episodes: list[int | None] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            episodes.append(int(json.loads(line)["episode_index"]))
        except (ValueError, KeyError, TypeError):
            episodes.append(None)
    return tuple(episodes)


def video_frames(path: Path) -> int | None:
    """How many frames a video file holds, counted from its packets without decoding them; None when the file
    cannot be read. *Measured* 2026-10-06: 40 ms for the 135 MB file of the clustered set, 18,351 frames, which is
    the sum of the lengths of the 26 episodes it holds."""
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return sum(1 for packet in container.demux(stream) if packet.dts is not None)
    except Exception:  # noqa: BLE001 -- whatever is wrong with the file, it is a file that cannot be read
        return None


def look(dataset_dir: Path, placement_log: Path) -> Found:
    """Read a set's folder and its placement log into plain counts. Nothing is opened for writing.

    Needs pyarrow and PyAV (the rig's interpreter). Tables are read without the reader's thread pool: with it, a
    process in the rig's environment that has read parquet files did not exit about half the time (2026-10-05).
    """
    import pyarrow.parquet as parquet

    def relative(path: Path) -> str:
        return path.relative_to(dataset_dir).as_posix()

    try:
        raw = json.loads((dataset_dir / INFO).read_text(encoding="utf-8"))
        info = {key: raw.get(key) for key in ("codebase_version", "total_episodes", "total_frames", "total_tasks",
                                              "splits", "fps")}
        cameras = tuple(name for name, feature in raw.get("features", {}).items() if feature.get("dtype") == "video")
        data_path, video_path = raw.get("data_path"), raw.get("video_path")
    except (OSError, ValueError, AttributeError):
        info, cameras, data_path, video_path = None, (), None, None

    rows: list[EpisodeRow] = []
    table_files: dict[str, int | None] = {}
    for path in sorted((dataset_dir / TABLE_FOLDER).rglob("*.parquet")):
        try:
            columns = [name for name in parquet.read_schema(path).names if not name.startswith("stats/")]
            table = parquet.read_table(path, columns=columns, use_threads=False).to_pylist()
        except Exception:  # noqa: BLE001 -- a table whose writer was never closed has no end
            table_files[relative(path)] = None
            continue
        table_files[relative(path)] = len(table)
        for row in table:
            segments = tuple(
                Segment(camera=camera,
                        file=video_path.format(video_key=camera, chunk_index=row[f"videos/{camera}/chunk_index"],
                                               file_index=row[f"videos/{camera}/file_index"]),
                        from_s=float(row[f"videos/{camera}/from_timestamp"]),
                        to_s=float(row[f"videos/{camera}/to_timestamp"]))
                for camera in cameras if row.get(f"videos/{camera}/file_index") is not None and video_path)
            rows.append(EpisodeRow(
                episode_index=int(row["episode_index"]), frames=int(row["length"]),
                first_frame=int(row["dataset_from_index"]), end_frame=int(row["dataset_to_index"]),
                table_file=relative(path),
                data_file=(data_path or "").format(chunk_index=row["data/chunk_index"],
                                                   file_index=row["data/file_index"]),
                segments=segments, tasks=tuple(row.get("tasks") or ())))

    data_files: dict[str, Mapping[int, int] | None] = {}
    for path in sorted((dataset_dir / "data").rglob("*.parquet")):
        try:
            column = parquet.read_table(path, columns=["episode_index"], use_threads=False).column("episode_index")
            data_files[relative(path)] = dict(sorted(Counter(int(e) for e in column.to_pylist()).items()))
        except Exception:  # noqa: BLE001
            data_files[relative(path)] = None
    video_files = {relative(path): video_frames(path) for path in sorted((dataset_dir / "videos").rglob("*.mp4"))}

    try:
        stats = json.loads((dataset_dir / STATS).read_text(encoding="utf-8"))
        # Every feature that is not a picture is counted once per frame; `index` is in every LeRobot set.
        stats_frames = int(stats["index"]["count"][0])
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        stats_frames = None

    known = {INFO, STATS, TASKS, *table_files, *data_files, *video_files}
    leftovers, strangers = [], []
    for entry in sorted(dataset_dir.iterdir()):
        if entry.name == PICTURES or (entry.is_dir() and entry.name.startswith(ENCODING_PREFIX)):
            leftovers.append(entry.name)
        elif entry.name not in ("data", "meta", "videos"):
            strangers.append(entry.name + ("/" if entry.is_dir() else ""))
    for folder in ("data", "meta", "videos"):
        strangers += [relative(path) for path in sorted((dataset_dir / folder).rglob("*"))
                      if path.is_file() and relative(path) not in known]
    return Found(info=info, cameras=cameras, rows=tuple(rows), table_files=table_files, data_files=data_files,
                 video_files=video_files, stats_frames=stats_frames, has_stats=(dataset_dir / STATS).is_file(),
                 has_tasks=(dataset_dir / TASKS).is_file(), leftovers=tuple(leftovers), strangers=tuple(strangers),
                 log_lines=log_episodes(placement_log))


# --------------------------------------------------------------------------------------------------------------------
# The statistics of the sound episodes (LeRobot's own aggregation)
# --------------------------------------------------------------------------------------------------------------------


def sound_stats(dataset_dir: Path, table_files: Sequence[str], order: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """The set's statistics over its sound episodes alone, as LeRobot would hold them had no other episode been
    saved. `table_files` are the readable episode-table files, in order; `order` is the order the features are given
    in (those it does not name follow).

    LeRobot keeps one running aggregate: at every save, `aggregate_stats([what it had, the new episode's])`, written
    to `meta/stats.json`. After a session that died, the file describes that session's episodes too, and nothing
    can be taken out of an aggregate. But each sound episode's own statistics are in its row of the episode table,
    so the aggregate is folded again from those rows, in their order, with LeRobot's own function.

    Two details make the result the very numbers the file held before (*measured* 2026-10-06: the 27 episodes of the
    clustered set folded this way give every number of its `stats.json`, to the last digit; folded in 64-bit numbers
    throughout they differ by 6e-9 of the value, and by 5e-8 with the types the features declare):

    - A session that resumes reads the file back as 64-bit numbers, and every session starts an episode-table file of
      its own. So the running aggregate is read back the same way wherever the table file changes.
    - While an episode is folded in, its standard deviation is squared in the precision it was computed in, which
      for a 32-bit feature and for pictures is 32 bits. The table stores every number as 64-bit. A list whose numbers
      are all exactly 32-bit numbers is taken to have been one (a 64-bit result is one only by a coincidence of 29
      bits, or when it is a small whole number, whose square is the same either way).

    The order of the features in LeRobot's file changes from run to run (its aggregate walks a set of names), so
    only the numbers can be compared, not the file's bytes.

    Needs the pinned LeRobot on the path (`lerobot101.use_pinned_lerobot`), NumPy and pyarrow.
    """
    import numpy as np
    import pyarrow.parquet as parquet
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import cast_stats_to_numpy
    from lerobot.datasets.utils import serialize_dict

    def of_episode(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for column, value in row.items():
            if not column.startswith("stats/"):
                continue
            feature, stat = column[len("stats/"):].rsplit("/", 1)
            array = np.array(value)
            if stat != "count":
                with np.errstate(over="ignore", invalid="ignore"):
                    narrow = array.astype(np.float32)
                if np.array_equal(narrow.astype(np.float64), array):
                    array = narrow
            stats.setdefault(feature, {})[stat] = array
        return stats

    running: dict[str, dict[str, Any]] | None = None
    for name in table_files:
        if running is not None:
            running = cast_stats_to_numpy(json.loads(json.dumps(serialize_dict(running))))
        for row in parquet.read_table(dataset_dir / name, use_threads=False).to_pylist():
            episode = of_episode(row)
            running = aggregate_stats([running, episode]) if running is not None else episode
    if running is None:
        raise ValueError("there is no sound episode to take statistics from")
    names = [name for name in order if name in running] + [name for name in running if name not in order]
    return {name: running[name] for name in names}


def _plain(value: Any) -> Any:
    """Statistics as plain lists and numbers (arrays have `tolist`)."""
    if isinstance(value, Mapping):
        return {key: _plain(inner) for key, inner in value.items()}
    return value.tolist() if hasattr(value, "tolist") else value


def stats_text(stats: Mapping[str, Mapping[str, Any]]) -> str:
    """The text of `meta/stats.json` for these statistics, in the form LeRobot's `write_stats` writes (arrays as
    lists, `json.dump(..., indent=4, ensure_ascii=False)`); the tests hold the two against each other."""
    return json.dumps(_plain(stats), indent=4, ensure_ascii=False)


def outside(fewer: Mapping[str, Mapping[str, Any]], more: Mapping[str, Mapping[str, Any]]) -> str | None:
    """A sentence when statistics over some episodes (`fewer`) do not lie inside statistics over those episodes
    and others (`more`), both as plain lists; None when they do, or when `more` does not have the feature.

    The file a repair replaces describes the sound episodes and the dead session's together, so every smallest value
    in it is at most the sound episodes' smallest, every largest at least their largest, and it counts at least as
    many. LeRobot aggregates all three without arithmetic, so the comparison is exact. A fold over the wrong rows
    fails it.
    """
    def flat(value: Any) -> list[float]:
        return [number for inner in value for number in flat(inner)] if isinstance(value, list) else [value]

    for feature, stats in fewer.items():
        over_more = more.get(feature)
        if not isinstance(stats, Mapping) or not isinstance(over_more, Mapping):
            continue
        for stat, holds in (("min", lambda a, b: a >= b), ("max", lambda a, b: a <= b), ("count", lambda a, b: a <= b)):
            if stat not in stats or stat not in over_more:
                continue
            mine, theirs = flat(_plain(stats[stat])), flat(_plain(over_more[stat]))
            if len(mine) != len(theirs) or not all(holds(a, b) for a, b in zip(mine, theirs, strict=True)):
                return (f"the {stat} of {feature} over fewer episodes ({mine}) does not lie inside the one over "
                        f"more ({theirs}): the statistics in the file and the sound episodes' own do not go "
                        "together")
    return None


# --------------------------------------------------------------------------------------------------------------------
# A data file without the rows of a take that was never saved (pyarrow)
# --------------------------------------------------------------------------------------------------------------------


def keep_episodes(source: Path, target: Path, episodes: Collection[int]) -> int:
    """Write the data file `source` again as `target` with the rows of `episodes` only, and return how many rows
    that is. `source` is only read.

    LeRobot puts each saved episode's rows into the file with one `write_table`, which makes one row group of them.
    So the file is copied row group by row group, a row group of a take that was never saved is left out, and
    nothing is encoded anew in another way: the writer is given LeRobot's own settings
    (`DatasetWriter._save_episode_data`: snappy, dictionaries) and the schema the file's own writer was given, which
    a parquet file keeps in its metadata (the reader alone names the joints' list differently from the writer).
    *Measured* 2026-10-06: both data files of the clustered set, copied this way with all their episodes, are the
    files byte for byte (36,957 and 866,528 bytes); with the first 13 of 26 episodes, the copy is the file up to the
    end of those episodes' rows.

    The result is then read back and held against the source: it must be exactly the source's rows of those
    episodes, in their order, with the source's schema. ValueError when it is not.
    """
    import base64

    import pyarrow as arrow
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    wanted = arrow.array(sorted(episodes), type=arrow.int64())
    with parquet.ParquetFile(source) as opened:
        stored = (opened.metadata.metadata or {}).get(b"ARROW:schema")
        schema = (arrow.ipc.read_schema(arrow.BufferReader(base64.b64decode(stored))) if stored
                  else opened.schema_arrow)
        with parquet.ParquetWriter(target, schema=schema, compression="snappy", use_dictionary=True) as writer:
            for group in range(opened.num_row_groups):
                table = opened.read_row_group(group, use_threads=False)
                keep = compute.is_in(table.column("episode_index"), value_set=wanted)
                kept = compute.sum(keep).as_py() or 0
                if kept:
                    writer.write_table((table if kept == table.num_rows else table.filter(keep)).cast(schema))
    whole = parquet.read_table(source, use_threads=False)
    expected = whole.filter(compute.is_in(whole.column("episode_index"), value_set=wanted))
    written = parquet.read_table(target, use_threads=False)
    if not written.equals(expected, check_metadata=True):
        raise ValueError(f"{target.name} written again does not hold exactly the rows of episodes "
                         f"{_run(sorted(episodes))} that {source.name} holds")
    return written.num_rows


# --------------------------------------------------------------------------------------------------------------------
# Texts (standard library only)
# --------------------------------------------------------------------------------------------------------------------


def info_text(old_text: str, wanted: Mapping[str, Any]) -> str:
    """The text of `meta/info.json` with the counts in `wanted` and everything else as it was, in the form LeRobot
    writes the file in (`json.dump(..., indent=4, ensure_ascii=False)`, keys in the file's own order)."""
    info = json.loads(old_text)
    for key, value in wanted.items():
        info[key] = value
    return json.dumps(info, indent=4, ensure_ascii=False)


#: How the placement log is read for cutting and written back: a byte that is not text comes through as it is.
AS_IT_IS = {"encoding": "utf-8", "errors": "surrogateescape", "newline": ""}


def split_log(text: str, keep: int) -> tuple[str, str]:
    """A placement log's text cut after its first `keep` lines that are not empty: (what stays, what is cut). The
    two together are the text as it was, byte for byte (read and written with `AS_IT_IS`)."""
    kept = 0
    position = 0
    for line in text.splitlines(keepends=True):
        if kept == keep:
            break
        position += len(line)
        kept += 1 if line.strip() else 0
    if kept < keep:
        raise ValueError(f"the placement log has {kept} lines, fewer than the {keep} to keep")
    return text[:position], text[position:]


# --------------------------------------------------------------------------------------------------------------------
# Is something still using the folder?
# --------------------------------------------------------------------------------------------------------------------


def session_processes(ps_text: str, own: Collection[int]) -> list[str]:
    """The running recorders and trial sessions in the output of `ps -axo pid=,command=`, as `process N: command`.

    A session is an interpreter started on one of the session scripts (`SESSION_SCRIPTS`), with `-m` on one of the
    session modules, or one of LeRobot's own commands. The script must be what the interpreter runs, not any word of
    the command line: an editor with `record_layout.py` open is not a session, and neither is this tool, whose own
    arguments name the set. Processes in `own` (this one and those it descends from) are left out.
    """
    found = []
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) in own:
            continue
        words = parts[1:]
        program, arguments = os.path.basename(words[0]), words[1:]
        session = program in SESSION_COMMANDS
        if program.lower().startswith("python"):
            for position, word in enumerate(arguments):
                if word == "-m":
                    session = "".join(arguments[position + 1 : position + 2]) in SESSION_MODULES
                    break
                if word in ("-c", "-"):
                    break
                if not word.startswith("-"):
                    session = os.path.basename(word) in SESSION_SCRIPTS + SESSION_COMMANDS
                    break
        if session:
            found.append(f"process {parts[0]}: {' '.join(words)[:160]}")
    return found


def writers(lsof_text: str, own: Collection[int]) -> list[str]:
    """The processes that hold a file open for writing, in the output of `lsof -F pcan`, as `process N (program):
    file`. A recorder holds its data file open from its first save until it closes the set."""
    found = []
    pid, program, access = None, "", ""
    for line in lsof_text.splitlines():
        if not line:
            continue
        kind, value = line[0], line[1:]
        if kind == "p":
            pid, program, access = int(value) if value.isdigit() else None, "", ""
        elif kind == "c":
            program = value
        elif kind == "a":
            access = value
        elif kind == "f":
            access = ""
        elif kind == "n" and pid is not None and pid not in own and ("w" in access or "u" in access):
            found.append(f"process {pid} ({program}): {value}")
    return found


def own_processes(ps_text: str, pid: int) -> set[int]:
    """`pid` and every process it descends from, read from the output of `ps -axo pid=,ppid=`."""
    parents = {}
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            parents[int(parts[0])] = int(parts[1])
    own = {pid}
    while pid in parents and parents[pid] not in own and parents[pid] > 1:
        pid = parents[pid]
        own.add(pid)
    return own


def newest_change_s(folder: Path) -> float | None:
    """When anything in a folder was last written, as seconds of the system's clock; None for an empty folder."""
    newest = None
    for directory, _folders, files in os.walk(folder):
        for name in (directory, *(os.path.join(directory, file) for file in files)):
            try:
                changed = os.stat(name).st_mtime
            except OSError:
                continue
            newest = changed if newest is None or changed > newest else newest
    return newest


def in_use_reason(dataset_dir: Path, *, quiet_s: float = QUIET_S, now_s: float | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> str | None:
    """Why the set's folder may be in use right now; None when nothing says so.

    A set that a running session is recording into looks like one a dead session left: the count ahead of the table,
    a data file with no end, pictures in `images/`. So before anything is moved, three things are asked: is a
    recorder or a trial session running (the system's list of processes); does any other process hold a file of the
    folder open for writing (`lsof`; where that command is missing, this one question goes unasked); and, when
    something in the folder was written less than `quiet_s` seconds ago, does the folder stay as it is until that
    long has passed (this waits for it). When the list of processes cannot be read, that is a reason too: a check
    that cannot be made is not passed.
    """
    try:
        family = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=True, timeout=30)
        listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True,
                                 timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return f"the system's list of running programs could not be read ({error}), so a session may be running"
    own = own_processes(family.stdout, os.getpid())
    running = session_processes(listing.stdout, own)
    if running:
        return ("a recording or trial session is running (" + "; ".join(running[:3]) + "). End it in its own window "
                "first (q, then wait for 'arm: torque ON'); if its window is gone, stop and ask")
    try:
        held = subprocess.run(["lsof", "-F", "pcan", "+D", str(dataset_dir)], capture_output=True, text=True,
                              check=False, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        held = ""
    writing = writers(held, own)
    if writing:
        return "a program is writing into the set's folder (" + "; ".join(writing[:3]) + "). Let it end first"
    newest = newest_change_s(dataset_dir)
    age_s = None if newest is None else (time.time() if now_s is None else now_s) - newest
    if age_s is not None and 0 <= age_s < quiet_s:
        # Written a moment ago: by an encoder the dead session left, or by a run of this tool. Watch it.
        sleep(quiet_s - age_s)
        if newest_change_s(dataset_dir) != newest:
            return (f"something is writing into the set's folder (it changed while this watched it for "
                    f"{quiet_s - age_s:.0f} s). If no session is running, run this again in a minute")
    return None


# --------------------------------------------------------------------------------------------------------------------
# The repair
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Plan:
    """A repair, decided and not yet done: where things go and what the rewritten files will say."""

    dataset_dir: Path
    placement_log: Path
    #: The folder beside the set that receives everything moved out, and a copy of everything rewritten.
    side_dir: Path
    #: Paths in the set's folder that are moved to `side_dir/moved/`, as they are.
    move_out: tuple[str, ...]
    #: Data files that are written again, each with the episodes it is to hold (`Verdict.rewrite_data`).
    rewrite_data: tuple[tuple[str, tuple[int, ...]], ...]
    #: The new text of `meta/stats.json`, of `meta/info.json` and of the placement log; None for one left as it is.
    stats_text: str | None
    info_text: str | None
    log_text: str | None
    #: The lines cut from the placement log's end, kept in `side_dir/cut_placements.jsonl`.
    cut_text: str


def side_dir_for(dataset_dir: Path, stamp: str) -> Path:
    """The folder beside a set that a repair of that moment moves things into: `<set>-unclosed-<stamp>`."""
    return dataset_dir.with_name(f"{dataset_dir.name}-{SIDE_LABEL}-{stamp}")


def make_plan(dataset_dir: Path, placement_log: Path, found: Found, verdict: Verdict, stamp: str,
              stats: Callable[[Path, Sequence[str], Sequence[str]], Mapping[str, Mapping[str, Any]]] | None = None,
              ) -> Plan:
    """Turn a verdict into the texts and moves of a repair. Reads the files that will be rewritten; changes nothing.

    `stats` computes the sound episodes' statistics (`sound_stats` when None, which needs LeRobot).
    """
    if verdict.refusal is not None:
        raise ValueError(f"there is no repair to plan: {verdict.refusal}")
    new_stats = None
    if verdict.rewrite_stats:
        try:
            old_stats = json.loads((dataset_dir / STATS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old_stats = {}
        if not isinstance(old_stats, dict):
            old_stats = {}
        tables = sorted(name for name, held in found.table_files.items() if held is not None)
        computed = _plain((stats or sound_stats)(dataset_dir, tables, list(old_stats)))
        try:
            counted = int(computed["index"]["count"][0])
        except (KeyError, TypeError, IndexError):
            counted = None
        if counted != verdict.sound_frames:
            raise ValueError(f"the sound episodes' own statistics describe {counted} frames, and the episodes have "
                             f"{verdict.sound_frames}")
        # The file describes the sound episodes and more (a session that died), or fewer of them (a save whose
        # last step, the statistics, was cut off): whichever covers fewer episodes lies inside the other.
        fewer_in_file = found.stats_frames is not None and found.stats_frames < verdict.sound_frames
        wrong = outside(old_stats, computed) if fewer_in_file else outside(computed, old_stats)
        if wrong:
            raise ValueError(wrong)
        new_stats = stats_text(computed)
    new_info = None
    if verdict.info_wanted is not None:
        new_info = info_text((dataset_dir / INFO).read_text(encoding="utf-8"), verdict.info_wanted)
    new_log, cut = None, ""
    if verdict.log_cut:
        with open(placement_log, **AS_IT_IS) as handle:
            new_log, cut = split_log(handle.read(), verdict.log_keep)
    return Plan(dataset_dir=dataset_dir, placement_log=placement_log, side_dir=side_dir_for(dataset_dir, stamp),
                move_out=verdict.move_out, rewrite_data=verdict.rewrite_data, stats_text=new_stats, info_text=new_info,
                log_text=new_log, cut_text=cut)


def _write_then_replace(text: str, staging: Path, target: Path) -> None:
    """Put `text` in place of `target` in one step: written and forced to disk beside it first, then renamed over
    it, so that the file is at every moment either the old one or the new one."""
    staging.parent.mkdir(parents=True, exist_ok=True)
    with open(staging, "w", **AS_IT_IS) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, target)


def carry_out(plan: Plan, rewrite: Callable[[Path, Path, Collection[int]], int] = keep_episodes) -> list[str]:
    """Do a planned repair, and return what was done, step by step. Nothing is deleted. `rewrite` stands in for
    `keep_episodes` in tests without a table reader.

    The order is chosen so that a set the recorder refuses stays refused until the repair is complete, and so that
    a repair that is cut short is finished by running it again (the plan is made from the folder as it is found):

    1. the folder beside the set is created (never one that exists), with a note of what it is and how to undo it;
    2. the files that will be rewritten are copied there, as they are (`before/`);
    3. the dead session's files and folders are moved there (`moved/`, same paths; a rename, no copying), and the
       folders this leaves empty in the set are removed;
    4. a data file that holds rows of a take that was never saved is written again without them beside the set,
       checked against the file, and put in its place;
    5. `meta/stats.json` is replaced;
    6. `meta/info.json` is replaced: until here the recorder refuses a set whose count is ahead of its table;
    7. the cut lines of the placement log are written beside the set, then the log is replaced: last, since from
       this step on the recorder accepts the set.

    Each step is added to `WHAT-WAS-DONE.txt` in that folder as it completes.
    """
    side = plan.side_dir
    side.mkdir(parents=False, exist_ok=False)
    record = side / "WHAT-WAS-DONE.txt"
    done: list[str] = []

    def note(line: str, *, step: bool = True) -> None:
        with open(record, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if step:
            done.append(line)

    log_name = plan.placement_log.name
    note(f"A repair of the recorded set {plan.dataset_dir.name} (scripts/rig/repair_set.py --apply), begun "
         f"{time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
         "A session ended without closing the set, or a save was cut short. This folder holds everything the\n"
         "repair took out of the set's folder or wrote over: nothing was deleted.\n\n"
         "  moved/    files and folders that belong to no sound episode, at the paths they had in the set's folder\n"
         "  before/   the files that were written again, as they were: meta/info.json, meta/stats.json, the\n"
         "            placement log, a data file\n"
         "  cut_placements.jsonl   the lines cut from the end of the placement log\n\n"
         "To undo by hand: move everything under moved/ back to the same path in the set's folder; copy\n"
         "everything under before/meta/ and before/data/ over the set's files of the same path; copy\n"
         f"before/{log_name} over the placement log.\n\n"
         "What was done, in order (a line is written when its step is complete):", step=False)

    rewritten = [name for name, _episodes in plan.rewrite_data]
    rewritten += [name for name, text in ((STATS, plan.stats_text), (INFO, plan.info_text)) if text is not None]
    for name in rewritten:
        if (plan.dataset_dir / name).is_file():
            (side / "before" / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan.dataset_dir / name, side / "before" / name)
            if (side / "before" / name).read_bytes() != (plan.dataset_dir / name).read_bytes():
                raise OSError(f"the copy of {name} in {side.name}/before is not the file")
            note(f"copied {name} as it was to before/{name}")
    if plan.log_text is not None:
        (side / "before").mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.placement_log, side / "before" / log_name)
        if (side / "before" / log_name).read_bytes() != plan.placement_log.read_bytes():
            raise OSError(f"the copy of {log_name} in {side.name}/before is not the file")
        note(f"copied the placement log {log_name} as it was to before/{log_name}")

    for name in plan.move_out:
        source, target = plan.dataset_dir / name, side / "moved" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        note(f"moved {name}{'/' if target.is_dir() else ''} to moved/{name}")
    for name in plan.move_out:
        # A folder the move left empty (a set none of whose episodes was closed): removed, up to the set's folder.
        folder = (plan.dataset_dir / name).parent
        while folder != plan.dataset_dir and folder.name != "meta" and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            note(f"removed the folder {folder.relative_to(plan.dataset_dir).as_posix()}/, which that left empty")
            folder = folder.parent

    for name, episodes in plan.rewrite_data:
        staged = side / "staging" / Path(name).name
        staged.parent.mkdir(parents=True, exist_ok=True)
        rows = rewrite(side / "before" / name, staged, episodes)
        with open(staged, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staged, plan.dataset_dir / name)
        note(f"wrote {name} again with the {rows} rows of episode{'' if len(episodes) == 1 else 's'} "
             f"{_run(episodes)} only")
    if plan.stats_text is not None:
        _write_then_replace(plan.stats_text, side / "staging" / "stats.json", plan.dataset_dir / STATS)
        note(f"wrote {STATS} again from the sound episodes' own statistics")
    if plan.info_text is not None:
        _write_then_replace(plan.info_text, side / "staging" / "info.json", plan.dataset_dir / INFO)
        counts = json.loads(plan.info_text)
        note(f"wrote {INFO} again: {counts['total_episodes']} episodes, {counts['total_frames']} frames")
    if (side / "staging").is_dir():
        (side / "staging").rmdir()
    if plan.log_text is not None:
        with open(side / "cut_placements.jsonl", "w", **AS_IT_IS) as handle:
            handle.write(plan.cut_text)
            handle.flush()
            os.fsync(handle.fileno())
        _write_then_replace(plan.log_text, plan.placement_log.with_name(log_name + ".repair-tmp"), plan.placement_log)
        cut = sum(1 for line in plan.cut_text.splitlines() if line.strip())
        note(f"cut the last {'line' if cut == 1 else f'{cut} lines'} of the placement log {log_name}; "
             f"{'it is' if cut == 1 else 'they are'} in cut_placements.jsonl")
    note(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}", step=False)
    return done


# --------------------------------------------------------------------------------------------------------------------
# What the person is told
# --------------------------------------------------------------------------------------------------------------------


def _size(path: Path) -> str:
    """A file's size, or what a folder holds, in a few words."""
    if path.is_dir():
        files = sum(len(names) for _directory, _folders, names in os.walk(path))
        return (f"a folder of {files} file{'' if files == 1 else 's'}" if files
                else "a folder with nothing in it but empty folders")
    try:
        return f"{path.stat().st_size:,} bytes"
    except OSError:
        return "not there"


def describe(dataset_dir: Path, placement_log: Path, found: Found, verdict: Verdict) -> list[str]:
    """What was found, in lines for the person: what is sound, and what the session that died left."""
    info = found.info or {}
    videos = "both videos" if len(found.cameras) == 2 else "every video"
    sound = (f"{verdict.sound_episodes} episodes, {verdict.sound_frames:,} frames (closed properly: each in the "
             f"episode table, its rows and its span of {videos} readable)" if verdict.sound_episodes
             else "no episode (none was closed properly)" if verdict.unclosed_episodes or verdict.move_out
             else "no episode (the set is empty, as it was created)")
    lines = [f"  The set at {dataset_dir}", f"    sound: {sound}",
             f"    meta/info.json says: {info.get('total_episodes')} episodes, {info.get('total_frames'):,} frames"]
    if verdict.unclosed_episodes:
        lines[-1] += (f" -- {verdict.unclosed_episodes} episodes ({verdict.unclosed_frames:,} frames) saved by a "
                      "session that ended without closing the set")
    if verdict.move_out:
        lines.append("    left behind, and no sound episode's:")
        held = {**found.table_files, **found.data_files, **found.video_files}
        for name in verdict.move_out:
            what = _size(dataset_dir / name)
            if name in held and held[name] is None:
                what += ", cannot be read (it has no end: its writer was never closed)"
            elif name in found.video_files:
                what += f", {held[name]} frames"
            elif name in held:
                what += ", readable, and no sound episode's"
            lines.append(f"      {name + ('/' if (dataset_dir / name).is_dir() else ''):58s} {what}")
    lines.append(f"    placement log {placement_log.name}: {len(found.log_lines)} lines"
                 + (f", {verdict.log_cut} more than the sound episodes" if verdict.log_cut else ""))
    lines += [f"    note: {note}" for note in verdict.notes]
    return lines


def steps(plan: Plan, found: Found, verdict: Verdict, side_name: str) -> list[str]:
    """What a repair does, in numbered lines, in the order `carry_out` does it. `side_name` is how the folder
    beside the set is called in them (its name is only fixed when the repair is done)."""
    info = found.info or {}
    lines: list[str] = []

    def step(text: str) -> None:
        lines.append(f"    {len(lines) + 1}. {text}")

    rewritten = [name for name, _episodes in plan.rewrite_data]
    rewritten += [name for name, text in ((STATS, plan.stats_text), (INFO, plan.info_text)) if text is not None]
    rewritten += [plan.placement_log.name] if plan.log_text is not None else []
    if rewritten:
        step(f"copy {', '.join(rewritten)} as {'it is' if len(rewritten) == 1 else 'they are'} into "
             f"{side_name}/before/")
    if plan.move_out:
        count = len(plan.move_out)
        step(f"move the {'one entry' if count == 1 else f'{count} entries'} listed above into {side_name}/moved/ "
             "(nothing is deleted)")
    for name, episodes in plan.rewrite_data:
        held = found.data_files.get(name) or {}
        left_out = sum(rows for episode, rows in held.items() if episode not in episodes)
        step(f"write {name} again with the rows of episode{'' if len(episodes) == 1 else 's'} {_run(episodes)} only "
             f"({left_out} rows of a take that was never saved are left out)")
    if plan.stats_text is not None:
        now = "no frames" if found.stats_frames is None else f"{found.stats_frames:,} frames"
        step(f"write {STATS} again from the {verdict.sound_episodes} sound episodes' own statistics (it describes "
             f"{now} now, and will describe {verdict.sound_frames:,})")
    if plan.info_text is not None:
        changes = [f"{key} {json.dumps(info.get(key))} -> {json.dumps(value)}"
                   for key, value in (verdict.info_wanted or {}).items() if info.get(key) != value]
        step(f"write {INFO} again: {'; '.join(changes)}")
    if plan.log_text is not None:
        cut = found.log_lines[verdict.log_keep:]
        step(f"cut {plan.placement_log.name} from {len(found.log_lines)} to {verdict.log_keep} lines; the "
             f"{'one cut line' if len(cut) == 1 else f'{len(cut)} cut lines'} "
             f"(episode{'' if len(cut) == 1 else 's'} {_run(cut)}) "
             f"{'is' if len(cut) == 1 else 'are'} kept in {side_name}/cut_placements.jsonl")
    return lines


@dataclass(frozen=True, slots=True)
class Report:
    """What a run of the repair has to say, and how it ended."""

    #: 0: the set adds up (it did already, or the repair was done). 1: refused, nothing changed. 2: a repair is
    #: needed and was only described (no --apply). 3: the repair was begun and the set still does not add up.
    code: int
    lines: tuple[str, ...]
    #: The folder beside the set that a repair moved things into; None when nothing was done.
    side_dir: Path | None = None


def repair(dataset_dir: Path, placement_log: Path, *, apply: bool, stamp: str,
           in_use: Callable[[Path], str | None] = in_use_reason,
           stats: Callable[[Path, Sequence[str], Sequence[str]], Mapping[str, Mapping[str, Any]]] | None = None,
           read: Callable[[Path, Path], Found] = look,
           rewrite: Callable[[Path, Path, Collection[int]], int] = keep_episodes) -> Report:
    """Look at a set, say what a repair would do, and with `apply` do it and look again.

    Refuses, changing nothing: when there is no such set; when a session appears to be using the folder (`in_use`);
    when what is sound cannot be told (`judge`); and, as nothing to do, when the set adds up already. `stamp` names
    the folder beside the set (the time of the run, from the caller: nothing here reads the clock for a name).
    `stats`, `read` and `rewrite` stand in for `sound_stats`, `look` and `keep_episodes` in tests without LeRobot
    and without a table reader.
    """
    nothing = "  Nothing was changed."
    ask = "  Do not delete anything, and say what this prints."
    if not (dataset_dir / "meta").is_dir():
        return Report(1, (f"  REFUSING -- there is no recorded set at {dataset_dir}.", nothing))
    busy = in_use(dataset_dir)
    if busy:
        return Report(1, (f"  REFUSING -- {busy}.", nothing))
    try:
        found = read(dataset_dir, placement_log)
        verdict = judge(found)
    except Exception as error:  # noqa: BLE001 -- nothing has been written yet: whatever it is, the set is as it was
        return Report(1, (f"  REFUSING -- the set at {dataset_dir} is not in the form this tool knows "
                          f"({type(error).__name__}: {error}).", nothing, ask))
    if verdict.refusal:
        return Report(1, (f"  REFUSING -- the set at {dataset_dir}: {verdict.refusal}.", nothing, ask))
    lines = describe(dataset_dir, placement_log, found, verdict)
    if verdict.sound_already:
        return Report(0, (*lines, "", f"  NOTHING TO REPAIR -- the set adds up: {verdict.sound_episodes} episodes, "
                                      f"{verdict.sound_frames:,} frames, a placement log of {len(found.log_lines)} "
                                      "lines.", nothing))
    side = side_dir_for(dataset_dir, stamp)
    if side.exists():
        return Report(1, (*lines, "", f"  REFUSING -- {side} is already there (the folder is named after the time): "
                                      "run this again.", nothing))
    try:
        plan = make_plan(dataset_dir, placement_log, found, verdict, stamp, stats)
    except Exception as error:  # noqa: BLE001 -- as above: a plan reads, and writes nothing
        return Report(1, (*lines, "", f"  REFUSING -- the repair cannot be prepared: {error}.", nothing, ask))
    lost = list(range(verdict.sound_episodes, verdict.sound_episodes + verdict.unclosed_episodes))
    again_words = (f"episode{'' if len(lost) == 1 else 's'} {_run(lost)}, which that session saved and never closed, "
                   f"{'is' if len(lost) == 1 else 'are'} recorded again")
    if not apply:
        after = (f"  Afterwards: {verdict.sound_episodes} episodes, {verdict.sound_frames:,} frames, a placement log "
                 f"of {verdict.log_keep} lines; the recorder's --resume goes on at episode {verdict.sound_episodes}"
                 if verdict.sound_episodes else
                 "  Afterwards: the empty set its first session created (meta/info.json and nothing else) and a "
                 "placement log of no lines; the recorder's --resume begins it again at episode 0")
        return Report(2, (*lines, "", "  WOULD DO (nothing was changed; run it again with --apply):",
                          *steps(plan, found, verdict, f"{dataset_dir.name}-{SIDE_LABEL}-<time>"),
                          after + (f" ({again_words})." if lost else "."), nothing))
    try:
        done = carry_out(plan, rewrite)
    except Exception as error:  # noqa: BLE001 -- whatever stopped it, what was done is on record and nothing is lost
        return Report(3, (*lines, "", f"  STOPPED PARTWAY -- {type(error).__name__}: {error}.",
                          f"  What was done so far is written in {side}/WHAT-WAS-DONE.txt; nothing was deleted.",
                          "  Do not record into the set. Run this again without --apply and say what it prints."),
                      side)
    told = [*lines, "", "  DONE (nothing was deleted):",
            *(f"    {number}. {line}" for number, line in enumerate(done, 1)),
            f"  Everything taken out of the set is in {side}", ""]
    try:
        # ... with this module's own rule, and with the recorder's own check before it appends to a set
        again = judge(read(dataset_dir, placement_log))
        left = again.refusal or record_layout.resume_problem(
            record_layout.dataset_counts(dataset_dir), len(log_episodes(placement_log)), placement_log.name)
    except Exception as error:  # noqa: BLE001 -- the second look is the proof: without it nothing is claimed
        again, left = None, f"the set could not be read again ({type(error).__name__}: {error})"
    if again is None or left or not again.sound_already:
        return Report(3, (*told, f"  THE SET STILL DOES NOT ADD UP -- {left or 'a repair is still needed'}.",
                          "  Do not record into it. Run this again without --apply and say what it prints."), side)
    return Report(0, (*told, f"  THE SET ADDS UP AGAIN: {again.sound_episodes} episodes, {again.sound_frames:,} "
                             f"frames, a placement log of {again.log_keep} lines. The recorder's --resume goes on "
                             f"at episode {again.sound_episodes}" + (f" ({again_words})." if lost else ".")), side)
