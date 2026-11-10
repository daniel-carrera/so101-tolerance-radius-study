"""The repair of a recorded set that a session left without closing (`rig/repair_set.py`).

Four kinds of test:

- the rule that says what is sound (`judge`), on plain counts: every way of dying that was measured with LeRobot's
  own writer on 2026-10-06, and everything the rule refuses. No LeRobot, no table reader: both environments;
- the texts and the file operations on small made-up folders, with the standard library only: both environments;
- the question whether something is still using the folder, on the text the system's tools print and on real
  child processes;
- the whole repair on a tiny set written by LeRobot's own writer, with a session that dies in it. These need the
  rig's interpreter and skip elsewhere. A session "dies" here in two ways: for the measured case a real process that
  ends by `os._exit(9)`; for the variations, the set's folder copied while the writer is still open, which is the
  bytes a kill leaves on disk (a parquet writer puts each episode's rows in the file as they are saved, and the end
  of the file only when it is closed).

Nothing here reads or writes the datasets of the home folder: every set is made up in a temporary folder.
"""

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import fakes_rig

from tolerance_radius.rig import record_layout, repair_set
from tolerance_radius.rig.repair_set import EpisodeRow, Found, Segment, Verdict, judge

CAMERAS = ("observation.images.front", "observation.images.wrist")
FPS = 30


def data_file(number: int) -> str:
    return f"data/chunk-000/file-{number:03d}.parquet"


def table_file(number: int) -> str:
    return f"meta/episodes/chunk-000/file-{number:03d}.parquet"


def video_file(camera: str, number: int) -> str:
    return f"videos/{camera}/chunk-000/file-{number:03d}.mp4"


def closed(sessions: tuple[int, ...] = (1, 26), frames: int = 20) -> Found:
    """A set as sessions that closed it leave it, in plain counts: each session saved so many episodes of `frames`
    frames into files of its own (file-000, file-001, ...), and the placement log has a line for each."""
    rows, tables, data, videos = [], {}, {}, {}
    for number, saved in enumerate(sessions):
        first = len(rows)
        for position in range(saved):
            episode = first + position
            rows.append(EpisodeRow(
                episode_index=episode, frames=frames, first_frame=episode * frames, end_frame=(episode + 1) * frames,
                table_file=table_file(number), data_file=data_file(number),
                segments=tuple(Segment(camera, video_file(camera, number), position * frames / FPS,
                                       (position + 1) * frames / FPS) for camera in CAMERAS)))
        tables[table_file(number)] = saved
        data[data_file(number)] = {first + position: frames for position in range(saved)}
        videos.update({video_file(camera, number): saved * frames for camera in CAMERAS})
    episodes = len(rows)
    info = {"codebase_version": "v3.0", "total_episodes": episodes, "total_frames": episodes * frames,
            "total_tasks": 1 if episodes else 0, "splits": {"train": f"0:{episodes}"} if episodes else {}, "fps": FPS}
    return Found(info=info, cameras=CAMERAS, rows=tuple(rows), table_files=tables, data_files=data,
                 video_files=videos, stats_frames=episodes * frames if episodes else None, has_stats=episodes > 0,
                 has_tasks=episodes > 0, log_lines=tuple(range(episodes)))


def died(before: Found, saves: int, *, lines: int | None = None, pictures: bool = True, frames: int = 20,
         table_written: bool = False) -> Found:
    """`before` after one more session that saved `saves` episodes and ended without closing the set: the count in
    info.json ahead, a data file with no end, a video file per camera, pictures in images/, `lines` placement lines
    (one per save unless said). With `table_written`, also an episode-table file with no end (ten saves or more)."""
    number = len(before.table_files)
    episodes = before.info["total_episodes"]
    info = {**before.info, "total_episodes": episodes + saves,
            "total_frames": before.info["total_frames"] + saves * frames, "total_tasks": 1,
            "splits": {"train": f"0:{episodes + saves}"}}
    extra_lines = saves if lines is None else lines
    found = replace(before, info=info if saves else before.info,
                    leftovers=("images",) if pictures else (),
                    log_lines=before.log_lines + tuple(range(episodes, episodes + extra_lines)))
    if saves:
        found = replace(found, data_files={**before.data_files, data_file(number): None},
                        video_files={**before.video_files,
                                     **{video_file(camera, number): saves * frames for camera in CAMERAS}},
                        stats_frames=(before.stats_frames or 0) + saves * frames, has_stats=True, has_tasks=True)
    if table_written:
        found = replace(found, table_files={**before.table_files, table_file(number): None})
    return found


# ====================================================================================================================
# The rule
# ====================================================================================================================


class SoundSetTests(unittest.TestCase):
    """A set that adds up is left alone."""

    def test_a_set_that_two_sessions_closed_has_nothing_to_repair(self):
        verdict = judge(closed())
        self.assertIsNone(verdict.refusal)
        self.assertTrue(verdict.sound_already)
        self.assertEqual((verdict.sound_episodes, verdict.sound_frames), (27, 540))
        self.assertEqual((verdict.move_out, verdict.info_wanted, verdict.rewrite_stats, verdict.log_cut),
                         ((), None, False, 0))
        self.assertEqual((verdict.unclosed_episodes, verdict.unclosed_frames, verdict.log_keep), (0, 0, 27))

    def test_a_set_created_empty_by_a_start_that_was_refused_has_nothing_to_repair(self):
        # A first sitting refused after the set was created (a camera that will not stream): meta/info.json only.
        verdict = judge(closed(sessions=()))
        self.assertTrue(verdict.sound_already)
        self.assertEqual((verdict.sound_episodes, verdict.sound_frames, verdict.log_keep), (0, 0, 0))

    def test_a_file_the_tool_does_not_know_is_left_alone_and_said(self):
        verdict = judge(replace(closed(), strangers=(".DS_Store", "meta/notes.txt")))
        self.assertTrue(verdict.sound_already)
        self.assertEqual(len(verdict.notes), 2)
        self.assertIn(".DS_Store is not a file this tool knows", verdict.notes[0])

    def test_a_video_span_longer_than_its_episode_is_said_and_not_refused(self):
        # Measured 2026-10-06: a session killed in its first take left 40 pictures in images/; the next session's
        # take of 25 frames was saved over them, and the episode's video span held 40 frames for 25 rows.
        whole = closed(sessions=(1, 2))
        last = whole.rows[-1]
        longer = replace(last, segments=tuple(replace(s, to_s=s.to_s + 15 / FPS) for s in last.segments))
        videos = {name: frames + (15 if name.endswith("file-001.mp4") else 0)
                  for name, frames in whole.video_files.items()}
        verdict = judge(replace(whole, rows=(*whole.rows[:-1], longer), video_files=videos))
        self.assertIsNone(verdict.refusal)
        self.assertTrue(verdict.sound_already)
        self.assertEqual(len(verdict.notes), 2)
        self.assertIn("holds 35 frames for 20 rows", verdict.notes[0])


class DeadSessionTests(unittest.TestCase):
    """Each way of dying that was measured, as the counts it leaves, and what the rule makes of it. The set before
    is the clustered set of 2026-10-06 in shape: 27 episodes, one from a first session and 26 from a second."""

    def test_killed_in_its_first_take_only_the_pictures_are_moved(self):
        verdict = judge(died(closed(), saves=0))
        self.assertIsNone(verdict.refusal)
        self.assertFalse(verdict.sound_already)
        self.assertEqual(verdict.move_out, ("images",))
        self.assertEqual((verdict.info_wanted, verdict.rewrite_stats, verdict.log_cut), (None, False, 0))
        self.assertEqual((verdict.sound_episodes, verdict.unclosed_episodes), (27, 0))

    def test_killed_after_one_save_its_files_go_and_the_counts_go_back(self):
        verdict = judge(died(closed(), saves=1))
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.move_out,
                         (data_file(2), video_file(CAMERAS[0], 2), video_file(CAMERAS[1], 2), "images"))
        self.assertEqual(verdict.info_wanted, {"total_episodes": 27, "total_frames": 540, "total_tasks": 1,
                                               "splits": {"train": "0:27"}})
        self.assertTrue(verdict.rewrite_stats)
        self.assertEqual((verdict.log_keep, verdict.log_cut), (27, 1))
        self.assertEqual((verdict.unclosed_episodes, verdict.unclosed_frames), (1, 20))

    def test_killed_after_two_saves_with_a_third_take_under_way(self):
        verdict = judge(died(closed(), saves=2))
        self.assertEqual((verdict.sound_episodes, verdict.sound_frames), (27, 540))
        self.assertEqual((verdict.unclosed_episodes, verdict.unclosed_frames), (2, 40))
        self.assertEqual(len(verdict.move_out), 4)
        self.assertEqual((verdict.log_keep, verdict.log_cut), (27, 2))
        self.assertTrue(verdict.rewrite_stats)

    def test_killed_between_a_save_and_its_placement_line_the_log_loses_one_line_less(self):
        verdict = judge(died(closed(), saves=2, lines=1))
        self.assertIsNone(verdict.refusal)
        self.assertEqual((verdict.unclosed_episodes, verdict.log_keep, verdict.log_cut), (2, 27, 1))
        # ... and with a single save and no line, the log is already the sound episodes'
        verdict = judge(died(closed(), saves=1, lines=0))
        self.assertEqual((verdict.unclosed_episodes, verdict.log_cut), (1, 0))
        self.assertFalse(verdict.sound_already)

    def test_killed_after_more_saves_than_the_table_buffers_its_unfinished_table_file_goes_too(self):
        # LeRobot writes episode rows ten at a time: after eleven saves a table file is there, with no end.
        verdict = judge(died(closed(), saves=11, table_written=True))
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.move_out, (data_file(2), table_file(2), video_file(CAMERAS[0], 2),
                                            video_file(CAMERAS[1], 2), "images"))
        self.assertEqual((verdict.unclosed_episodes, verdict.log_cut), (11, 11))

    def test_killed_while_a_save_encodes_the_count_never_moved_and_the_encoding_folders_go(self):
        before = closed()
        found = replace(before, data_files={**before.data_files, data_file(2): None},
                        leftovers=("images", "tmp7j5hlu11", "tmpx61j0798"))
        verdict = judge(found)
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.move_out, (data_file(2), "images", "tmp7j5hlu11", "tmpx61j0798"))
        self.assertEqual((verdict.info_wanted, verdict.rewrite_stats, verdict.log_cut), (None, False, 0))

    def test_a_new_set_whose_first_session_was_killed_goes_back_to_the_empty_set(self):
        verdict = judge(died(closed(sessions=()), saves=2))
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.sound_episodes, 0)
        self.assertEqual(verdict.move_out, (data_file(0), video_file(CAMERAS[0], 0), video_file(CAMERAS[1], 0),
                                            "images", "meta/stats.json", "meta/tasks.parquet"))
        self.assertEqual(verdict.info_wanted, {"total_episodes": 0, "total_frames": 0, "total_tasks": 0, "splits": {}})
        self.assertFalse(verdict.rewrite_stats)                   # an empty set has no statistics file
        self.assertEqual((verdict.log_keep, verdict.log_cut), (0, 2))

    def test_a_dead_session_whose_video_passed_the_size_limit_leaves_two_files_a_camera_and_both_go(self):
        # A long sitting starts a second video file for a camera when the first passes 200 MB (2026-10-06, on a copy
        # with the limit set to 1 MB: the front camera's video was split several times within one session).
        before = closed()
        dead = died(before, saves=12, table_written=True)
        videos = {**dead.video_files, video_file(CAMERAS[0], 2): 160, video_file(CAMERAS[0], 3): 80}
        verdict = judge(replace(dead, video_files=videos))
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.move_out, (data_file(2), table_file(2), video_file(CAMERAS[0], 2),
                                            video_file(CAMERAS[0], 3), video_file(CAMERAS[1], 2), "images"))

    def test_sound_episodes_whose_session_split_a_video_are_sound_and_the_dead_files_come_after_them(self):
        whole = closed(sessions=(1, 4))
        # the second session's last two episodes are in a second video file of the front camera
        rows = tuple(replace(row, segments=tuple(
            replace(s, file=video_file(CAMERAS[0], 2), from_s=(row.episode_index - 3) * 20 / FPS,
                    to_s=(row.episode_index - 2) * 20 / FPS) if s.camera == CAMERAS[0] and row.episode_index >= 3 else s
            for s in row.segments)) for row in whole.rows)
        videos = {**whole.video_files, video_file(CAMERAS[0], 1): 40, video_file(CAMERAS[0], 2): 40}
        split = replace(whole, rows=rows, video_files=videos)
        self.assertTrue(judge(split).sound_already)
        # a session that dies after it: its front video is numbered after the split one, its wrist video after 1
        dead = replace(split, info={**split.info, "total_episodes": 6, "total_frames": 120, "splits": {"train": "0:6"}},
                       data_files={**split.data_files, data_file(2): None},
                       video_files={**videos, video_file(CAMERAS[0], 3): 20, video_file(CAMERAS[1], 2): 20},
                       log_lines=tuple(range(6)), stats_frames=120)
        verdict = judge(dead)
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.move_out, (data_file(2), video_file(CAMERAS[0], 3), video_file(CAMERAS[1], 2)))
        self.assertEqual((verdict.sound_episodes, verdict.unclosed_episodes, verdict.log_cut), (5, 1, 1))

    def test_killed_between_closing_the_data_file_and_the_episode_table(self):
        # finalize() closes the data file first: a readable data file whose episodes have no row in the table.
        before = closed()
        found = died(before, saves=2)
        found = replace(found, data_files={**before.data_files, data_file(2): {27: 20, 28: 20}})
        verdict = judge(found)
        self.assertIsNone(verdict.refusal)
        self.assertIn(data_file(2), verdict.move_out)
        self.assertEqual(verdict.unclosed_episodes, 2)

    def test_a_first_save_cut_short_in_a_session_that_closed_the_set(self):
        # Measured 2026-10-06: Ctrl-C while the session's first save encoded. The set was closed on the way out:
        # the counts agree, the recorder's own check finds nothing, and a readable data file holds 60 rows of an
        # episode the table does not have, beside two encoding folders.
        before = closed()
        found = replace(before, data_files={**before.data_files, data_file(2): {27: 20}},
                        leftovers=("tmpaf4l6r29", "tmpcc442tea"))
        verdict = judge(found)
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.move_out, (data_file(2), "tmpaf4l6r29", "tmpcc442tea"))
        self.assertEqual((verdict.info_wanted, verdict.rewrite_stats, verdict.log_cut), (None, False, 0))


class CutShortSaveTests(unittest.TestCase):
    """A save cut short while it encoded, in a session that went on to close the set: the take's rows are in a
    data file that sound episodes are in too. Measured 2026-10-06 (Ctrl-C while the session's second save encoded):
    the counts agree, the recorder's own check finds nothing, and resumed as it was the next episode's rows were
    read from the cut take (19,282 rows for 19,222 frames, 45 index values held twice)."""

    def test_the_file_is_to_be_written_again_with_its_sound_episode_only(self):
        before = closed(sessions=(1, 26, 1))
        mixed = {**before.data_files, data_file(2): {27: 20, 28: 20}}
        verdict = judge(replace(before, data_files=mixed, leftovers=("tmppdcakh28", "tmpt_mx272f")))
        self.assertIsNone(verdict.refusal)
        self.assertFalse(verdict.sound_already)
        self.assertEqual(verdict.rewrite_data, ((data_file(2), (27,)),))
        self.assertEqual(verdict.move_out, ("tmppdcakh28", "tmpt_mx272f"))
        self.assertEqual((verdict.sound_episodes, verdict.info_wanted, verdict.rewrite_stats, verdict.log_cut),
                         (28, None, False, 0))
        self.assertIn(f"{data_file(2)} holds 20 rows of episode 28 beside its sound episode 27", verdict.notes[0])
        self.assertIn("a take that was never saved", verdict.notes[0])

    def test_after_a_later_session_took_the_cut_take_s_number_the_rows_are_still_told_apart(self):
        # The next session's episode 28 is in its own file; the rows numbered 28 in the earlier file are the cut
        # take's, because the table gives that file episode 27 only.
        before = closed(sessions=(1, 26, 1, 1))
        mixed = {**before.data_files, data_file(2): {27: 20, 28: 20}}
        verdict = judge(replace(before, data_files=mixed))
        self.assertIsNone(verdict.refusal)
        self.assertEqual(verdict.rewrite_data, ((data_file(2), (27,)),))
        self.assertEqual((verdict.sound_episodes, verdict.move_out), (29, ()))

    def test_a_file_of_several_sound_episodes_keeps_them_all(self):
        before = closed()
        mixed = {**before.data_files, data_file(1): {**before.data_files[data_file(1)], 27: 20}}
        verdict = judge(replace(before, data_files=mixed))
        self.assertEqual(verdict.rewrite_data, ((data_file(1), tuple(range(1, 27))),))
        self.assertIn("beside its sound episodes 1 to 26", verdict.notes[0])

    def test_rows_that_come_before_the_file_s_own_episodes_are_not_a_cut_take_s_and_are_refused(self):
        before = closed()
        mixed = {**before.data_files, data_file(1): {0: 20, **before.data_files[data_file(1)]}}
        verdict = judge(replace(before, data_files=mixed))
        self.assertIn(f"{data_file(1)} holds rows of episode 0, which the episode table does not give to it",
                      verdict.refusal)
        self.assertEqual(verdict.rewrite_data, ())

    def test_the_cut_take_s_video_left_at_the_end_of_a_file_is_said_and_left(self):
        # the interrupt between the two cameras' videos: one of them already joined to the session's file
        before = closed(sessions=(1, 26, 1))
        longer = {**before.video_files, video_file(CAMERAS[0], 2): 40}
        verdict = judge(replace(before, video_files=longer))
        self.assertTrue(verdict.sound_already)
        self.assertIn(f"{video_file(CAMERAS[0], 2)} holds 40 frames, 20 more than its episodes' spans",
                      verdict.notes[0])


class HalfDoneRepairTests(unittest.TestCase):
    """A repair that was cut short is finished by running it again: the rule is asked of the folder as it is."""

    def test_with_the_files_moved_the_counts_and_the_log_are_still_to_do(self):
        dead = died(closed(), saves=2)
        moved = replace(dead, data_files=closed().data_files, video_files=closed().video_files, leftovers=())
        verdict = judge(moved)
        self.assertEqual(verdict.move_out, ())
        self.assertIsNotNone(verdict.info_wanted)
        self.assertTrue(verdict.rewrite_stats)
        self.assertEqual(verdict.log_cut, 2)

    def test_with_the_counts_written_only_the_log_is_still_to_do(self):
        verdict = judge(replace(closed(), log_lines=tuple(range(29))))
        self.assertIsNone(verdict.refusal)
        self.assertFalse(verdict.sound_already)
        self.assertEqual((verdict.move_out, verdict.info_wanted, verdict.rewrite_stats), ((), None, False))
        self.assertEqual((verdict.log_keep, verdict.log_cut), (27, 2))

    def test_statistics_that_describe_other_frames_than_the_sound_ones_are_written_again(self):
        verdict = judge(replace(closed(), stats_frames=580))
        self.assertTrue(verdict.rewrite_stats)
        self.assertFalse(verdict.sound_already)
        self.assertTrue(judge(replace(closed(), has_stats=False, stats_frames=None)).rewrite_stats)


class RefusalTests(unittest.TestCase):
    """What the rule cannot make sense of is refused, with a sentence that says what was seen."""

    def refused(self, found: Found, words: str) -> Verdict:
        verdict = judge(found)
        self.assertIsNotNone(verdict.refusal, "the rule went on where it should have refused")
        self.assertIn(words, verdict.refusal)
        self.assertEqual((verdict.move_out, verdict.info_wanted, verdict.rewrite_stats, verdict.log_cut),
                         ((), None, False, 0))
        self.assertFalse(verdict.sound_already)
        return verdict

    def test_an_info_file_that_cannot_be_read(self):
        self.refused(replace(closed(), info=None), "meta/info.json cannot be read")
        self.refused(replace(closed(), info={**closed().info, "total_episodes": None}), "meta/info.json cannot be read")

    def test_another_layout_or_no_frame_rate(self):
        self.refused(replace(closed(), info={**closed().info, "codebase_version": "v2.1"}), "knows only v3.0")
        self.refused(replace(closed(), info={**closed().info, "fps": None}), "no frame rate")

    def test_a_session_recorded_on_top_of_an_unclosed_one(self):
        # The dry run of 2026-10-06, before the recorder refused such a set: the next session went on at episode
        # 29 with 27 rows in the table. Its episodes were closed properly and are not this tool's to move.
        before = closed()
        row = replace(before.rows[-1], episode_index=29, table_file=table_file(2), data_file=data_file(2))
        found = replace(before, rows=(*before.rows, row), info={**before.info, "total_episodes": 30})
        verdict = self.refused(found, "not an unbroken run from 0")
        self.assertIn("a session recorded on top of an unclosed one", verdict.refusal)

    def test_frame_numbers_that_are_not_one_unbroken_run(self):
        before = closed()
        rows = (*before.rows[:-1], replace(before.rows[-1], first_frame=600, end_frame=620))
        self.refused(replace(before, rows=rows), "frame numbers are not one unbroken run")
        rows = (*before.rows[:-1], replace(before.rows[-1], frames=19))
        self.refused(replace(before, rows=rows), "frame numbers are not one unbroken run")

    def test_a_sound_episode_whose_rows_are_not_there(self):
        before = closed()
        missing = {name: held for name, held in before.data_files.items() if name != data_file(1)}
        self.refused(replace(before, data_files=missing), f"its data file {data_file(1)} is not there")
        self.refused(replace(before, data_files={**before.data_files, data_file(1): None}), "cannot be read")
        short = {**before.data_files, data_file(1): {**before.data_files[data_file(1)], 26: 19}}
        self.refused(replace(before, data_files=short), "episode 26 has 20 frames in the episode table and 19 rows")

    def test_a_sound_episode_whose_pictures_are_not_there(self):
        before = closed()
        gone = {name: frames for name, frames in before.video_files.items() if name != video_file(CAMERAS[1], 1)}
        self.refused(replace(before, video_files=gone), f"{video_file(CAMERAS[1], 1)} is not there")
        self.refused(replace(before, video_files={**before.video_files, video_file(CAMERAS[0], 0): None}),
                     f"{video_file(CAMERAS[0], 0)} cannot be read")
        self.refused(replace(before, video_files={**before.video_files, video_file(CAMERAS[0], 1): 519}),
                     "does not hold them in a file of 519 frames")
        one_camera = (*before.rows[:-1], replace(before.rows[-1], segments=before.rows[-1].segments[:1]))
        self.refused(replace(before, rows=one_camera), f"no video span for the camera {CAMERAS[1]}")
        last = before.rows[-1]
        short = replace(last, segments=tuple(replace(s, to_s=s.to_s - 1 / FPS) for s in last.segments))
        self.refused(replace(before, rows=(*before.rows[:-1], short)), "19 frames) does not hold them")

    def test_counts_in_the_info_file_that_a_dead_session_does_not_leave(self):
        before = closed()
        self.refused(replace(before, info={**before.info, "total_episodes": 26}), "never behind")
        self.refused(replace(before, info={**before.info, "total_frames": 560}), "the two counts do not go together")
        self.refused(replace(before, info={**before.info, "total_episodes": 28}), "the two counts do not go together")
        self.refused(replace(before, info={**before.info, "total_episodes": 28, "total_frames": 500}),
                     "the two counts do not go together")

    def test_a_file_of_no_sound_episode_that_comes_before_the_sound_ones(self):
        before = closed(sessions=(1, 26))
        # the second session's files renumbered 2, with unpointed files left at 1
        rows = tuple(replace(row, data_file=data_file(2) if row.episode_index else row.data_file)
                     for row in before.rows)
        data = {data_file(0): before.data_files[data_file(0)], data_file(1): None,
                data_file(2): before.data_files[data_file(1)]}
        self.refused(replace(before, rows=rows, data_files=data), f"the data file {data_file(1)} belongs to no sound")
        tables = {table_file(0): 1, table_file(1): None, table_file(2): 26}
        rows = tuple(replace(row, table_file=table_file(2) if row.episode_index else row.table_file)
                     for row in before.rows)
        self.refused(replace(before, rows=rows, table_files=tables), f"the episode-table file {table_file(1)} belongs")
        videos = {video_file(camera, number): frames for camera in CAMERAS
                  for number, frames in ((0, 20), (1, 20), (2, 520))}
        rows = tuple(replace(row, segments=tuple(replace(s, file=s.file.replace("file-001", "file-002"))
                                                 for s in row.segments)) for row in before.rows)
        self.refused(replace(before, rows=rows, video_files=videos),
                     f"the video file {video_file(CAMERAS[0], 1)} belongs to no sound episode and comes before")

    def test_a_table_file_that_can_be_read_and_holds_no_episode(self):
        before = closed()
        self.refused(replace(before, table_files={**before.table_files, table_file(2): 0}), "holds no episode")

    def test_a_placement_log_that_is_not_the_sound_episodes_in_order(self):
        before = closed()
        verdict = self.refused(replace(before, log_lines=tuple(range(26))), "1 sound episode has no placement line")
        self.assertIn("never writes a placement line", verdict.refusal)
        self.refused(replace(before, log_lines=()), "the placement log has 0 lines: 27 sound episodes have no")
        swapped = (*range(25), 26, 25)
        self.refused(replace(before, log_lines=swapped), "line 26 of the placement log is not episode 25's")
        self.refused(replace(before, log_lines=(*range(26), None)), "line 27 of the placement log is not episode 26's")

    def test_lines_after_the_sound_ones_that_are_not_one_dead_session_s(self):
        dead = died(closed(), saves=2)
        self.refused(replace(dead, log_lines=(*range(27), 27, 27)), "are not the episodes 27 onwards in order")
        self.refused(replace(dead, log_lines=(*range(27), 3)), "they say 3")
        self.refused(replace(dead, log_lines=(*range(27), None)), "they say ?")


# ====================================================================================================================
# The texts (standard library only)
# ====================================================================================================================


class TextTests(unittest.TestCase):
    def test_the_info_file_gets_its_counts_and_keeps_everything_else_in_its_order(self):
        old = json.dumps({"codebase_version": "v3.0", "fps": 30, "features": {"action": {"shape": [6]}},
                          "total_episodes": 29, "total_frames": 19237, "total_tasks": 1, "chunks_size": 1000,
                          "robot_type": "so_follower", "splits": {"train": "0:29"}}, indent=4, ensure_ascii=False)
        new = repair_set.info_text(old, {"total_episodes": 27, "total_frames": 19117, "total_tasks": 1,
                                         "splits": {"train": "0:27"}})
        self.assertEqual(new, old.replace("29", "27").replace("19237", "19117"))
        self.assertEqual(list(json.loads(new)), list(json.loads(old)))
        self.assertFalse(new.endswith("\n"))                      # LeRobot's own file has no last newline

    def test_a_log_is_cut_after_so_many_lines_and_the_two_parts_are_the_log(self):
        text = '{"episode_index": 0}\n{"episode_index": 1}\n\n{"episode_index": 2}\n{"episode_index": 3}\n'
        kept, cut = repair_set.split_log(text, 2)
        self.assertEqual(kept, '{"episode_index": 0}\n{"episode_index": 1}\n')
        self.assertEqual(kept + cut, text)
        self.assertEqual(repair_set.split_log(text, 4), (text, ""))
        self.assertEqual(repair_set.split_log(text, 0), ("", text))
        # a last line that was never ended (the process died while writing it) is cut whole
        kept, cut = repair_set.split_log('{"episode_index": 0}\n{"episode_ind', 1)
        self.assertEqual((kept, cut), ('{"episode_index": 0}\n', '{"episode_ind'))
        with self.assertRaisesRegex(ValueError, "fewer than the 5 to keep"):
            repair_set.split_log(text, 5)

    def test_a_log_s_lines_are_read_as_episode_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clustered_placements.jsonl"
            self.assertEqual(repair_set.log_episodes(path), ())
            path.write_text('{"episode_index": 0, "cell": "C2"}\n\n{"episode_index": 1}\nnot a line\n{"cell": "B1"}\n',
                            encoding="utf-8")
            self.assertEqual(repair_set.log_episodes(path), (0, 1, None, None))
            self.assertEqual(len(repair_set.log_episodes(path)), record_layout.placement_lines(path))
            # bytes that are not text: the line is still a line, and still an episode's when it reads as one
            path.write_bytes(b'{"episode_index": 0, "cell": "C\xff2"}\n\xff\xfe\n{"episode_index": 2}\n')
            self.assertEqual(repair_set.log_episodes(path), (0, None, 2))

    def test_statistics_are_written_as_lerobot_writes_them(self):
        stats = {"action": {"min": [0.5, -1.0], "count": [40]}, "observation.images.front": {"max": [[[1.0]], [[0.5]]]}}
        text = repair_set.stats_text(stats)
        self.assertEqual(json.loads(text), stats)
        self.assertEqual(text, json.dumps(stats, indent=4, ensure_ascii=False))

    def test_statistics_over_fewer_episodes_lie_inside_those_over_more(self):
        more = {"action": {"min": [-3.0, 0.0], "max": [5.0, 9.0], "count": [600], "mean": [1.0, 2.0]},
                "observation.images.front": {"min": [[[0.0]], [[0.1]]], "max": [[[1.0]], [[0.9]]], "count": [300]}}
        fewer = {"action": {"min": [-3.0, 0.5], "max": [4.0, 9.0], "count": [540], "mean": [9.0, 9.0]},
                 "observation.images.front": {"min": [[[0.0]], [[0.2]]], "max": [[[1.0]], [[0.9]]], "count": [280]}}
        self.assertIsNone(repair_set.outside(fewer, more))
        self.assertIsNone(repair_set.outside(more, more))
        self.assertIsNone(repair_set.outside(fewer, {}))          # no file to hold them against
        self.assertIn("min of action", repair_set.outside({"action": {**fewer["action"], "min": [-3.5, 0.5]}}, more))
        self.assertIn("max of action", repair_set.outside({"action": {**fewer["action"], "max": [4.0, 9.5]}}, more))
        self.assertIn("count of action", repair_set.outside({"action": {**fewer["action"], "count": [601]}}, more))
        self.assertIn("max of observation.images.front",
                      repair_set.outside({"observation.images.front": {"max": [[[1.0]], [[0.95]]]}}, more))
        self.assertIn("do not go together", repair_set.outside(more, fewer))
        self.assertIsNone(repair_set.outside({"action": "not statistics"}, more))


# ====================================================================================================================
# Is something still using the folder?
# ====================================================================================================================


PS_TEXT = """\
    1 /sbin/launchd
  412 /bin/zsh -il
  901 /home/someone/miniconda3/envs/lerobot/bin/python scripts/rig/record_layout.py --sitting 1 --resume
  902 /usr/bin/vim src/tolerance_radius/rig/record_layout.py
  903 /home/someone/venv/bin/python -m pytest tests/test_rig_record_layout.py
  904 /home/someone/miniconda3/envs/lerobot/bin/python scripts/rig/repair_set.py clustered --name tolrad-clustered-v1
  905 python3 -u /somewhere/scripts/rig/run_trials.py --blocks T1a --log data/trials/main.jsonl
  906 /home/someone/miniconda3/envs/lerobot/bin/python -m tolerance_radius.rig.record_layout --sitting 2
  907 /home/someone/miniconda3/envs/lerobot/bin/python /home/someone/miniconda3/envs/lerobot/bin/lerobot-record --x
  908 /home/someone/miniconda3/envs/lerobot/bin/python -c from multiprocessing.spawn import spawn_main; spawn_main()
  909 /home/someone/miniconda3/envs/lerobot/bin/python scripts/rig/audit_dataset.py clustered
  910 tail -f data/logs/record_layout.py.log
"""


class InUseTextTests(unittest.TestCase):
    def test_a_session_is_an_interpreter_started_on_a_session_script(self):
        found = repair_set.session_processes(PS_TEXT, own=())
        self.assertEqual([line.split(":")[0] for line in found],
                         ["process 901", "process 905", "process 906", "process 907"])
        self.assertIn("record_layout.py --sitting 1 --resume", found[0])

    def test_an_editor_a_test_run_a_reader_and_this_tool_itself_are_not_sessions(self):
        found = " ".join(repair_set.session_processes(PS_TEXT, own=()))
        for pid in (902, 903, 904, 908, 909, 910):
            self.assertNotIn(f"process {pid}", found)

    def test_its_own_processes_are_left_out_whatever_they_are_called(self):
        self.assertEqual([line.split(":")[0] for line in repair_set.session_processes(PS_TEXT, own={901, 906})],
                         ["process 905", "process 907"])

    def test_a_process_and_those_it_descends_from(self):
        family = "    1     0\n  412     1\n  901   412\n  950   901\n  951   950\n  700     1\n"
        self.assertEqual(repair_set.own_processes(family, 951), {951, 950, 901, 412})
        self.assertEqual(repair_set.own_processes(family, 4242), {4242})
        self.assertEqual(repair_set.own_processes("garbage\n  5  5\n", 5), {5})

    def test_only_a_process_that_holds_a_file_open_for_writing_counts(self):
        listing = ("p901\ncPython\nf3\naw\nn/sets/tolrad/data/chunk-000/file-002.parquet\n"
                   "f4\nar\nn/sets/tolrad/meta/info.json\n"
                   "p77\ncmdworker\nf9\nar\nn/sets/tolrad/videos/a.mp4\n"
                   "p88\ncother\nf5\nau\nn/sets/tolrad/meta/stats.json\n")
        self.assertEqual(repair_set.writers(listing, own=()),
                         ["process 901 (Python): /sets/tolrad/data/chunk-000/file-002.parquet",
                          "process 88 (other): /sets/tolrad/meta/stats.json"])
        self.assertEqual(repair_set.writers(listing, own={901, 88}), [])
        self.assertEqual(repair_set.writers("", own=()), [])


class InUseTests(unittest.TestCase):
    """The same question asked of the real system, with child processes that only sleep."""

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.folder = Path(self._folder.name).resolve() / "tolrad-test-v1"
        (self.folder / "meta").mkdir(parents=True)
        (self.folder / "meta" / "info.json").write_text("{}", encoding="utf-8")
        long_ago = time.time() - 3600
        for path in (self.folder / "meta" / "info.json", self.folder / "meta", self.folder):
            os.utime(path, (long_ago, long_ago))

    def child(self, *arguments: str) -> subprocess.Popen:
        process = subprocess.Popen([sys.executable, *arguments], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        self.addCleanup(process.wait)
        self.addCleanup(process.kill)
        self.addCleanup(process.stdout.close)
        self.assertEqual(process.stdout.readline().strip(), b"ready")
        return process

    def sessions_now(self) -> list[str]:
        """The sessions the system's list shows at this moment (a test must not fail because one is running)."""
        if shutil.which("ps") is None:
            self.skipTest("no ps on this machine")
        listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True).stdout
        return repair_set.session_processes(listing, own={os.getpid()})

    def no_session_of_someone_else(self) -> None:
        running = self.sessions_now()
        if running:
            self.skipTest("a session is running on this machine, so the folder cannot be shown free: "
                          + running[0][:60])

    def test_a_folder_nobody_touches_is_free(self):
        self.no_session_of_someone_else()
        self.assertIsNone(repair_set.in_use_reason(self.folder))

    def test_a_running_recorder_is_a_reason(self):
        script = Path(self._folder.name) / "record_layout.py"
        script.write_text("import sys, time\nprint('ready'); sys.stdout.flush()\ntime.sleep(60)\n", encoding="utf-8")
        process = self.child(str(script), "--sitting", "1")
        self.assertIn(f"process {process.pid}:", " ".join(self.sessions_now()))
        reason = repair_set.in_use_reason(self.folder)
        self.assertIn("a recording or trial session is running (process ", reason)
        self.assertIn("End it in its own window", reason)
        process.kill()
        process.wait()
        self.assertNotIn(f"process {process.pid}:", " ".join(self.sessions_now()))

    def test_a_process_writing_into_the_folder_is_a_reason(self):
        if shutil.which("lsof") is None:
            self.skipTest("no lsof on this machine")
        self.no_session_of_someone_else()
        held = self.folder / "data.parquet"
        code = ("import sys, time\nhandle = open(sys.argv[1], 'wb')\nhandle.write(b'PAR1'); handle.flush()\n"
                "print('ready'); sys.stdout.flush()\ntime.sleep(60)\n")
        process = self.child("-c", code, str(held))
        reason = repair_set.in_use_reason(self.folder, quiet_s=0)
        self.assertIn("a program is writing into the set's folder", reason)
        self.assertIn(f"process {process.pid} ", reason)
        self.assertIn("data.parquet", reason)
        # a reader is not a writer: this process holds a file of the folder open for reading, and is left out anyway
        process.kill()
        process.wait()
        with open(self.folder / "meta" / "info.json", encoding="utf-8"):
            self.assertIsNone(repair_set.in_use_reason(self.folder, quiet_s=0))

    def test_a_folder_written_a_moment_ago_is_watched_and_is_free_when_it_stays_as_it_is(self):
        self.no_session_of_someone_else()
        (self.folder / "meta" / "stats.json").write_text("{}", encoding="utf-8")
        waited = []
        self.assertIsNone(repair_set.in_use_reason(self.folder, sleep=waited.append))
        self.assertEqual(len(waited), 1)
        self.assertTrue(0 < waited[0] <= repair_set.QUIET_S, waited)
        # a folder that has been quiet long enough is not waited for
        self.assertIsNone(repair_set.in_use_reason(self.folder, now_s=time.time() + repair_set.QUIET_S + 1,
                                                   sleep=waited.append))
        self.assertIsNone(repair_set.in_use_reason(self.folder, quiet_s=0, sleep=waited.append))
        self.assertEqual(len(waited), 1)

    def test_a_folder_that_changes_while_it_is_watched_is_a_reason(self):
        self.no_session_of_someone_else()
        (self.folder / "meta" / "stats.json").write_text("{}", encoding="utf-8")

        def a_take_goes_on(seconds: float) -> None:
            picture = self.folder / "images" / "frame-000001.png"
            picture.parent.mkdir(exist_ok=True)
            picture.write_bytes(b"a picture")
            later = time.time() + 2
            os.utime(picture, (later, later))

        reason = repair_set.in_use_reason(self.folder, sleep=a_take_goes_on)
        self.assertIn("something is writing into the set's folder (it changed while this watched it for ", reason)

    def test_a_list_of_processes_that_cannot_be_read_is_a_reason(self):
        with mock.patch.object(repair_set.subprocess, "run", side_effect=OSError("no such program")):
            self.assertIn("could not be read", repair_set.in_use_reason(self.folder))


# ====================================================================================================================
# The file operations, on made-up folders (standard library only)
# ====================================================================================================================


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def every_file(folder: Path) -> dict[str, str]:
    """Every file under a folder with the hash of what it holds."""
    return {path.relative_to(folder).as_posix(): digest(path) for path in sorted(folder.rglob("*")) if path.is_file()}


def made_up_stats(frames: int) -> dict:
    return {"index": {"min": [0], "max": [frames - 1], "count": [frames]},
            "action": {"min": [-1.0], "max": [2.0], "count": [frames]}}


class MadeUpSet:
    """A folder that holds what a `Found` says, in files a few bytes long, and a reader that turns it back into
    the same counts: a table file is a small JSON text under a parquet name, and one "with no end" is not JSON."""

    def __init__(self, base: Path, found: Found):
        self.dataset = base / "home" / "owner" / "tolrad-clustered-v1"
        self.log = base / "recording" / "clustered_placements.jsonl"
        self.cameras, self.rows = found.cameras, {row.episode_index: row for row in found.rows}
        (self.dataset / "meta").mkdir(parents=True)
        self.log.parent.mkdir(parents=True)
        info = {"codebase_version": found.info["codebase_version"], "fps": found.info["fps"],
                "features": {camera: {"dtype": "video"} for camera in found.cameras},
                **{key: found.info[key] for key in ("total_episodes", "total_frames", "total_tasks")},
                "chunks_size": 1000, "robot_type": "so_follower", "splits": found.info["splits"]}
        (self.dataset / "meta" / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False),
                                                         encoding="utf-8")
        if found.has_stats:
            (self.dataset / "meta" / "stats.json").write_text(json.dumps(made_up_stats(found.stats_frames), indent=4),
                                                              encoding="utf-8")
        if found.has_tasks:
            (self.dataset / "meta" / "tasks.parquet").write_bytes(b"PAR1 one task PAR1")
        for name, held in found.table_files.items():
            episodes = [row.episode_index for row in found.rows if row.table_file == name]
            self.write(name, None if held is None else {"episodes": episodes})
        for name, held in found.data_files.items():
            self.write(name, None if held is None else {"rows": {str(episode): rows for episode, rows in held.items()}})
        for name, frames in found.video_files.items():
            self.write(name, None if frames is None else {"frames": frames})
        for name in found.leftovers:
            (self.dataset / name / "observation.images.front" / "episode-000029").mkdir(parents=True)
            (self.dataset / name / "observation.images.front" / "episode-000029" / "frame-000000.png").write_bytes(
                b"a picture in " + name.encode())
        for name in found.strangers:
            (self.dataset / name).write_text("something else", encoding="utf-8")
        self.log.write_text("".join(json.dumps({"episode_index": episode, "cell": "B1"}, sort_keys=True) + "\n"
                                    for episode in found.log_lines), encoding="utf-8")

    def write(self, name: str, held: dict | None) -> None:
        path = self.dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"PAR1 {name} and no end" if held is None else json.dumps({"file": name, **held}),
                        encoding="utf-8")

    def read(self, dataset_dir: Path, placement_log: Path) -> Found:
        """What `repair_set.look` does, for these made-up files."""
        def held(folder: str, suffix: str, key: str):
            files = {}
            for path in sorted((dataset_dir / folder).rglob(f"*{suffix}")):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))[key]
                except (ValueError, KeyError):
                    value = None
                files[path.relative_to(dataset_dir).as_posix()] = value
            return files

        info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        tables = held("meta/episodes", ".parquet", "episodes")
        rows = tuple(self.rows[episode] for name in sorted(tables) for episode in (tables[name] or []))
        data = {name: None if value is None else {int(episode): count for episode, count in value.items()}
                for name, value in held("data", ".parquet", "rows").items()}
        try:
            stats = json.loads((dataset_dir / "meta" / "stats.json").read_text(encoding="utf-8"))
            stats_frames = stats["index"]["count"][0]
        except (OSError, ValueError, KeyError):
            stats_frames = None
        top = sorted(path.name for path in dataset_dir.iterdir() if path.name not in ("data", "meta", "videos"))
        return Found(info={key: info.get(key) for key in ("codebase_version", "total_episodes", "total_frames",
                                                          "total_tasks", "splits", "fps")},
                     cameras=self.cameras, rows=rows,
                     table_files={name: None if value is None else len(value) for name, value in tables.items()},
                     data_files=data, video_files=held("videos", ".mp4", "frames"), stats_frames=stats_frames,
                     has_stats=(dataset_dir / "meta" / "stats.json").is_file(),
                     has_tasks=(dataset_dir / "meta" / "tasks.parquet").is_file(),
                     leftovers=tuple(name for name in top if name == "images" or name.startswith("tmp")),
                     strangers=tuple(name for name in top if name != "images" and not name.startswith("tmp")),
                     log_lines=repair_set.log_episodes(placement_log))

    def counts(self, dataset_dir: Path) -> dict:
        """What `record_layout.dataset_counts` says, for these made-up files."""
        found = self.read(dataset_dir, self.log)
        unreadable = [name for name, held in {**found.table_files, **found.data_files}.items() if held is None]
        return {"said": found.info["total_episodes"], "rows": len(found.rows), "unreadable": unreadable}

    def stats(self, dataset_dir: Path, tables: list[str], order: list[str]) -> dict:
        """In place of LeRobot's aggregation: statistics of as many frames as the readable tables' episodes have."""
        self.asked_for = (list(tables), list(order))
        frames = sum(self.rows[episode].frames for name in tables
                     for episode in json.loads((dataset_dir / name).read_text(encoding="utf-8"))["episodes"])
        return made_up_stats(frames)

    @staticmethod
    def rewrite(source: Path, target: Path, episodes) -> int:
        """In place of `repair_set.keep_episodes`, for these made-up files."""
        held = json.loads(source.read_text(encoding="utf-8"))
        held["rows"] = {episode: rows for episode, rows in held["rows"].items() if int(episode) in episodes}
        target.write_text(json.dumps(held), encoding="utf-8")
        return sum(held["rows"].values())

    def repair(self, *, apply: bool, stamp: str = "20261006_230000", in_use=lambda folder: None) -> repair_set.Report:
        with mock.patch.object(record_layout, "dataset_counts", self.counts):
            return repair_set.repair(self.dataset, self.log, apply=apply, stamp=stamp, in_use=in_use, stats=self.stats,
                                     read=self.read, rewrite=self.rewrite)

    def problem(self) -> str | None:
        """The recorder's own check before it appends to a set."""
        return record_layout.resume_problem(self.counts(self.dataset), record_layout.placement_lines(self.log),
                                            self.log.name)


class FolderCase(unittest.TestCase):
    def made_up(self, found: Found) -> MadeUpSet:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return MadeUpSet(Path(folder.name), found)

    def side(self, made: MadeUpSet, stamp: str = "20261006_230000") -> Path:
        return made.dataset.with_name(f"tolrad-clustered-v1-unclosed-{stamp}")


class CarryOutTests(FolderCase):
    """What a repair does to the folder: the dead session's files moved, three files rewritten, nothing deleted."""

    def test_the_measured_case_is_put_back_and_everything_taken_out_is_beside_the_set(self):
        sound = self.made_up(closed())
        made = self.made_up(died(closed(), saves=2))
        before = every_file(made.dataset)
        old_log = made.log.read_bytes()
        self.assertIn("cannot be read", made.problem())

        report = made.repair(apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        self.assertEqual(report.side_dir, self.side(made))
        self.assertIsNone(made.problem())
        # the set's folder is, file for file, the folder of the set that was never damaged
        self.assertEqual(every_file(made.dataset), every_file(sound.dataset))
        self.assertEqual(made.log.read_bytes(), sound.log.read_bytes())
        # ... and beside it: what was moved, at its path; what was rewritten, as it was; the cut lines
        side = self.side(made)
        kept = every_file(side)
        for name in (data_file(2), video_file(CAMERAS[0], 2), video_file(CAMERAS[1], 2),
                     "images/observation.images.front/episode-000029/frame-000000.png"):
            self.assertEqual(kept[f"moved/{name}"], before[name])
        for name in ("meta/info.json", "meta/stats.json"):
            self.assertEqual(kept[f"before/{name}"], before[name])
        self.assertEqual((side / "before" / "clustered_placements.jsonl").read_bytes(), old_log)
        self.assertEqual(sound.log.read_bytes() + (side / "cut_placements.jsonl").read_bytes(), old_log)
        self.assertEqual([json.loads(line)["episode_index"]
                          for line in (side / "cut_placements.jsonl").read_text().splitlines()], [27, 28])
        self.assertEqual(sorted(kept), sorted([
            "WHAT-WAS-DONE.txt", "cut_placements.jsonl", "before/clustered_placements.jsonl", "before/meta/info.json",
            "before/meta/stats.json", f"moved/{data_file(2)}", f"moved/{video_file(CAMERAS[0], 2)}",
            f"moved/{video_file(CAMERAS[1], 2)}",
            "moved/images/observation.images.front/episode-000029/frame-000000.png"]))
        # nothing deleted: every file that was there is still somewhere, with what it held
        after = set(every_file(made.dataset.parent).values()) | {digest(made.log)}
        self.assertLessEqual(set(before.values()), after)
        self.assertIn(hashlib.sha256(old_log).hexdigest(), after)
        # the statistics were asked of the readable tables, in the order of the features in the old file
        self.assertEqual(made.asked_for, ([table_file(0), table_file(1)], ["index", "action"]))

    def test_the_record_beside_the_set_says_what_it_is_how_to_undo_it_and_each_step(self):
        made = self.made_up(died(closed(), saves=2))
        report = made.repair(apply=True)
        record = (self.side(made) / "WHAT-WAS-DONE.txt").read_text(encoding="utf-8")
        self.assertIn("nothing was deleted", record)
        self.assertIn("To undo by hand", record)
        self.assertIn("everything under before/meta/ and before/data/ over the set's files of the same path", record)
        self.assertIn("before/clustered_placements.jsonl over the placement log", record)
        steps = [line for line in record.splitlines()
                 if line.startswith(("copied", "moved", "wrote", "cut", "removed"))]
        self.assertEqual([step.split()[0] for step in steps],
                         ["copied", "copied", "copied", "moved", "moved", "moved", "moved", "wrote", "wrote", "cut"])
        self.assertIn("wrote meta/info.json again: 27 episodes, 540 frames", steps[-2])
        self.assertTrue(record.rstrip().splitlines()[-1].startswith("finished "))
        for step in steps:
            self.assertIn(step, "\n".join(report.lines))
        self.assertNotIn(str(made.dataset.parent), record)          # no machine path in the record

    def test_the_repair_can_be_undone_by_hand_as_the_record_says(self):
        whole = closed(sessions=(1, 26, 1))
        cases = {"a session that died": died(closed(), saves=2),
                 "a save cut short": replace(whole, data_files={**whole.data_files, data_file(2): {27: 20, 28: 20}},
                                             leftovers=("tmppdcakh28",))}
        for name, found in cases.items():
            with self.subTest(case=name):
                made = self.made_up(found)
                before, old_log = every_file(made.dataset), made.log.read_bytes()
                self.assertEqual(made.repair(apply=True).code, 0)
                side = self.side(made)
                # move everything under moved/ back to the same path in the set's folder
                for path in sorted((side / "moved").rglob("*")):
                    if path.is_file():
                        target = made.dataset / path.relative_to(side / "moved")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(path, target)
                # copy everything under before/meta/ and before/data/ over the set's files of the same path
                for folder in ("meta", "data"):
                    for path in sorted((side / "before" / folder).rglob("*")):
                        if path.is_file():
                            shutil.copy2(path, made.dataset / path.relative_to(side / "before"))
                # copy before/<the log> over the placement log
                if (side / "before" / made.log.name).is_file():
                    shutil.copy2(side / "before" / made.log.name, made.log)
                self.assertEqual(every_file(made.dataset), before)
                self.assertEqual(made.log.read_bytes(), old_log)

    def test_every_measured_way_of_dying_ends_as_the_set_that_was_never_damaged(self):
        cases = {
            "killed in its first take": died(closed(), saves=0),
            "killed after one save": died(closed(), saves=1),
            "killed between a save and its line": died(closed(), saves=2, lines=1),
            "killed after eleven saves": died(closed(), saves=11, table_written=True),
            "killed while a save encodes": replace(closed(), data_files={**closed().data_files, data_file(2): None},
                                                   leftovers=("images", "tmp7j5hlu11", "tmpx61j0798")),
            "a first save cut short, the set closed": replace(
                closed(), data_files={**closed().data_files, data_file(2): {27: 20}}, leftovers=("tmpaf4l6r29",)),
        }
        sound = self.made_up(closed())
        for name, found in cases.items():
            with self.subTest(case=name):
                made = self.made_up(found)
                before = set(every_file(made.dataset).values()) | {digest(made.log)}
                report = made.repair(apply=True)
                self.assertEqual(report.code, 0, "\n".join(report.lines))
                self.assertEqual(every_file(made.dataset), every_file(sound.dataset))
                self.assertEqual(made.log.read_bytes(), sound.log.read_bytes())
                self.assertIsNone(made.problem())
                self.assertLessEqual(before, set(every_file(made.dataset.parent).values()) | {digest(made.log)})
                self.assertIn("THE SET ADDS UP AGAIN: 27 episodes, 540 frames, a placement log of 27 lines. The "
                              "recorder's --resume goes on at episode 27", report.lines[-1])

    def test_rows_of_a_take_that_was_never_saved_are_taken_out_of_the_file_they_share_with_a_sound_episode(self):
        sound = self.made_up(closed(sessions=(1, 26, 1)))
        before_found = closed(sessions=(1, 26, 1))
        made = self.made_up(replace(before_found, leftovers=("tmppdcakh28",),
                                    data_files={**before_found.data_files, data_file(2): {27: 20, 28: 20}}))
        self.assertIsNone(made.problem())                          # the recorder's own check finds nothing
        before = every_file(made.dataset)
        report = made.repair(apply=False)
        self.assertEqual(report.code, 2)
        self.assertEqual(every_file(made.dataset), before)
        text = "\n".join(report.lines)
        self.assertIn(f"note: {data_file(2)} holds 20 rows of episode 28 beside its sound episode 27", text)
        self.assertIn(f"copy {data_file(2)} as it is into tolrad-clustered-v1-unclosed-<time>/before/", text)
        self.assertIn(f"write {data_file(2)} again with the rows of episode 27 only (20 rows of a take that was never "
                      "saved are left out)", text)

        report = made.repair(apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        self.assertEqual(every_file(made.dataset), every_file(sound.dataset))
        kept = every_file(self.side(made))
        self.assertEqual(kept[f"before/{data_file(2)}"], before[data_file(2)])
        self.assertIn("moved/tmppdcakh28/observation.images.front/episode-000029/frame-000000.png", kept)
        self.assertIn(f"wrote {data_file(2)} again with the 20 rows of episode 27 only", "\n".join(report.lines))
        self.assertTrue(made.repair(apply=False).code == 0)

    def test_a_file_written_again_that_does_not_check_out_stops_the_repair_with_the_set_s_file_as_it_was(self):
        before_found = closed(sessions=(1, 26, 1))
        made = self.made_up(replace(before_found,
                                    data_files={**before_found.data_files, data_file(2): {27: 20, 28: 20}}))
        before = every_file(made.dataset)

        def failing(source, target, episodes):
            raise ValueError("file-002.parquet written again does not hold exactly the rows")

        with mock.patch.object(made, "rewrite", failing):
            report = made.repair(apply=True)
        self.assertEqual(report.code, 3)
        self.assertIn("STOPPED PARTWAY -- ValueError: file-002.parquet written again does not hold",
                      "\n".join(report.lines))
        self.assertEqual(every_file(made.dataset), before)

    def test_a_log_is_cut_byte_for_byte_whatever_its_lines_hold(self):
        # line ends of another system, and a byte that is not text, in a line that stays and in one that is cut
        made = self.made_up(died(closed(sessions=(1, 2)), saves=2))
        lines = made.log.read_bytes().splitlines()
        lines[1] = lines[1].replace(b'"B1"', b'"B\xff1"') + b"\r"
        lines[4] = lines[4].replace(b'"B1"', b'"\xfe"')
        old = b"\n".join(lines) + b"\n"
        made.log.write_bytes(old)
        report = made.repair(apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        side = self.side(made)
        self.assertEqual(made.log.read_bytes(), b"\n".join(lines[:3]) + b"\n")
        self.assertEqual((side / "cut_placements.jsonl").read_bytes(), b"\n".join(lines[3:]) + b"\n")
        self.assertEqual(made.log.read_bytes() + (side / "cut_placements.jsonl").read_bytes(), old)
        self.assertEqual((side / "before" / made.log.name).read_bytes(), old)

    def test_a_file_that_needs_no_change_is_not_written_again(self):
        made = self.made_up(died(closed(), saves=0))            # only pictures were left
        stamps = {name: (made.dataset / name).stat().st_mtime_ns for name in ("meta/info.json", "meta/stats.json")}
        log_stamp = made.log.stat().st_mtime_ns
        report = made.repair(apply=True)
        self.assertEqual(report.code, 0)
        self.assertEqual({name: (made.dataset / name).stat().st_mtime_ns for name in stamps}, stamps)
        self.assertEqual(made.log.stat().st_mtime_ns, log_stamp)
        self.assertEqual(sorted(every_file(self.side(made))),
                         ["WHAT-WAS-DONE.txt", "moved/images/observation.images.front/episode-000029/frame-000000.png"])

    def test_a_new_set_whose_first_session_died_is_left_as_the_empty_set_it_was_created_as(self):
        made = self.made_up(died(closed(sessions=()), saves=2))
        report = made.repair(apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        self.assertEqual(sorted(path.relative_to(made.dataset).as_posix() for path in made.dataset.rglob("*")),
                         ["meta", "meta/info.json"])
        info = json.loads((made.dataset / "meta" / "info.json").read_text())
        self.assertEqual((info["total_episodes"], info["total_frames"], info["total_tasks"], info["splits"]),
                         (0, 0, 0, {}))
        self.assertEqual(made.log.read_text(), "")
        self.assertIsNone(made.problem())
        moved = [name for name in every_file(self.side(made)) if name.startswith("moved/")]
        self.assertIn("moved/meta/stats.json", moved)
        self.assertIn("moved/meta/tasks.parquet", moved)
        self.assertIn("removed the folder data/", "\n".join(report.lines))

    def test_the_set_does_not_pass_for_sound_until_the_last_step_and_a_cut_short_repair_is_finished_by_another(self):
        # Stop the repair at each of its renames in turn (three moved files, the pictures, then three files put in
        # place): the recorder's own check must still refuse the set, and a second run must finish the job.
        sound = self.made_up(closed())
        for stop_at in range(1, 8):
            with self.subTest(stopped_at_rename=stop_at):
                made = self.made_up(died(closed(), saves=2))
                before = set(every_file(made.dataset).values()) | {digest(made.log)}
                calls = {"n": 0}
                real_rename, real_replace = os.rename, os.replace

                def failing(real, calls=calls, stop_at=stop_at):
                    def step(source, target):
                        calls["n"] += 1
                        if calls["n"] == stop_at:
                            raise OSError("the disk went away")
                        return real(source, target)
                    return step

                with mock.patch.object(repair_set.os, "rename", failing(real_rename)), \
                        mock.patch.object(repair_set.os, "replace", failing(real_replace)):
                    report = made.repair(apply=True, stamp="20261006_230000")
                self.assertEqual(report.code, 3)
                self.assertIn("STOPPED PARTWAY -- OSError: the disk went away", "\n".join(report.lines))
                self.assertIsNotNone(made.problem(), "a half-done repair passed the recorder's check")
                everywhere = set(every_file(made.dataset.parent).values()) | {digest(made.log)}
                self.assertLessEqual(before, everywhere | {digest(path) for path in made.log.parent.iterdir()})

                again = made.repair(apply=True, stamp="20261006_230100")
                self.assertEqual(again.code, 0, "\n".join(again.lines))
                self.assertEqual(every_file(made.dataset), every_file(sound.dataset))
                self.assertEqual(made.log.read_bytes(), sound.log.read_bytes())
                self.assertIsNone(made.problem())


class RepairFlowTests(FolderCase):
    """What a run says and does not do: the dry run, the refusals, the exit codes."""

    def test_without_apply_nothing_is_changed_and_what_would_be_done_is_said(self):
        made = self.made_up(died(closed(), saves=2))
        before = every_file(made.dataset.parent.parent.parent)
        report = made.repair(apply=False)
        self.assertEqual(report.code, 2)
        self.assertIsNone(report.side_dir)
        self.assertEqual(every_file(made.dataset.parent.parent.parent), before)
        self.assertEqual([path.name for path in made.dataset.parent.iterdir()], ["tolrad-clustered-v1"])
        text = "\n".join(report.lines)
        self.assertIn("sound: 27 episodes, 540 frames", text)
        self.assertIn("meta/info.json says: 29 episodes, 580 frames -- 2 episodes (40 frames) saved by a session that "
                      "ended without closing the set", text)
        self.assertIn(f"{data_file(2)}", text)
        self.assertIn("cannot be read (it has no end: its writer was never closed)", text)
        self.assertIn("placement log clustered_placements.jsonl: 29 lines, 2 more than the sound episodes", text)
        self.assertIn("WOULD DO (nothing was changed; run it again with --apply):", text)
        self.assertIn("move the 4 entries listed above into tolrad-clustered-v1-unclosed-<time>/moved/", text)
        self.assertIn("(it describes 580 frames now, and will describe 540)", text)
        self.assertIn('total_episodes 29 -> 27; total_frames 580 -> 540; splits {"train": "0:29"} -> {"train": "0:27"}',
                      text)
        self.assertIn("cut clustered_placements.jsonl from 29 to 27 lines; the 2 cut lines (episodes 27, 28) are kept",
                      text)
        self.assertIn("the recorder's --resume goes on at episode 27 (episodes 27, 28, which that session saved and "
                      "never closed, are recorded again).", text)
        self.assertEqual(report.lines[-1], "  Nothing was changed.")

    def test_a_set_that_adds_up_is_refused_as_nothing_to_repair(self):
        for found in (closed(), closed(sessions=())):
            made = self.made_up(found)
            before = every_file(made.dataset.parent.parent.parent)
            for apply in (False, True):
                report = made.repair(apply=apply)
                self.assertEqual(report.code, 0)
                self.assertIn("NOTHING TO REPAIR -- the set adds up", "\n".join(report.lines))
                self.assertEqual(report.lines[-1], "  Nothing was changed.")
                self.assertEqual(every_file(made.dataset.parent.parent.parent), before)
                self.assertEqual([path.name for path in made.dataset.parent.iterdir()], ["tolrad-clustered-v1"])

    def test_a_set_it_cannot_make_sense_of_is_refused_and_nothing_is_changed(self):
        # a set whose placement log is a line short: a sound episode without its measured position
        made = self.made_up(replace(closed(), log_lines=tuple(range(26))))
        before = every_file(made.dataset.parent.parent.parent)
        for apply in (False, True):
            report = made.repair(apply=apply)
            self.assertEqual(report.code, 1)
            self.assertIn("REFUSING -- the set at", report.lines[0])
            self.assertIn("1 sound episode has no placement line", report.lines[0])
            self.assertEqual(report.lines[1:],
                             ("  Nothing was changed.", "  Do not delete anything, and say what this prints."))
            self.assertEqual(every_file(made.dataset.parent.parent.parent), before)

    def test_a_folder_that_a_session_is_using_is_refused_before_anything_is_read(self):
        made = self.made_up(died(closed(), saves=2))
        before = every_file(made.dataset.parent.parent.parent)
        asked = []

        def in_use(folder):
            asked.append(folder)
            return "a recording or trial session is running (process 901: python record_layout.py)"

        with mock.patch.object(made, "read", side_effect=AssertionError("the set was read while in use")):
            report = made.repair(apply=True, in_use=in_use)
        self.assertEqual(report.code, 1)
        self.assertEqual(asked, [made.dataset])
        self.assertEqual(report.lines, ("  REFUSING -- a recording or trial session is running (process 901: python "
                                        "record_layout.py).", "  Nothing was changed."))
        self.assertEqual(every_file(made.dataset.parent.parent.parent), before)

    def test_a_set_that_is_not_there_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            report = repair_set.repair(Path(directory) / "tolrad-spread-v1",
                                       Path(directory) / "spread_placements.jsonl", apply=True,
                                       stamp="20261006_230000", in_use=lambda folder: None)
        self.assertEqual(report.code, 1)
        self.assertIn("there is no recorded set at", report.lines[0])

    def test_a_folder_of_that_time_that_is_already_there_is_never_written_into(self):
        made = self.made_up(died(closed(), saves=2))
        self.side(made).mkdir()
        (self.side(made) / "earlier.txt").write_text("an earlier repair's", encoding="utf-8")
        before = every_file(made.dataset.parent.parent.parent)
        report = made.repair(apply=True)
        self.assertEqual(report.code, 1)
        self.assertIn("is already there", "\n".join(report.lines))
        self.assertEqual(every_file(made.dataset.parent.parent.parent), before)

    def test_statistics_that_do_not_fit_the_sound_episodes_stop_the_repair_before_it_begins(self):
        made = self.made_up(died(closed(), saves=2))
        before = every_file(made.dataset.parent.parent.parent)
        with mock.patch.object(made, "stats", lambda *_arguments: made_up_stats(541)):
            report = made.repair(apply=True)
        self.assertEqual(report.code, 1)
        self.assertIn("the repair cannot be prepared: the sound episodes' own statistics describe 541 frames, and the "
                      "episodes have 540", "\n".join(report.lines))
        wide = {**made_up_stats(540), "action": {"min": [-9.0], "max": [2.0], "count": [540]}}
        with mock.patch.object(made, "stats", lambda *_arguments: wide):
            report = made.repair(apply=True)
        self.assertEqual(report.code, 1)
        self.assertIn("the min of action over fewer episodes ([-9.0]) does not lie inside the one over more ([-1.0])",
                      "\n".join(report.lines))
        self.assertEqual(every_file(made.dataset.parent.parent.parent), before)

    def test_statistics_left_one_episode_behind_are_written_again_for_all_the_sound_episodes(self):
        # A save cut off between its last two steps: the count written, the statistics not yet. The set closed.
        made = self.made_up(replace(closed(), stats_frames=520))
        report = made.repair(apply=False)
        self.assertEqual(report.code, 2, "\n".join(report.lines))
        self.assertIn("(it describes 520 frames now, and will describe 540)", "\n".join(report.lines))
        self.assertEqual(made.repair(apply=True).code, 0)
        self.assertEqual(json.loads((made.dataset / "meta" / "stats.json").read_text())["index"]["count"], [540])
        self.assertEqual(sorted(every_file(self.side(made))), ["WHAT-WAS-DONE.txt", "before/meta/stats.json"])
        # a file that is not statistics at all is replaced too, and kept
        made = self.made_up(closed())
        (made.dataset / "meta" / "stats.json").write_text("[1, 2", encoding="utf-8")
        self.assertEqual(made.repair(apply=True).code, 0)
        self.assertEqual((self.side(made) / "before" / "meta" / "stats.json").read_text(), "[1, 2")
        self.assertEqual(json.loads((made.dataset / "meta" / "stats.json").read_text())["index"]["count"], [540])

    def test_the_words_for_a_set_with_no_sound_episode(self):
        made = self.made_up(died(closed(sessions=()), saves=2))
        text = "\n".join(made.repair(apply=False).lines)
        self.assertIn("sound: no episode (none was closed properly)", text)
        self.assertIn("total_episodes 2 -> 0; total_frames 40 -> 0; total_tasks 1 -> 0", text)
        self.assertIn("Afterwards: the empty set its first session created", text)
        self.assertIn("the recorder's --resume begins it again at episode 0 (episodes 0, 1, which that session saved "
                      "and never closed, are recorded again).", text)

    def test_a_set_whose_tables_are_not_as_expected_is_refused_and_not_a_crash(self):
        made = self.made_up(died(closed(), saves=2))
        before = every_file(made.dataset.parent.parent.parent)
        with mock.patch.object(made, "read", side_effect=KeyError("videos/observation.images.front/file_index")):
            report = made.repair(apply=True)
        self.assertEqual(report.code, 1)
        self.assertIn("is not in the form this tool knows (KeyError: 'videos/observation.images.front/file_index')",
                      report.lines[0])
        self.assertEqual(every_file(made.dataset.parent.parent.parent), before)

    def test_a_repair_stopped_by_anything_at_all_says_so_and_what_it_did_is_on_record(self):
        made = self.made_up(died(closed(), saves=2))

        def failing(source, target):
            raise RuntimeError("something nobody thought of")

        with mock.patch.object(repair_set.os, "replace", failing):
            report = made.repair(apply=True)
        self.assertEqual(report.code, 3)
        self.assertIn("STOPPED PARTWAY -- RuntimeError: something nobody thought of", "\n".join(report.lines))
        record = (self.side(made) / "WHAT-WAS-DONE.txt").read_text()
        self.assertIn("moved images/ to moved/images", record)
        self.assertNotIn("finished", record)
        self.assertIsNotNone(made.problem())

    def test_one_thing_is_said_as_one(self):
        text = "\n".join(self.made_up(died(closed(), saves=0)).repair(apply=False).lines)
        self.assertIn("move the one entry listed above", text)
        self.assertIn("a folder of 1 file", text)
        made = self.made_up(died(closed(), saves=1))
        text = "\n".join(made.repair(apply=False).lines)
        self.assertIn("the one cut line (episode 27) is kept in", text)
        self.assertIn("at episode 27 (episode 27, which that session saved and never closed, is recorded again).", text)
        self.assertTrue(made.repair(apply=True).lines[-1].endswith(
            "at episode 27 (episode 27, which that session saved and never closed, is recorded again)."))


class ScriptTests(unittest.TestCase):
    """The script beside the others: it parses, and its first lines are the launcher the hygiene test allows."""

    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rig" / "repair_set.py"

    def test_it_starts_as_the_other_rig_scripts_do(self):
        other = (self.SCRIPT.parent / "set_aside.py").read_text(encoding="utf-8").splitlines()[:7]
        self.assertEqual(self.SCRIPT.read_text(encoding="utf-8").splitlines()[:7], other)
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))

    def test_it_compiles_and_says_its_two_commands_and_its_exit_codes(self):
        source = self.SCRIPT.read_text(encoding="utf-8")
        compile(source, str(self.SCRIPT), "exec")
        for words in ("scripts/rig/repair_set.py clustered --apply", "changes nothing", "Deletes nothing",
                      "Exit 0", "--placement-log"):
            self.assertIn(words, source)


# ====================================================================================================================
# The whole repair on a set written by LeRobot's own writer (the rig's interpreter)
# ====================================================================================================================

WIDTH, HEIGHT = 96, 64
TASK = "pick up the orange cube and place it in the basket"
REPO_ID = "someone/tolrad-test-v1"
#: The program of a session that dies, run by a real second process: LeRobot's writer as the recorder calls it, two
#: episodes saved with their placement lines, a third take under way, and `os._exit(9)`.
DYING_SESSION = textwrap.dedent("""
    import json, os, sys
    sys.path[:0] = [sys.argv[5], sys.argv[4]]
    import test_rig_repair_set as tests
    from tolerance_radius.rig import lerobot101, record_layout
    lerobot101.use_pinned_lerobot()
    from lerobot.datasets import LeRobotDataset, VideoEncodingManager
    dataset = LeRobotDataset.resume(sys.argv[1], root=sys.argv[2], image_writer_processes=0, image_writer_threads=4)
    already = int(dataset.meta.total_episodes)
    with VideoEncodingManager(dataset):
        for n in range(2):
            tests.take(dataset, frames=8, seed=100 + n)
            dataset.save_episode(parallel_encoding=False)
            record_layout.append_placement(tests.Path(sys.argv[3]), {"episode_index": already + n, "cell": "B1"})
        tests.take(dataset, frames=5, seed=109)
        dataset.writer._wait_image_writer()
        os._exit(9)
""")


def take(dataset, frames: int, seed: int) -> None:
    """A short made-up take: a square that moves over a grey picture, and six joints that drift."""
    import numpy as np

    rng = np.random.default_rng(seed)
    for index in range(frames):
        picture = np.full((HEIGHT, WIDTH, 3), 40 + 3 * index + seed % 7, dtype=np.uint8)
        left = (7 * index + seed) % (WIDTH - 12)
        picture[20:32, left:left + 12] = (255, 128, 0)
        joints = (np.arange(6) * 10 + index + rng.normal(0, 0.1, 6)).astype(np.float32)
        dataset.add_frame({"action": joints, "observation.state": joints.copy(), "observation.images.front": picture,
                           "observation.images.wrist": picture[::-1].copy(), "task": TASK})


@contextlib.contextmanager
def quietly():
    """Run LeRobot's writer without its progress bars and the encoder's remarks on the test run's output."""
    import datasets

    saved = os.dup(2)
    was = datasets.utils.logging.is_progress_bar_enabled()
    datasets.utils.logging.disable_progress_bar()
    sys.stderr.flush()
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        if was:
            datasets.utils.logging.enable_progress_bar()


@unittest.skipUnless(fakes_rig.READY, fakes_rig.WHY_NOT)
class WriterTests(unittest.TestCase):
    """The repair on a tiny set (three episodes of 8 frames from two sessions, two cameras of 96 x 64) that LeRobot's
    own writer wrote, called as the recorder calls it, and on what a session that dies leaves in it."""

    @classmethod
    def setUpClass(cls):
        try:
            fakes_rig.prepare()
        except Exception as error:  # noqa: BLE001 -- the tooling is not the tested version here
            raise unittest.SkipTest(f"the arm tooling or LeRobot is not the tested version here: {error}") from error
        import datasets

        cls._folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._folder.cleanup)
        base = Path(cls._folder.name).resolve()
        # The encoder's own log, its level set before it starts; the reader's cache, kept out of the home folder;
        # and no way to the hub (LeRobot asks it when a set does not load from disk).
        for name, value in (("SVT_LOG", "1"), ("HF_HUB_OFFLINE", "1")):
            patch = mock.patch.dict(os.environ, {name: value})
            patch.start()
            cls.addClassCleanup(patch.stop)
        patch = mock.patch.object(datasets.config, "HF_DATASETS_CACHE", base / "reader-cache")
        patch.start()
        cls.addClassCleanup(patch.stop)
        from lerobot.datasets import dataset_metadata, lerobot_dataset

        def no_hub(*_arguments, **_keywords):
            raise AssertionError("a test reached for the hub")

        for module in (dataset_metadata, lerobot_dataset):
            for name in ("snapshot_download", "get_safe_version"):
                patch = mock.patch.object(module, name, no_hub)
                patch.start()
                cls.addClassCleanup(patch.stop)

        cls.base = base
        cls.sound_dir, cls.sound_log = cls.new_set(base / "sound", sessions=(1, 2))

    # ------------------------------------------------------------------------------------------------ building sets

    @staticmethod
    def features() -> dict:
        joints = [f"joint_{number}.pos" for number in range(6)]
        picture = {"dtype": "video", "shape": (HEIGHT, WIDTH, 3), "names": ["height", "width", "channels"]}
        return {"action": {"dtype": "float32", "shape": (6,), "names": joints},
                "observation.state": {"dtype": "float32", "shape": (6,), "names": joints},
                "observation.images.front": dict(picture), "observation.images.wrist": dict(picture)}

    @classmethod
    def new_set(cls, folder: Path, sessions: tuple[int, ...]) -> tuple[Path, Path]:
        """A set recorded by sessions of so many episodes each, every one closed, and its placement log."""
        from lerobot.datasets import LeRobotDataset

        dataset_dir, log = folder / "home" / REPO_ID, folder / "recording" / "test_placements.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text("", encoding="utf-8")
        with quietly():
            created = LeRobotDataset.create(REPO_ID, FPS, features=cls.features(), root=dataset_dir,
                                            robot_type="so_follower", use_videos=True, image_writer_processes=0,
                                            image_writer_threads=4)
            created.finalize()
        for number, episodes in enumerate(sessions):
            cls.session(dataset_dir, log, saves=episodes, seed=10 * number)
        return dataset_dir, log

    @staticmethod
    def session(dataset_dir: Path, log: Path, saves: int, seed: int, *, close: bool = True, unsaved_frames: int = 0,
                lines: int | None = None, rows_at_a_time: int | None = None, cut_short: bool = False):
        """One recording session as the recorder runs it: resume, takes, a save and a placement line for each, and
        (with `close`) the set closed. Without `close` the writer is returned still open, for the folder to be
        copied as a kill would leave it. `cut_short` ends the last save as Ctrl-C does while it encodes."""
        from lerobot.datasets import LeRobotDataset, VideoEncodingManager

        with quietly():
            dataset = LeRobotDataset.resume(REPO_ID, root=dataset_dir, image_writer_processes=0, image_writer_threads=4)
            if rows_at_a_time is not None:
                dataset.meta._metadata_buffer_size = rows_at_a_time
            already = int(dataset.meta.total_episodes)

            def record() -> None:
                for number in range(saves):
                    take(dataset, frames=8, seed=seed + number)
                    if cut_short and number == saves - 1:
                        # the take's rows are written, then the interrupt arrives while its videos are encoded
                        with mock.patch.object(dataset.writer, "_save_episode_video", side_effect=KeyboardInterrupt):
                            dataset.save_episode(parallel_encoding=False)
                        raise AssertionError("the save was not cut short")
                    dataset.save_episode(parallel_encoding=False)
                    if lines is None or number < lines:
                        record_layout.append_placement(log, {"episode_index": already + number, "cell": "B1"})
                if unsaved_frames:
                    take(dataset, frames=unsaved_frames, seed=seed + 99)
                    dataset.writer._wait_image_writer()

            if not close:
                record()
                return dataset
            try:
                with VideoEncodingManager(dataset):
                    record()
            except KeyboardInterrupt:
                pass
            finally:
                dataset.finalize()
        return None

    def copy_of_sound(self) -> tuple[Path, Path]:
        folder = Path(tempfile.mkdtemp(dir=self.base))
        shutil.copytree(self.sound_dir.parents[2], folder, dirs_exist_ok=True)
        return folder / "home" / REPO_ID, folder / "recording" / self.sound_log.name

    def left_by_a_kill(self, **session) -> tuple[Path, Path]:
        """A copy of the sound set with one more session's files as a kill leaves them: the session runs on one
        copy, and that copy's folder is copied while its writer is still open."""
        scratch_dir, scratch_log = self.copy_of_sound()
        writer = self.session(scratch_dir, scratch_log, close=False, **session)
        folder = Path(tempfile.mkdtemp(dir=self.base))
        shutil.copytree(scratch_dir.parents[2], folder, dirs_exist_ok=True)
        with quietly():
            writer.finalize()                                      # into the scratch copy, which is not looked at again
        dataset_dir, log = folder / "home" / REPO_ID, folder / "recording" / scratch_log.name
        return dataset_dir, log

    def repair(self, dataset_dir: Path, log: Path, *, apply: bool, stamp: str = "20261006_230000") -> repair_set.Report:
        """The repair with the real reader and LeRobot's statistics. Whether a session is running on this machine is
        not asked here (`InUseTests` ask it): a test must not fail because one is."""
        return repair_set.repair(dataset_dir, log, apply=apply, stamp=stamp, in_use=lambda folder: None)

    def loaded(self, dataset_dir: Path):
        from lerobot.datasets import LeRobotDataset

        with quietly():
            return LeRobotDataset(REPO_ID, root=dataset_dir)

    def problem(self, dataset_dir: Path, log: Path) -> str | None:
        return record_layout.resume_problem(record_layout.dataset_counts(dataset_dir),
                                            record_layout.placement_lines(log), log.name)

    def assert_is_the_sound_set(self, dataset_dir: Path, log: Path) -> None:
        """Every file is byte for byte the sound set's, but for the statistics file, whose numbers are the same
        (LeRobot writes its features in an order that changes from process to process)."""
        mine, sound = every_file(dataset_dir), every_file(self.sound_dir)
        self.assertEqual(sorted(mine), sorted(sound))
        self.assertEqual({name: value for name, value in mine.items() if name != "meta/stats.json"},
                         {name: value for name, value in sound.items() if name != "meta/stats.json"})
        self.assertEqual(json.loads((dataset_dir / "meta" / "stats.json").read_text()),
                         json.loads((self.sound_dir / "meta" / "stats.json").read_text()))
        self.assertEqual(log.read_bytes(), self.sound_log.read_bytes())

    # ------------------------------------------------------------------------------------------------------ the tests

    def test_the_sound_set_is_read_into_the_counts_the_rule_is_given(self):
        found = repair_set.look(self.sound_dir, self.sound_log)
        self.assertEqual(found.info, {"codebase_version": "v3.0", "total_episodes": 3, "total_frames": 24,
                                      "total_tasks": 1, "splits": {"train": "0:3"}, "fps": FPS})
        self.assertEqual(found.cameras, CAMERAS)
        self.assertEqual([(row.episode_index, row.frames, row.first_frame, row.end_frame, row.table_file, row.data_file)
                          for row in found.rows],
                         [(0, 8, 0, 8, table_file(0), data_file(0)), (1, 8, 8, 16, table_file(1), data_file(1)),
                          (2, 8, 16, 24, table_file(1), data_file(1))])
        self.assertEqual(found.rows[2].segments[1].file, video_file(CAMERAS[1], 1))
        self.assertAlmostEqual(found.rows[2].segments[1].from_s, 8 / FPS)
        self.assertAlmostEqual(found.rows[2].segments[1].to_s, 16 / FPS)
        self.assertEqual(found.rows[0].tasks, (TASK,))
        self.assertEqual(found.table_files, {table_file(0): 1, table_file(1): 2})
        self.assertEqual(found.data_files, {data_file(0): {0: 8}, data_file(1): {1: 8, 2: 8}})
        self.assertEqual(found.video_files, {video_file(camera, number): frames for camera in CAMERAS
                                             for number, frames in ((0, 8), (1, 16))})
        self.assertEqual((found.stats_frames, found.has_stats, found.has_tasks), (24, True, True))
        self.assertEqual((found.leftovers, found.strangers, found.log_lines), ((), (), (0, 1, 2)))
        self.assertTrue(judge(found).sound_already)

    def test_the_sound_episodes_statistics_are_the_numbers_lerobot_s_file_holds(self):
        # Two sessions wrote the file: the fold reads the aggregate back where the second one began, as LeRobot did.
        written = json.loads((self.sound_dir / "meta" / "stats.json").read_text())
        folded = repair_set.sound_stats(self.sound_dir, [table_file(0), table_file(1)], list(written))
        self.assertEqual(json.loads(repair_set.stats_text(folded)), written)
        self.assertEqual(repair_set.stats_text(folded), (self.sound_dir / "meta" / "stats.json").read_text())
        # ... and the text is LeRobot's own writer's, byte for byte, for the same statistics
        from lerobot.datasets.io_utils import write_stats

        with tempfile.TemporaryDirectory() as directory:
            write_stats(folded, Path(directory))
            self.assertEqual((Path(directory) / "meta" / "stats.json").read_text(), repair_set.stats_text(folded))
        with self.assertRaisesRegex(ValueError, "no sound episode"):
            repair_set.sound_stats(self.sound_dir, [], [])

    def test_a_session_killed_by_the_system_after_two_saves_is_repaired_and_recorded_on(self):
        dataset_dir, log = self.copy_of_sound()
        tests = Path(__file__).resolve().parent
        with fakes_rig.located_tooling():
            died_at = subprocess.run(
                [sys.executable, "-c", DYING_SESSION, REPO_ID, str(dataset_dir), str(log), str(tests),
                 str(tests.parent / "src")],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
                env={**os.environ, "HF_LEROBOT_HOME": str(self.base / "home-of-the-dying"), "HF_HUB_OFFLINE": "1",
                     "HF_DATASETS_CACHE": str(self.base / "reader-cache"), "SVT_LOG": "1"})
        self.assertEqual(died_at.returncode, 9, died_at.stderr[-2000:])

        # what the session left: the picture measured on the real set on 2026-10-06
        found = repair_set.look(dataset_dir, log)
        self.assertEqual((found.info["total_episodes"], found.info["total_frames"], len(found.rows)), (5, 40, 3))
        self.assertEqual(found.data_files[data_file(2)], None)
        self.assertEqual({name: frames for name, frames in found.video_files.items() if "file-002" in name},
                         {video_file(camera, 2): 16 for camera in CAMERAS})
        self.assertEqual((found.leftovers, found.log_lines, found.stats_frames), (("images",), (0, 1, 2, 3, 4), 40))
        self.assertEqual(len(list((dataset_dir / "images").rglob("*.png"))), 10)
        self.assertIn(f"1 of its table files cannot be read ({data_file(2)})", self.problem(dataset_dir, log))

        # the dry run says so and changes nothing
        before = every_file(dataset_dir.parents[2])
        report = self.repair(dataset_dir, log, apply=False)
        self.assertEqual(report.code, 2, "\n".join(report.lines))
        self.assertEqual(every_file(dataset_dir.parents[2]), before)
        text = "\n".join(report.lines)
        self.assertIn("sound: 3 episodes, 24 frames", text)
        self.assertIn("meta/info.json says: 5 episodes, 40 frames -- 2 episodes (16 frames) saved by a session", text)
        self.assertIn("(it describes 40 frames now, and will describe 24)", text)
        self.assertIn("cut test_placements.jsonl from 5 to 3 lines", text)

        # the repair puts the set back, byte for byte, and keeps everything it took out
        report = self.repair(dataset_dir, log, apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        self.assertIn("THE SET ADDS UP AGAIN: 3 episodes, 24 frames, a placement log of 3 lines", report.lines[-1])
        self.assertIsNone(self.problem(dataset_dir, log))
        self.assert_is_the_sound_set(dataset_dir, log)
        self.assertLessEqual(set(before.values()), set(every_file(dataset_dir.parents[2]).values()))
        side = every_file(report.side_dir)
        self.assertIn(f"moved/{data_file(2)}", side)
        self.assertEqual(len([name for name in side if name.startswith("moved/images/")]), 10)
        self.assertEqual(self.repair(dataset_dir, log, apply=True).code, 0)       # and a second run finds nothing
        self.assertIn("NOTHING TO REPAIR", "\n".join(self.repair(dataset_dir, log, apply=False).lines))

        # it loads, and a new session resumes, saves two episodes and closes
        loaded = self.loaded(dataset_dir)
        self.assertEqual((loaded.num_episodes, loaded.num_frames), (3, 24))
        self.session(dataset_dir, log, saves=2, seed=50)
        control_dir, control_log = self.copy_of_sound()            # the same session on a set that was never damaged
        self.session(control_dir, control_log, saves=2, seed=50)
        loaded = self.loaded(dataset_dir)
        self.assertEqual((loaded.num_episodes, loaded.num_frames), (5, 40))
        self.assertIsNone(self.problem(dataset_dir, log))
        mine = json.loads((dataset_dir / "meta" / "stats.json").read_text())
        self.assertEqual(mine, json.loads((control_dir / "meta" / "stats.json").read_text()))
        self.assertEqual(mine["action"]["count"], [40])
        tables = sorted(path.relative_to(dataset_dir).as_posix()
                        for path in (dataset_dir / "meta" / "episodes").rglob("*.parquet"))
        folded = repair_set.sound_stats(dataset_dir, tables, list(mine))
        self.assertEqual(json.loads(repair_set.stats_text(folded)), mine)
        for name in (data_file(2), table_file(2)):
            self.assertEqual(digest(dataset_dir / name), digest(control_dir / name), name)
        # a frame of the last sound episode and one of the first new episode, from both videos
        for index, episode in ((23, 2), (24, 3)):
            with quietly():
                frame = loaded[index]
            self.assertEqual((int(frame["episode_index"]), int(frame["index"])), (episode, index))
            for camera in CAMERAS:
                self.assertEqual(tuple(frame[camera].shape), (3, HEIGHT, WIDTH))

    def test_the_other_ways_of_dying_end_as_the_set_that_was_never_damaged(self):
        cases = {
            "killed in its first take": ({"saves": 0, "seed": 60, "unsaved_frames": 5}, [("images", None)]),
            "killed after one save": ({"saves": 1, "seed": 61}, [(data_file(2), None), ("images", None)]),
            "killed between a save and its line": ({"saves": 2, "seed": 62, "lines": 1}, [(data_file(2), None)]),
            "killed after its episode table began": ({"saves": 3, "seed": 63, "rows_at_a_time": 2},
                                                     [(data_file(2), None), (table_file(2), None)]),
        }
        for name, (session, left) in cases.items():
            with self.subTest(case=name):
                dataset_dir, log = self.left_by_a_kill(**session)
                found = repair_set.look(dataset_dir, log)
                verdict = judge(found)
                self.assertIsNone(verdict.refusal)
                self.assertEqual((verdict.sound_episodes, verdict.unclosed_episodes), (3, session["saves"]))
                for entry, _held in left:
                    self.assertIn(entry, verdict.move_out)
                    if entry.endswith(".parquet"):
                        self.assertIsNone({**found.table_files, **found.data_files}[entry])
                before = set(every_file(dataset_dir.parents[2]).values())
                self.assertEqual(self.repair(dataset_dir, log, apply=False).code, 2)
                report = self.repair(dataset_dir, log, apply=True)
                self.assertEqual(report.code, 0, "\n".join(report.lines))
                self.assert_is_the_sound_set(dataset_dir, log)
                self.assertIsNone(self.problem(dataset_dir, log))
                self.assertLessEqual(before, set(every_file(dataset_dir.parents[2]).values()))
                loaded = self.loaded(dataset_dir)
                self.assertEqual((loaded.num_episodes, loaded.num_frames), (3, 24))

    def test_a_set_killed_in_its_first_take_is_refused_for_its_pictures_and_repaired(self):
        dataset_dir, log = self.left_by_a_kill(saves=0, seed=64, unsaved_frames=5)
        # the counts agree: only pictures were left, and the recorder's check goes by them
        self.assertIn("10 pictures of a take that was never saved are left in its images folder",
                      self.problem(dataset_dir, log))
        report = self.repair(dataset_dir, log, apply=False)
        self.assertEqual(report.code, 2)
        self.assertIn("a folder of 10 files", "\n".join(report.lines))

    def test_a_new_set_whose_first_session_was_killed_is_put_back_to_empty_and_recorded_again(self):
        from lerobot.datasets import LeRobotDataset

        folder = Path(tempfile.mkdtemp(dir=self.base))
        dataset_dir, log = folder / "home" / REPO_ID, folder / "recording" / "test_placements.jsonl"
        log.parent.mkdir(parents=True)
        with quietly():
            dataset = LeRobotDataset.create(REPO_ID, FPS, features=self.features(), root=dataset_dir,
                                            robot_type="so_follower", use_videos=True, image_writer_processes=0,
                                            image_writer_threads=4)
            for number in range(2):
                take(dataset, frames=8, seed=70 + number)
                dataset.save_episode(parallel_encoding=False)
                record_layout.append_placement(log, {"episode_index": number, "cell": "C2"})
        dead = Path(tempfile.mkdtemp(dir=self.base))
        shutil.copytree(folder, dead, dirs_exist_ok=True)
        with quietly():
            dataset.finalize()
        dataset_dir, log = dead / "home" / REPO_ID, dead / "recording" / log.name
        self.assertIn("cannot be read", self.problem(dataset_dir, log))

        report = self.repair(dataset_dir, log, apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        self.assertEqual(sorted(path.relative_to(dataset_dir).as_posix() for path in dataset_dir.rglob("*")),
                         ["meta", "meta/info.json"])
        self.assertIsNone(self.problem(dataset_dir, log))
        self.assertEqual(log.read_text(), "")
        # LeRobot's own resume takes the empty set, and the set it then records loads
        self.session(dataset_dir, log, saves=2, seed=70)
        loaded = self.loaded(dataset_dir)
        self.assertEqual((loaded.num_episodes, loaded.num_frames), (2, 16))
        self.assertEqual(json.loads((dataset_dir / "meta" / "stats.json").read_text())["action"]["count"], [16])
        self.assertIsNone(self.problem(dataset_dir, log))

    def test_a_first_save_cut_short_in_a_session_that_closed_the_set_is_moved_aside(self):
        dataset_dir, log = self.copy_of_sound()
        self.assertIsNone(record_layout.deeper_look(dataset_dir, log))          # a sound set, looked through
        self.session(dataset_dir, log, saves=1, seed=80, cut_short=True, lines=0)
        self.assertIsNone(self.problem(dataset_dir, log))            # the recorder's counts see nothing ...
        self.assertIn("files or folders of a session that did not end properly",      # ... its look through does
                      record_layout.deeper_look(dataset_dir, log))
        found = repair_set.look(dataset_dir, log)
        self.assertEqual(found.data_files[data_file(2)], {3: 8})
        self.assertEqual(self.repair(dataset_dir, log, apply=True).code, 0)
        self.assert_is_the_sound_set(dataset_dir, log)

    def test_a_later_save_cut_short_leaves_rows_that_are_taken_out_of_the_session_s_file(self):
        # one episode saved, then the next save cut short while it encoded; the set closed on the way out
        dataset_dir, log = self.copy_of_sound()
        self.session(dataset_dir, log, saves=2, seed=80, cut_short=True, lines=1)
        control_dir, control_log = self.copy_of_sound()            # the same session with its one save and no other
        self.session(control_dir, control_log, saves=1, seed=80)
        self.assertIsNone(self.problem(dataset_dir, log))            # every count agrees ...
        self.assertEqual(record_layout.deeper_look(dataset_dir, log),              # ... and the look through names it
                         f"its files hold rows of a take that was never saved in {data_file(2)}")
        self.assertIsNone(record_layout.deeper_look(control_dir, control_log))
        self.assertEqual(repair_set.look(dataset_dir, log).data_files[data_file(2)], {3: 8, 4: 8})
        cut_file = digest(dataset_dir / data_file(2))
        before = every_file(dataset_dir)

        report = self.repair(dataset_dir, log, apply=False)
        self.assertEqual(report.code, 2, "\n".join(report.lines))
        self.assertEqual(every_file(dataset_dir), before)
        self.assertIn(f"write {data_file(2)} again with the rows of episode 3 only (8 rows of a take that was never "
                      "saved are left out)", "\n".join(report.lines))

        report = self.repair(dataset_dir, log, apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        # the file written again is, byte for byte, the file LeRobot writes when the session saves only that episode
        self.assertEqual(digest(dataset_dir / data_file(2)), digest(control_dir / data_file(2)))
        self.assertEqual(digest(report.side_dir / "before" / data_file(2)), cut_file)
        after = every_file(dataset_dir)
        self.assertEqual({name: value for name, value in after.items() if name != data_file(2)},
                         {name: value for name, value in before.items() if name != data_file(2)})
        loaded = self.loaded(dataset_dir)
        self.assertEqual((loaded.num_episodes, loaded.num_frames, len(loaded.hf_dataset)), (4, 32, 32))
        self.assertTrue(judge(repair_set.look(dataset_dir, log)).sound_already)
        # and the next session's episode is read from its own rows
        self.session(dataset_dir, log, saves=1, seed=90)
        self.session(control_dir, control_log, saves=1, seed=90)
        loaded = self.loaded(dataset_dir)
        self.assertEqual((loaded.num_episodes, loaded.num_frames, len(loaded.hf_dataset)), (5, 40, 40))
        self.assertEqual(loaded.hf_dataset["index"], list(range(40)))
        for name in (data_file(2), data_file(3), table_file(3)):
            self.assertEqual(digest(dataset_dir / name), digest(control_dir / name), name)

    def test_a_set_recorded_on_after_a_cut_short_save_reads_the_cut_take_s_rows_until_it_is_repaired(self):
        import numpy as np

        dataset_dir, log = self.copy_of_sound()
        self.session(dataset_dir, log, saves=2, seed=80, cut_short=True, lines=1)
        self.session(dataset_dir, log, saves=1, seed=90)            # the recorder resumes: its check found nothing
        control_dir, control_log = self.copy_of_sound()
        self.session(control_dir, control_log, saves=1, seed=80)
        self.session(control_dir, control_log, saves=1, seed=90)
        right = np.array(self.loaded(control_dir).hf_dataset["action"])
        # the damage, as LeRobot's own reader shows it: 8 rows too many, and episode 4 read from the cut take
        loaded = self.loaded(dataset_dir)
        self.assertEqual((loaded.num_episodes, loaded.num_frames, len(loaded.hf_dataset)), (5, 40, 48))
        self.assertFalse(np.array_equal(np.array(loaded.hf_dataset["action"])[:40], right))
        self.assertIsNone(self.problem(dataset_dir, log))

        report = self.repair(dataset_dir, log, apply=True)
        self.assertEqual(report.code, 0, "\n".join(report.lines))
        loaded = self.loaded(dataset_dir)
        self.assertEqual((loaded.num_episodes, loaded.num_frames, len(loaded.hf_dataset)), (5, 40, 40))
        self.assertTrue(np.array_equal(np.array(loaded.hf_dataset["action"]), right))
        self.assertEqual({name: value for name, value in every_file(dataset_dir).items() if name != "meta/stats.json"},
                         {name: value for name, value in every_file(control_dir).items() if name != "meta/stats.json"})

    def test_a_data_file_is_written_again_with_some_of_its_episodes_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "file-001.parquet"
            self.assertEqual(repair_set.keep_episodes(self.sound_dir / data_file(1), target, (1,)), 8)
            import pyarrow.parquet as parquet

            written, whole = parquet.read_table(target), parquet.read_table(self.sound_dir / data_file(1))
            self.assertEqual(written.column("episode_index").to_pylist(), [1] * 8)
            self.assertTrue(written.equals(whole.slice(0, 8), check_metadata=True))
            self.assertEqual(parquet.read_metadata(target).num_row_groups, 1)
            # all of its episodes: the file itself, byte for byte
            again = Path(directory) / "again.parquet"
            self.assertEqual(repair_set.keep_episodes(self.sound_dir / data_file(1), again, (1, 2)), 16)
            self.assertEqual(digest(again), digest(self.sound_dir / data_file(1)))


if __name__ == "__main__":
    unittest.main()
