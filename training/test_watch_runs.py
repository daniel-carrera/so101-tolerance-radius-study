"""The training watch: what it reads out of the launch output and a file list, and what it calls a problem."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import watch_runs as w  # noqa: E402


def look(at, step, *, loss=0.5, state="running", beat=20.0, checkpoints=(), layout="clustered"):
    return w.Look(at=float(at), layout=layout, state=state, heartbeat_age_s=beat, step=step,
                  losses={} if loss is None else {"train/loss": loss, "train/l1_loss": loss * 0.9},
                  checkpoints=None if checkpoints is None else list(checkpoints))


def steady(count, *, start_step=0, every_s=300.0, steps_per_s=4.3, loss=0.5):
    """Looks of a healthy run, oldest first."""
    return [look(i * every_s, int(start_step + i * every_s * steps_per_s), loss=loss) for i in range(count)]


class ReadingTests(unittest.TestCase):
    def test_the_run_s_name_is_the_last_one_the_launch_output_prints(self):
        text = "x\nfor the watchers:  WANDB_RUN=someone/tolrad-act/abc123\nlater WANDB_RUN=someone/tolrad-act/def456 \n"
        self.assertEqual(w.wandb_run_in(text), "someone/tolrad-act/def456")

    def test_a_launch_output_without_a_run_names_none(self):
        self.assertIsNone(w.wandb_run_in("NOT STARTED\nWANDB_RUN=\n"))
        self.assertIsNone(w.wandb_run_in("WANDB_RUN=only/two\n"))

    def test_the_key_is_read_from_either_form_of_the_secret_file(self):
        self.assertEqual(w.key_in("# note\nWANDB_API_KEY='abc'\n"), "abc")
        self.assertEqual(w.key_in("\nabc\n"), "abc")
        self.assertIsNone(w.key_in("OTHER=1\n"))
        self.assertIsNone(w.key_in(""))

    def test_a_checkpoint_counts_when_its_model_file_is_there(self):
        files = ["README.md", "checkpoints/010000/pretrained_model/model.safetensors",
                 "checkpoints/010000/training_state/optimizer_state.safetensors",
                 "checkpoints/020000/training_state/optimizer_state.safetensors",
                 "checkpoints/030000/pretrained_model/model.safetensors", "model.safetensors"]
        self.assertEqual(w.checkpoints_in(files), ["010000", "030000"])

    def test_the_rate_is_steps_over_seconds_between_two_looks_that_read_a_step(self):
        self.assertAlmostEqual(w.rate(look(0, 1000), look(300, 2290)), 4.3)
        self.assertIsNone(w.rate(None, look(300, 2290)))
        self.assertIsNone(w.rate(look(0, None), look(300, 2290)))

    def test_the_rate_goes_by_w_and_b_s_own_clock_for_the_two_steps(self):
        before, now = look(0, 1000), look(300, 2200)
        before.logged_at, now.logged_at = 1000.0, 1260.0      # this computer looked 300 s apart; the steps are 260 s apart
        self.assertAlmostEqual(w.rate(before, now), 1200 / 260)
        again = look(600, 2200)
        again.logged_at = 1260.0                               # the same logged step seen a second time
        self.assertIsNone(w.rate(now, again))

    def test_a_run_with_no_launch_output_is_found_by_asking_for_the_newest_running_one(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            for layout in ("clustered", "spread"):
                (local / f"{layout}.params.json").write_text(
                    '{"NAMESPACE": "owner", "RUN_NAME": "act-tolrad-%s-v1"}' % layout)
            (local / "launch-clustered-attempt1.txt").write_text("WANDB_RUN=ent/tolrad-act/old1\n")
            asked = []

            def discover(name):
                asked.append(name)
                return "ent/tolrad-act/new2" if "spread" in name else None

            self.assertEqual(w.project_in(local), "ent/tolrad-act")
            self.assertEqual(w.runs_to_watch(local, discover=discover),
                             {"spread": ("ent/tolrad-act/new2", "owner/act-tolrad-spread-v1")})
            self.assertEqual(asked, ["act-tolrad-clustered-v1", "act-tolrad-spread-v1"])

    def test_a_run_that_was_watched_stays_watched_when_nothing_names_a_running_one(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            (local / "spread.params.json").write_text('{"NAMESPACE": "owner", "RUN_NAME": "act-tolrad-spread-v1"}')
            current = {"spread": ("ent/tolrad-act/new2", "owner/act-tolrad-spread-v1")}
            self.assertEqual(w.runs_to_watch(local, discover=lambda name: None, current=current), current)
            newer = w.runs_to_watch(local, discover=lambda name: "ent/tolrad-act/new3", current=current)
            self.assertEqual(newer["spread"][0], "ent/tolrad-act/new3")
        self.assertIsNone(w.project_in(Path(folder)))

    def test_a_run_is_taken_up_only_while_w_and_b_still_hears_from_it(self):
        self.assertTrue(w.alive("running", 8.0))
        self.assertFalse(w.alive("running", 433.0))            # the machine is gone; W&B has not said so yet
        self.assertFalse(w.alive("running", None))
        self.assertFalse(w.alive("crashed", 8.0))
        self.assertAlmostEqual(w.heartbeat_age("2026-10-07T23:01:26Z", 1791414086.0 + 60.0), 60.0)
        self.assertIsNone(w.heartbeat_age(None, 0.0))

    def test_looks_at_the_run_before_are_not_this_run_s(self):
        old, new = look(0, 5400), look(900, 200)
        old.run, new.run = "old1", "new2"
        kept = w.of_this_run([old, new, look(1000, 400)], "ent/tolrad-act/new2")
        self.assertEqual([one.step for one in kept], [200])
        self.assertEqual(w.problems(kept[:0], new), [])       # step 200 after 5,400 is no step that stood still

    def test_a_look_takes_what_the_reader_gives_and_names_its_run(self):
        def reader(name, key, at):
            return {"state": "running", "heartbeat_age_s": 4.0, "step": 600, "losses": {"train/loss": 1.9},
                    "logged_at": at - 50.0}

        got = w.look_at("spread", "e/p/zt4", "o/act", "key", clock=lambda: 1000.0, wandb_reader=reader,
                        hub_reader=lambda repository: ["010000"])
        self.assertEqual((got.run, got.state, got.step, got.logged_at, got.checkpoints),
                         ("zt4", "running", 600, 950.0, ["010000"]))

    def test_runs_are_found_from_the_launch_output_and_the_parameter_file(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder)
            (local / "clustered.params.json").write_text('{"NAMESPACE": "owner", "RUN_NAME": "act-tolrad-clustered-v1"}')
            (local / "launch-clustered.txt").write_text("WANDB_RUN=ent/tolrad-act/r1\n")
            (local / "spread.params.json").write_text('{"NAMESPACE": "owner", "RUN_NAME": "act-tolrad-spread-v1"}')
            (local / "launch-spread.txt").write_text("NOT STARTED\n")
            self.assertEqual(w.runs_to_watch(local),
                             {"clustered": ("ent/tolrad-act/r1", "owner/act-tolrad-clustered-v1")})

    def test_a_look_that_cannot_be_read_says_so_and_does_not_raise(self):
        def no_wandb(*_):
            raise TimeoutError("slow")

        def no_hub(_):
            raise ConnectionError("down")

        got = w.look_at("spread", "e/p/r", "o/act", "key", clock=lambda: 10.0, wandb_reader=no_wandb, hub_reader=no_hub)
        self.assertIsNone(got.state)
        self.assertIsNone(got.checkpoints)
        self.assertEqual(got.unread, ["W&B: TimeoutError", "Hub: ConnectionError"])
        self.assertIn("not read", w.line(got, None))

    def test_the_watch_s_memory_gives_back_the_looks_it_was_given(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.jsonl"
            first, second = look(0, 100, checkpoints=[]), look(300, 1400, checkpoints=["010000"], layout="spread")
            w.remember(first, path)
            with path.open("a") as out:
                out.write("not json\n")
            w.remember(second, path)
            got = w.recall(path)
            self.assertEqual(got["clustered"][0].step, 100)
            self.assertEqual(got["spread"][0].checkpoints, ["010000"])
            self.assertEqual(got["spread"][0].loss, 0.5)
        self.assertEqual(w.recall(Path(folder) / "gone.jsonl"), {})


class JudgingTests(unittest.TestCase):
    def test_a_healthy_run_has_no_problem(self):
        looks = steady(8)
        self.assertEqual(w.problems(looks[:-1], looks[-1]), [])

    def test_a_run_w_and_b_calls_crashed_is_a_stopped_trainer(self):
        looks = steady(3)
        now = look(900, 3870, state="crashed")
        self.assertIn("the trainer has stopped", w.problems(looks, now)[0])

    def test_an_old_heartbeat_is_named_with_its_age(self):
        looks = steady(3)
        now = look(900, 3870, beat=660.0)
        self.assertIn("no heartbeat for 11 minutes (limit 3)", w.problems(looks, now)[0])
        self.assertEqual(w.problems(looks, look(900, 3870, beat=170.0)), [])

    def test_a_step_that_stands_still_is_named(self):
        looks = steady(3)
        now = look(900, looks[-1].step)
        self.assertTrue(any("has not advanced in 5 minutes" in said for said in w.problems(looks, now)))

    def test_a_step_standing_at_the_last_step_is_the_run_finishing(self):
        up = [f"{k:06d}" for k in range(10_000, 80_000, 10_000)]
        before = [look(0, 78_700, checkpoints=up), look(300, 80_000, checkpoints=up)]
        self.assertEqual(w.problems(before, look(600, 80_000, checkpoints=up)), [])
        self.assertTrue(any("has not advanced" in said
                            for said in w.problems(before[:1], look(300, 78_700, checkpoints=up))))

    def test_a_loss_that_is_not_a_number_is_named(self):
        looks = steady(3)
        now = look(900, 3870, loss=math.nan)
        self.assertTrue(any("train/loss is nan" in said for said in w.problems(looks, now)))

    def test_a_loss_twice_its_usual_is_a_jump_only_with_enough_looks_before(self):
        looks = steady(6, loss=0.05)
        self.assertTrue(any("the loss jumped to 0.1500, 3.0 times" in said
                            for said in w.problems(looks, look(1800, 7740, loss=0.15))))
        self.assertEqual(w.problems(looks, look(1800, 7740, loss=0.09)), [])
        self.assertEqual(w.problems(steady(3, loss=0.05), look(900, 3870, loss=0.5)), [])

    def test_a_falling_loss_is_never_a_jump(self):
        looks = [look(i * 300, i * 1290, loss=2.0 / (i + 1)) for i in range(8)]
        self.assertEqual(w.problems(looks[:-1], looks[-1]), [])

    def test_a_checkpoint_is_overdue_two_thousand_steps_past_it(self):
        before = [look(0, 10_500, checkpoints=[]), look(300, 11_790, checkpoints=[])]
        self.assertEqual(w.problems(before, look(360, 11_999, checkpoints=[])), [])
        said = w.problems(before, look(600, 13_080, checkpoints=[]))
        self.assertTrue(any("does not hold checkpoints/010000" in text for text in said))
        self.assertEqual(w.problems(before, look(600, 13_080, checkpoints=["010000"])), [])

    def test_a_hub_that_was_not_read_is_not_a_missing_checkpoint(self):
        before = [look(0, 10_500), look(300, 11_790)]
        self.assertEqual(w.problems(before, look(600, 13_080, checkpoints=None)), [])

    def test_three_slow_looks_in_a_row_are_named_and_two_are_not(self):
        slow = [look(i * 300, i * 600) for i in range(4)]          # 2 steps a second
        self.assertTrue(any("under 3.0 steps a second for 3 looks" in said for said in w.problems(slow[:-1], slow[-1])))
        mixed = [look(0, 0), look(300, 1290), look(600, 1890), look(900, 2490)]
        self.assertEqual(w.problems(mixed[:-1], mixed[-1]), [])

    def test_looks_under_two_minutes_apart_do_not_count_as_slow(self):
        close = [look(0, 0), look(60, 100), look(120, 200), look(180, 300)]   # 1.7 steps a second, a minute apart
        self.assertEqual(w.problems(close[:-1], close[-1]), [])

    def test_a_line_s_rate_is_taken_over_up_to_twenty_minutes(self):
        looks = steady(8)                                      # 0 to 2100 s, 300 s apart
        now = look(2400, 10_320)
        self.assertEqual(w.look_to_rate_from(looks, now).at, 1200.0)
        self.assertEqual(w.look_to_rate_from(looks[:1], look(5000, 20_000)).at, 0.0)
        self.assertIsNone(w.look_to_rate_from([], now))

    def test_three_unread_looks_in_a_row_say_the_watch_is_blind(self):
        blind = [w.Look(at=i * 300.0, layout="clustered", unread=["W&B: TimeoutError"]) for i in range(3)]
        self.assertEqual(w.problems(blind[:1], blind[1]), [])
        self.assertIn("the watch is blind", w.problems(blind[:2], blind[2])[0])

    def test_a_finished_run_is_no_problem_once_its_last_checkpoint_is_up(self):
        done = look(9000, 80_000, state="finished", beat=4000.0, checkpoints=["070000", "080000"])
        self.assertTrue(w.finished(done))
        self.assertEqual(w.problems(steady(3), done), [])
        early = look(9000, 80_000, state="finished", checkpoints=["070000"])
        self.assertIn("does not hold checkpoints/080000 yet", w.problems(steady(3), early)[0])

    def test_a_run_that_says_finished_before_its_last_step_has_stopped(self):
        short = look(900, 41_000, state="finished")
        self.assertFalse(w.finished(short))
        self.assertIn("the trainer has stopped", w.problems(steady(3), short)[0])

    def test_new_checkpoints_are_those_the_hub_did_not_hold_at_the_last_look(self):
        self.assertEqual(w.new_checkpoints(look(0, 1, checkpoints=["010000"]),
                                           look(300, 2, checkpoints=["010000", "020000"])), ["020000"])
        self.assertEqual(w.new_checkpoints(look(0, 1, checkpoints=["010000"]), look(300, 2, checkpoints=None)), [])

    def test_a_line_carries_the_figures_and_no_account_name(self):
        now = look(1_800_000_000, 20_000, loss=0.0421, checkpoints=["010000"])
        text = w.line(now, 4.3)
        for part in ("clustered", "running", "heartbeat 20 s ago", "step 20,000 of 80,000 (25.0 %)", "4.30 steps/s",
                     "ends about", "loss 0.0421", "Hub: 010000"):
            self.assertIn(part, text)
        self.assertNotIn("/act-", text)


if __name__ == "__main__":
    unittest.main()
