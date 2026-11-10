"""The settings check: which differences between two training configurations pass, and which must not."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_train_configs as ctc  # noqa: E402


def run_config(dataset: str, run: str, run_id: str, **changes: object) -> dict:
    """A small training configuration shaped like LeRobot's, for one made-up run."""
    config: dict = {
        "dataset": {
            "repo_id": f"owner/{dataset}",
            "episodes": None,
            "image_transforms": {"enable": False, "tfs": {"hue": {"kwargs": {"hue": [-0.05, 0.05]}}}},
            "eval_split": 0.0,
        },
        "policy": {
            "type": "act",
            "repo_id": f"owner/{run}",
            "chunk_size": 100,
            "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
        },
        "output_dir": f"/content/train/{run}",
        "job_name": run,
        "seed": 1000,
        "batch_size": 32,
        "steps": 80000,
        "eval_steps": 0,
        "save_freq": 10000,
        "optimizer": {"type": "adamw", "lr": 1e-05, "betas": [0.9, 0.999]},
        "scheduler": None,
        "eval": {"n_episodes": 50, "batch_size": 8},
        "wandb": {"enable": True, "project": "project", "entity": "team", "run_id": run_id, "notes": None},
        "rename_map": {},
    }
    for dotted, value in changes.items():
        target = config
        *parents, leaf = dotted.split("__")
        for name in parents:
            target = target[name]
        target[leaf] = value
    return config


CLUSTERED = run_config("set-clustered", "act-clustered", "aaaa1111")
SPREAD = run_config("set-spread", "act-spread", "bbbb2222")


class Files(unittest.TestCase):
    """Writes configurations to a temporary folder and runs the command on them."""

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write(self, name: str, content: object) -> str:
        path = self.folder / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return str(path)

    def run_main(self, *arguments: str) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                code = ctc.main(list(arguments))
            except SystemExit as stop:  # argparse leaves through here on a bad command line
                code = stop.code if isinstance(stop.code, int) else 1
        return code, output.getvalue()

    def compare(self, first: object, second: object, *options: str) -> tuple[int, str]:
        return self.run_main(self.write("first.json", first), self.write("second.json", second), *options)


class FlattenTests(unittest.TestCase):
    def test_nested_settings_become_dotted_keys(self):
        flat = ctc.flatten({"optimizer": {"lr": 1e-05, "betas": [0.9, 0.999]}, "steps": 80000})
        self.assertEqual(flat, {("optimizer", "lr"): 1e-05, ("optimizer", "betas"): [0.9, 0.999], ("steps",): 80000})

    def test_a_list_and_an_empty_group_are_single_values(self):
        flat = ctc.flatten({"rename_map": {}, "tags": [], "episodes": None})
        self.assertEqual(flat, {("rename_map",): {}, ("tags",): [], ("episodes",): None})

    def test_a_name_containing_a_dot_cannot_be_mistaken_for_a_nested_group(self):
        dotted_name = ctc.flatten({"features": {"observation.state": 1}})
        nested_group = ctc.flatten({"features": {"observation": {"state": 1}}})
        self.assertNotEqual(set(dotted_name), set(nested_group))


class SameValueTests(unittest.TestCase):
    def test_a_whole_number_written_as_a_float_is_the_same_setting(self):
        self.assertTrue(ctc.same_value(80000, 80000.0))

    def test_true_is_not_the_number_one(self):
        self.assertFalse(ctc.same_value(True, 1))
        self.assertFalse(ctc.same_value(0, False))
        self.assertTrue(ctc.same_value(True, True))

    def test_lists_are_compared_in_order_and_in_depth(self):
        self.assertTrue(ctc.same_value([0.9, 0.999], [0.9, 0.999]))
        self.assertFalse(ctc.same_value([0.9, 0.999], [0.999, 0.9]))
        self.assertFalse(ctc.same_value([0.9, 0.999], [0.9]))
        self.assertFalse(ctc.same_value([[1, True]], [[1, 1]]))

    def test_null_is_only_equal_to_null(self):
        self.assertTrue(ctc.same_value(None, None))
        self.assertFalse(ctc.same_value(None, 0))
        self.assertFalse(ctc.same_value(None, ""))

    def test_groups_inside_a_list_are_compared_in_depth(self):
        self.assertTrue(ctc.same_value([{"weight": 1, "on": True}], [{"on": True, "weight": 1.0}]))
        self.assertFalse(ctc.same_value([{"weight": 1, "on": True}], [{"weight": 1, "on": 1}]))
        self.assertFalse(ctc.same_value([{"weight": 1}], [{"weight": 1, "on": True}]))

    def test_text_is_compared_exactly(self):
        self.assertTrue(ctc.same_value("adamw", "adamw"))
        self.assertFalse(ctc.same_value("adamw", "AdamW"))
        self.assertFalse(ctc.same_value("1", 1))


class AllowedTests(unittest.TestCase):
    def test_an_allowed_key_covers_itself_and_what_is_under_it(self):
        self.assertTrue(ctc.is_allowed("wandb.run_id", ["wandb.run_id"]))
        self.assertTrue(ctc.is_allowed("wandb.run_id", ["wandb"]))

    def test_an_allowed_key_does_not_cover_a_longer_name_that_starts_the_same(self):
        self.assertFalse(ctc.is_allowed("eval_steps", ["eval"]))
        self.assertFalse(ctc.is_allowed("policy.repo_id_extra", ["policy.repo_id"]))
        self.assertFalse(ctc.is_allowed("dataset.repo_id", ["policy.repo_id"]))

    def test_the_default_list_is_the_one_the_paper_states(self):
        self.assertEqual(
            set(ctc.ALLOWED_TO_DIFFER),
            {
                "dataset.repo_id",
                "output_dir",
                "job_name",
                "policy.repo_id",
                "wandb.entity",
                "wandb.project",
                "wandb.run_id",
            },
        )


class TwoRunsTests(Files):
    def test_two_runs_that_differ_only_where_allowed_pass(self):
        code, text = self.compare(CLUSTERED, SPREAD)
        self.assertEqual(code, 0, text)
        self.assertIn("identical apart from 5 allowed key(s)", text)
        self.assertNotIn("DIFFERS", text)

    def test_the_allowed_keys_are_named_but_their_values_stay_hidden(self):
        code, text = self.compare(CLUSTERED, SPREAD)
        self.assertEqual(code, 0, text)
        for key in ("dataset.repo_id", "output_dir", "job_name", "policy.repo_id", "wandb.run_id"):
            self.assertIn(f"allowed  {key}", text)
        for hidden in ("owner", "set-clustered", "act-spread", "aaaa1111", "/content/train"):
            self.assertNotIn(hidden, text)

    def test_show_allowed_prints_those_values(self):
        code, text = self.compare(CLUSTERED, SPREAD, "--show-allowed")
        self.assertEqual(code, 0, text)
        self.assertIn('allowed  dataset.repo_id: "owner/set-clustered" | "owner/set-spread"', text)

    def test_a_setting_that_differs_fails_and_is_printed_with_both_values(self):
        other = run_config("set-spread", "act-spread", "bbbb2222", batch_size=64)
        code, text = self.compare(CLUSTERED, other)
        self.assertEqual(code, 1, text)
        self.assertIn("DIFFERS  batch_size: 32 | 64", text)
        self.assertIn("1 key(s) differ outside the allowed list", text)

    def test_a_nested_setting_that_differs_is_named_by_its_full_key(self):
        other = run_config("set-spread", "act-spread", "bbbb2222", optimizer__lr=5e-05, dataset__eval_split=0.1)
        code, text = self.compare(CLUSTERED, other)
        self.assertEqual(code, 1, text)
        self.assertIn("DIFFERS  optimizer.lr: 1e-05 | 5e-05", text)
        self.assertIn("DIFFERS  dataset.eval_split: 0.0 | 0.1", text)
        self.assertIn("2 key(s) differ outside the allowed list", text)

    def test_a_difference_deep_inside_a_group_is_found(self):
        other = run_config("set-spread", "act-spread", "bbbb2222")
        other["dataset"]["image_transforms"]["tfs"]["hue"]["kwargs"]["hue"] = [-0.1, 0.1]
        other["policy"]["input_features"]["observation.state"]["shape"] = [7]
        code, text = self.compare(CLUSTERED, other)
        self.assertEqual(code, 1, text)
        self.assertIn("DIFFERS  dataset.image_transforms.tfs.hue.kwargs.hue: [-0.05, 0.05] | [-0.1, 0.1]", text)
        self.assertIn("DIFFERS  policy.input_features.observation.state.shape: [6] | [7]", text)

    def test_a_key_present_in_one_file_only_fails(self):
        other = run_config("set-spread", "act-spread", "bbbb2222")
        other["ema"] = {"enable": False}
        del other["scheduler"]
        code, text = self.compare(CLUSTERED, other)
        self.assertEqual(code, 1, text)
        self.assertIn("DIFFERS  ema.enable: <absent> | false", text)
        self.assertIn("DIFFERS  scheduler: null | <absent>", text)

    def test_a_resumed_run_is_reported_not_passed_over(self):
        resumed = run_config("set-spread", "act-spread", "bbbb2222", resume=True, checkpoint_path="/content/x/040000")
        code, text = self.compare(CLUSTERED, resumed)
        self.assertEqual(code, 1, text)
        self.assertIn("DIFFERS  resume: <absent> | true", text)
        self.assertIn("DIFFERS  checkpoint_path:", text)


class OptionTests(Files):
    def test_allow_lets_a_named_key_differ(self):
        earlier = run_config("set-earlier", "act-earlier", "cccc3333", steps=100000, eval_steps=5000, save_freq=5000)
        earlier["dataset"]["eval_split"] = 0.1
        code, text = self.compare(earlier, CLUSTERED)
        self.assertEqual(code, 1, text)
        self.assertIn("4 key(s) differ outside the allowed list", text)

        changed_on_purpose = ("steps", "eval_steps", "save_freq", "dataset.eval_split")
        options = [part for key in changed_on_purpose for part in ("--allow", key)]
        code, text = self.compare(earlier, CLUSTERED, *options)
        self.assertEqual(code, 0, text)
        self.assertIn("identical apart from 9 allowed key(s)", text)
        self.assertIn("also allowed here: steps, eval_steps, save_freq, dataset.eval_split", text)

    def test_a_setting_allowed_on_purpose_is_printed_with_its_values_and_a_repository_is_not(self):
        earlier = run_config("set-earlier", "act-earlier", "cccc3333", steps=100000)
        code, text = self.compare(earlier, CLUSTERED, "--allow", "steps")
        self.assertEqual(code, 0, text)
        self.assertIn("allowed  steps: 100000 | 80000", text)
        self.assertIn("allowed  dataset.repo_id (values not shown)", text)
        self.assertNotIn("owner", text)

    def test_allow_does_not_hide_a_neighbouring_key(self):
        other = run_config("set-spread", "act-spread", "bbbb2222", eval_steps=5000)
        code, text = self.compare(CLUSTERED, other, "--allow", "eval")
        self.assertEqual(code, 1, text)
        self.assertIn("DIFFERS  eval_steps: 0 | 5000", text)

    def test_allow_on_a_group_covers_the_keys_inside_it(self):
        other = run_config("set-spread", "act-spread", "bbbb2222", eval__batch_size=1, wandb__notes="second VM")
        code, text = self.compare(CLUSTERED, other, "--allow", "eval", "--allow", "wandb")
        self.assertEqual(code, 0, text)
        self.assertIn("allowed  eval.batch_size", text)

    def test_must_differ_fails_when_the_two_files_are_the_same_run(self):
        code, text = self.compare(CLUSTERED, CLUSTERED, "--must-differ", "dataset.repo_id")
        self.assertEqual(code, 1, text)
        self.assertIn("SAME     dataset.repo_id", text)

    def test_must_differ_passes_for_two_different_runs(self):
        code, text = self.compare(CLUSTERED, SPREAD, "--must-differ", "dataset.repo_id", "--must-differ", "job_name")
        self.assertEqual(code, 0, text)
        self.assertNotIn("SAME", text)

    def test_must_differ_fails_for_a_key_that_neither_file_has(self):
        code, text = self.compare(CLUSTERED, SPREAD, "--must-differ", "dataset.repo")
        self.assertEqual(code, 1, text)
        self.assertIn("SAME     dataset.repo:", text)
        self.assertIn("1 key(s) that must differ are the same", text)

    def test_without_must_differ_a_file_compared_with_itself_passes_and_asks_the_question(self):
        code, text = self.compare(CLUSTERED, CLUSTERED)
        self.assertEqual(code, 0, text)
        self.assertIn("identical apart from 0 allowed key(s)", text)
        self.assertIn("Is this one run compared with itself?", text)

    def test_the_output_states_which_keys_were_allowed(self):
        code, text = self.compare(CLUSTERED, SPREAD)
        self.assertEqual(code, 0, text)
        self.assertIn("allowed to differ: " + ", ".join(ctc.ALLOWED_TO_DIFFER), text)
        self.assertNotIn("also allowed here", text)


class BadInputTests(Files):
    def test_a_missing_file_is_an_error_not_a_pass(self):
        code, text = self.run_main(self.write("first.json", CLUSTERED), str(self.folder / "nowhere.json"))
        self.assertEqual(code, 2, text)
        self.assertIn("cannot read", text)

    def test_a_file_that_is_not_json_is_an_error(self):
        code, text = self.compare(CLUSTERED, "{not json")
        self.assertEqual(code, 2, text)
        self.assertIn("cannot read", text)

    def test_a_file_that_is_not_a_configuration_is_an_error(self):
        code, text = self.compare(CLUSTERED, [1, 2, 3])
        self.assertEqual(code, 2, text)
        self.assertIn("not a JSON object", text)

    def test_an_empty_configuration_is_an_error(self):
        code, text = self.compare(CLUSTERED, {})
        self.assertEqual(code, 2, text)
        self.assertIn("holds no settings", text)

    def test_one_file_alone_is_a_usage_error(self):
        code, _ = self.run_main(self.write("first.json", CLUSTERED))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
