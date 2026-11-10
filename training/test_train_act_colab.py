"""The training notebook's logic, run cell by cell against stand-ins for Colab, the Hub, the dataset and the trainer.

The notebook cannot be run here: it needs a Colab machine, an account and a GPU. What can be checked is everything it
decides: which parameters it accepts, which command it builds, when its gates open, and that it starts no trainer unless
all of them have. Each test executes cells of `train_act_colab.ipynb` as they are on disk, in a plain namespace, with
`/content` moved to a temporary folder. Nothing here touches the network.

    python -m unittest discover -s training
"""

import contextlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "train_act_colab.ipynb"
#: The earlier project's notebook, when it is checked out beside this repository: its training cell is the reference.
EARLIER_NOTEBOOK = HERE.parent.parent / "lerobot-101" / "notebooks" / "train_act_colab.ipynb"

with open(NOTEBOOK, encoding="utf-8") as _handle:
    _NOTEBOOK = json.load(_handle)
CELLS = [(cell["id"], cell["cell_type"], "".join(cell["source"])) for cell in _NOTEBOOK["cells"]]
CODE = {cell_id: source for cell_id, kind, source in CELLS if kind == "code"}
ORDER = [cell_id for cell_id, kind, _ in CELLS if kind == "code"]
PARAMS_LINE = 'PARAMS_FILE = "/content/run_params.json"'


def run_cell(namespace: dict, cell_id: str, replace: tuple[tuple[str, str], ...] = ()) -> str:
    """Execute one code cell in the namespace and return what it printed."""
    source = CODE[cell_id]
    for old, new in replace:
        assert old in source, (cell_id, old)
        source = source.replace(old, new)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(source, cell_id, "exec"), namespace)
    return printed.getvalue()


def run_failing_cell(namespace: dict, cell_id: str) -> tuple[BaseException, str]:
    """Execute a cell that must stop with an error; return the error and what the cell printed before it."""
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(CODE[cell_id], cell_id, "exec"), namespace)
    except (AssertionError, RuntimeError) as error:
        return error, printed.getvalue()
    raise AssertionError(f"cell {cell_id} did not stop with an error; it printed:\n{printed.getvalue()}")


class NotebookCase(unittest.TestCase):
    """A fresh namespace, a temporary folder in place of /content, and waits cut short."""

    sleep_cap_s = 0.05

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ns: dict = {"__name__": "__main__"}
        real_sleep = time.sleep
        self.real_sleep = real_sleep
        time.sleep = lambda seconds: real_sleep(min(seconds, self.sleep_cap_s))
        self.addCleanup(setattr, time, "sleep", real_sleep)

    def replace_modules(self, **modules: types.ModuleType) -> None:
        """Put stand-in modules into `sys.modules` for this test only."""
        for name, module in modules.items():
            original = sys.modules.get(name)
            sys.modules[name] = module
            if original is None:
                self.addCleanup(sys.modules.pop, name, None)
            else:
                self.addCleanup(sys.modules.__setitem__, name, original)

    def params(self, overrides: dict | None) -> str:
        """Run the parameter cell, with the override file it would find in /content (or none)."""
        path = self.folder / "run_params.json"
        if overrides is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(json.dumps(overrides), encoding="utf-8")
        return run_cell(self.ns, "parameters", ((PARAMS_LINE, f"PARAMS_FILE = {str(path)!r}"),))

    def refused_params(self, overrides: dict | None) -> str:
        with self.assertRaises(AssertionError) as caught:
            self.params(overrides)
        return str(caught.exception)

    def through_helpers(self, **overrides: object) -> dict:
        """In a fresh namespace, run the parameter, settings and helper cells, with every path under the folder."""
        self.ns = {"__name__": "__main__"}
        self.params({"NAMESPACE": "owner", **overrides})
        run_cell(self.ns, "settings")
        work = str(self.folder)
        self.ns.update(
            WORK_DIR=work,
            OUTPUT_DIR=f"{work}/train/{self.ns['RUN_NAME']}",
            TRAIN_LOG=f"{work}/train.log",
            PID_FILE=f"{work}/train.pid",
            FREEZE_FILE=f"{work}/pip-freeze.txt",
            HF_TOKEN_UPLOAD=f"{work}/hf_token_file",
            WANDB_KEY_UPLOAD=f"{work}/wandb_key_file",
            WANDB_KEY_STORE=f"{work}/config/wandb_api_key",
        )
        run_cell(self.ns, "helpers")
        self.ns.update(shutil=shutil, subprocess=subprocess, sys=sys)  # what the machine cell imports
        return self.ns


class ParameterTests(NotebookCase):
    def test_the_placeholder_namespace_is_refused_and_no_gate_opens(self):
        self.assertIn("set NAMESPACE", self.refused_params(None))
        self.assertEqual(self.ns["GATES"], {})

    def test_the_defaults_are_the_protocols_and_name_the_clustered_run(self):
        printed = self.params({"NAMESPACE": "owner"})
        self.assertIn("dataset tolrad-clustered-v1 | run act-tolrad-clustered-v1 | 80,000 steps", printed)
        self.assertEqual((self.ns["STEPS"], self.ns["SAVE_FREQ"], self.ns["EXPECTED_EPISODES"]), (80_000, 10_000, 55))
        self.assertEqual((self.ns["MODE"], self.ns["FOLLOW"], self.ns["EXPECTED_FRAMES"]), ("fresh", False, None))
        self.assertEqual(self.ns["GATES"], {"parameters": True})
        self.assertNotIn("owner", printed)

    def test_the_second_run_changes_two_names(self):
        self.params({"NAMESPACE": "owner", "DATASET_NAME": "tolrad-spread-v1", "RUN_NAME": "act-tolrad-spread-v1"})
        self.assertEqual(self.ns["RUN_NAME"], "act-tolrad-spread-v1")

    def test_a_run_name_left_from_the_other_run_is_refused(self):
        message = self.refused_params({"NAMESPACE": "owner", "DATASET_NAME": "tolrad-spread-v1"})
        self.assertIn("does not contain DATASET_NAME", message)
        self.assertNotIn("parameters", self.ns["GATES"])

    def test_wrong_values_are_refused(self):
        cases = [
            ({"STEP": 5}, "unknown parameter"),
            ({"MODE": "resume"}, "unknown MODE"),
            ({"STEPS": 85_000}, "whole number of SAVE_FREQ"),
            ({"STEPS": True}, "whole number of SAVE_FREQ"),
            ({"DATASET_NAME": "owner/tolrad-clustered-v1"}, "without the namespace"),
            ({"EXPECTED_EPISODES": 0}, "count of episodes"),
            ({"FOLLOW": "yes"}, "True or False"),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                self.assertIn(expected, self.refused_params({"NAMESPACE": "owner", **overrides}))

    def test_a_failed_pass_shuts_the_gates_an_earlier_pass_opened(self):
        self.params({"NAMESPACE": "owner"})
        self.ns["GATES"]["dataset"] = True
        self.refused_params({"NAMESPACE": "owner", "MODE": "nope"})
        self.assertEqual(self.ns["GATES"], {})

    def test_a_short_rehearsal_on_a_small_dataset_can_be_asked_for(self):
        self.params(
            {
                "NAMESPACE": "owner",
                "DATASET_NAME": "tolrad-practice-v1",
                "RUN_NAME": "act-tolrad-practice-v1",
                "STEPS": 200,
                "SAVE_FREQ": 200,
                "EXPECTED_EPISODES": 10,
            }
        )
        self.assertTrue(self.ns["GATES"]["parameters"])


class SettingsAndHelperTests(NotebookCase):
    def test_settings_derive_the_names_and_hold_the_protocols_values(self):
        self.params({"NAMESPACE": "owner"})
        printed = run_cell(self.ns, "settings")
        self.assertEqual(self.ns["DATASET_REPO"], "owner/tolrad-clustered-v1")
        self.assertEqual(self.ns["POLICY_REPO"], "owner/act-tolrad-clustered-v1")
        self.assertEqual(self.ns["OUTPUT_DIR"], "/content/train/act-tolrad-clustered-v1")
        self.assertEqual((self.ns["EVAL_SPLIT"], self.ns["EVAL_STEPS"]), (0.0, 0))
        self.assertEqual((self.ns["BATCH_SIZE"], self.ns["NUM_WORKERS"], self.ns["LEROBOT_VERSION"]), (32, 8, "0.6.1"))
        self.assertIn("80,000 steps x batch 32 = 2,560,000 samples", printed)
        self.assertIn("5.1 h, about 27 compute units; a checkpoint about every 38 min", printed)

    def test_a_secret_is_read_in_each_form_the_files_come_in(self):
        read_secret = self.through_helpers()["read_secret"]
        path = self.folder / "secret"
        good = [
            ("abc123\n", ("HF_TOKEN",), "abc123"),
            ("HF_TOKEN=abc123\n", ("HF_TOKEN",), "abc123"),
            ('# a comment\n\nexport WANDB_API_KEY="k-e-y"\nOTHER=1\n', ("WANDB_API_KEY",), "k-e-y"),
            ("OTHER=1\nHF_TOKEN = 'tok'\n", ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"), "tok"),
        ]
        for text, names, expected in good:
            path.write_text(text, encoding="utf-8")
            self.assertEqual(read_secret(path, names), expected)
        for text in ("", "OTHER=1\n", "two words\n", "first\nsecond\n"):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_secret(path, ("HF_TOKEN",))

    def test_checkpoint_steps_are_the_numbered_folders_only(self):
        checkpoint_steps = self.through_helpers()["checkpoint_steps"]
        files = [
            ".gitattributes",
            "model.safetensors",
            "checkpoints/020000/pretrained_model/config.json",
            "checkpoints/010000/pretrained_model/model.safetensors",
            "checkpoints/010000/training_state/training_step.json",
            "checkpoints/last/pretrained_model/config.json",
            "checkpoints/100000/training_state/rng_state.safetensors",
            "checkpoints/note.txt",
        ]
        self.assertEqual(checkpoint_steps(files), ["010000", "020000", "100000"])
        self.assertEqual(checkpoint_steps([]), [])

    def test_only_the_trainer_counts_as_a_running_trainer(self):
        trainer_pids = self.through_helpers()["trainer_pids"]
        proc = self.folder / "proc"
        commands = {
            "101": b"/usr/bin/python3\0/usr/local/bin/lerobot-train\0--steps=80000\0",
            "102": b"python\0-c\0from multiprocessing.spawn import spawn_main\0",
            "103": b"",  # an ended process that has not been collected
            "self": b"lerobot-train",
        }
        for pid, command in commands.items():
            (proc / pid).mkdir(parents=True)
            (proc / pid / "cmdline").write_bytes(command)
        (proc / "104").mkdir()  # a process that ended between the listing and the read
        self.assertEqual(trainer_pids(str(proc)), [101])
        self.assertEqual(trainer_pids(str(self.folder / "absent")), [])

    def test_the_log_is_read_as_the_trainer_writes_it(self):
        ns = self.through_helpers()
        log = self.folder / "train.log"
        log.write_text(
            "\n===== launch 2026-10-07T03:00:00+00:00 mode=fresh run=act-tolrad-clustered-v1 =====\n"
            "INFO Track this run --> \x1b[1m\x1b[33mhttps://wandb.ai/team/tolrad-act/runs/ab12cd34\x1b[0m\n"
            "INFO ot_train.py:405 dataset.num_frames=41250 (41K)\n"
            "INFO ot_train.py:406 dataset.num_episodes=55\n"
            "Upload 3 LFS files: 100%|##########| 3/3 [00:10<00:00,  3.4s/it]\n"
            "Training:   0%|          | 0/80000 [00:00<?, ?step/s]\r"
            "Training:   0%|          | 200/80000 [00:46<5:07:11,  4.33step/s]\r"
            "INFO ot_train.py:606 step:200 smpl:6K ep:9 epch:0.16 loss:2.871 grdn:71.3 lr:1.0e-05 updt_s:0.214\n"
            "Training:   2%|1         | 1400/80000 [05:23<5:02:31,  4.33step/s]\r"
            "INFO ot_train.py:606 step:1K smpl:45K ep:60 epch:1.09 loss:0.9 grdn:30 lr:1.0e-05 updt_s:0.21\n"
            "INFO ot_train.py:651 Checkpoint policy after step 10000\n"
            "Upload 2 LFS files: 100%|##########| 2/2 [00:10<00:00,  5.0s/it]\n",
            encoding="utf-8",
        )
        status = ns["log_status"](ns["read_log"](str(log)))
        self.assertEqual(status["launches"], [("2026-10-07T03:00:00+00:00", "fresh", "act-tolrad-clustered-v1")])
        self.assertEqual(status["wandb_run"], "team/tolrad-act/ab12cd34")
        self.assertEqual((status["episodes"], status["frames"]), (55, 41250))
        self.assertEqual(len(status["step_lines"]), 2)
        self.assertTrue(status["step_lines"][0].startswith("step:200 smpl:6K"))
        self.assertEqual(status["progress"], ("1400", "80000", "05:23<5:02:31,  4.33step/s"))
        self.assertEqual(status["checkpoints"], [10000])
        self.assertEqual((status["held_out"], status["ended"], status["pushed"]), ([], False, False))

    def test_an_empty_log_a_held_out_split_and_an_ended_run_are_told_apart(self):
        log_status = self.through_helpers()["log_status"]
        empty = log_status("")
        self.assertEqual((empty["step_lines"], empty["progress"], empty["wandb_run"]), ([], None, None))
        self.assertIsNone(empty["episodes"])
        held_out = log_status("INFO Train/eval split: 49 train, 6 eval (eval_split=0.1, 1 tasks)\n")
        self.assertEqual(len(held_out["held_out"]), 1)
        ended = log_status("INFO End of training\nINFO Model pushed to https://example.invalid/x\n")
        self.assertTrue(ended["ended"] and ended["pushed"])

    def test_only_the_last_start_is_reported(self):
        log_status = self.through_helpers()["log_status"]
        first = (
            "\n===== launch 2026-10-07T03:00:00+00:00 mode=fresh run=r =====\n"
            "INFO dataset.num_episodes=55\nINFO step:200 smpl:6K ep:9 epch:0.16 loss:2.8\n"
            "INFO Checkpoint policy after step 10000\nINFO End of training\n"
        )
        second = "\n===== launch 2026-10-07T05:00:00+00:00 mode=resume_local run=r =====\nINFO Creating dataset\n"
        status = log_status(first + second)
        self.assertEqual(len(status["launches"]), 2)
        self.assertEqual((status["step_lines"], status["checkpoints"], status["ended"]), ([], [], False))
        resumed = "INFO Resuming data order at epoch 7, sample 12\nINFO step:10K smpl:326K ep:435 epch:7.9 loss:0.1\n"
        status = log_status(first + second + resumed)
        self.assertEqual((len(status["step_lines"]), len(status["resumed"])), (1, 1))


class CommandTests(NotebookCase):
    def flags(self, **overrides: object) -> dict[str, str]:
        ns = self.through_helpers(**overrides)
        ns["OUTPUT_DIR"] = f"/content/train/{ns['RUN_NAME']}"
        run_cell(ns, "command")
        command = ns["CMD"]
        flags = dict(part[2:].split("=", 1) for part in command[1:])
        self.assertEqual(len(flags), len(command) - 1, "a flag is given twice")
        return flags

    def test_the_command_carries_the_protocols_flags(self):
        flags = self.flags()
        expected = {
            "dataset.repo_id": "owner/tolrad-clustered-v1",
            "dataset.eval_split": "0.0",
            "policy.type": "act",
            "policy.device": "cuda",
            "policy.repo_id": "owner/act-tolrad-clustered-v1",
            "policy.private": "true",
            "output_dir": "/content/train/act-tolrad-clustered-v1",
            "job_name": "act-tolrad-clustered-v1",
            "steps": "80000",
            "batch_size": "32",
            "num_workers": "8",
            "save_freq": "10000",
            "save_checkpoint_to_hub": "true",
            "eval_steps": "0",
            "max_eval_samples": "512",
            "log_freq": "200",
            "wandb.enable": "true",
            "wandb.project": "tolrad-act",
        }
        self.assertEqual(flags, expected)
        self.assertTrue(self.ns["GATES"]["command"])

    @unittest.skipUnless(EARLIER_NOTEBOOK.exists(), "the earlier project is not checked out beside this one")
    def test_it_is_the_earlier_command_flag_for_flag_in_the_same_order(self):
        with open(EARLIER_NOTEBOOK, encoding="utf-8") as handle:
            training_cell = "".join(json.load(handle)["cells"][9]["source"])
        earlier = [line.strip().rstrip("\\").strip() for line in training_cell.splitlines()]
        earlier = [line[2:].split("=", 1)[0] for line in earlier if line.startswith("--")]
        self.flags()
        ours = [part[2:].split("=", 1)[0] for part in self.ns["CMD"][1:]]
        self.assertEqual(ours, [name for name in earlier if name != "wandb.entity"])
        self.assertEqual(earlier.count("wandb.entity"), 1)

    def test_the_two_runs_differ_in_exactly_four_flags(self):
        clustered = self.flags()
        spread = self.flags(DATASET_NAME="tolrad-spread-v1", RUN_NAME="act-tolrad-spread-v1")
        differing = sorted(name for name in clustered if clustered[name] != spread[name])
        self.assertEqual(differing, ["dataset.repo_id", "job_name", "output_dir", "policy.repo_id"])
        self.assertEqual(set(clustered), set(spread))

    def test_a_weights_and_biases_account_is_passed_only_when_set(self):
        self.assertNotIn("wandb.entity", self.flags())
        self.assertEqual(self.flags(WANDB_ENTITY="team")["wandb.entity"], "team")

    def test_a_seed_is_passed_only_when_set_and_only_under_a_run_name_of_its_own(self):
        self.assertNotIn("seed", self.flags())                       # the paper's two runs: LeRobot's default
        seeded = self.flags(SEED=2000, RUN_NAME="act-tolrad-clustered-v1-s2000")
        self.assertEqual(seeded["seed"], "2000")
        self.assertEqual(seeded["policy.repo_id"], "owner/act-tolrad-clustered-v1-s2000")
        plain = self.flags()
        self.assertEqual({name for name in seeded if seeded[name] != plain.get(name)},
                         {"seed", "job_name", "output_dir", "policy.repo_id"})
        with self.assertRaisesRegex(AssertionError, "needs a RUN_NAME of its own"):
            self.flags(SEED=2000)                                    # ... would push into the paper's repository
        with self.assertRaisesRegex(AssertionError, "SEED is a whole number"):
            self.flags(SEED="2000", RUN_NAME="act-tolrad-clustered-v1-s2000")

    def test_the_resume_commands(self):
        ns = self.through_helpers(MODE="resume_hub")
        run_cell(ns, "command")
        out = ns["OUTPUT_DIR"]
        self.assertEqual(
            ns["CMD"][1:], ["--config_path=owner/act-tolrad-clustered-v1", "--resume=true", f"--output_dir={out}"]
        )
        ns["MODE"] = "resume_local"
        run_cell(ns, "command")
        last = f"{out}/checkpoints/last/pretrained_model/train_config.json"
        self.assertEqual(ns["CMD"][1:], [f"--config_path={last}", "--resume=true"])


class MachineAndInstallTests(NotebookCase):
    def machine(self, gpu_line: str | None, free_gb: float = 200.0) -> str:
        ns = self.through_helpers()
        self.replace_modules(
            shutil=types.SimpleNamespace(
                which=lambda name: "/usr/bin/nvidia-smi" if gpu_line is not None else None,
                disk_usage=lambda path: types.SimpleNamespace(free=free_gb * 1e9),
            ),
            subprocess=types.SimpleNamespace(
                run=lambda command, **options: types.SimpleNamespace(stdout=f"{gpu_line or ''}\n")
            ),
        )
        return run_cell(ns, "machine")

    def test_an_a100_with_room_on_its_disk_passes(self):
        printed = self.machine("NVIDIA A100-SXM4-40GB, 40960 MiB")
        self.assertIn("gpu: NVIDIA A100-SXM4-40GB, 40960 MiB", printed)
        self.assertTrue(self.ns["GATES"]["machine"])

    def test_another_gpu_no_gpu_or_a_full_disk_is_refused(self):
        cases = [
            ("Tesla T4, 15360 MiB", 200.0, "not an A100"),
            (None, 200.0, "no GPU on this runtime"),
            ("NVIDIA A100-SXM4-40GB, 40960 MiB", 9.0, "less than 15 GB free"),
        ]
        for gpu_line, free_gb, expected in cases:
            with self.subTest(gpu=gpu_line, free_gb=free_gb):
                with self.assertRaises(AssertionError) as caught:
                    self.machine(gpu_line, free_gb)
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn("machine", self.ns["GATES"])

    def install(self, messages: str, return_code: int) -> tuple[list, dict]:
        ns = self.through_helpers()
        commands: list = []

        def run(command: list[str], **options: object) -> types.SimpleNamespace:
            commands.append(command)
            return types.SimpleNamespace(stdout="", stderr=messages, returncode=return_code)

        ns["subprocess"] = types.SimpleNamespace(run=run)
        return commands, ns

    def test_the_pinned_version_is_installed_and_the_gate_opens(self):
        commands, ns = self.install("", 0)
        printed = run_cell(ns, "install")
        self.assertEqual(commands[0][1:], ["-m", "pip", "install", "-q", "lerobot[training]==0.6.1"])
        self.assertIn("pip had nothing to say", printed)
        self.assertTrue(ns["GATES"]["install"])

    def test_a_failed_install_shows_what_pip_said_and_opens_no_gate(self):
        _, ns = self.install("ERROR: No matching distribution found", 1)
        error, printed = run_failing_cell(ns, "install")
        self.assertIn("pip could not install", str(error))
        self.assertIn("No matching distribution", printed)
        self.assertNotIn("install", ns["GATES"])


class EnvironmentTests(NotebookCase):
    def environment(
        self, lerobot_version: str = "0.6.1", torch_version: str = "2.11.0+cu128", cuda: bool = True
    ) -> dict:
        """A namespace in which stand-ins for torch, lerobot and pip describe the given installation."""
        ns = self.through_helpers()
        torch = types.ModuleType("torch")
        torch.__version__ = torch_version
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda index: "NVIDIA A100")
        torch.version = types.SimpleNamespace(cuda="12.8")
        lerobot = types.ModuleType("lerobot")
        lerobot.__version__ = lerobot_version
        self.replace_modules(torch=torch, lerobot=lerobot)
        ns["subprocess"] = types.SimpleNamespace(
            run=lambda command, **options: types.SimpleNamespace(stdout="b==2\na==1\n", stderr="", returncode=0)
        )
        installed = {"numpy": "2.2.6", "torchvision": "0.26.0", "accelerate": "1.14"}
        installed["torch"] = torch_version.split("+")[0]
        for name in ("numpy", "torchvision", "pandas", "pyarrow", "huggingface_hub", "datasets"):
            loaded = getattr(sys.modules.get(name), "__version__", None)
            if loaded:  # whatever this interpreter has already imported counts as installed at that version
                installed[name] = loaded.split("+")[0]
        real_version = importlib.metadata.version

        def version(distribution: str) -> str:
            if distribution in installed:
                return installed[distribution]
            raise importlib.metadata.PackageNotFoundError(distribution)

        importlib.metadata.version = version
        self.addCleanup(setattr, importlib.metadata, "version", real_version)
        return ns

    def test_the_fingerprint_names_versions_and_no_person(self):
        numpy, torchvision = types.ModuleType("numpy"), types.ModuleType("torchvision")
        numpy.__version__, torchvision.__version__ = "2.2.6", "0.26.0"
        self.replace_modules(numpy=numpy, torchvision=torchvision)
        ns = self.environment()
        printed = run_cell(ns, "environment")
        pattern = (
            r"^python=3\.\d+\.\d+ lerobot=0\.6\.1 torch=2\.11\.0\+cu128 torchvision=0\.26\.0 accelerate=1\.14 "
            r"numpy=2\.2\.6 cuda=12\.8 gpu='NVIDIA A100' processors=\d+ pip_freeze_sha256=[0-9a-f]{16}$"
        )
        self.assertRegex(ns["ENV_LINE"], pattern)
        self.assertIn("ENV python=", printed)
        self.assertEqual(Path(ns["FREEZE_FILE"]).read_text(encoding="utf-8"), "b==2\na==1\n")
        self.assertTrue(ns["GATES"]["environment"])

    def test_the_package_list_hash_does_not_depend_on_the_order_pip_printed(self):
        ns = self.environment()
        run_cell(ns, "environment")
        first = ns["ENV_LINE"]
        ns["subprocess"] = types.SimpleNamespace(
            run=lambda command, **options: types.SimpleNamespace(stdout="a==1\nb==2\n", stderr="", returncode=0)
        )
        run_cell(ns, "environment")
        self.assertEqual(ns["ENV_LINE"], first)

    def test_a_wrong_install_is_refused(self):
        cases = [
            ({"lerobot_version": "0.6.0"}, "lerobot 0.6.0 is installed, not 0.6.1"),
            ({"torch_version": "2.11.0+cpu"}, "CPU-only torch"),
            ({"cuda": False}, "cannot see the GPU"),
        ]
        for options, expected in cases:
            with self.subTest(options=options):
                error, _ = run_failing_cell(self.environment(**options), "environment")
                self.assertIn(expected, str(error))
                self.assertNotIn("environment", self.ns["GATES"])

    def test_a_module_loaded_before_pip_replaced_it_asks_for_a_restart(self):
        ns = self.environment()
        stale = types.ModuleType("numpy")
        stale.__version__ = "0.0.1"
        self.replace_modules(numpy=stale)
        error, _ = run_failing_cell(ns, "environment")
        self.assertIn("restart the kernel", str(error))
        self.assertIn("numpy 0.0.1 is loaded", str(error))


class FakeHub(types.ModuleType):
    """A stand-in for `huggingface_hub`: one account, some repositories, and a record of what was asked of it."""

    def __init__(self, me: dict, repos: dict, stored_token: str | None = None) -> None:
        super().__init__("huggingface_hub")
        self.calls: list[tuple] = []
        self.repos = repos
        hub = self

        class HfApi:
            def dataset_info(self, repo: str) -> types.SimpleNamespace:
                return types.SimpleNamespace(private=hub.repos["dataset", repo]["private"])

            def model_info(self, repo: str) -> types.SimpleNamespace:
                return types.SimpleNamespace(private=hub.repos["model", repo]["private"])

            def create_repo(self, repo: str, repo_type: str, private: bool, exist_ok: bool) -> None:
                hub.calls.append(("create_repo", repo, repo_type, private, exist_ok))
                hub.repos.setdefault((repo_type, repo), {"private": private, "files": []})

            def list_repo_files(self, repo: str, repo_type: str) -> list[str]:
                return hub.repos[repo_type, repo]["files"]

        def login(token: str) -> None:
            hub.calls.append(("login", token))
            hub.stored_token = token

        self.stored_token = stored_token
        self.HfApi = HfApi
        self.login = login
        self.get_token = lambda: hub.stored_token
        self.whoami = lambda: me


MEMBER = {"name": "someone", "orgs": [{"name": "owner"}], "auth": {"accessToken": {"role": "write"}}}
DATASET = ("dataset", "owner/tolrad-clustered-v1")
POLICY = ("model", "owner/act-tolrad-clustered-v1")
WHOLE_CHECKPOINT = (
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "training_state/training_step.json",
)


class SignInTests(NotebookCase):
    def signed_out(self, mode: str = "fresh", me: dict = MEMBER, repos: dict | None = None) -> tuple[dict, FakeHub]:
        ns = self.through_helpers(MODE=mode)
        hub = FakeHub(me, repos if repos is not None else {DATASET: {"private": True, "files": []}})
        self.replace_modules(huggingface_hub=hub)
        Path(ns["HF_TOKEN_UPLOAD"]).write_text("HF_TOKEN=hf_abc\n", encoding="utf-8")
        Path(ns["WANDB_KEY_UPLOAD"]).write_text("WANDB_API_KEY=" + "k" * 40 + "\n", encoding="utf-8")
        ns["trainer_pids"] = lambda proc=None: []
        return ns, hub

    def test_a_new_run_signs_in_stores_the_key_and_makes_a_private_repository(self):
        ns, hub = self.signed_out()
        printed = run_cell(ns, "sign-in")
        created = ("create_repo", "owner/act-tolrad-clustered-v1", "model", True, True)
        self.assertEqual(hub.calls, [("login", "hf_abc"), created])
        self.assertFalse(os.path.exists(ns["HF_TOKEN_UPLOAD"]))
        self.assertFalse(os.path.exists(ns["WANDB_KEY_UPLOAD"]))
        self.assertEqual(Path(ns["WANDB_KEY_STORE"]).read_text(encoding="utf-8"), "k" * 40)
        self.assertEqual(stat.S_IMODE(os.stat(ns["WANDB_KEY_STORE"]).st_mode), 0o600)
        for secret_or_name in ("hf_abc", "k" * 40, "someone", "owner"):
            self.assertNotIn(secret_or_name, printed)
        self.assertIsNone(ns["_secret"])
        self.assertTrue(ns["GATES"]["sign_in"])

    def test_a_second_pass_needs_no_files(self):
        ns, hub = self.signed_out()
        run_cell(ns, "sign-in")
        hub.calls.clear()
        run_cell(ns, "sign-in")
        self.assertEqual([call[0] for call in hub.calls], ["create_repo"])

    def test_a_missing_file_is_named(self):
        ns, _ = self.signed_out()
        os.remove(ns["HF_TOKEN_UPLOAD"])
        error, _ = run_failing_cell(ns, "sign-in")
        self.assertIn("no Hub token: upload it to", str(error))
        ns, _ = self.signed_out()
        os.remove(ns["WANDB_KEY_UPLOAD"])
        error, _ = run_failing_cell(ns, "sign-in")
        self.assertIn("no Weights & Biases key: upload it to", str(error))
        self.assertNotIn("sign_in", ns["GATES"])

    def test_a_token_of_another_account_is_refused_before_anything_is_created(self):
        stranger = {"name": "someone", "orgs": [], "auth": {"accessToken": {"role": "write"}}}
        ns, hub = self.signed_out(me=stranger)
        error, _ = run_failing_cell(ns, "sign-in")
        self.assertIn("belongs neither to NAMESPACE", str(error))
        self.assertEqual([call[0] for call in hub.calls], ["login"])

    def test_a_read_only_token_is_refused(self):
        reader = {"name": "owner", "orgs": [], "auth": {"accessToken": {"role": "read"}}}
        ns, hub = self.signed_out(me=reader)
        error, _ = run_failing_cell(ns, "sign-in")
        self.assertIn("read-only", str(error))
        self.assertNotIn("create_repo", [call[0] for call in hub.calls])

    def test_a_public_dataset_is_refused(self):
        ns, hub = self.signed_out(repos={DATASET: {"private": False, "files": []}})
        error, _ = run_failing_cell(ns, "sign-in")
        self.assertIn("dataset repository is public", str(error))
        self.assertNotIn("create_repo", [call[0] for call in hub.calls])

    def test_a_new_run_refuses_a_repository_that_already_holds_checkpoints(self):
        files = ["checkpoints/010000/pretrained_model/model.safetensors"]
        repos = {DATASET: {"private": True, "files": []}, POLICY: {"private": True, "files": files}}
        ns, _ = self.signed_out(repos=repos)
        error, _ = run_failing_cell(ns, "sign-in")
        self.assertIn("already holds checkpoints ['010000']", str(error))
        ns["trainer_pids"] = lambda proc=None: [4242]  # a second pass while that very run is going is fine
        run_cell(ns, "sign-in")
        self.assertTrue(ns["GATES"]["sign_in"])

    def test_check_mode_creates_nothing(self):
        ns, hub = self.signed_out(mode="check")
        printed = run_cell(ns, "sign-in")
        self.assertEqual([call[0] for call in hub.calls], ["login"])
        self.assertIn("neither created nor read", printed)

    def test_resuming_from_the_hub_needs_a_checkpoint_there(self):
        repos = {DATASET: {"private": True, "files": []}, POLICY: {"private": True, "files": []}}
        ns, hub = self.signed_out(mode="resume_hub", repos=repos)
        error, _ = run_failing_cell(ns, "sign-in")
        self.assertIn("nothing to resume", str(error))
        self.assertNotIn("create_repo", [call[0] for call in hub.calls])
        repos[POLICY]["files"] = ["checkpoints/040000/training_state/training_step.json"]
        self.assertIn("checkpoints on the Hub: ['040000']", run_cell(ns, "sign-in"))

    def test_the_hub_view_lists_what_is_there_what_is_incomplete_and_what_is_to_come(self):
        repos = {DATASET: {"private": True, "files": []}, POLICY: {"private": True, "files": []}}
        ns, _ = self.signed_out(repos=repos)
        run_cell(ns, "sign-in")
        printed = run_cell(ns, "hub-view")
        self.assertIn("checkpoints on the Hub: none yet", printed)
        self.assertIn("The first checkpoint is due about 38 min", printed)
        files = [f"checkpoints/{step}/{part}" for step in ("010000", "020000") for part in WHOLE_CHECKPOINT]
        repos[POLICY]["files"] = [*files, "checkpoints/030000/pretrained_model/model.safetensors"]
        printed = run_cell(ns, "hub-view")
        self.assertIn("checkpoints on the Hub: ['010000', '020000', '030000']", printed)
        self.assertIn("INCOMPLETE on the Hub (a file is missing): ['030000']", printed)
        self.assertIn("still to come: ['040000', '050000', '060000', '070000', '080000']", printed)
        self.assertIn("final policy at the top of the repository: False", printed)


class FakeDataset:
    """A stand-in for a loaded dataset: counts, camera names, and items with the shapes the notebook checks."""

    def __init__(
        self,
        episodes: int = 55,
        frames: int = 41250,
        cameras: tuple[str, ...] = ("observation.images.front", "observation.images.wrist"),
        fps: int = 30,
        tasks: tuple[str, ...] = ("pick up the cube",),
        picture: tuple[int, int, int] = (3, 480, 640),
    ) -> None:
        self.num_episodes, self.num_frames, self.fps = episodes, frames, fps
        self.meta = types.SimpleNamespace(camera_keys=list(cameras), tasks=types.SimpleNamespace(index=list(tasks)))
        self.picture = picture
        self.asked: list[int] = []

    def __getitem__(self, index: int) -> dict:
        self.asked.append(index)
        item = {camera: types.SimpleNamespace(shape=self.picture) for camera in self.meta.camera_keys}
        item["action"] = types.SimpleNamespace(shape=(6,))
        item["observation.state"] = types.SimpleNamespace(shape=(6,))
        return item


class DatasetTests(NotebookCase):
    def with_dataset(
        self, dataset: FakeDataset | None = None, error: Exception | None = None, **overrides: object
    ) -> tuple[dict, list[str]]:
        ns = self.through_helpers(**overrides)
        asked_for: list[str] = []

        def load(repo: str) -> FakeDataset:
            asked_for.append(repo)
            if error is not None:
                raise error
            return dataset or FakeDataset()

        datasets = types.ModuleType("lerobot.datasets")
        datasets.LeRobotDataset = load
        self.replace_modules(**{"lerobot": types.ModuleType("lerobot"), "lerobot.datasets": datasets})
        return ns, asked_for

    def test_the_recorded_dataset_passes_and_is_decoded_at_both_ends(self):
        ns, asked_for = self.with_dataset()
        printed = run_cell(ns, "dataset")
        self.assertEqual(asked_for, ["owner/tolrad-clustered-v1"])
        self.assertIn("55 episodes, 41250 frames, 30 fps", printed)
        self.assertEqual(ns["ds"].asked, [0, 41249])
        self.assertEqual((ns["DATASET_EPISODES"], ns["DATASET_FRAMES"]), (55, 41250))
        self.assertTrue(ns["GATES"]["dataset"])
        self.assertIn("ms a sample for one worker", run_cell(ns, "decode-speed"))
        self.assertNotIn("decode", " ".join(ns["GATES"]))

    def test_a_dataset_that_is_not_the_recorded_one_is_refused(self):
        cases = [
            (FakeDataset(episodes=54), "54 episodes on the Hub, 55 expected"),
            (FakeDataset(cameras=("observation.images.front",)), "unexpected cameras"),
            (FakeDataset(cameras=("observation.images.wrist", "observation.images.top")), "unexpected cameras"),
            (FakeDataset(fps=15), "15 frames a second"),
            (FakeDataset(picture=(3, 720, 1280)), "not 640x480"),
            (FakeDataset(tasks=("one", "two")), "2 task strings"),
        ]
        for dataset, expected in cases:
            with self.subTest(expected=expected):
                ns, _ = self.with_dataset(dataset)
                error, _ = run_failing_cell(ns, "dataset")
                self.assertIn(expected, str(error))
                self.assertNotIn("dataset", ns["GATES"])

    def test_the_frame_count_is_compared_when_it_is_given(self):
        ns, _ = self.with_dataset(EXPECTED_FRAMES=41251)
        error, _ = run_failing_cell(ns, "dataset")
        self.assertIn("41250 frames on the Hub, 41251 counted at the rig", str(error))

    def test_a_missing_version_tag_is_explained(self):
        ns, _ = self.with_dataset(error=RuntimeError("Your dataset must be tagged with a codebase version."))
        _, printed = run_failing_cell(ns, "dataset")
        self.assertIn("has no version tag", printed)


TRAINER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    for line in {lines!r}:
        sys.stdout.write(line)
        sys.stdout.flush()
        time.sleep(0.02)
    time.sleep({linger_s})
    sys.exit({exit_status})
    """
)
HEALTHY_LINES = [
    "INFO Track this run --> \x1b[1m\x1b[33mhttps://wandb.ai/team/tolrad-act/runs/ab12cd34\x1b[0m\n",
    "INFO ot_train.py:405 dataset.num_frames=41250 (41K)\n",
    "INFO ot_train.py:406 dataset.num_episodes=55\n",
    "Training:   0%|          | 0/80000 [00:00<?, ?step/s]\r",
    "Training:   0%|          | 200/80000 [00:46<5:07:11,  4.33step/s]\r",
    "INFO ot_train.py:606 step:200 smpl:6K ep:9 epch:0.16 loss:2.871 grdn:71.3 lr:1.0e-05 updt_s:0.214 data_s:0.570\n",
]
CHECK_GATES = ("machine", "install", "environment", "sign_in", "dataset", "command")


class LaunchTests(NotebookCase):
    """The launch and report cells, with a small script standing in for `lerobot-train`."""

    def ready(
        self, mode: str = "fresh", lines: list[str] | None = None, linger_s: float = 3, exit_status: int = 0
    ) -> dict:
        """A namespace in which every check has passed and a small script stands in for the trainer."""
        ns = self.through_helpers(MODE=mode)
        for gate in CHECK_GATES:
            ns["GATES"][gate] = True
        key_store = Path(ns["WANDB_KEY_STORE"])
        key_store.parent.mkdir(parents=True, exist_ok=True)
        key_store.write_text("w" * 40, encoding="utf-8")
        script = self.folder / "stand-in-trainer.py"
        script.write_text(
            TRAINER_SCRIPT.format(
                lines=HEALTHY_LINES if lines is None else lines, linger_s=linger_s, exit_status=exit_status
            ),
            encoding="utf-8",
        )
        ns.update(CMD=[sys.executable, str(script)], ENV_LINE="python=3.12 lerobot=0.6.1")
        ns.update(DATASET_EPISODES=55, DATASET_FRAMES=41250)

        def trainer_pids(proc: str | None = None) -> list[int]:
            trainer = ns.get("TRAINER")
            return [trainer.pid] if trainer is not None and trainer.poll() is None else []

        def stop_trainer() -> None:
            trainer = ns.get("TRAINER")
            if trainer is not None:
                if trainer.poll() is None:
                    trainer.kill()
                trainer.wait()

        ns["trainer_pids"] = trainer_pids
        self.addCleanup(stop_trainer)
        return ns

    def log(self) -> str:
        return Path(self.ns["TRAIN_LOG"]).read_text(encoding="utf-8")

    def test_a_healthy_start_is_reported_with_its_first_real_values(self):
        ns = self.ready()
        printed = run_cell(ns, "launch")
        self.assertIn("STARTED at ", printed)
        self.assertEqual(Path(ns["PID_FILE"]).read_text(encoding="utf-8"), str(ns["TRAINER"].pid))
        printed = run_cell(ns, "first-value")
        self.assertIn("trainer running: True", printed)
        self.assertIn("first logged step:  step:200 smpl:6K ep:9", printed)
        self.assertIn("progress since the last start: 200/80000 [00:46<5:07:11,  4.33step/s]", printed)
        self.assertIn("the trainer counted: 55 episodes, 41250 frames", printed)
        self.assertIn("for the watchers:  WANDB_RUN=team/tolrad-act/ab12cd34", printed)
        record = printed.split("RUN RECORD")[1]
        self.assertIn("dataset tolrad-clustered-v1: 55 episodes, 41250 frames", record)
        self.assertIn("run act-tolrad-clustered-v1: 80,000 steps, a checkpoint every 10,000, mode fresh", record)
        self.assertNotIn("owner", record)
        self.assertNotIn("team", record)

    def test_a_second_pass_starts_nothing(self):
        ns = self.ready()
        run_cell(ns, "launch")
        printed = run_cell(ns, "launch")
        self.assertIn("already running", printed)
        self.assertNotIn("STARTED", printed)
        self.assertEqual(self.log().count("===== launch"), 1)

    def test_the_key_reaches_the_trainer_through_its_environment_and_nowhere_else(self):
        ns = self.ready()
        probe = self.folder / "probe.py"
        probe.write_text(
            "import os, sys, time\n"
            "print('KEY LENGTH', len(os.environ.get('WANDB_API_KEY', '')), 'ARGUMENTS', len(sys.argv) - 1)\n"
            "sys.stdout.flush()\ntime.sleep(2)\n",
            encoding="utf-8",
        )
        ns["CMD"] = [sys.executable, str(probe)]
        self.sleep_cap_s = 1.0
        printed = run_cell(ns, "launch")
        self.assertIn("KEY LENGTH 40 ARGUMENTS 0", self.log())
        self.assertNotIn("w" * 40, printed + self.log())
        self.assertIsNone(ns["_environment"])

    def test_a_trainer_that_stops_at_once_is_not_called_started(self):
        lines = ["Traceback (most recent call last):\n", "wandb.errors.UsageError: api_key not configured\n"]
        ns = self.ready(lines=lines, linger_s=0, exit_status=1)
        self.sleep_cap_s = 1.5
        error, printed = run_failing_cell(ns, "launch")
        self.assertIn("NOT RUNNING. The trainer stopped at once with exit status 1", str(error))
        self.assertIn("api_key not configured", printed)

    def test_a_very_short_run_that_has_already_ended_well_is_not_called_a_failure(self):
        ns = self.ready(lines=[*HEALTHY_LINES, "INFO End of training\n"], linger_s=0)
        self.sleep_cap_s = 1.5
        printed = run_cell(ns, "launch")
        self.assertIn("STARTED at ", printed)
        self.assertIn("It has already ended without an error", printed)
        printed = run_cell(ns, "first-value")
        self.assertIn("trainer running: False", printed)
        self.assertIn("the log says 'End of training'", printed)

    def test_a_shut_gate_starts_nothing(self):
        ns = self.ready()
        del ns["GATES"]["dataset"]
        error, _ = run_failing_cell(ns, "launch")
        self.assertIn("NOT STARTED. These checks have not passed: ['dataset']", str(error))
        self.assertNotIn("TRAINER", ns)
        self.assertFalse(os.path.exists(ns["TRAIN_LOG"]))

    def test_check_mode_starts_nothing(self):
        ns = self.ready(mode="check")
        printed = "".join(run_cell(ns, cell) for cell in ("launch", "first-value", "follow", "hub-view"))
        self.assertIn("nothing was started", printed)
        self.assertNotIn("TRAINER", ns)
        self.assertFalse(os.path.exists(ns["TRAIN_LOG"]))

    def test_checkpoints_already_on_this_machine_block_a_new_start(self):
        ns = self.ready()
        os.makedirs(os.path.join(ns["OUTPUT_DIR"], "checkpoints", "010000"))
        error, _ = run_failing_cell(ns, "launch")
        self.assertIn("already holds checkpoints ['010000']", str(error))
        self.assertNotIn("TRAINER", ns)

    def test_a_folder_left_by_a_failed_start_is_cleared(self):
        ns = self.ready()
        os.makedirs(os.path.join(ns["OUTPUT_DIR"], "wandb"))
        printed = run_cell(ns, "launch")
        self.assertIn("removing", printed)
        self.assertIn("STARTED", printed)
        self.assertFalse(os.path.exists(ns["OUTPUT_DIR"]))

    def test_resuming_on_this_machine_needs_a_checkpoint_here_and_keeps_the_folder(self):
        ns = self.ready(mode="resume_local")
        error, _ = run_failing_cell(ns, "launch")
        self.assertIn("holds no checkpoint to resume from", str(error))
        os.makedirs(os.path.join(ns["OUTPUT_DIR"], "checkpoints", "040000"))
        self.assertIn("STARTED", run_cell(ns, "launch"))
        self.assertTrue(os.path.isdir(os.path.join(ns["OUTPUT_DIR"], "checkpoints", "040000")))

    def test_held_out_episodes_stop_the_report(self):
        split = "INFO Train/eval split: 49 train, 6 eval (eval_split=0.1, 1 tasks)\n"
        ns = self.ready(lines=[split, *HEALTHY_LINES])
        run_cell(ns, "launch")
        error, _ = run_failing_cell(ns, "first-value")
        self.assertIn("STOP THIS RUN. The trainer held episodes out", str(error))

    def test_a_wrong_episode_count_stops_the_report(self):
        ns = self.ready(lines=[line.replace("num_episodes=55", "num_episodes=49") for line in HEALTHY_LINES])
        run_cell(ns, "launch")
        error, _ = run_failing_cell(ns, "first-value")
        self.assertIn("counted 49 episodes, not 55", str(error))

    def test_a_trainer_that_dies_before_its_first_step_is_loud(self):
        ns = self.ready(lines=["INFO Creating dataset\n"], linger_s=0.3, exit_status=3)
        self.sleep_cap_s = 0.01
        run_cell(ns, "launch")
        self.real_sleep(0.8)
        error, printed = run_failing_cell(ns, "first-value")
        self.assertIn("NO LOGGED STEP: the trainer has stopped", str(error))
        self.assertIn("Creating dataset", printed)

    def test_a_trainer_that_dies_after_some_steps_is_loud(self):
        ns = self.ready(linger_s=0.2, exit_status=1)
        self.sleep_cap_s = 0.01
        run_cell(ns, "launch")
        self.real_sleep(0.8)
        error, _ = run_failing_cell(ns, "first-value")
        self.assertIn("THE TRAINER HAS STOPPED without 'End of training'", str(error))

    def test_a_finished_run_reports_its_end(self):
        ending = ["INFO Checkpoint policy after step 80000\n", "INFO End of training\n", "INFO Model pushed to x\n"]
        ns = self.ready(lines=[*HEALTHY_LINES, *ending], linger_s=0.1)
        self.sleep_cap_s = 0.01
        run_cell(ns, "launch")
        self.real_sleep(0.8)
        printed = run_cell(ns, "first-value")
        self.assertIn("trainer running: False", printed)
        self.assertIn("checkpoints announced since the last start: [80000]", printed)
        self.assertIn("the log says 'End of training', and the final policy was pushed", printed)

    def test_following_waits_for_the_end_when_asked(self):
        ns = self.ready(lines=[*HEALTHY_LINES, "INFO End of training\n"], linger_s=0.3)
        ns["FOLLOW"] = True
        run_cell(ns, "launch")
        printed = run_cell(ns, "follow")
        self.assertIn("The trainer has exited, and the log says 'End of training'.", printed)
        self.assertEqual(ns["trainer_pids"](), [])


class WholeNotebookTests(unittest.TestCase):
    def test_it_is_a_notebook_with_no_outputs(self):
        self.assertEqual((_NOTEBOOK["nbformat"], _NOTEBOOK["nbformat_minor"]), (4, 5))
        for cell in _NOTEBOOK["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None), cell["id"])
        self.assertEqual(len({cell_id for cell_id, _, _ in CELLS}), len(CELLS), "cell ids repeat")

    def test_every_code_cell_is_plain_python(self):
        for cell_id in ORDER:
            compile(CODE[cell_id], cell_id, "exec")
            for line in CODE[cell_id].splitlines():
                self.assertFalse(line.lstrip().startswith(("!", "%")), f"{cell_id}: a shell escape: {line}")

    def test_every_gate_the_launch_needs_is_opened_once_by_an_earlier_cell_on_its_last_line(self):
        needed = re.search(r"REQUIRED_GATES = \((.*?)\)", CODE["launch"], re.DOTALL).group(1)
        needed = re.findall(r'"(\w+)"', needed)
        opened: dict[str, list[int]] = {}
        for position, cell_id in enumerate(ORDER):
            lines = CODE[cell_id].splitlines()
            for number, line in enumerate(lines):
                gate = re.match(r'GATES\["(\w+)"\] = True', line)
                if gate:
                    opened.setdefault(gate.group(1), []).append(position)
                    self.assertEqual(number, len(lines) - 1, f"{cell_id} opens its gate before its last check")
        self.assertEqual(sorted(needed), sorted(opened))
        for gate in needed:
            self.assertEqual(len(opened[gate]), 1, gate)
            self.assertLess(opened[gate][0], ORDER.index("launch"), gate)

    def test_it_names_no_earlier_dataset_and_no_account(self):
        everything = "\n".join(source for _, _, source in CELLS)
        for leftover in ("marker", "so101", "39627", "39,627", "53 episodes", "35127", "35,127", "075000"):
            self.assertNotIn(leftover, everything)
        self.assertIn('NAMESPACE = "NAMESPACE"', CODE["parameters"])
        self.assertIn("WANDB_ENTITY = None", CODE["parameters"])


if __name__ == "__main__":
    unittest.main()
