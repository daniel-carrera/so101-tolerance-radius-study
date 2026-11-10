#!/usr/bin/env python3
"""Watch this week's two training runs: one line per run every few minutes, and a stop when a person is needed.

    "$RIG_PY" training/watch_runs.py                 # every 5 minutes until something is worth a look
    "$RIG_PY" training/watch_runs.py --once          # one look at both runs
    "$RIG_PY" training/watch_runs.py --keep-going    # checkpoints are logged and the watch goes on

It only reads: a run's state, heartbeat, step and losses from W&B, and the checkpoint folders of its policy repository
on the Hub. It never calls the `colab` command (`README.md`, step 4: polling a training machine with it is how August
lost a run). Every look is appended to `data/logs/training_watch.log`, which git ignores.

What it takes from where, so that no account name stands in a tracked file:

- a run's W&B name from `training/local/launch-<layout>.txt`, the line the notebook prints after `WANDB_RUN=`; where
  a launch from a browser left no such file, the newest run of the layout's name that W&B calls running, in the
  project that any kept launch output names;
- the Hub owner from `training/local/<layout>.params.json`;
- the W&B key from the earlier project's `secrets/wandb.venv` (never printed), or `WANDB_API_KEY`;
- the Hub login from the `hf` command's own stored login.

**What it can and cannot show.** There is no evaluation loss this week (protocol, section 5: every recorded episode is
trained on), so nothing here measures overfitting: the training loss only falls. This watches whether a run is alive,
advancing at the expected rate, numerically sound, and leaving its checkpoints.

With nothing to watch yet it waits, a look a minute, until a running run appears. A run started again under the
same name is taken up as a new run: its looks are kept apart from those at the run before it.

Exit status: 0 both runs finished; 2 something is wrong (the lines above say what); 3 a new checkpoint arrived (so
that whoever started the watch looks, compares, and starts it again); 4 no W&B key, or `--once` with nothing to
watch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LOCAL = HERE / "local"
LOG = REPO / "data" / "logs" / "training_watch.log"
SECRET = REPO.parent / "lerobot-101" / "secrets" / "wandb.venv"

LAYOUTS = ("clustered", "spread")
STEPS = 80_000
SAVE_FREQ = 10_000

#: A heartbeat older than this is a run W&B has stopped hearing from. A live run's heartbeat was 1 to 10 seconds old
#: at every look of 2026-10-07; the two machines that Colab took back that day stopped theirs at once.
HEARTBEAT_LIMIT_S = 180.0
#: A checkpoint is overdue when the run is this many steps past it and the Hub does not hold it (about 8 minutes).
CHECKPOINT_GRACE_STEPS = 2_000
#: August's average was 4.33 steps a second on the same machine type.
SLOW_STEPS_PER_S = 3.0
SLOW_LOOKS = 3
#: A loss this many times the median of the looks before it is called a jump.
LOSS_JUMP = 2.0
LOSS_HISTORY = 10
LOSS_MIN_HISTORY = 5
#: Looks that could not be read, in a row, before the watch says it is blind.
BLIND_LOOKS = 3


@dataclass(slots=True)
class Look:
    """One look at one run."""

    at: float
    layout: str
    #: None when W&B could not be read at this look.
    state: str | None = None
    heartbeat_age_s: float | None = None
    step: int | None = None
    losses: dict[str, float] = field(default_factory=dict)
    #: The checkpoint folders the Hub holds, as their names ("010000"); None when the Hub could not be read.
    checkpoints: list[str] | None = None
    #: Why a source could not be read, for the log.
    unread: list[str] = field(default_factory=list)
    #: The W&B run's own id (the last part of its name: no account in it). A run started again has another.
    run: str | None = None
    #: W&B's own clock for the step it shows (`_timestamp`, seconds since 1970): when the trainer logged it.
    logged_at: float | None = None

    @property
    def loss(self) -> float | None:
        if "train/loss" in self.losses:
            return self.losses["train/loss"]
        return next(iter(self.losses.values()), None)


def wandb_run_in(launch_text: str) -> str | None:
    """The run's W&B name, `entity/project/id`, from the launch output: the last `WANDB_RUN=` line with three parts."""
    found = re.findall(r"WANDB_RUN=([^\s/'\"]+/[^\s/'\"]+/[^\s/'\"]+)", launch_text)
    return found[-1] if found else None


def key_in(secret_text: str) -> str | None:
    """The W&B key from the secret file: a line `WANDB_API_KEY=...`, or a bare value on a line of its own."""
    lines = [line.strip() for line in secret_text.splitlines() if line.strip() and not line.strip().startswith("#")]
    for line in lines:
        if line.startswith("WANDB_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'") or None
    if len(lines) == 1 and "=" not in lines[0]:
        return lines[0]
    return None


def checkpoints_in(files: Sequence[str]) -> list[str]:
    """The checkpoint folders that hold a model, from a repository's file list. An upload is one commit, so a folder
    that is there is whole (`README.md`, step 4.3)."""
    found = {parts[1] for name in files
             if len(parts := name.split("/")) >= 4 and parts[0] == "checkpoints"
             and parts[2] == "pretrained_model" and parts[3] == "model.safetensors"}
    return sorted(found)


def rate(before: Look | None, now: Look) -> float | None:
    """Steps a second between two looks, when both read a step. By W&B's own clock for the two steps where both
    looks have it: the step it shows is the last one logged, up to a minute old, so this computer's clock would
    count time in which nothing new could be seen. The same logged step seen twice gives no rate."""
    if before is None or before.step is None or now.step is None:
        return None
    if before.logged_at is not None and now.logged_at is not None:
        if now.logged_at <= before.logged_at:
            return None
        return (now.step - before.step) / (now.logged_at - before.logged_at)
    if now.at <= before.at:
        return None
    return (now.step - before.step) / (now.at - before.at)


def look_to_rate_from(history: Sequence[Look], now: Look, span_s: float = 1200.0) -> Look | None:
    """The look a line's rate is taken from: the oldest of the last twenty minutes that read a step, or else the
    last that did. W&B's step moves 200 at a time and some seconds late, so two looks a minute apart say little
    (2026-10-07: 2.49 steps a second from 200 steps in 80 seconds, in a run doing 4.5)."""
    read = [look for look in history if look.step is not None and look.at < now.at]
    recent = [look for look in read if now.at - look.at <= span_s]
    return recent[0] if recent else (read[-1] if read else None)


def finished(look: Look, steps: int = STEPS) -> bool:
    return look.state == "finished" and look.step is not None and look.step >= steps


def problems(history: Sequence[Look], now: Look, *, steps: int = STEPS, save_freq: int = SAVE_FREQ) -> list[str]:
    """What is wrong with a run at this look, given the looks before it (oldest first). Empty when nothing is."""
    said: list[str] = []
    recent = [*history, now]

    blind = 0
    for look in reversed(recent):
        if look.state is None:
            blind += 1
        else:
            break
    if blind >= BLIND_LOOKS:
        said.append(f"W&B could not be read {blind} looks in a row ({'; '.join(now.unread) or 'no reason given'}): "
                    "the watch is blind, which says nothing about the run")
    if now.state is None:
        return said

    if finished(now, steps):
        if now.checkpoints is not None and f"{steps:06d}" not in now.checkpoints:
            said.append(f"finished at step {now.step:,}, and the Hub does not hold checkpoints/{steps:06d} yet")
        return said
    if now.state != "running":
        said.append(f"W&B says the run is {now.state!r} at step {now.step if now.step is not None else '?'} of "
                    f"{steps:,}: the trainer has stopped")
        return said

    if now.heartbeat_age_s is not None and now.heartbeat_age_s > HEARTBEAT_LIMIT_S:
        said.append(f"no heartbeat for {now.heartbeat_age_s / 60:.0f} minutes (limit {HEARTBEAT_LIMIT_S / 60:.0f}): "
                    "W&B has stopped hearing from the trainer")

    read = [look for look in history if look.step is not None]
    if (read and now.step is not None and now.step < steps and now.at - read[-1].at >= 120
            and now.step <= read[-1].step):
        said.append(f"the step has not advanced in {(now.at - read[-1].at) / 60:.0f} minutes (still {now.step:,})")

    for name, value in now.losses.items():
        if not math.isfinite(value):
            said.append(f"{name} is {value}: not a number the trainer can go on from")
    before = [look.loss for look in read if look.loss is not None and math.isfinite(look.loss)][-LOSS_HISTORY:]
    if now.loss is not None and math.isfinite(now.loss) and len(before) >= LOSS_MIN_HISTORY:
        usual = statistics.median(before)
        if usual > 0 and now.loss > LOSS_JUMP * usual:
            said.append(f"the loss jumped to {now.loss:.4f}, {now.loss / usual:.1f} times the median of the "
                        f"{len(before)} looks before ({usual:.4f})")

    if now.step is not None and now.checkpoints is not None:
        due = [k for k in range(save_freq, steps + 1, save_freq) if now.step >= k + CHECKPOINT_GRACE_STEPS]
        missing = [f"{k:06d}" for k in due if f"{k:06d}" not in now.checkpoints]
        if missing:
            said.append(f"at step {now.step:,} the Hub does not hold checkpoints/{', '.join(missing)}")

    rates = []
    for earlier, later in zip(recent[:-1], recent[1:]):   # looks under two minutes apart say nothing of the rate
        rates.append(rate(earlier, later) if later.at - earlier.at >= 120 else None)
    last = [r for r in rates[-SLOW_LOOKS:]]
    if len(last) == SLOW_LOOKS and all(r is not None and r < SLOW_STEPS_PER_S for r in last):
        said.append(f"under {SLOW_STEPS_PER_S} steps a second for {SLOW_LOOKS} looks ({last[-1]:.2f} now; August's "
                    "average was 4.33): the run will end later than planned")
    return said


def new_checkpoints(before: Look | None, now: Look) -> list[str]:
    """Checkpoint folders the Hub holds now and did not at the last look that read it."""
    if now.checkpoints is None:
        return []
    had = set(before.checkpoints or []) if before is not None else set()
    return [name for name in now.checkpoints if name not in had]


def line(now: Look, steps_per_s: float | None, *, steps: int = STEPS) -> str:
    """One look as one line of the log. No account name in it."""
    when = dt.datetime.fromtimestamp(now.at).strftime("%Y-%m-%d %H:%M:%S")
    if now.state is None:
        return f"{when} | {now.layout:9} | not read: {'; '.join(now.unread) or 'no reason given'}"
    parts = [when, f"{now.layout:9}", f"{now.state}"]
    parts.append("heartbeat ?" if now.heartbeat_age_s is None else f"heartbeat {now.heartbeat_age_s:.0f} s ago")
    if now.step is None:
        parts.append("step ?")
    else:
        parts.append(f"step {now.step:,} of {steps:,} ({100 * now.step / steps:.1f} %)")
    parts.append("rate ?" if steps_per_s is None else f"{steps_per_s:.2f} steps/s")
    if steps_per_s and now.step is not None and steps_per_s > 0 and now.step < steps:
        parts.append(f"ends about {dt.datetime.fromtimestamp(now.at + (steps - now.step) / steps_per_s):%H:%M}")
    parts.append(", ".join(f"{name.split('/')[-1]} {value:.4f}" for name, value in sorted(now.losses.items()))
                 or "no loss read")
    parts.append("Hub: not read" if now.checkpoints is None
                 else f"Hub: {', '.join(now.checkpoints) if now.checkpoints else 'no checkpoint yet'}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------------------------------------------
# Reading. Everything below touches the network or the disk and is not what the tests judge.


def heartbeat_age(beat: str | None, now: float) -> float | None:
    """Seconds since a W&B heartbeat (`2026-10-07T23:01:26Z`), or None when the run has none."""
    if not beat:
        return None
    stamp = dt.datetime.strptime(beat[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return now - stamp.timestamp()


def read_wandb(run_name: str, key: str, now: float) -> dict[str, Any]:
    """A run's state, heartbeat, last logged step with W&B's clock for it, and losses."""
    import wandb

    run = wandb.Api(api_key=key, timeout=30).run(run_name)
    age = heartbeat_age(run._attrs.get("heartbeatAt"), now)
    summary = run.summary
    step, logged = summary.get("_step"), summary.get("_timestamp")
    losses = {}
    for name in sorted(summary.keys()):
        if "loss" in name.lower():
            try:
                losses[name] = float(summary[name])
            except (TypeError, ValueError):
                continue
    return {"state": run.state, "heartbeat_age_s": age, "step": None if step is None else int(step),
            "losses": losses, "logged_at": None if logged is None else float(logged)}


def read_hub(repository: str) -> list[str]:
    from huggingface_hub import HfApi

    return checkpoints_in(HfApi().list_repo_files(repository, repo_type="model"))


def alive(state: str | None, heartbeat_age_s: float | None) -> bool:
    """Whether a run is one to take up: W&B calls it running and has heard from it lately. W&B goes on calling a run
    running for some minutes after its machine is gone (2026-10-07: seven minutes after the last heartbeat)."""
    return state == "running" and heartbeat_age_s is not None and heartbeat_age_s <= HEARTBEAT_LIMIT_S


def newest_running(project: str, run_name: str, key: str, clock: Callable[[], float] = time.time) -> str | None:
    """The newest run of that name that is alive, as `entity/project/id`; None when there is none. For a launch from
    a browser, which leaves no launch output on this computer."""
    import wandb

    for run in wandb.Api(api_key=key, timeout=30).runs(project, filters={"display_name": run_name},
                                                       order="-created_at", per_page=10):
        if alive(run.state, heartbeat_age(run._attrs.get("heartbeatAt"), clock())):
            return f"{project}/{run.id}"
    return None


def look_at(layout: str, run_name: str, repository: str, key: str, *, clock: Callable[[], float] = time.time,
            wandb_reader: Callable[..., Any] = read_wandb, hub_reader: Callable[[str], list[str]] = read_hub) -> Look:
    now = Look(at=clock(), layout=layout, run=run_name.rsplit("/", 1)[-1])
    try:
        read = wandb_reader(run_name, key, now.at)
        now.state, now.heartbeat_age_s, now.step = read["state"], read["heartbeat_age_s"], read["step"]
        now.losses, now.logged_at = read["losses"], read.get("logged_at")
    except Exception as error:  # noqa: BLE001: a look that fails is a look that says so
        now.unread.append(f"W&B: {type(error).__name__}")
    try:
        now.checkpoints = hub_reader(repository)
    except Exception as error:  # noqa: BLE001
        now.unread.append(f"Hub: {type(error).__name__}")
    return now


def remember(look: Look, path: Path) -> None:
    """Append a look to the watch's own memory, so that a watch started again knows the looks before it."""
    with path.open("a") as out:
        out.write(json.dumps({"at": look.at, "layout": look.layout, "state": look.state,
                              "heartbeat_age_s": look.heartbeat_age_s, "step": look.step, "losses": look.losses,
                              "checkpoints": look.checkpoints, "unread": look.unread, "run": look.run,
                              "logged_at": look.logged_at}) + "\n")


def recall(path: Path, keep: int = 40) -> dict[str, list[Look]]:
    """The last looks of every layout from the watch's memory, oldest first. A line that does not read is left out."""
    looks: dict[str, list[Look]] = {}
    if not path.exists():
        return looks
    for text in path.read_text().splitlines():
        try:
            looks.setdefault(json.loads(text)["layout"], []).append(Look(**json.loads(text)))
        except (ValueError, TypeError, KeyError):
            continue
    return {layout: found[-keep:] for layout, found in looks.items()}


def of_this_run(looks: Sequence[Look], run_name: str) -> list[Look]:
    """The looks that belong to this run: a run started again begins at step 0, and the looks at the run before it
    would make it seem to have gone backwards."""
    run_id = run_name.rsplit("/", 1)[-1]
    return [look for look in looks if look.run == run_id]


def project_in(local: Path = LOCAL) -> str | None:
    """The W&B project of this week's runs, `entity/project`, from any launch output kept in the folder (the outputs
    of launches that were lost are kept under other names)."""
    newest = sorted(local.glob("launch-*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in newest:
        run_name = wandb_run_in(path.read_text(errors="replace"))
        if run_name is not None:
            return run_name.rsplit("/", 1)[0]
    return None


def runs_to_watch(local: Path = LOCAL, *, discover: Callable[[str], str | None] | None = None,
                  current: dict[str, tuple[str, str]] | None = None) -> dict[str, tuple[str, str]]:
    """For every layout that has been launched: its W&B run name and its policy repository.

    A run is named by `launch-<layout>.txt` when a launch from this computer left one. Otherwise `discover` is asked
    for the newest running run of the layout's name (a launch from a browser). A layout that was being watched
    (`current`) and for which neither names a run stays what it was: a run that has stopped is then still looked at,
    and called stopped."""
    found = dict(current or {})
    for layout in LAYOUTS:
        launch, parameters = local / f"launch-{layout}.txt", local / f"{layout}.params.json"
        if not parameters.exists():
            continue
        told = json.loads(parameters.read_text())
        run_name = wandb_run_in(launch.read_text(errors="replace")) if launch.exists() else None
        if run_name is None and discover is not None:
            run_name = discover(told["RUN_NAME"])
        if run_name is not None:
            found[layout] = (run_name, f"{told['NAMESPACE']}/{told['RUN_NAME']}")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=float, default=300.0, help="seconds between looks (default 300)")
    parser.add_argument("--once", action="store_true", help="one look, then stop")
    parser.add_argument("--keep-going", action="store_true", help="do not stop when a checkpoint arrives")
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--memory", type=Path, default=None,
                        help="the looks so far, one a line (default: beside the log, .jsonl)")
    args = parser.parse_args(argv)

    import os

    key = os.environ.get("WANDB_API_KEY") or (key_in(SECRET.read_text()) if SECRET.exists() else None)
    if not key:
        print("no W&B key: neither WANDB_API_KEY nor the earlier project's secrets/wandb.venv")
        return 4
    project = project_in()

    def discover(run_name: str) -> str | None:
        if project is None:
            return None
        try:
            return newest_running(project, run_name, key)
        except Exception:  # noqa: BLE001: W&B not answering is not a run to watch, and not a reason to stop
            return None

    args.log.parent.mkdir(parents=True, exist_ok=True)
    memory = args.memory or args.log.with_suffix(".jsonl")
    recalled = recall(memory)
    runs: dict[str, tuple[str, str]] = {}
    history: dict[str, list[Look]] = {}
    watched: dict[str, str] = {}
    waiting_said = False
    while True:
        try:
            runs = runs_to_watch(discover=discover, current=runs)
        except (OSError, ValueError, KeyError):  # a launch output or a parameter file half written: next look
            pass
        if not runs:
            if args.once:
                print("nothing to watch yet: no launch output names a run, and W&B shows none running")
                return 4
            if not waiting_said:
                print("nothing to watch yet: waiting for a running run to appear on W&B", flush=True)
                waiting_said = True
            time.sleep(min(args.every, 60.0))
            continue
        wrong, arrived = [], []
        for layout, (run_name, repository) in runs.items():
            if watched.get(layout) != run_name:
                watched[layout] = run_name
                history[layout] = of_this_run(recalled.get(layout, []) + history.get(layout, []), run_name)
                with args.log.open("a") as log:
                    log.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} | {layout:9} | the watch takes up run "
                              f"{run_name.rsplit('/', 1)[-1]} ({len(history[layout])} earlier looks at it)\n")
            now = look_at(layout, run_name, repository, key)
            text = line(now, rate(look_to_rate_from(history[layout], now), now))
            hub_before = next((look for look in reversed(history[layout]) if look.checkpoints is not None), None)
            fresh = new_checkpoints(hub_before, now) if hub_before is not None else []
            found = problems(history[layout], now)
            with args.log.open("a") as log:
                log.write(text + "\n")
                for name in fresh:
                    log.write(f"{text[:19]} | {layout:9} | NEW CHECKPOINT {name} on the Hub\n")
                for sentence in found:
                    log.write(f"{text[:19]} | {layout:9} | PROBLEM: {sentence}\n")
            print(text, flush=True)
            arrived += [f"{layout}: checkpoint {name} is on the Hub" for name in fresh]
            wrong += [f"{layout}: {sentence}" for sentence in found]
            history[layout].append(now)
            remember(now, memory)
        if wrong:
            print("\nSOMETHING IS WRONG:\n  " + "\n  ".join(wrong), flush=True)
            return 2
        if all(history[layout] and finished(history[layout][-1]) for layout in runs) and len(runs) == len(LAYOUTS):
            print("\nBOTH RUNS FINISHED", flush=True)
            return 0
        if arrived and not args.keep_going:
            print("\n" + "\n".join(arrived), flush=True)
            return 3
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
