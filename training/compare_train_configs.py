#!/usr/bin/env python3
"""Check that two training runs used the same settings, from their `train_config.json` files.

    python training/compare_train_configs.py FIRST/train_config.json SECOND/train_config.json

LeRobot saves the full configuration of a run beside every checkpoint. This command reads two of those files and
prints every key whose value differs. A key in `ALLOWED_TO_DIFFER` says where a run reads and writes, not how it
trains, and may differ; any other difference makes the command fail. It is the check behind the sentence "the two
policies were trained with identical settings".

Exit status
    0  nothing differs outside the allowed keys
    1  at least one other key differs, or a `--must-differ` key is the same in both files
    2  a file is missing, is not JSON, or is not a configuration

Options
    --allow KEY        also let KEY differ, with everything under it (repeatable). Use it to compare this week's
                       configuration with an earlier one, naming each setting that was changed on purpose.
    --must-differ KEY  fail if KEY has the same value in both files (repeatable). `--must-differ dataset.repo_id`
                       catches a file compared with itself, and two runs started on the same dataset.
    --show-allowed     also print the values of the keys in `ALLOWED_TO_DIFFER`. They hold repository names: by
                       default they are hidden, so that the output can be pasted into a record that must not
                       name anyone. The values of a key allowed with `--allow` are always printed.

A nested setting is named by its full key (`optimizer.lr`). A list is one value: the same items in another order
differ. `80000` and `80000.0` are the same setting; `true` and `1` are not.

Standard library only, so it runs at the rig as it is. It needs Python 3.10 or newer; it is tested with 3.12.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Keys that may differ between two runs with identical settings: they name the run's inputs and outputs.
ALLOWED_TO_DIFFER: tuple[str, ...] = (
    "dataset.repo_id",  # the dataset repository
    "output_dir",  # the output directory on the training machine
    "job_name",  # the job name
    "policy.repo_id",  # the policy repository
    "wandb.entity",  # Weights & Biases: the account, the project and the run the curves are logged to
    "wandb.project",
    "wandb.run_id",
)

KeyPath = tuple[str, ...]
_ABSENT = object()


@dataclass(frozen=True)
class Difference:
    """One key whose value is not the same in the two files."""

    key: str
    first: Any
    second: Any


@dataclass(frozen=True)
class Report:
    """What comparing two configurations found."""

    unexpected: tuple[Difference, ...]
    allowed: tuple[Difference, ...]
    same_but_must_differ: tuple[str, ...]
    keys_compared: int

    @property
    def passed(self) -> bool:
        """True when the two runs differ only where they are allowed to, and wherever they are required to."""
        return not self.unexpected and not self.same_but_must_differ


def flatten(config: Mapping[str, Any], prefix: KeyPath = ()) -> dict[KeyPath, Any]:
    """Turn a nested configuration into one entry per setting, keyed by the path of names that leads to it.

    A group of settings (a JSON object) is walked into; anything else is a value, including a list and an empty
    group. Paths are tuples, so a name that contains a dot cannot be mistaken for a nested group.
    """
    flat: dict[KeyPath, Any] = {}
    for name, value in config.items():
        path = (*prefix, str(name))
        if isinstance(value, Mapping) and value:
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def same_value(first: Any, second: Any) -> bool:
    """Whether two JSON values are the same setting: numbers by value, booleans only with booleans, lists in order."""
    if isinstance(first, bool) or isinstance(second, bool):
        return isinstance(first, bool) and isinstance(second, bool) and first == second
    if isinstance(first, int | float) and isinstance(second, int | float):
        return first == second
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(same_value(a, b) for a, b in zip(first, second, strict=True))
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(same_value(first[name], second[name]) for name in first)
    return type(first) is type(second) and first == second


def is_allowed(key: str, allowed: Iterable[str]) -> bool:
    """Whether a dotted key is one of the allowed keys, or lies under one of them."""
    return any(key == entry or key.startswith(entry + ".") for entry in allowed)


def compare(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    allowed: Iterable[str] = ALLOWED_TO_DIFFER,
    must_differ: Iterable[str] = (),
) -> Report:
    """Compare two configurations key by key and sort the differences into allowed and unexpected."""
    allowed = tuple(allowed)
    flat_first, flat_second = flatten(first), flatten(second)
    paths = sorted(set(flat_first) | set(flat_second))
    unexpected: list[Difference] = []
    permitted: list[Difference] = []
    for path in paths:
        value_first, value_second = flat_first.get(path, _ABSENT), flat_second.get(path, _ABSENT)
        if value_first is not _ABSENT and value_second is not _ABSENT and same_value(value_first, value_second):
            continue
        difference = Difference(".".join(path), value_first, value_second)
        (permitted if is_allowed(difference.key, allowed) else unexpected).append(difference)
    differing = [d.key for d in (*unexpected, *permitted)]
    same = [key for key in must_differ if not any(is_allowed(other, [key]) for other in differing)]
    return Report(tuple(unexpected), tuple(permitted), tuple(same), len(paths))


def _show(value: Any) -> str:
    return "<absent>" if value is _ABSENT else json.dumps(value)


def format_report(report: Report, show_allowed: bool = False) -> list[str]:
    """The lines to print: unexpected differences first, then the allowed ones, then the verdict.

    The values of the keys in `ALLOWED_TO_DIFFER` are hidden unless `show_allowed` is set: they are repository
    names. A key allowed with `--allow` is a setting changed on purpose, and its two values are always printed.
    """
    lines = [f"DIFFERS  {d.key}: {_show(d.first)} | {_show(d.second)}" for d in report.unexpected]
    for key in report.same_but_must_differ:
        lines.append(f"SAME     {key}: must differ between these two files, and does not")
    for d in report.allowed:
        hidden = is_allowed(d.key, ALLOWED_TO_DIFFER) and not show_allowed
        values = " (values not shown)" if hidden else f": {_show(d.first)} | {_show(d.second)}"
        lines.append(f"allowed  {d.key}{values}")
    if report.unexpected:
        lines.append(f"RESULT: {len(report.unexpected)} key(s) differ outside the allowed list.")
    elif report.same_but_must_differ:
        lines.append(f"RESULT: {len(report.same_but_must_differ)} key(s) that must differ are the same.")
    else:
        note = "" if report.allowed else " Is this one run compared with itself?"
        lines.append(
            f"RESULT: identical apart from {len(report.allowed)} allowed key(s), "
            f"{report.keys_compared} key(s) compared.{note}"
        )
    return lines


def load(path: str) -> dict[str, Any]:
    """Read one configuration file; raise ValueError with a plain reason when it is not one."""
    try:
        with open(path, encoding="utf-8") as handle:
            content = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(content, dict):
        raise ValueError(f"{path} is not a JSON object, so it is not a training configuration")
    if not content:
        raise ValueError(f"{path} holds no settings")
    return content


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison from the command line and return the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Allowed to differ by default: " + ", ".join(ALLOWED_TO_DIFFER),
    )
    parser.add_argument("first", help="a train_config.json")
    parser.add_argument("second", help="the train_config.json to compare it with")
    parser.add_argument("--allow", action="append", default=[], metavar="KEY", help="also let this key differ")
    parser.add_argument(
        "--must-differ", action="append", default=[], metavar="KEY", help="fail if this key is the same in both"
    )
    parser.add_argument("--show-allowed", action="store_true", help="print the values of the allowed keys")
    args = parser.parse_args(argv)

    try:
        first, second = load(args.first), load(args.second)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    report = compare(first, second, allowed=(*ALLOWED_TO_DIFFER, *args.allow), must_differ=args.must_differ)
    print(f"first:   {args.first}")
    print(f"second:  {args.second}")
    print("allowed to differ: " + ", ".join(ALLOWED_TO_DIFFER))
    if args.allow:
        print("also allowed here: " + ", ".join(args.allow))
    for line in format_report(report, show_allowed=args.show_allowed):
        print(line)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
