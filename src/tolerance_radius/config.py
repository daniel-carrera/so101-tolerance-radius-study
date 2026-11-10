"""The protocol's parameters, read from `config/protocol.toml`.

One file holds every number the planner and the analysis share, so a parameter cannot differ between the tool that ran
the trials and the script that analysed them. The file is returned as plain nested dictionaries: its sections and keys
are documented in the file itself.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = ("protocol", "table", "placement", "datasets", "staircase", "blocks", "analysis")

#: The repository root, found from this file's place in it (`src/tolerance_radius/config.py`).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "config" / "protocol.toml"
#: Overrides for pilot runs: only what differs from the protocol. See the file's own header.
PILOT_OVERLAY_PATH = REPO_ROOT / "config" / "pilot.toml"


class ConfigError(ValueError):
    """The protocol file is missing something the tools rely on."""


def load_protocol(path: str | Path | None = None, *, overlay: str | Path | None = None) -> dict[str, Any]:
    """Read the protocol file and check that its sections are present.

    `overlay` names a second file whose entries replace the protocol's: tables are merged key by key, anything else
    (a number, a string, a list) is replaced whole. It is how a pilot run changes a block's size without a second copy
    of everything else.
    """
    path = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    with open(path, "rb") as handle:
        protocol = tomllib.load(handle)
    if overlay is not None:
        with open(overlay, "rb") as handle:
            protocol = _merged(protocol, tomllib.load(handle))
    missing = [name for name in REQUIRED_SECTIONS if name not in protocol]
    if missing:
        raise ConfigError(f"{path}: missing sections {missing}")
    if "version" not in protocol["protocol"]:
        raise ConfigError(f"{path}: [protocol] needs a version")
    return protocol


def _merged(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """`base` with `changes` laid over it. Neither argument is modified."""
    result = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merged(result[key], value)
        else:
            result[key] = value
    return result
