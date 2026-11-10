"""Leaving a rig command for good.

Two things in the rig's environment keep a finished process from exiting by itself: the viewer's exit handler (met in
August, which is why the arm tooling's launchers end with a hard exit) and the threaded reader of dataset tables
(measured 2026-10-05: about half of such processes never exit). A session has already settled the arm and written its
records by the time it gets here, so the process is ended without running exit handlers.

Buffered output is flushed first. Skipping that step loses everything printed when the output is a pipe or a file.
"""

from __future__ import annotations

import logging
import os
import sys


def leave(code: int | None) -> None:
    """Flush what has been printed and logged, then end the process with `code` without running exit handlers."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    logging.shutdown()
    os._exit(int(code or 0))
