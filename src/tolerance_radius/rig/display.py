"""Showing the placement picture from inside an arm session, without opening a window in that process.

An arm session already has a Rerun viewer open, running as its own process. Sending the placement picture there keeps
the session process free of a second window system beside its global key listener.

**Where the picture goes.** LeRobot lays the viewer out as one view per camera, rooted at the camera's own path
(`observation.front`, `observation.wrist`), plus the plots; nothing outside those roots is shown. So the picture is
logged as a child of the front camera's path, where that view picks it up, and drawn over the camera picture. When the
placement is over it is cleared, or it would hide the live camera for the rest of the session.

The picture goes on a timeline and compressed, because this Rerun version hangs on compressed pictures logged as static
and floods on raw ones (both met in August, both in the arm tooling's notes).

Display is guidance, never a gate: a failure here is logged once and then ignored, and the placement carries on from
its measurements and its spoken prompts.

**Seen in a real viewer once, and it did not work** (2026-10-06, the recorder's smoke test): after the first placement
the viewer's camera pictures stood still for the rest of the session. The picture was logged on a timeline of its own
then; that is gone, but nobody has looked at the viewer since. The sessions show the placement picture in a window of
its own by default (`PlacementPhase`'s own), and this class is what `--placement-view rerun` still gives.
"""

from __future__ import annotations

import logging
import time

import numpy as np

#: The front camera's path in the viewer, as LeRobot names it: the root of the view the operator looks at.
FRONT_VIEW = "observation.front"
ENTITY = f"{FRONT_VIEW}/placement"
TIMELINE = "placement"
#: Above the camera picture, which the library draws at its default of -10.
DRAW_ORDER = 10.0


class RerunDisplay:
    """`show(title, picture)`, `wait_key(ms)` and `clear()` for `PlacementPhase`, backed by the session's viewer."""

    def __init__(self, jpeg_quality: int = 70, sleep=time.sleep) -> None:
        self._jpeg_quality, self._sleep = jpeg_quality, sleep
        self._frame = 0
        self._failed = False

    def _give_up(self, error: Exception) -> None:
        self._failed = True
        logging.warning("placement picture not shown (%s: %s); carrying on without it", type(error).__name__, error)

    def show(self, _title: str, picture_bgr: np.ndarray) -> None:
        if self._failed:
            return
        try:
            import cv2
            import rerun as rr

            self._frame += 1
            # No timeline of its own. A timeline set here stays set for everything this thread logs afterwards,
            # the cameras' frames among them, and the viewer goes over to it: in the first session shown in a real
            # viewer (2026-10-06) the live cameras stood still from the first placement on. The picture goes on the
            # time everything else is logged on.
            # Encoded here and not by the library's `Image.compress`, which drops the draw order.
            done, jpeg = cv2.imencode(".jpg", picture_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            if not done:
                raise ValueError("the picture could not be encoded")
            rr.log(ENTITY, rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg", draw_order=DRAW_ORDER))
        except Exception as error:  # noqa: BLE001 -- guidance must never cost a session
            self._give_up(error)

    def clear(self) -> None:
        """Take the picture away, so that the live camera shows again in its view."""
        if self._failed:
            return
        try:
            import rerun as rr

            rr.log(ENTITY, rr.Clear(recursive=False))
        except Exception as error:  # noqa: BLE001
            self._give_up(error)

    def wait_key(self, ms: int) -> int:
        """No keys are read here: the session's own listener owns the keyboard. Just keep the loop's pace."""
        self._sleep(ms / 1000.0)
        return -1
