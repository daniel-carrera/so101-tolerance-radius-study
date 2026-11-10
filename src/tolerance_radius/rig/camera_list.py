"""Which cameras the computer offers, by name, without opening any of them.

The sessions name their two cameras by number: 0 is the front camera, 1 the wrist camera (the arm tooling's
`rig.py`). The numbers are positions in the list of cameras the system offers, so they belong to whatever is plugged
in at the moment. On 2026-10-06 the wrist camera did not start after the computer had been left alone (it was on the
USB bus and had not been configured), the list held the front camera and the laptop's own, and a recording session
took the laptop's camera for the wrist: number 1 was simply the next one in the list. The operator saw it in the
viewer and stopped before anything was recorded. Nothing in the session would have.

This reads the system's own list (`system_profiler SPCameraDataType`, which opens no camera and needs no permission)
and says which of the two cameras the study uses are in it. A missing wrist camera is certain to be wrong. The order
of the numbers is *inferred*: OpenCV's macOS backend sorts the cameras by their unique identifier, which for a USB
camera begins with the port it is plugged into; that held in every session of 2026-10-06. A session refuses on
either, a camera that is missing and a camera at another number (`check_for_session`; `--any-cameras` goes on all the
same): two cameras at each other's numbers were the second fault of that evening. A list that cannot be read stops
nothing.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

#: The two cameras of this rig, as the system names them (*read* from its list on 2026-10-06).
FRONT_NAME = "HD Pro Webcam C920"
WRIST_NAME = "USB2.0_CAM1"


class CameraListError(RuntimeError):
    """The system's list of cameras could not be read."""


@dataclass(frozen=True, slots=True)
class SystemCamera:
    name: str
    #: What the cameras are sorted by; kept out of anything printed or logged (it is the machine's own).
    unique_id: str


def parse(listing_json: str) -> list[SystemCamera]:
    """The cameras in `system_profiler SPCameraDataType -json`'s output, in the order the sessions number them."""
    try:
        entries = json.loads(listing_json).get("SPCameraDataType", [])
    except (json.JSONDecodeError, AttributeError) as error:
        raise CameraListError(f"the system's camera list could not be read: {error}") from error
    cameras = [SystemCamera(str(entry.get("_name", "?")), str(entry.get("spcamera_unique-id", "")))
               for entry in entries if isinstance(entry, dict)]
    return sorted(cameras, key=lambda camera: camera.unique_id)


def system_cameras(timeout_s: float = 10.0) -> list[SystemCamera]:
    """The cameras the system offers now. Opens none of them."""
    try:
        done = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True,
                              timeout=timeout_s, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CameraListError(f"the system's camera list could not be read: {error}") from error
    if done.returncode != 0:
        raise CameraListError(f"the system's camera list could not be read (exit {done.returncode})")
    return parse(done.stdout)


def problems(cameras: Sequence[SystemCamera], front_index: int, wrist_index: int,
             front_name: str = FRONT_NAME, wrist_name: str = WRIST_NAME) -> tuple[list[str], list[str]]:
    """What is wrong with the list for a session that takes `front_index` and `wrist_index`: (certain, likely).

    Certain: a camera of the study is not in the list at all, so its number is another camera's or nobody's.
    Likely: both are there and the order the numbers are taken to follow puts another camera at a number.
    """
    names = [camera.name for camera in cameras]
    certain, likely = [], []
    for role, name, index in (("front", front_name, front_index), ("wrist", wrist_name, wrist_index)):
        if name not in names:
            taken = f"camera {index} is now '{names[index]}'" if index < len(names) else f"there is no camera {index}"
            certain.append(f"the {role} camera ('{name}') is not among the cameras the system offers: {taken}. "
                           "Is it plugged in, and has the system started it?")
        elif index >= len(names) or names[index] != name:
            at = names.index(name)
            likely.append(f"the {role} camera ('{name}') looks to be camera {at}, and the session takes camera "
                          f"{index} for it")
    return certain, likely


def lines(cameras: Sequence[SystemCamera], front_index: int, wrist_index: int,
          front_name: str = FRONT_NAME, wrist_name: str = WRIST_NAME) -> list[str]:
    """The list as the operator reads it: one line per camera, in the sessions' order, with what each is taken for."""
    told = []
    for index, camera in enumerate(cameras):
        role = {front_index: "front", wrist_index: "wrist"}.get(index)
        right = {front_index: front_name, wrist_index: wrist_name}.get(index)
        if role is None:
            note = "not used"
        elif camera.name == right:
            note = f"the session's {role} camera"
        else:
            note = f"the session would take this for the {role} camera: WRONG, that is '{right}'"
        told.append(f"camera {index}: {camera.name}  ({note})")
    if not cameras:
        told.append("the system offers no camera")
    return told


def check_for_session(front_index: int, wrist_index: int) -> tuple[list[str], str | None]:
    """What a session writes into its log about the cameras, and the sentence to refuse with.

    The sentence is None when both cameras of the study sit at the numbers the session takes. It is also None when
    the system's list cannot be read: the lines then say that the cameras were not checked by name, and the
    session's own checks of the pictures go on as before.
    """
    try:
        cameras = system_cameras()
    except CameraListError as error:
        return [f"cameras by name: not checked ({error})"], None
    told = [f"cameras by name: {line}" for line in lines(cameras, front_index, wrist_index)]
    certain, likely = problems(cameras, front_index, wrist_index)
    return told, "; ".join(certain + likely) or None
