"""The count of USB transfer errors on the cameras' links (`tolerance_radius.usb_errors`): the log's lines as the
system wrote them on 2026-10-06, with the times and ports of that evening."""

from __future__ import annotations

import unittest

from tolerance_radius import usb_errors

FRONT, WRIST = "HD Pro Webcam C920", "USB2.0_CAM1"


def error(clock: str, name: str, port: str) -> str:
    return (f"2026-10-06 {clock} E  kernel[0:1d0114c] (IOUSBHostFamily) AppleUSBIORequest: AppleUSBIORequest::complete: "
            f"device 3 ({name}@{port}) endpoint 0x81: status 0xe00002ed (transaction error)")


LINES = [
    "Timestamp               Ty Process[PID:TID]",
    error("20:52:45.144", FRONT, "00110000"),
    "2026-10-06 20:52:45.144 Df UVCAssistant[535:1d10932] [com.apple.UVCFamily:device] <UVCPipeStatusMonitor: 0x7f> "
    "scheduled action tigger block 0x7fcc06450",
    error("20:52:45.375", FRONT, "00110000"),
    error("20:52:59.870", FRONT, "00110000"),
    error("20:53:01.058", FRONT, "00110000"),
    error("20:53:20.997", WRIST, "02100000"),
    error("21:10:03.100", WRIST, "02100000"),
    # another device's error is not a camera's
    error("21:10:04.100", "USB Single Serial", "01110000"),
    "",
]


class UsbErrorTests(unittest.TestCase):
    def test_errors_are_counted_by_camera_and_minute(self):
        counts = usb_errors.per_minute(LINES, [FRONT, WRIST])
        self.assertEqual(dict(counts[FRONT]), {"20:52": 3, "20:53": 1})
        self.assertEqual(dict(counts[WRIST]), {"20:53": 1, "21:10": 1})

    def test_a_camera_without_a_line_has_none_and_other_devices_are_left_out(self):
        counts = usb_errors.per_minute([LINES[0], LINES[2], LINES[-2]], [FRONT, WRIST])
        self.assertEqual((sum(counts[FRONT].values()), sum(counts[WRIST].values())), (0, 0))
        self.assertNotIn("USB Single Serial", counts)

    def test_the_ports_a_camera_was_in_are_given(self):
        moved = [*LINES, error("21:40:00.000", FRONT, "01130000")]
        self.assertEqual(usb_errors.ports(moved, [FRONT, WRIST]),
                         {FRONT: {"00110000", "01130000"}, WRIST: {"02100000"}})

    def test_a_session_s_sentence_names_each_camera_s_errors_and_never_fails(self):
        from unittest import mock

        names = {FRONT: "front camera", WRIST: "wrist camera"}
        asked = []
        with mock.patch.object(usb_errors, "system_log", lambda minutes, timeout_s=0.0: asked.append(minutes) or LINES):
            told = usb_errors.over_session(1000.0, 1125.0, names)       # the session's own two readings of the clock
        self.assertEqual(asked, [4], "the session's two minutes and five seconds, rounded up, and one more")
        self.assertEqual(told, "USB transfer errors on the cameras' links in the last 4 minutes: front camera 4, "
                               "wrist camera 2")

        def broken(minutes, timeout_s=0.0):
            raise usb_errors.LogError("the system's log could not be read: no such program")

        with mock.patch.object(usb_errors, "system_log", broken):
            self.assertIn("not read (the system's log could not be read", usb_errors.over_session(1000.0, 1001.0, names))

    def test_the_errors_since_a_moment_are_counted_for_one_device(self):
        self.assertEqual(usb_errors.since(LINES, WRIST, "2026-10-06 00:00:00"), 2)
        self.assertEqual(usb_errors.since(LINES, FRONT, "2026-10-06 00:00:00"), 4)
        self.assertEqual(usb_errors.since(LINES, WRIST, "2099-01-01 00:00:00"), 0)
        times = sorted(line[:19] for line in LINES if WRIST in line)
        self.assertEqual(usb_errors.since(LINES, WRIST, times[-1]), 1 + sum(t == times[-1] for t in times[:-1]))
        self.assertEqual(usb_errors.since(["not a line of the log"], WRIST, "2026-10-06 00:00:00"), 0)

    def test_a_log_command_that_failed_is_not_read_as_no_errors(self):
        # An answer that failed is not an empty answer: "no errors" would be the wrong thing to say of it (the
        # review of the night of 2026-10-06 took the check of the exit code away, and no test said so).
        from types import SimpleNamespace
        from unittest import mock

        asked = []

        def run(command, **kwargs):
            asked.append(command)
            return SimpleNamespace(returncode=64, stdout="", stderr="log: bad predicate")

        with mock.patch.object(usb_errors.subprocess, "run", run):
            with self.assertRaisesRegex(usb_errors.LogError, "exit 64"):
                usb_errors.system_log(3)
            told = usb_errors.over_session(1000.0, 1100.0, {FRONT: "front camera", WRIST: "wrist camera"})
        self.assertIn("not read", told)
        self.assertNotIn("front camera 0", told)
        self.assertEqual(asked[0][:4], ["/usr/bin/log", "show", "--last", "3m"])

        def gone(command, **kwargs):
            raise FileNotFoundError("/usr/bin/log")

        with mock.patch.object(usb_errors.subprocess, "run", gone), self.assertRaises(usb_errors.LogError):
            usb_errors.system_log(3)
        ok = lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="\n".join(LINES), stderr="")  # noqa: E731
        with mock.patch.object(usb_errors.subprocess, "run", ok):
            self.assertEqual(usb_errors.system_log(3), "\n".join(LINES).splitlines())

    def test_the_question_asked_of_the_log_is_for_transfer_errors_of_the_kernel(self):
        self.assertIn('process == "kernel"', usb_errors.PREDICATE)
        self.assertIn("transaction error", usb_errors.PREDICATE)


if __name__ == "__main__":
    unittest.main()
