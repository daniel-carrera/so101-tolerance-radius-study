"""Tools that run at the arm.

Everything that opens a camera, opens a serial port or commands a motor lives here and nowhere else. These modules wrap
the gated primitives of the earlier SO-101 project (see `lerobot101`) and never re-implement a control loop.

The core of the package does not import this subpackage, so the core stays importable and testable with no hardware and
no arm tooling present.
"""
