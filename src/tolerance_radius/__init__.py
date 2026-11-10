"""Measuring how far from its demonstrations an imitation policy still works.

The package has two halves that never import each other in the wrong direction:

* the core (`schema`, `config`, geometry, layouts, the planner, the statistics): pure Python and NumPy, no hardware, so
  it imports and is tested anywhere;
* `tolerance_radius.rig`: the tools that run at the arm. They wrap the gated primitives of the existing SO-101 tooling
  and are the only modules allowed to open a camera or talk to a motor.

Nothing in the core imports `tolerance_radius.rig`.
"""

__version__ = "0.1.0"
