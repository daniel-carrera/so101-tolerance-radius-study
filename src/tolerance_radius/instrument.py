"""Checking the measuring instrument against a physical scale.

The camera's reading of the cube's position is the instrument behind every distance in the study, so before any trial
it is compared with a ruler. The cube is moved along a straight scale in equal known steps and read at each stop; this
module turns those readings into errors: how far each measured step is from the true step, and how far the measured
total is from the true total.

A constant offset in the readings does not matter (it cancels in every difference); what is tested is whether a known
displacement on the table reads as that displacement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tolerance_radius.schema import Point


@dataclass(frozen=True, slots=True)
class ScaleCheck:
    """The readings of one run along the scale, compared with the true step."""

    true_step_mm: float
    #: Measured length of each step, in order.
    measured_steps_mm: tuple[float, ...]
    #: Measured minus true, for each step.
    step_errors_mm: tuple[float, ...]
    #: Straight-line distance from the first reading to the last, and what it should be.
    measured_span_mm: float
    true_span_mm: float
    #: Largest distance of any reading from the straight line through the first and last readings.
    max_off_line_mm: float

    @property
    def mean_abs_step_error_mm(self) -> float:
        return sum(abs(e) for e in self.step_errors_mm) / len(self.step_errors_mm)

    @property
    def max_abs_step_error_mm(self) -> float:
        return max(abs(e) for e in self.step_errors_mm)

    @property
    def span_error_mm(self) -> float:
        return self.measured_span_mm - self.true_span_mm

    @property
    def scale_error_percent(self) -> float:
        """How much the instrument stretches or shrinks distances along this run."""
        return 100.0 * self.span_error_mm / self.true_span_mm

    def passes(self, mean_abs_limit_mm: float, max_abs_limit_mm: float) -> bool:
        """The protocol's pass line: the mean step error and the worst step error both within their limits."""
        return self.mean_abs_step_error_mm <= mean_abs_limit_mm and self.max_abs_step_error_mm <= max_abs_limit_mm


def check_against_scale(readings: Sequence[Point], true_step_mm: float) -> ScaleCheck:
    """Compare readings taken at equal known steps along a straight scale with the true step."""
    if len(readings) < 2:
        raise ValueError("a scale check needs at least two readings")
    if true_step_mm <= 0:
        raise ValueError("the true step is positive")
    steps = tuple(readings[i].distance_to(readings[i + 1]) for i in range(len(readings) - 1))
    first, last = readings[0], readings[-1]
    span = first.distance_to(last)
    return ScaleCheck(
        true_step_mm=true_step_mm,
        measured_steps_mm=steps,
        step_errors_mm=tuple(step - true_step_mm for step in steps),
        measured_span_mm=span,
        true_span_mm=true_step_mm * (len(readings) - 1),
        max_off_line_mm=max((_distance_to_line(p, first, last) for p in readings), default=0.0) if span > 0 else 0.0,
    )


def scatter_mm(readings: Sequence[Point]) -> tuple[float, float]:
    """Repeatability of readings meant to be the same place: (root-mean-square, largest) distance from their mean."""
    if not readings:
        raise ValueError("no readings")
    mean = Point(sum(p.x_mm for p in readings) / len(readings), sum(p.y_mm for p in readings) / len(readings))
    distances = [p.distance_to(mean) for p in readings]
    return math.sqrt(sum(d * d for d in distances) / len(distances)), max(distances)


def _distance_to_line(point: Point, a: Point, b: Point) -> float:
    length = a.distance_to(b)
    cross = (b.x_mm - a.x_mm) * (a.y_mm - point.y_mm) - (a.x_mm - point.x_mm) * (b.y_mm - a.y_mm)
    return abs(cross) / length
