"""A tag's outline measured from the picture: four straight sides, each a line through sub-pixel edge positions.

Why this exists. The tag detector reports the four corners of a tag. At the rig's resolution (a sheet tag is about
32 pixels across, 4 pixels to a module) those corners sit about half a pixel inside the tag's real outline, and one
of them can jump by more than a pixel when the picture changes by a single grey level (measured on 2026-10-05:
`docs/journal.md`). Half a pixel is half a millimetre there. For telling where the camera looks that is harmless; for
measuring where the table is, it is an error of half a millimetre that need not be there. (It is not the largest
one: on the first sitting the sheet itself was not flat, and that cost more.)

A side of a tag is a long straight edge between the white margin and the black border. This module scans across each
side at many places, finds the edge on each scan to a fraction of a pixel, fits a straight line through those
positions, and takes the corners where neighbouring lines cross. A side of 30 pixels gives about 20 edge positions
instead of one 3 x 3 window per corner.

Where the edge is on one scan: the place where the white-equivalent length of the scan ends, that is, the sum of
(light - black) / (white - black) along the scan. For a pixel that the edge cuts in two this is exact, and a blur that
is the same on both sides of the edge does not move it. It must be done on light, not on stored grey levels: a camera
stores roughly the square root of the light, and the middle grey level then lies on the dark side of the true edge.
`linear_light` undoes the standard curve (sRGB). A camera's own curve differs a little from the standard; what that
leaves is a tag that reads a few tenths of a pixel larger or smaller on every side alike, which moves neither its
centre nor its shape.

NumPy only: no camera, no OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A tag is 8 modules across: 6 of data and a black border of one module on each side.
MODULES = 8
#: Scans are taken along this share of a side, away from the corners, where the neighbouring side's edge comes in.
SIDE_SHARE = (0.18, 0.82)
#: Distance between samples along a scan, in pixels.
SCAN_STEP_PX = 0.25
#: The white level is read this far outside the edge and the black level this far inside it (pixels), at most.
LEVEL_NEAR_PX = 1.0
LEVEL_FAR_PX = 2.5
#: The black border is one module wide: the black level is read no deeper than this share of a module.
BORDER_SHARE = 0.62
#: The first pass starts from the detector's corners, which may be this far off.
FIRST_REACH_PX = 2.5
#: An edge needs at least this contrast, as a share of its white level, to be measured.
MIN_CONTRAST = 0.3
#: A side needs this many edge positions.
MIN_POINTS = 6
#: A side whose edge positions scatter more than this about their line is not a clean straight edge.
MAX_SIDE_RMS_PX = 0.35
#: An outline further than this from the detector's corners is not the same tag.
MAX_MOVE_PX = 4.0
#: The white around a tag at or above this grey level is clipped: the camera is over-exposed there, the edge of the
#: black is eaten into, and the outline reads small.
CLIPPED_GREY = 252.0


def linear_light(gray: np.ndarray) -> np.ndarray:
    """Grey levels 0 to 255 as light between 0 and 1, undoing the sRGB curve."""
    v = np.asarray(gray, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The picture between its pixels: `x`, `y` are pixel-centre coordinates."""
    height, width = image.shape
    x0 = np.clip(np.floor(x).astype(int), 0, width - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, height - 2)
    fx = np.clip(x - x0, 0.0, 1.0)
    fy = np.clip(y - y0, 0.0, 1.0)
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
            + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)


@dataclass(frozen=True)
class Outline:
    """One tag's outline."""

    #: The four corners in pixels, in the detector's order.
    corners: np.ndarray
    #: How far each side's edge positions scatter about its line, root mean square, pixels.
    side_rms_px: tuple[float, ...]
    #: How many edge positions each side's line went through.
    side_points: tuple[int, ...]


def _edge_positions(light: np.ndarray, start: np.ndarray, end: np.ndarray, inward: np.ndarray, module_px: float,
                    reach_px: float) -> np.ndarray:
    """Sub-pixel positions of the white-to-black edge along the side from `start` to `end`, as an n x 2 array.

    `inward` is the unit normal that points into the tag; the true edge is taken to lie within `reach_px` of the side.
    """
    length = float(np.linalg.norm(end - start))
    inside_px = min(LEVEL_FAR_PX, BORDER_SHARE * module_px)
    scan = np.arange(-(reach_px + LEVEL_FAR_PX), reach_px + inside_px + 1e-9, SCAN_STEP_PX)
    shares = np.linspace(SIDE_SHARE[0], SIDE_SHARE[1], max(MIN_POINTS + 2, int(round(0.64 * length))))
    bases = start[None, :] + shares[:, None] * (end - start)[None, :]
    points = bases[:, None, :] + scan[None, :, None] * inward[None, None, :]
    height, width = light.shape
    if (points[..., 0].min() < 0.0 or points[..., 1].min() < 0.0 or points[..., 0].max() > width - 1.0
            or points[..., 1].max() > height - 1.0):
        return np.empty((0, 2))            # a scan would leave the picture: this side cannot be measured
    profiles = _bilinear(light, points[..., 0], points[..., 1])           # one row per scan, outside to inside
    found = []
    # The steepest fall of the light, looked for where both level windows still fit inside the scan.
    slope = np.gradient(profiles, SCAN_STEP_PX, axis=1)
    smooth = (slope[:, :-2] + 2 * slope[:, 1:-1] + slope[:, 2:]) / 4.0
    allowed = (scan[1:-1] >= -reach_px) & (scan[1:-1] <= reach_px)
    for base, profile, row in zip(bases, profiles, smooth, strict=True):
        steepest = float(scan[1:-1][allowed][int(np.argmin(row[allowed]))])
        outside = (scan >= steepest - LEVEL_FAR_PX) & (scan <= steepest - LEVEL_NEAR_PX)
        inside = (scan >= steepest + LEVEL_NEAR_PX) & (scan <= steepest + inside_px)
        if not outside.any() or not inside.any():
            continue
        white, black = float(np.median(profile[outside])), float(np.median(profile[inside]))
        if not (white > 0.0 and white - black >= MIN_CONTRAST * white):
            continue                   # no edge here to speak of (or nothing finite to measure)
        window = (scan >= steepest - LEVEL_FAR_PX) & (scan <= steepest + inside_px)
        first = float(scan[window][0])
        share = (profile[window] - black) / (white - black)
        edge = first - SCAN_STEP_PX / 2.0 + float(share.sum()) * SCAN_STEP_PX
        found.append(base + edge * inward)
    return np.array(found).reshape(-1, 2)


def _line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """A straight line through edge positions, with the ones far from it left out: point, direction, rms, count."""
    kept = points
    for _ in range(3):
        if len(kept) < MIN_POINTS:
            return None
        mean = kept.mean(axis=0)
        _, _, axes = np.linalg.svd(kept - mean)
        distance = (kept - mean) @ axes[1]
        limit = max(2.5 * 1.4826 * float(np.median(np.abs(distance))), 0.12)
        near = np.abs(distance) <= limit
        if near.all():
            break
        kept = kept[near]
    if len(kept) < MIN_POINTS:
        return None
    mean = kept.mean(axis=0)
    _, _, axes = np.linalg.svd(kept - mean)
    distance = (kept - mean) @ axes[1]
    if not np.all(np.isfinite(axes)):
        return None
    return mean, axes[0], float(np.sqrt(np.mean(distance**2))), len(kept)


def _crossing(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    (p1, d1), (p2, d2) = first, second
    system = np.array([d1, -d2]).T
    if abs(np.linalg.det(system)) < 1e-9:
        return None
    return p1 + np.linalg.solve(system, p2 - p1)[0] * d1


def tag_outline(light: np.ndarray, quad: np.ndarray, *, passes: int = 3) -> Outline | None:
    """The outline of the tag whose corners the detector reported as `quad`, or None when it cannot be measured.

    `light` is the picture as light (`linear_light`), `quad` the detector's four corners in pixels. The first pass
    looks for each side near the detector's; later passes start from the outline found so far, so a corner the
    detector misplaced by a pixel or two does not tilt the scans. None means that a side had too little contrast, too
    few clean edge positions, or was not straight: a tag in glare, partly covered, or too small.
    """
    start_quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    try:
        return _outline(light, start_quad, passes)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return None        # a region with nothing to measure in it (all one level, or not finite)


def _outline(light: np.ndarray, start_quad: np.ndarray, passes: int) -> Outline | None:
    current = start_quad
    result: Outline | None = None
    for attempt in range(passes):
        centre = current.mean(axis=0)
        sides_px = [float(np.linalg.norm(current[k] - current[(k + 1) % 4])) for k in range(4)]
        if min(sides_px) < 2.0 * MODULES:
            return None
        module_px = float(np.mean(sides_px)) / MODULES
        reach_px = FIRST_REACH_PX if attempt == 0 else 1.0
        lines = []
        for k in range(4):
            start, end = current[k], current[(k + 1) % 4]
            along = (end - start) / np.linalg.norm(end - start)
            inward = np.array([-along[1], along[0]])
            if float(np.dot(centre - start, inward)) < 0.0:
                inward = -inward
            fitted = _line(_edge_positions(light, start, end, inward, module_px, reach_px))
            if fitted is None:
                return None
            lines.append(fitted)
        corners = []
        for k in range(4):                         # corner k is where side k-1 meets side k
            crossing = _crossing(lines[(k - 1) % 4][:2], lines[k][:2])
            if crossing is None:
                return None
            corners.append(crossing)
        current = np.array(corners)
        if float(np.abs(current - start_quad).max()) > MAX_MOVE_PX:
            return None
        result = Outline(corners=current, side_rms_px=tuple(line[2] for line in lines),
                         side_points=tuple(line[3] for line in lines))
    if result is None or max(result.side_rms_px) > MAX_SIDE_RMS_PX:
        return None
    return result


def tag_outlines(gray: np.ndarray, found: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """The outline corners of every tag the detector found in one grey picture; a tag that cannot be measured is
    left out."""
    light = linear_light(gray)
    outlines = {}
    for tag_id, quad in found.items():
        measured = tag_outline(light, quad)
        if measured is not None:
            outlines[int(tag_id)] = measured.corners
    return outlines


def white_level(gray: np.ndarray, quad: np.ndarray) -> float:
    """The grey level (0 to 255) of the white around and inside a tag: the median of the brighter half of the pixels
    in the tag's box, widened by one module. At `CLIPPED_GREY` or above the camera is over-exposed there."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    module = float(np.mean([np.linalg.norm(quad[k] - quad[(k + 1) % 4]) for k in range(4)])) / MODULES
    height, width = gray.shape[:2]
    x0, y0 = np.floor(quad.min(axis=0) - module).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0) + module).astype(int) + 1
    box = np.asarray(gray, dtype=np.float64)[max(y0, 0):min(y1, height), max(x0, 0):min(x1, width)]
    if box.size == 0:
        return 0.0
    return float(np.median(box[box >= np.median(box)]))


def crossing_of_diagonals(quad: np.ndarray) -> np.ndarray:
    """Where a quadrilateral's diagonals cross: the picture of the tag's centre, whatever the perspective."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    crossing = _crossing((quad[0], quad[2] - quad[0]), (quad[1], quad[3] - quad[1]))
    return quad.mean(axis=0) if crossing is None else crossing


def inset_px(quad: np.ndarray, outline: np.ndarray) -> float:
    """How far, on average, the corners of `quad` lie inside `outline`, along the outline's diagonals, in pixels."""
    outline = np.asarray(outline, dtype=np.float64).reshape(4, 2)
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = crossing_of_diagonals(outline)
    return float(np.mean(np.linalg.norm(outline - centre, axis=1) - np.linalg.norm(quad - centre, axis=1)))
