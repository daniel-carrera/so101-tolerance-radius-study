"""The success curve and its grid posterior: the model, the fit on easy and on awkward data, and what it reports."""

import decimal
import math
import unittest
import warnings
from unittest import mock

import numpy as np

from tolerance_radius import config, psychometric
from tolerance_radius.psychometric import (
    ThresholdPosterior,
    ThresholdSummary,
    distance_at_probability,
    fit_threshold,
    grids_from_protocol,
    log_likelihood_grid,
    success_probability,
)

try:  # SciPy is a cross-check only. The rig's interpreter has none, and those tests then skip.
    from scipy import special as scipy_special
    from scipy import stats as scipy_stats
except ImportError:
    scipy_special = scipy_stats = None

# The grids of protocol version 0.1, written out so that these tests do not move when the protocol file does.
R50_GRID_MM = np.arange(0.0, 80.25, 0.5)
WIDTH_GRID_MM = np.geomspace(0.5, 20.0, 30)
LAPSE_VALUES = np.array([0.0, 0.02, 0.05, 0.10])
GRIDS = {"r50_grid_mm": R50_GRID_MM, "width_grid_mm": WIDTH_GRID_MM, "lapse_values": LAPSE_VALUES}
# The same with a floor in the model: a chance of success that remains however far the cube is.
FLOOR_VALUES = np.array([0.0, 0.05, 0.10])
FLOOR_GRIDS = GRIDS | {"floor_values": FLOOR_VALUES}


def draw_trials(
    seed: int, n: int, r50_mm: float, width_mm: float, lapse: float, max_mm: float = 60.0, floor: float = 0.0
):
    """Simulated trials at distances spread evenly at random from 0 to `max_mm`."""
    rng = np.random.default_rng(seed)
    distances = rng.uniform(0.0, max_mm, size=n)
    successes = rng.random(n) < success_probability(distances, r50_mm, width_mm, lapse, floor)
    return distances, successes


def repeat(levels_mm, times: int) -> list[float]:
    return [float(level) for level in levels_mm for _ in range(times)]


def direct_log_likelihood(
    distances, successes, r50_mm: float, width_mm: float, lapse: float, floor: float = 0.0
) -> float:
    """The log-likelihood written straight from the model, one trial at a time, with the standard library."""
    total = 0.0
    for distance, success in zip(distances, successes, strict=True):
        p = floor + (1.0 - lapse - floor) / (1.0 + math.exp((distance - r50_mm) / width_mm))
        total += math.log(p) if success else math.log(1.0 - p)
    return total


def sixty_digit_log_likelihood(distances, successes, r50_mm, width_mm, lapse, floor) -> float:
    """The log-likelihood in 60-digit decimal arithmetic, for any distance and width.

    The two halves of the curve, `1 / (1 + exp(z))` and `1 / (1 + exp(-z))`, are computed separately, so that neither
    is the difference of two nearly equal numbers; success and failure are then sums of positive terms.
    """
    exact = decimal.Decimal
    with decimal.localcontext() as context:
        context.prec = 60
        lapse, floor = exact(float(lapse)), exact(float(floor))
        total = exact(0)
        for distance, success in zip(distances, successes, strict=True):
            z = (exact(float(distance)) - exact(float(r50_mm))) / exact(float(width_mm))
            near, far = 1 / (1 + z.exp()), 1 / (1 + (-z).exp())
            probability = (1 - lapse) * near + floor * far if success else lapse * near + (1 - floor) * far
            total += probability.ln()
        return float(total)


class SuccessProbabilityTests(unittest.TestCase):
    def test_success_is_half_the_plateau_at_r50(self):
        self.assertAlmostEqual(float(success_probability(20.0, 20.0, 3.0)), 0.5, places=15)
        self.assertAlmostEqual(float(success_probability(20.0, 20.0, 3.0, lapse=0.1)), 0.45, places=15)

    def test_it_falls_from_the_plateau_to_zero_as_distance_grows(self):
        distances = np.linspace(0.0, 80.0, 161)
        curve = success_probability(distances, 40.0, 2.0, lapse=0.05)
        self.assertTrue(np.all(np.diff(curve) < 0))
        self.assertAlmostEqual(curve[0], 0.95, places=8)
        self.assertAlmostEqual(curve[-1], 0.0, places=8)

    def test_width_is_the_scale_of_the_fall(self):
        # Three quarters of the plateau at r50 - width * ln 3, one quarter at r50 + width * ln 3.
        for width in (0.5, 4.0, 20.0):
            with self.subTest(width=width):
                near, far = 30.0 - width * math.log(3.0), 30.0 + width * math.log(3.0)
                self.assertAlmostEqual(float(success_probability(near, 30.0, width, 0.02)), 0.75 * 0.98, places=12)
                self.assertAlmostEqual(float(success_probability(far, 30.0, width, 0.02)), 0.25 * 0.98, places=12)

    def test_huge_arguments_neither_overflow_nor_warn(self):
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            far = success_probability(np.array([1e3, 1e9, 1e300]), 10.0, 1e-6)
            near = success_probability(0.0, np.array([1e3, 1e9, 1e300]), 1e-6, lapse=0.1)
        self.assertTrue(np.array_equal(far, [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(near, [0.9, 0.9, 0.9], rtol=0, atol=1e-15))

    def test_the_far_tail_is_right_even_where_exp_alone_would_overflow(self):
        # 720 widths beyond r50 the probability is exp(-720): about 2e-313, tiny but not zero. exp(720) overflows.
        tail = float(success_probability(720.0, 0.0, 1.0))
        self.assertGreater(tail, 0.0)
        self.assertAlmostEqual(tail / math.exp(-720.0), 1.0, places=6)

    def test_the_result_has_the_broadcast_shape_of_its_arguments(self):
        self.assertEqual(success_probability(np.zeros((3, 4)), 10.0, 2.0).shape, (3, 4))
        self.assertEqual(success_probability(np.zeros((3, 1)), np.array([10.0, 20.0]), 2.0).shape, (3, 2))
        self.assertEqual(success_probability(5.0, 10.0, 2.0).shape, ())
        self.assertIsInstance(success_probability([1.0, 2.0], 10.0, 2.0), np.ndarray)

    def test_impossible_parameters_are_refused(self):
        for width, lapse in ((0.0, 0.0), (-1.0, 0.0), (2.0, -0.01), (2.0, 1.0)):
            with self.subTest(width=width, lapse=lapse), self.assertRaises(ValueError):
                success_probability(5.0, 10.0, width, lapse)

    def test_a_floor_is_what_remains_far_out_and_r50_stays_the_halfway_point(self):
        distances = np.linspace(0.0, 80.0, 161)
        curve = success_probability(distances, 30.0, 2.0, lapse=0.05, floor=0.08)
        self.assertTrue(np.all(np.diff(curve) < 0))
        self.assertAlmostEqual(curve[0], 0.95, places=6)  # the plateau is still 1 - lapse
        self.assertAlmostEqual(curve[-1], 0.08, places=8)  # and the far end is the floor
        self.assertAlmostEqual(float(success_probability(30.0, 30.0, 2.0, 0.05, 0.08)), (0.95 + 0.08) / 2, places=15)
        # Three quarters of the way from floor to plateau at r50 - width * ln 3, one quarter at r50 + width * ln 3.
        near, far = 30.0 - 2.0 * math.log(3.0), 30.0 + 2.0 * math.log(3.0)
        self.assertAlmostEqual(float(success_probability(near, 30.0, 2.0, 0.05, 0.08)), 0.08 + 0.75 * 0.87, places=12)
        self.assertAlmostEqual(float(success_probability(far, 30.0, 2.0, 0.05, 0.08)), 0.08 + 0.25 * 0.87, places=12)
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            self.assertEqual(float(success_probability(1e300, 10.0, 1e-6, 0.05, 0.08)), 0.08)

    def test_without_a_floor_the_curve_is_bit_for_bit_what_it_was(self):
        distances = np.linspace(0.0, 80.0, 161)
        for lapse in (0.0, 0.05):
            with self.subTest(lapse=lapse):
                plain = (1.0 - lapse) * np.exp(-np.logaddexp(0.0, (distances - 22.0) / 3.0))
                self.assertTrue(np.array_equal(success_probability(distances, 22.0, 3.0, lapse), plain))
                self.assertTrue(np.array_equal(success_probability(distances, 22.0, 3.0, lapse, floor=0.0), plain))

    def test_a_floor_that_cannot_be_is_refused(self):
        for lapse, floor in ((0.0, -0.01), (0.0, 1.0), (0.5, 0.5), (0.3, 0.8)):
            with self.subTest(lapse=lapse, floor=floor), self.assertRaises(ValueError):
                success_probability(5.0, 10.0, 2.0, lapse, floor)


class DistanceAtProbabilityTests(unittest.TestCase):
    """The model read the other way: the distance at which success has a given probability."""

    def test_it_undoes_the_model(self):
        for r50, width, lapse in ((20.0, 3.0, 0.0), (5.0, 10.0, 0.05), (40.0, 1.0, 0.1)):
            plateau = 1.0 - lapse
            probabilities = plateau * np.array([0.001, 0.1, 0.25, 0.5, 0.9, 0.999])
            with self.subTest(r50=r50, width=width, lapse=lapse):
                distances = distance_at_probability(probabilities, r50, width, lapse)
                back = success_probability(distances, r50, width, lapse)
                self.assertTrue(np.allclose(back, probabilities, rtol=1e-12))
                self.assertTrue(np.all(np.diff(distances) < 0))  # a higher probability is found closer in
                self.assertAlmostEqual(float(distance_at_probability(plateau / 2.0, r50, width, lapse)), r50, places=12)

    def test_no_distance_is_as_good_as_the_plateau_and_none_is_hopeless(self):
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            distances = distance_at_probability(np.array([0.0, 0.9, 0.95, 1.0]), 20.0, 3.0, lapse=0.1)
        self.assertEqual(distances[0], np.inf)  # success with probability 0 is found only infinitely far out
        self.assertTrue(np.all(distances[1:] == -np.inf))  # the plateau is 0.9: it is reached at no distance

    def test_a_trial_succeeds_exactly_when_it_falls_short_of_its_own_limit(self):
        # Drawing a uniform number u and calling the trial a success when u < P(success at d) is the same as calling
        # it a success when d < the distance at which P(success) = u, because success falls with distance.
        rng = np.random.default_rng(17)
        distances = rng.uniform(0.0, 60.0, size=20000)
        uniforms = rng.random(20000)
        for r50, width, lapse in ((20.0, 3.0, 0.05), (10.0, 10.0, 0.0), (30.0, 1.0, 0.1)):
            with self.subTest(r50=r50, width=width, lapse=lapse):
                by_probability = uniforms < success_probability(distances, r50, width, lapse)
                by_limit = distances < distance_at_probability(uniforms, r50, width, lapse)
                self.assertTrue(np.array_equal(by_probability, by_limit))
                self.assertTrue(0.05 < by_probability.mean() < 0.95)

    def test_with_a_floor_it_still_undoes_the_model_and_a_number_under_the_floor_always_succeeds(self):
        r50, width, lapse, floor = 20.0, 3.0, 0.05, 0.08
        probabilities = floor + (1.0 - lapse - floor) * np.array([0.001, 0.25, 0.5, 0.9, 0.999])
        distances = distance_at_probability(probabilities, r50, width, lapse, floor)
        back = success_probability(distances, r50, width, lapse, floor)
        self.assertTrue(np.allclose(back, probabilities, rtol=1e-12))
        self.assertAlmostEqual(float(distances[2]), r50, places=12)
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            ends = distance_at_probability(np.array([0.0, 0.03, 0.08, 0.95, 0.99]), r50, width, lapse, floor)
        self.assertTrue(np.all(ends[:3] == np.inf))  # at or under the floor: success at any distance
        self.assertTrue(np.all(ends[3:] == -np.inf))  # at or over the plateau: success at none

    def test_with_a_floor_a_trial_still_succeeds_exactly_when_it_falls_short_of_its_limit(self):
        rng = np.random.default_rng(18)
        distances = rng.uniform(0.0, 60.0, size=20000)
        uniforms = rng.random(20000)
        by_probability = uniforms < success_probability(distances, 20.0, 3.0, 0.05, 0.08)
        by_limit = distances < distance_at_probability(uniforms, 20.0, 3.0, 0.05, 0.08)
        self.assertTrue(np.array_equal(by_probability, by_limit))
        far = distances > 45.0
        self.assertAlmostEqual(float(by_limit[far].mean()), 0.08, delta=0.02)  # the lucky successes far out

    def test_the_result_has_the_broadcast_shape_of_its_arguments(self):
        self.assertEqual(distance_at_probability(0.3, 20.0, 3.0).shape, ())
        self.assertEqual(distance_at_probability(np.full((4, 1), 0.3), np.array([10.0, 20.0]), 3.0).shape, (4, 2))

    def test_impossible_arguments_are_refused(self):
        for probability, width, lapse in ((-0.1, 3.0, 0.0), (1.1, 3.0, 0.0), (0.5, 0.0, 0.0), (0.5, 3.0, 1.0)):
            with self.subTest(probability=probability, width=width, lapse=lapse), self.assertRaises(ValueError):
                distance_at_probability(probability, 20.0, width, lapse)
        for lapse, floor in ((0.0, -0.1), (0.6, 0.4)):
            with self.subTest(lapse=lapse, floor=floor), self.assertRaises(ValueError):
                distance_at_probability(0.5, 20.0, 3.0, lapse, floor)


class LogLikelihoodTests(unittest.TestCase):
    DISTANCES = [0.0, 3.2, 11.9, 18.4, 21.0, 26.7, 33.3, 41.0]
    SUCCESSES = [True, True, False, True, True, False, False, False]
    POINTS = ((10, 4, 0), (40, 0, 0), (40, 12, 1), (41, 25, 2), (80, 29, 3), (160, 18, 3), (0, 29, 0))

    def test_it_matches_the_model_written_out_trial_by_trial(self):
        grid = log_likelihood_grid(self.DISTANCES, self.SUCCESSES, **FLOOR_GRIDS)
        shape = (R50_GRID_MM.size, WIDTH_GRID_MM.size, LAPSE_VALUES.size, FLOOR_VALUES.size)
        self.assertEqual(grid.shape, shape)
        for i, j, k in self.POINTS:
            for m, floor in enumerate(FLOOR_VALUES):
                with self.subTest(r50=R50_GRID_MM[i], width=WIDTH_GRID_MM[j], lapse=LAPSE_VALUES[k], floor=floor):
                    expected = direct_log_likelihood(
                        self.DISTANCES, self.SUCCESSES, R50_GRID_MM[i], WIDTH_GRID_MM[j], LAPSE_VALUES[k], floor
                    )
                    self.assertAlmostEqual(grid[i, j, k, m], expected, delta=1e-10 * max(1.0, abs(expected)))

    def test_it_matches_sixty_digit_arithmetic_on_random_trials_and_grids(self):
        # Random trials (some on a 5 mm lattice, so that distances repeat), random grids, with and without floors,
        # and every fourth case with widths so small that exp(z) would overflow and the always-safe route is taken.
        # (A longer run of the same check, 1800 grid points, agreed to 2 parts in 10^15.)
        rng = np.random.default_rng(12345)
        checked = 0
        for case in range(48):
            n = int(rng.integers(1, 90))
            distances = rng.uniform(0.0, 70.0, n)
            if case % 3 == 0:
                distances = np.round(distances / 5.0) * 5.0
            grids = {
                "r50_grid_mm": np.sort(rng.uniform(0.0, 80.0, 5)),
                "width_grid_mm": np.sort(rng.uniform(0.3, 25.0, 4) if case % 4 else rng.uniform(0.01, 0.2, 4)),
                "lapse_values": np.unique(np.concatenate(([0.0], rng.uniform(0.0, 0.3, 2)))),
                "floor_values": np.unique(np.concatenate(([0.0], rng.uniform(0.0, 0.3, 2)))) if case % 5 else [0.0],
            }
            truth = (rng.uniform(5.0, 45.0), rng.uniform(1.0, 10.0), rng.uniform(0.0, 0.1), rng.uniform(0.0, 0.1))
            successes = rng.random(n) < success_probability(distances, *truth)
            grid = log_likelihood_grid(distances, successes, **grids)
            self.assertTrue(np.all(np.isfinite(grid)))
            for _ in range(3):
                i, j, k, m = (int(rng.integers(size)) for size in grid.shape)
                point = (grids["r50_grid_mm"][i], grids["width_grid_mm"][j], grids["lapse_values"][k])
                expected = sixty_digit_log_likelihood(distances, successes, *point, grids["floor_values"][m])
                with self.subTest(case=case, point=point, floor=grids["floor_values"][m]):
                    self.assertAlmostEqual(grid[i, j, k, m], expected, delta=1e-12 * max(1.0, abs(expected)))
                checked += 1
        self.assertEqual(checked, 144)

    def test_without_floor_values_the_grid_has_one_floor_of_zero(self):
        plain = log_likelihood_grid(self.DISTANCES, self.SUCCESSES, **GRIDS)
        self.assertEqual(plain.shape, (R50_GRID_MM.size, WIDTH_GRID_MM.size, LAPSE_VALUES.size, 1))
        with_floors = log_likelihood_grid(self.DISTANCES, self.SUCCESSES, **FLOOR_GRIDS)
        self.assertTrue(np.array_equal(plain[..., 0], with_floors[..., 0]))  # the same numbers, to the last bit

    def curves_on_the_grid(self, distance: float) -> np.ndarray:
        return success_probability(
            distance,
            R50_GRID_MM[:, None, None, None],
            WIDTH_GRID_MM[None, :, None, None],
            LAPSE_VALUES[None, None, :, None],
            FLOOR_VALUES[None, None, None, :],
        )

    def test_one_success_is_the_logarithm_of_the_curve_everywhere_on_the_grid(self):
        for distance in (0.0, 17.3, 64.0):
            with self.subTest(distance=distance):
                grid = log_likelihood_grid([distance], [True], **FLOOR_GRIDS)
                self.assertTrue(np.allclose(grid, np.log(self.curves_on_the_grid(distance)), rtol=1e-12, atol=1e-12))

    def test_one_failure_is_the_logarithm_of_one_minus_the_curve(self):
        for distance in (0.0, 17.3, 64.0):
            with self.subTest(distance=distance):
                grid = log_likelihood_grid([distance], [False], **FLOOR_GRIDS)
                curve = self.curves_on_the_grid(distance)
                # Where the curve is within rounding of 1, `1 - curve` has lost its digits: compare the rest.
                safe = curve < 0.999
                self.assertGreater(safe.sum(), safe.size // 4)
                self.assertTrue(np.allclose(grid[safe], np.log1p(-curve[safe]), rtol=1e-10, atol=1e-12))

    def test_a_cliff_far_from_the_trials_is_handled_exactly(self):
        # Width 0.001 mm: a success 30 mm beyond r50 has log-probability log(1 - lapse) - 30000, a failure 30 mm
        # short of r50 has log-probability -30000 without lapses and log(lapse) with them. Nothing may overflow.
        grids = {"r50_grid_mm": [10.0, 40.0, 70.0], "width_grid_mm": [0.001], "lapse_values": [0.0, 0.1]}
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            success = log_likelihood_grid([40.0], [True], **grids)[:, 0, :, 0]
            failure = log_likelihood_grid([40.0], [False], **grids)[:, 0, :, 0]
        self.assertAlmostEqual(success[0, 0], -30000.0, places=6)
        self.assertAlmostEqual(success[0, 1], math.log(0.9) - 30000.0, places=6)
        self.assertAlmostEqual(success[1, 0], math.log(0.5), places=12)
        self.assertAlmostEqual(success[2, 1], math.log(0.9), places=12)
        self.assertAlmostEqual(failure[2, 0], -30000.0, places=6)
        self.assertAlmostEqual(failure[2, 1], math.log(0.1), places=12)
        self.assertAlmostEqual(failure[1, 1], math.log(0.55), places=12)
        self.assertAlmostEqual(failure[0, 0], 0.0, places=12)

    def test_a_cliff_far_from_the_trials_is_handled_exactly_with_a_floor(self):
        # The same cliff with a floor of 0.2: far beyond r50 a success has probability 0.2 and a failure 0.8; far
        # short of it a success has probability 1 - lapse and a failure has probability lapse, or, without lapses,
        # 0.8 * exp(-30000).
        grids = {"r50_grid_mm": [10.0, 40.0, 70.0], "width_grid_mm": [0.001], "lapse_values": [0.0, 0.1]}
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            success = log_likelihood_grid([40.0], [True], **grids, floor_values=[0.0, 0.2])[:, 0, :, 1]
            failure = log_likelihood_grid([40.0], [False], **grids, floor_values=[0.0, 0.2])[:, 0, :, 1]
        self.assertAlmostEqual(success[0, 0], math.log(0.2), places=12)
        self.assertAlmostEqual(success[0, 1], math.log(0.2), places=12)
        self.assertAlmostEqual(success[1, 1], math.log(0.2 + 0.7 / 2), places=12)
        self.assertAlmostEqual(success[2, 0], 0.0, places=12)
        self.assertAlmostEqual(success[2, 1], math.log(0.9), places=12)
        self.assertAlmostEqual(failure[0, 0], math.log(0.8), places=12)
        self.assertAlmostEqual(failure[1, 0], math.log(0.8 / 2), places=12)
        self.assertAlmostEqual(failure[2, 0], math.log(0.8) - 30000.0, places=6)
        self.assertAlmostEqual(failure[2, 1], math.log(0.1), places=12)

    def test_the_quick_route_and_the_always_safe_route_give_the_same_values(self):
        # With grids like the protocol's, exp(z) is built as one factor per trial times one per grid value, which is
        # quick. Where that could overflow, exp(-|z|) is computed directly. Forbidding the quick route must change
        # nothing but the last digits (and it does change those, which shows that both routes ran).
        distances, successes = draw_trials(seed=13, n=90, r50_mm=20.0, width_mm=4.0, lapse=0.05, floor=0.08)
        quick = log_likelihood_grid(distances, successes, **FLOOR_GRIDS)
        with mock.patch.object(psychometric, "_SPLIT_EXPONENT_LIMIT", 0.0):
            safe = log_likelihood_grid(distances, successes, **FLOOR_GRIDS)
        self.assertTrue(np.allclose(quick, safe, rtol=1e-11, atol=1e-10))
        self.assertFalse(np.array_equal(quick, safe))

    def test_many_trials_are_summed_without_losing_any(self):
        # More trials than one internal block holds, at one distance: the total is the count times one trial's value.
        one = log_likelihood_grid([12.0, 12.0], [True, False], **FLOOR_GRIDS)
        many = log_likelihood_grid([12.0] * 600, [True, False] * 300, **FLOOR_GRIDS)
        self.assertTrue(np.allclose(many, 300 * one, rtol=1e-10, atol=1e-9))

    def test_tiny_lapse_and_floor_values_do_not_underflow(self):
        # A product of many probabilities as small as 1e-200 would underflow; they must be multiplied a few at a time.
        # The tiny value is once a lapse and once a floor, so that each alone must keep the blocks short.
        grids = {"r50_grid_mm": [10.0, 30.0], "width_grid_mm": [0.5]}
        failures = log_likelihood_grid([0.0] * 40, [False] * 40, **grids, lapse_values=[0.0, 1e-200])
        successes = log_likelihood_grid(
            [60.0] * 40, [True] * 40, **grids, lapse_values=[0.0, 0.02], floor_values=[0.0, 1e-200]
        )
        self.assertTrue(np.all(np.isfinite(failures)) and np.all(np.isfinite(successes)))
        # 40 failures 20 widths short of r50 = 10, with a lapse of 1e-200 and no floor: each has probability
        # 1e-200 + exp(-20). And 40 successes 60 widths beyond r50 = 30 with a floor of 1e-200: each exp(-60) + 1e-200.
        self.assertAlmostEqual(failures[0, 0, 1, 0], 40 * math.log(1e-200 + math.exp(-20.0)), places=6)
        self.assertAlmostEqual(successes[1, 0, 0, 1], 40 * math.log(1e-200 + math.exp(-60.0)), places=6)

    @unittest.skipIf(scipy_stats is None, "SciPy is not installed in this environment")
    def test_it_matches_scipy_at_grid_points_across_the_grid(self):
        distances, successes = draw_trials(seed=7, n=40, r50_mm=22.0, width_mm=5.0, lapse=0.05)
        grid = log_likelihood_grid(distances, successes, **GRIDS)
        for i in (0, 13, 44, 45, 90, 160):
            for j in (0, 9, 17, 29):
                for k in range(LAPSE_VALUES.size):
                    with self.subTest(r50=R50_GRID_MM[i], width=WIDTH_GRID_MM[j], lapse=LAPSE_VALUES[k]):
                        z = (distances - R50_GRID_MM[i]) / WIDTH_GRID_MM[j]
                        log_p = math.log1p(-LAPSE_VALUES[k]) + scipy_special.log_expit(-z)
                        log_q = np.log(LAPSE_VALUES[k] * scipy_special.expit(-z) + scipy_special.expit(z))
                        expected = float(np.where(successes, log_p, log_q).sum())
                        self.assertAlmostEqual(grid[i, j, k, 0], expected, delta=1e-9 * max(1.0, abs(expected)))

    @unittest.skipIf(scipy_stats is None, "SciPy is not installed in this environment")
    def test_with_a_floor_it_matches_scipy_at_grid_points_across_the_grid(self):
        # Success is (1 - lapse) s + floor q and failure is lapse s + (1 - floor) q, where s and q = 1 - s are the
        # two logistic halves: sums of positive terms, so SciPy's expit gives them without cancellation.
        distances, successes = draw_trials(seed=8, n=40, r50_mm=22.0, width_mm=5.0, lapse=0.05, floor=0.08)
        grid = log_likelihood_grid(distances, successes, **FLOOR_GRIDS)
        for i in (0, 13, 44, 45, 90, 160):
            for j in (0, 9, 17, 29):
                z = (distances - R50_GRID_MM[i]) / WIDTH_GRID_MM[j]
                near, far = scipy_special.expit(-z), scipy_special.expit(z)
                for k, lapse in enumerate(LAPSE_VALUES):
                    for m, floor in enumerate(FLOOR_VALUES):
                        if floor == 0.0:
                            continue  # the test above covers the model without a floor, on the logarithmic scale
                        with self.subTest(r50=R50_GRID_MM[i], width=WIDTH_GRID_MM[j], lapse=lapse, floor=floor):
                            log_p = np.log((1.0 - lapse) * near + floor * far)
                            log_q = np.log(lapse * near + (1.0 - floor) * far)
                            expected = float(np.where(successes, log_p, log_q).sum())
                            self.assertAlmostEqual(grid[i, j, k, m], expected, delta=1e-9 * max(1.0, abs(expected)))

    @unittest.skipIf(scipy_stats is None, "SciPy is not installed in this environment")
    def test_it_matches_scipy_bernoulli_where_the_plain_formula_is_accurate(self):
        distances, successes = draw_trials(seed=11, n=30, r50_mm=15.0, width_mm=3.0, lapse=0.0)
        grid = log_likelihood_grid(distances, successes, **GRIDS)
        for i, j, k in ((30, 16, 0), (30, 16, 2), (50, 24, 1), (20, 29, 3)):
            with self.subTest(r50=R50_GRID_MM[i], width=WIDTH_GRID_MM[j], lapse=LAPSE_VALUES[k]):
                p = (1.0 - LAPSE_VALUES[k]) * scipy_special.expit(-(distances - R50_GRID_MM[i]) / WIDTH_GRID_MM[j])
                expected = float(scipy_stats.bernoulli.logpmf(successes.astype(int), p).sum())
                self.assertAlmostEqual(grid[i, j, k, 0], expected, delta=1e-9 * max(1.0, abs(expected)))


class NoFloorByDefaultTests(unittest.TestCase):
    """The floor was added to the model later. Without `floor_values` the fit must be what it was before."""

    DISTANCES = [0.0, 0.4, 5.3, 9.8, 10.2, 14.7, 15.1, 19.6, 20.5, 24.9, 30.2, 50.0]
    SUCCESSES = [True, True, True, True, True, False, True, False, True, False, False, False]
    # Written down from the fit as it stood before the floor existed (NumPy 2.5 and 2.2 agreed to the last digit
    # or the one before it).
    PROBABILITIES = {
        (40, 10, 0): 0.00031971897962830326,
        (36, 4, 1): 5.210393777039229e-05,
        (44, 20, 3): 0.00039580681799149654,
        (60, 29, 2): 8.199973832270365e-05,
        (0, 0, 0): 7.714524683966482e-56,
        (160, 29, 3): 1.977832227590866e-06,
    }
    LOG_LIKELIHOODS = {
        (40, 10, 0): -4.810680570243255,
        (36, 4, 1): -6.624882464633075,
        (44, 20, 3): -4.597196732671636,
        (60, 29, 2): -6.171406935469285,
        (0, 0, 0): -123.66427276666381,
        (160, 29, 3): -9.89612158115468,
    }

    def setUp(self):
        self.posterior = fit_threshold(self.DISTANCES, self.SUCCESSES, **GRIDS)

    def test_the_posterior_and_the_likelihood_are_the_numbers_they_were(self):
        self.assertEqual(self.posterior.probabilities.shape, (161, 30, 4, 1))
        log_likelihood = log_likelihood_grid(self.DISTANCES, self.SUCCESSES, **GRIDS)
        for index, expected in self.PROBABILITIES.items():
            with self.subTest(index=index):
                self.assertAlmostEqual(self.posterior.probabilities[(*index, 0)] / expected, 1.0, places=12)
                self.assertAlmostEqual(log_likelihood[(*index, 0)], self.LOG_LIKELIHOODS[index], places=11)

    def test_the_summary_is_the_one_it_was_with_a_floor_of_zero(self):
        summary = self.posterior.summary()
        self.assertEqual((summary.n_trials, summary.n_success, summary.level), (12, 7, 0.95))
        self.assertEqual((summary.r50_low_mm, summary.r50_median_mm, summary.r50_high_mm), (10.0, 19.5, 36.5))
        self.assertEqual((summary.reaches_lower_edge, summary.reaches_upper_edge), (False, False))
        self.assertAlmostEqual(summary.width_median_mm, 4.346189116633815, places=12)
        self.assertAlmostEqual(summary.width_low_mm, 0.8316561362471894, places=12)
        self.assertAlmostEqual(summary.width_high_mm, 17.61110261860995, places=12)
        self.assertAlmostEqual(summary.lapse_mean, 0.04213026496216757, places=13)
        self.assertEqual(summary.floor_mean, 0.0)

    def test_the_predicted_curve_and_its_band_are_the_ones_they_were(self):
        prediction = self.posterior.predict([0.0, 10.0, 20.0, 30.0])
        expected = [0.9063852729175312, 0.8107818656036784, 0.44734239139681714, 0.1513573912889525]
        self.assertTrue(np.allclose(prediction, expected, rtol=1e-12, atol=0.0))
        low, high = self.posterior.band([15.0, 25.0])
        self.assertTrue(np.allclose(low, [0.308033174560273, 0.0019022248644786593], rtol=1e-12, atol=0.0))
        self.assertTrue(np.allclose(high, [0.94897185706045, 0.7133469061918843], rtol=1e-12, atol=0.0))

    def test_naming_a_single_floor_of_zero_changes_nothing_at_all(self):
        named = fit_threshold(self.DISTANCES, self.SUCCESSES, **GRIDS, floor_values=(0.0,))
        self.assertTrue(np.array_equal(named.probabilities, self.posterior.probabilities))
        self.assertEqual(named.summary(), self.posterior.summary())
        self.assertTrue(np.array_equal(self.posterior.floor_values, [0.0]))

    def test_the_no_floor_part_of_a_grid_with_floors_is_the_same_likelihood(self):
        # Adding floor values adds combinations; it does not touch the ones that were there.
        plain = log_likelihood_grid(self.DISTANCES, self.SUCCESSES, **GRIDS)
        wider = log_likelihood_grid(self.DISTANCES, self.SUCCESSES, **FLOOR_GRIDS)
        self.assertTrue(np.array_equal(wider[..., 0], plain[..., 0]))


class FloorTests(unittest.TestCase):
    """A floor in the model: a chance of success that remains far from every demonstration."""

    def test_a_floor_in_the_data_is_recovered_and_ignoring_it_pushes_the_radius_out(self):
        truth = {"r50_mm": 20.0, "width_mm": 3.0, "lapse": 0.05, "floor": 0.08}
        distances, successes = draw_trials(seed=200, n=2500, **truth)
        with_floor = fit_threshold(distances, successes, **FLOOR_GRIDS).summary()
        without = fit_threshold(distances, successes, **GRIDS).summary()
        self.assertAlmostEqual(with_floor.r50_median_mm, 20.0, delta=1.5)
        self.assertAlmostEqual(with_floor.width_median_mm / 3.0, 1.0, delta=0.30)
        self.assertAlmostEqual(with_floor.floor_mean, 0.08, delta=0.03)
        self.assertAlmostEqual(with_floor.lapse_mean, 0.05, delta=0.03)
        # Without a floor the model has to stretch the curve to reach the lucky successes far out.
        self.assertGreater(without.width_median_mm, 2.0 * with_floor.width_median_mm)
        self.assertEqual(without.floor_mean, 0.0)

    def test_one_lucky_success_far_out_no_longer_sets_the_radius(self):
        # The pattern of the August trials: success near the demonstrations, eleven failures from 31 to 47 mm, and one
        # success at 40 mm among them.
        near = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0, 25.0, 25.0]
        near_outcomes = [True, True, True, True, True, True, True, True, True, False, True, False, False, False]
        far = [31.0, 33.0, 35.0, 36.0, 38.0, 40.0, 41.0, 43.0, 44.0, 46.0, 47.0]
        clean = (near + far, near_outcomes + [False] * len(far))
        lucky = (near + far + [40.0], near_outcomes + [False] * len(far) + [True])
        reference = fit_threshold(*clean, **GRIDS).summary()
        moved = fit_threshold(*lucky, **GRIDS).summary()
        robust = fit_threshold(*lucky, **FLOOR_GRIDS).summary()
        self.assertGreater(moved.r50_median_mm, reference.r50_median_mm + 2.0)  # without a floor: pulled outward
        self.assertLess(abs(robust.r50_median_mm - reference.r50_median_mm), 2.0)  # with one: it stays
        self.assertLess(robust.r50_high_mm, moved.r50_high_mm)
        # A single far success says little about how large the floor is: its posterior mean stays near its prior's.
        self.assertTrue(0.03 < robust.floor_mean < 0.09, robust)

    def test_floor_values_that_cannot_be_are_refused(self):
        bad_floors = (
            [],
            [-0.01],
            [0.0, 1.0],
            [0.05, 0.05],  # the same value twice would count double
            [0.0, 0.9],  # with the largest lapse of 0.10 nothing would be left between plateau and floor
            [float("nan")],
        )
        for floors in bad_floors:
            with self.subTest(floors=floors), self.assertRaises(ValueError):
                fit_threshold([5.0, 30.0], [True, False], **GRIDS, floor_values=floors)
        with self.assertRaises(ValueError):
            log_likelihood_grid([5.0], [True], **(GRIDS | {"lapse_values": [0.0, 0.5]}), floor_values=[0.0, 0.5])

    def test_the_posterior_keeps_the_floor_values_and_their_probabilities(self):
        distances, successes = draw_trials(seed=31, n=60, r50_mm=20.0, width_mm=3.0, lapse=0.05, floor=0.08)
        posterior = fit_threshold(distances, successes, **FLOOR_GRIDS)
        self.assertEqual(posterior.probabilities.shape, (161, 30, 4, 3))
        self.assertAlmostEqual(posterior.probabilities.sum(), 1.0, places=12)
        values, marginal = posterior.floor_marginal()
        self.assertTrue(np.array_equal(values, FLOOR_VALUES))
        self.assertTrue(np.allclose(marginal, posterior.probabilities.sum(axis=(0, 1, 2)), rtol=1e-12))
        self.assertAlmostEqual(posterior.summary().floor_mean, float(marginal @ FLOOR_VALUES), places=12)
        with self.assertRaises(ValueError):
            posterior.floor_values[0] = 0.5

    def test_no_data_gives_back_the_prior_with_equal_weight_on_each_floor(self):
        posterior = fit_threshold([], [], **FLOOR_GRIDS)
        self.assertTrue(np.allclose(posterior.probabilities, 1.0 / posterior.probabilities.size, rtol=1e-12))
        summary = posterior.summary()
        self.assertAlmostEqual(summary.floor_mean, FLOOR_VALUES.mean(), places=12)
        self.assertEqual((summary.r50_low_mm, summary.r50_median_mm, summary.r50_high_mm), (2.0, 40.0, 78.0))
        # Far from every demonstration the prior prediction is the prior mean of the floor.
        self.assertAlmostEqual(float(posterior.predict(1e6)), FLOOR_VALUES.mean(), places=12)

    def test_the_order_of_the_trials_still_does_not_change_one_bit(self):
        distances, successes = draw_trials(seed=3, n=34, r50_mm=18.0, width_mm=3.0, lapse=0.05, floor=0.08)
        shuffled = np.random.default_rng(5).permutation(distances.size)
        reference = fit_threshold(distances, successes, **FLOOR_GRIDS)
        again = fit_threshold(distances[shuffled], successes[shuffled], **FLOOR_GRIDS)
        self.assertTrue(np.array_equal(reference.probabilities, again.probabilities))


class FitTests(unittest.TestCase):
    def test_known_parameters_are_recovered_from_a_large_sample(self):
        # Tolerances: r50 within 1.5 mm, width within 30 % (its grid moves in steps of 14 %), lapse within 0.03.
        for seed, (r50, width, lapse) in enumerate(((20.0, 4.0, 0.05), (10.0, 1.0, 0.0), (35.0, 8.0, 0.10))):
            with self.subTest(r50=r50, width=width, lapse=lapse):
                distances, successes = draw_trials(seed=100 + seed, n=2500, r50_mm=r50, width_mm=width, lapse=lapse)
                summary = fit_threshold(distances, successes, **GRIDS).summary()
                self.assertAlmostEqual(summary.r50_median_mm, r50, delta=1.5)
                self.assertAlmostEqual(summary.width_median_mm / width, 1.0, delta=0.30)
                self.assertAlmostEqual(summary.lapse_mean, lapse, delta=0.03)
                self.assertLess(summary.r50_high_mm - summary.r50_low_mm, 6.0)
                self.assertFalse(summary.reaches_lower_edge or summary.reaches_upper_edge)

    def test_the_posterior_is_the_likelihood_rescaled_to_sum_to_one(self):
        # Equal prior weight on every combination of the grids: flat in r50 on an even grid, flat in log(width) on a
        # logarithmic one, and the lapse values equally likely.
        distances, successes = draw_trials(seed=4, n=30, r50_mm=20.0, width_mm=3.0, lapse=0.02, max_mm=50.0)
        log_likelihood = log_likelihood_grid(distances, successes, **GRIDS)
        expected = np.exp(log_likelihood - log_likelihood.max())
        expected /= expected.sum()
        posterior = fit_threshold(distances, successes, **GRIDS)
        self.assertTrue(np.allclose(posterior.probabilities, expected, rtol=1e-12, atol=0.0))
        self.assertEqual((posterior.n_trials, posterior.n_success), (30, int(successes.sum())))

    def test_perfectly_separated_data_give_a_proper_answer_between_the_two_groups(self):
        distances = repeat([0, 5, 10, 15, 20], 2) + repeat([25, 30, 35, 40], 2)
        successes = [True] * 10 + [False] * 8
        posterior = fit_threshold(distances, successes, **GRIDS)
        self.assertTrue(np.all(np.isfinite(posterior.probabilities)))
        self.assertAlmostEqual(posterior.probabilities.sum(), 1.0, places=12)
        summary = posterior.summary()
        self.assertTrue(20.0 <= summary.r50_median_mm <= 25.0, summary)
        self.assertTrue(15.0 <= summary.r50_low_mm <= summary.r50_high_mm <= 30.0, summary)
        self.assertFalse(summary.reaches_lower_edge or summary.reaches_upper_edge)
        # Nothing in such data limits how sharp the cliff is: the width's interval starts at the grid's smallest.
        self.assertEqual(summary.width_low_mm, WIDTH_GRID_MM[0])

    def test_all_successes_bound_the_radius_from_below_only(self):
        distances = repeat([0, 5, 10, 15, 20, 25, 30, 35, 40], 2)
        posterior = fit_threshold(distances, [True] * len(distances), **GRIDS)
        self.assertAlmostEqual(posterior.probabilities.sum(), 1.0, places=12)
        summary = posterior.summary()
        self.assertTrue(summary.reaches_upper_edge)
        self.assertFalse(summary.reaches_lower_edge)
        self.assertGreater(summary.r50_low_mm, 35.0)
        self.assertGreater(summary.r50_high_mm, 75.0)
        self.assertEqual((summary.n_trials, summary.n_success), (18, 18))

    def test_all_failures_bound_the_radius_from_above_only(self):
        distances = repeat([10, 20, 30, 40, 50], 2)
        posterior = fit_threshold(distances, [False] * len(distances), **GRIDS)
        self.assertAlmostEqual(posterior.probabilities.sum(), 1.0, places=12)
        summary = posterior.summary()
        self.assertTrue(summary.reaches_lower_edge)
        self.assertFalse(summary.reaches_upper_edge)
        self.assertLess(summary.r50_median_mm, 10.0)
        self.assertLess(summary.r50_high_mm, 25.0)
        self.assertEqual((summary.n_trials, summary.n_success), (10, 0))

    def test_no_data_gives_back_the_prior(self):
        posterior = fit_threshold([], [], **GRIDS)
        self.assertTrue(np.allclose(posterior.probabilities, 1.0 / posterior.probabilities.size, rtol=1e-12))
        grid, marginal = posterior.r50_marginal()
        self.assertTrue(np.array_equal(grid, R50_GRID_MM))
        self.assertTrue(np.allclose(marginal, 1.0 / R50_GRID_MM.size, rtol=1e-12))
        summary = posterior.summary()
        self.assertEqual((summary.n_trials, summary.n_success), (0, 0))
        # A flat prior from 0 to 80 mm in steps of 0.5: its middle, and the points that leave 2.5 % outside.
        self.assertEqual((summary.r50_low_mm, summary.r50_median_mm, summary.r50_high_mm), (2.0, 40.0, 78.0))
        self.assertTrue(summary.reaches_lower_edge and summary.reaches_upper_edge)
        self.assertAlmostEqual(summary.lapse_mean, LAPSE_VALUES.mean(), places=12)
        self.assertEqual(summary.width_median_mm, WIDTH_GRID_MM[14])  # half of 30 equal weights is reached at the 15th

    def test_the_order_of_the_trials_does_not_change_one_bit(self):
        distances, successes = draw_trials(seed=3, n=34, r50_mm=18.0, width_mm=3.0, lapse=0.05)
        reference = fit_threshold(distances, successes, **GRIDS)
        shuffled = np.random.default_rng(5).permutation(distances.size)
        again = fit_threshold(distances[shuffled], successes[shuffled], **GRIDS)
        self.assertTrue(np.array_equal(reference.probabilities, again.probabilities))
        self.assertEqual(reference.summary(), again.summary())

    def test_the_interval_narrows_as_trials_are_added(self):
        distances, successes = draw_trials(seed=21, n=320, r50_mm=25.0, width_mm=4.0, lapse=0.02, max_mm=50.0)
        widths = []
        for n in (20, 80, 320):
            summary = fit_threshold(distances[:n], successes[:n], **GRIDS).summary()
            widths.append(summary.r50_high_mm - summary.r50_low_mm)
        self.assertTrue(widths[0] > widths[1] > widths[2], widths)

    def test_outcomes_may_be_booleans_or_zeros_and_ones(self):
        distances = [0.0, 10.0, 20.0, 30.0]
        as_bool = fit_threshold(distances, [True, True, False, False], **GRIDS)
        as_int = fit_threshold(distances, [1, 1, 0, 0], **GRIDS)
        as_array = fit_threshold(np.array(distances), np.array([1.0, 1.0, 0.0, 0.0]), **GRIDS)
        self.assertTrue(np.array_equal(as_bool.probabilities, as_int.probabilities))
        self.assertTrue(np.array_equal(as_bool.probabilities, as_array.probabilities))

    def test_the_posterior_keeps_its_grids_and_cannot_be_written_to(self):
        posterior = fit_threshold([5.0, 30.0], [True, False], **GRIDS)
        self.assertIsInstance(posterior, ThresholdPosterior)
        self.assertTrue(np.array_equal(posterior.r50_grid_mm, R50_GRID_MM))
        self.assertTrue(np.array_equal(posterior.width_grid_mm, WIDTH_GRID_MM))
        self.assertTrue(np.array_equal(posterior.lapse_values, LAPSE_VALUES))
        with self.assertRaises(ValueError):
            posterior.probabilities[0, 0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            posterior.r50_grid_mm[0] = 1.0

    def test_trials_that_cannot_be_are_refused(self):
        bad_trials = (
            ([1.0, 2.0], [True]),  # one outcome missing
            ([-1.0], [True]),  # a distance is not negative
            ([float("nan")], [True]),
            ([float("inf")], [False]),
            ([1.0, 2.0], [1, 2]),  # an outcome is a success or a failure
            ([1.0], [0.5]),
            ([1.0], ["success"]),
            ([[1.0, 2.0]], [[True, False]]),  # one list of trials, not a table
        )
        for distances, successes in bad_trials:
            with self.subTest(distances=distances, successes=successes), self.assertRaises(ValueError):
                fit_threshold(distances, successes, **GRIDS)

    def test_grids_that_cannot_be_are_refused(self):
        bad_grids = (
            {"r50_grid_mm": []},
            {"r50_grid_mm": [0.0, 10.0, 5.0]},  # not increasing
            {"r50_grid_mm": [0.0, float("nan")]},
            {"width_grid_mm": [0.0, 1.0]},  # a width is positive
            {"width_grid_mm": [2.0, 1.0]},
            {"width_grid_mm": []},
            {"lapse_values": []},
            {"lapse_values": [0.0, 1.0]},  # a lapse rate of 1 leaves no successes to explain
            {"lapse_values": [-0.1]},
            {"lapse_values": [0.05, 0.05]},  # the same value twice would count double
        )
        for change in bad_grids:
            with self.subTest(change=change), self.assertRaises(ValueError):
                fit_threshold([5.0, 30.0], [True, False], **(GRIDS | change))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        distances, successes = draw_trials(seed=42, n=60, r50_mm=22.0, width_mm=4.0, lapse=0.05, max_mm=50.0)
        self.posterior = fit_threshold(distances, successes, **FLOOR_GRIDS)
        self.summary = self.posterior.summary()

    def test_it_is_a_record_of_plain_numbers(self):
        self.assertIsInstance(self.summary, ThresholdSummary)
        self.assertEqual(self.summary.level, 0.95)
        self.assertEqual(self.summary.n_trials, 60)
        for name in ("r50_median_mm", "r50_low_mm", "r50_high_mm", "width_median_mm", "lapse_mean", "floor_mean"):
            self.assertIs(type(getattr(self.summary, name)), float, name)
        for name in ("reaches_lower_edge", "reaches_upper_edge"):
            self.assertIs(type(getattr(self.summary, name)), bool, name)
        self.assertEqual(self.summary.r50_interval_width_mm, self.summary.r50_high_mm - self.summary.r50_low_mm)

    def test_reported_values_are_values_of_the_grids(self):
        for value in (self.summary.r50_low_mm, self.summary.r50_median_mm, self.summary.r50_high_mm):
            self.assertIn(value, R50_GRID_MM)
        for value in (self.summary.width_low_mm, self.summary.width_median_mm, self.summary.width_high_mm):
            self.assertIn(value, WIDTH_GRID_MM)
        self.assertTrue(self.summary.r50_low_mm <= self.summary.r50_median_mm <= self.summary.r50_high_mm)
        self.assertTrue(self.summary.width_low_mm <= self.summary.width_median_mm <= self.summary.width_high_mm)
        self.assertTrue(LAPSE_VALUES.min() <= self.summary.lapse_mean <= LAPSE_VALUES.max())
        self.assertTrue(FLOOR_VALUES.min() <= self.summary.floor_mean <= FLOOR_VALUES.max())

    def test_a_summary_written_by_hand_without_a_floor_still_holds(self):
        # The floor's mean was added last and defaults to 0, so records built before it existed stay valid.
        by_hand = ThresholdSummary(0.95, 10, 5, 20.0, 15.0, 25.0, False, False, 3.0, 1.0, 9.0, 0.02)
        self.assertEqual(by_hand.floor_mean, 0.0)

    def test_each_bound_leaves_just_under_its_share_of_probability_outside(self):
        # Equal tails: each bound is the first grid value at which the running total reaches its mark.
        grid, marginal = self.posterior.r50_marginal()
        self.assertAlmostEqual(marginal.sum(), 1.0, places=12)
        for value, mark in (
            (self.summary.r50_low_mm, 0.025),
            (self.summary.r50_median_mm, 0.5),
            (self.summary.r50_high_mm, 0.975),
        ):
            with self.subTest(mark=mark):
                self.assertLess(marginal[grid < value].sum(), mark)
                self.assertGreaterEqual(marginal[grid <= value].sum(), mark)

    def test_the_marginals_are_the_sums_of_the_joint_posterior(self):
        joint = self.posterior.probabilities
        self.assertTrue(np.allclose(self.posterior.r50_marginal()[1], joint.sum(axis=(1, 2, 3)), rtol=1e-12))
        width_grid, width_marginal = self.posterior.width_marginal()
        self.assertTrue(np.array_equal(width_grid, WIDTH_GRID_MM))
        self.assertTrue(np.allclose(width_marginal, joint.sum(axis=(0, 2, 3)), rtol=1e-12))
        lapse_grid, lapse_marginal = self.posterior.lapse_marginal()
        self.assertTrue(np.array_equal(lapse_grid, LAPSE_VALUES))
        self.assertTrue(np.allclose(lapse_marginal, joint.sum(axis=(0, 1, 3)), rtol=1e-12))
        self.assertAlmostEqual(self.summary.lapse_mean, float(lapse_marginal @ LAPSE_VALUES), places=12)
        floor_grid, floor_marginal = self.posterior.floor_marginal()
        self.assertTrue(np.array_equal(floor_grid, FLOOR_VALUES))
        self.assertTrue(np.allclose(floor_marginal, joint.sum(axis=(0, 1, 2)), rtol=1e-12))
        self.assertAlmostEqual(self.summary.floor_mean, float(floor_marginal @ FLOOR_VALUES), places=12)

    def test_a_lower_level_gives_an_interval_inside_the_wider_one(self):
        half = self.posterior.summary(level=0.5)
        self.assertEqual(half.level, 0.5)
        self.assertEqual(half.r50_median_mm, self.summary.r50_median_mm)
        self.assertTrue(self.summary.r50_low_mm <= half.r50_low_mm <= half.r50_high_mm <= self.summary.r50_high_mm)
        self.assertLess(half.r50_interval_width_mm, self.summary.r50_interval_width_mm)

    def test_the_level_is_a_share_strictly_between_zero_and_one(self):
        for level in (0.0, 1.0, -0.5, 95):
            with self.subTest(level=level), self.assertRaises(ValueError):
                self.posterior.summary(level=level)


class EdgeFlagTests(unittest.TestCase):
    """When the report must say that the data do not bound the radius on one side."""

    def test_the_rule_is_half_the_height_of_the_interval_s_own_end(self):
        self.assertEqual(psychometric.EDGE_RATIO, 0.5)
        cases = (
            draw_trials(seed=1, n=24, r50_mm=5.0, width_mm=3.0, lapse=0.0, max_mm=30.0),
            draw_trials(seed=2, n=24, r50_mm=40.0, width_mm=10.0, lapse=0.05, max_mm=50.0),
            draw_trials(seed=3, n=34, r50_mm=20.0, width_mm=1.0, lapse=0.0, max_mm=50.0),
            ([], []),
        )
        for number, (distances, successes) in enumerate(cases):
            with self.subTest(case=number):
                posterior = fit_threshold(distances, successes, **GRIDS)
                summary = posterior.summary()
                grid, marginal = posterior.r50_marginal()
                at_low = marginal[grid == summary.r50_low_mm][0]
                at_high = marginal[grid == summary.r50_high_mm][0]
                self.assertEqual(summary.reaches_lower_edge, bool(marginal[0] >= 0.5 * at_low))
                self.assertEqual(summary.reaches_upper_edge, bool(marginal[-1] >= 0.5 * at_high))

    def test_a_clear_threshold_in_the_middle_is_bounded_on_both_sides(self):
        distances = repeat([0, 5, 10, 15, 20], 3) + repeat([30, 35, 40, 45, 50], 3)
        summary = fit_threshold(distances, [True] * 15 + [False] * 15, **GRIDS).summary()
        self.assertFalse(summary.reaches_lower_edge)
        self.assertFalse(summary.reaches_upper_edge)
        self.assertTrue(15.0 < summary.r50_low_mm < summary.r50_high_mm < 35.0, summary)

    def test_one_stray_failure_does_not_bound_the_radius_from_above(self):
        # Fourteen successes out to 30 mm and a single failure at 35 mm. The failure may be the edge of the radius or
        # one of the failures that have nothing to do with distance, so the posterior keeps a level tail out to the
        # end of the grid and the upper bound is set by where the grid stops.
        distances = repeat([0, 5, 10, 15, 20, 25, 30], 2) + [35.0]
        summary = fit_threshold(distances, [True] * 14 + [False], **GRIDS).summary()
        self.assertTrue(summary.reaches_upper_edge, summary)
        self.assertGreater(summary.r50_high_mm, 50.0)
        self.assertFalse(summary.reaches_lower_edge)

    def test_a_radius_near_zero_is_not_bounded_from_below(self):
        distances = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 10.0, 10.0, 15.0, 20.0]
        successes = [True, True, False, False, False, False, False, False, False, False]
        summary = fit_threshold(distances, successes, **GRIDS).summary()
        self.assertTrue(summary.reaches_lower_edge, summary)
        self.assertFalse(summary.reaches_upper_edge)


class PredictTests(unittest.TestCase):
    SMALL = {
        "r50_grid_mm": [10.0, 20.0, 30.0],
        "width_grid_mm": [1.0, 5.0],
        "lapse_values": [0.0, 0.1],
        "floor_values": [0.0, 0.2],
    }

    def brute_force(self, posterior: ThresholdPosterior, distance: float) -> list[tuple[float, float]]:
        """Every curve of the grid at one distance, with its posterior probability."""
        pairs = []
        for i, r50 in enumerate(posterior.r50_grid_mm):
            for j, width in enumerate(posterior.width_grid_mm):
                for k, lapse in enumerate(posterior.lapse_values):
                    for m, floor in enumerate(posterior.floor_values):
                        value = floor + (1.0 - lapse - floor) / (1.0 + math.exp((distance - r50) / width))
                        pairs.append((value, float(posterior.probabilities[i, j, k, m])))
        return pairs

    def test_the_prediction_is_the_posterior_average_of_the_curves(self):
        for distances, successes in (([], []), ([5.0, 12.0, 18.0, 26.0], [True, True, False, False])):
            posterior = fit_threshold(distances, successes, **self.SMALL)
            for distance in (0.0, 14.0, 20.0, 33.0):
                with self.subTest(trials=len(distances), distance=distance):
                    expected = sum(value * weight for value, weight in self.brute_force(posterior, distance))
                    self.assertAlmostEqual(float(posterior.predict(distance)), expected, places=12)

    def test_with_much_data_the_prediction_follows_the_true_curve(self):
        for floor, grids in ((0.0, GRIDS), (0.08, FLOOR_GRIDS)):
            with self.subTest(floor=floor):
                distances, successes = draw_trials(seed=8, n=2500, r50_mm=24.0, width_mm=5.0, lapse=0.05, floor=floor)
                posterior = fit_threshold(distances, successes, **grids)
                at = np.linspace(0.0, 60.0, 61)
                truth = success_probability(at, 24.0, 5.0, 0.05, floor)
                self.assertLess(np.abs(posterior.predict(at) - truth).max(), 0.05)
                low, high = posterior.band(at)
                self.assertLess((high - low).max(), 0.15)

    def test_the_prediction_is_a_probability_that_falls_with_distance(self):
        distances, successes = draw_trials(seed=9, n=30, r50_mm=15.0, width_mm=2.0, lapse=0.0, max_mm=40.0)
        prediction = fit_threshold(distances, successes, **GRIDS).predict(np.linspace(0.0, 80.0, 81))
        self.assertTrue(np.all((prediction >= 0.0) & (prediction <= 1.0)))
        self.assertTrue(np.all(np.diff(prediction) <= 1e-12))  # never rising, to within rounding
        self.assertLess(prediction[-1], 0.01)
        self.assertGreater(prediction[0], 0.9)

    def test_the_result_has_the_shape_of_the_distances_however_many(self):
        posterior = fit_threshold([5.0, 30.0], [True, False], **GRIDS)
        self.assertEqual(posterior.predict(12.0).shape, ())
        self.assertEqual(posterior.predict([1.0, 2.0, 3.0]).shape, (3,))
        self.assertEqual(posterior.predict(np.zeros((2, 3))).shape, (2, 3))
        many = np.linspace(0.0, 80.0, 700)
        prediction = posterior.predict(many)
        self.assertEqual(prediction.shape, (700,))
        self.assertTrue(np.allclose(prediction[[0, 350, 699]], posterior.predict(many[[0, 350, 699]]), rtol=1e-12))

    def test_the_band_holds_the_stated_share_of_the_curves_at_each_distance(self):
        posterior = fit_threshold([5.0, 12.0, 18.0, 26.0], [True, True, False, False], **self.SMALL)
        at = [0.0, 14.0, 20.0, 33.0]
        for level in (0.5, 0.95):
            low, high = posterior.band(at, level=level)
            tail = (1.0 - level) / 2.0
            for index, distance in enumerate(at):
                with self.subTest(level=level, distance=distance):
                    pairs = self.brute_force(posterior, distance)
                    below_low = sum(weight for value, weight in pairs if value < low[index] - 1e-12)
                    up_to_low = sum(weight for value, weight in pairs if value <= low[index] + 1e-12)
                    below_high = sum(weight for value, weight in pairs if value < high[index] - 1e-12)
                    up_to_high = sum(weight for value, weight in pairs if value <= high[index] + 1e-12)
                    self.assertLess(below_low, tail + 1e-12)
                    self.assertGreaterEqual(up_to_low, tail - 1e-12)
                    self.assertLess(below_high, 1.0 - tail + 1e-12)
                    self.assertGreaterEqual(up_to_high, 1.0 - tail - 1e-12)

    def test_the_band_is_ordered_nested_and_shaped_like_the_distances(self):
        distances, successes = draw_trials(seed=10, n=30, r50_mm=20.0, width_mm=4.0, lapse=0.05, max_mm=45.0)
        posterior = fit_threshold(distances, successes, **GRIDS)
        at = np.linspace(0.0, 60.0, 13)
        low, high = posterior.band(at)
        inner_low, inner_high = posterior.band(at, level=0.5)
        self.assertEqual((low.shape, high.shape), ((13,), (13,)))
        self.assertTrue(np.all((0.0 <= low) & (low <= inner_low) & (inner_low <= inner_high)))
        self.assertTrue(np.all((inner_high <= high) & (high <= 1.0)))
        self.assertEqual(posterior.band(7.0)[0].shape, ())
        with self.assertRaises(ValueError):
            posterior.band(at, level=1.0)

    def test_without_data_the_band_says_almost_anything_is_possible(self):
        low, high = fit_threshold([], [], **GRIDS).band(40.0)
        self.assertLess(float(low), 0.05)
        self.assertGreater(float(high), 0.9)


class CountDistributionTests(unittest.TestCase):
    """The predicted number of successes among trials at given distances: the lookup prediction's interval."""

    SMALL = PredictTests.SMALL

    def by_listing_every_outcome(self, posterior: ThresholdPosterior, distances: list[float]) -> list[float]:
        """The same distribution the slow way: every curve of the grid, and every way the trials can turn out."""
        totals = [0.0] * (len(distances) + 1)
        for i, r50 in enumerate(posterior.r50_grid_mm):
            for j, width in enumerate(posterior.width_grid_mm):
                for k, lapse in enumerate(posterior.lapse_values):
                    for m, floor in enumerate(posterior.floor_values):
                        chances = [
                            floor + (1.0 - lapse - floor) / (1.0 + math.exp((distance - r50) / width))
                            for distance in distances
                        ]
                        weight = float(posterior.probabilities[i, j, k, m])
                        for outcome in range(2 ** len(distances)):
                            happened = [(outcome >> trial) & 1 for trial in range(len(distances))]
                            chance = math.prod(p if hit else 1.0 - p for p, hit in zip(chances, happened, strict=True))
                            totals[sum(happened)] += weight * chance
        return totals

    def test_it_is_the_mixture_over_the_grid_of_every_way_the_trials_can_turn_out(self):
        distances = [0.0, 8.0, 14.0, 19.5, 26.0, 33.0]
        for trials, outcomes in (([], []), ([5.0, 12.0, 18.0, 26.0], [True, True, False, False])):
            posterior = fit_threshold(trials, outcomes, **self.SMALL)
            with self.subTest(trials=len(trials)):
                counts = posterior.count_distribution(distances)
                self.assertEqual(counts.shape, (7,))
                expected = self.by_listing_every_outcome(posterior, distances)
                self.assertTrue(np.allclose(counts, expected, rtol=0.0, atol=1e-14), (counts, expected))

    def test_it_matches_a_direct_simulation_on_the_protocol_s_grid(self):
        trials, outcomes = draw_trials(seed=21, n=34, r50_mm=21.0, width_mm=3.0, lapse=0.03, max_mm=50.0)
        posterior = fit_threshold(trials, outcomes, **FLOOR_GRIDS)
        distances = np.random.default_rng(22).uniform(0.0, 36.0, size=20)
        counts = posterior.count_distribution(distances)
        # Draw a curve from the posterior, then the trials under it, many times over.
        rng = np.random.default_rng(23)
        draws = 150_000
        picked = rng.choice(posterior.probabilities.size, size=draws, p=posterior.probabilities.reshape(-1))
        i, j, k, m = np.unravel_index(picked, posterior.probabilities.shape)
        chances = success_probability(
            distances[None, :],
            posterior.r50_grid_mm[i][:, None],
            posterior.width_grid_mm[j][:, None],
            posterior.lapse_values[k][:, None],
            posterior.floor_values[m][:, None],
        )
        simulated = np.bincount((rng.random(chances.shape) < chances).sum(axis=1), minlength=21) / draws
        standard_error = np.sqrt(counts * (1.0 - counts) / draws)
        self.assertTrue(np.all(np.abs(simulated - counts) < 5.0 * standard_error + 1e-5), (simulated, counts))

    def test_it_sums_to_one_and_its_mean_is_the_sum_of_the_predicted_probabilities(self):
        trials, outcomes = draw_trials(seed=24, n=34, r50_mm=18.0, width_mm=4.0, lapse=0.05, max_mm=50.0)
        posterior = fit_threshold(trials, outcomes, **FLOOR_GRIDS)
        distances = np.linspace(0.0, 45.0, 26)
        counts = posterior.count_distribution(distances)
        self.assertEqual(counts.shape, (27,))
        self.assertTrue(np.all(counts >= 0.0))
        self.assertAlmostEqual(float(counts.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(counts @ np.arange(27)), float(posterior.predict(distances).sum()), places=10)

    def test_one_trial_succeeds_with_the_predicted_probability(self):
        posterior = fit_threshold([5.0, 30.0], [True, False], **FLOOR_GRIDS)
        counts = posterior.count_distribution([17.0])
        chance = float(posterior.predict(17.0))
        self.assertTrue(np.allclose(counts, [1.0 - chance, chance], rtol=0.0, atol=1e-13))
        self.assertTrue(np.array_equal(posterior.count_distribution([]), [1.0]))

    def test_under_one_curve_it_is_the_count_of_independent_trials(self):
        one_curve = {"r50_grid_mm": [20.0], "width_grid_mm": [4.0], "lapse_values": [0.05], "floor_values": [0.1]}
        posterior = fit_threshold([], [], **one_curve)
        counts = posterior.count_distribution([20.0, 20.0, 20.0])
        p = 0.1 + 0.85 / 2.0  # halfway between the floor and the plateau at r50
        binomial = [(1 - p) ** 3, 3 * p * (1 - p) ** 2, 3 * p**2 * (1 - p), p**3]
        self.assertTrue(np.allclose(counts, binomial, rtol=0.0, atol=1e-15))

    def test_it_is_wider_than_a_count_predicted_from_the_mean_curve_alone(self):
        # What the fit leaves uncertain moves every trial's chance the same way, which spreads the count.
        trials, outcomes = draw_trials(seed=25, n=22, r50_mm=20.0, width_mm=3.0, lapse=0.0, max_mm=45.0)
        posterior = fit_threshold(trials, outcomes, **FLOOR_GRIDS)
        distances = np.full(20, 19.0)
        counts = posterior.count_distribution(distances)
        mean = float(counts @ np.arange(21))
        variance = float(counts @ (np.arange(21) - mean) ** 2)
        chance = float(posterior.predict(19.0))
        self.assertGreater(variance, 1.5 * 20 * chance * (1.0 - chance))

    def test_the_order_of_the_distances_does_not_change_one_bit(self):
        posterior = fit_threshold([5.0, 12.0, 18.0, 26.0], [True, True, False, False], **FLOOR_GRIDS)
        distances = np.random.default_rng(26).uniform(0.0, 40.0, size=12)
        shuffled = np.random.default_rng(27).permutation(distances)
        counts = posterior.count_distribution(distances)
        self.assertTrue(np.array_equal(counts, posterior.count_distribution(shuffled)))
        self.assertTrue(np.array_equal(counts, posterior.count_distribution(distances.reshape(3, 4))))

    def test_a_distance_that_cannot_be_is_refused(self):
        posterior = fit_threshold([5.0, 30.0], [True, False], **GRIDS)
        for bad in ([-1.0], [float("nan")], [3.0, float("inf")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                posterior.count_distribution(bad)

    @unittest.skipIf(scipy_stats is None or not hasattr(scipy_stats, "poisson_binom"), "needs SciPy's poisson_binom")
    def test_under_one_curve_it_matches_scipy_s_poisson_binomial(self):
        one_curve = {"r50_grid_mm": [22.0], "width_grid_mm": [3.0], "lapse_values": [0.02], "floor_values": [0.05]}
        posterior = fit_threshold([], [], **one_curve)
        distances = np.linspace(2.0, 40.0, 20)
        chances = success_probability(distances, 22.0, 3.0, 0.02, 0.05)
        expected = scipy_stats.poisson_binom.pmf(np.arange(21), chances)
        self.assertTrue(np.allclose(posterior.count_distribution(distances), expected, rtol=0.0, atol=1e-13))


class GridsFromProtocolTests(unittest.TestCase):
    ANALYSIS = {
        "r50_grid_mm": {"min": 0.0, "max": 2.0, "step": 0.5},
        "width_grid_mm": {"min": 1.0, "max": 16.0, "count": 5},
        "lapse_values": [0.0, 0.1],
        "interval_level": 0.95,
    }

    def test_it_builds_an_even_grid_a_logarithmic_grid_and_the_listed_lapses(self):
        grids = grids_from_protocol(self.ANALYSIS)
        self.assertTrue(np.allclose(grids.r50_grid_mm, [0.0, 0.5, 1.0, 1.5, 2.0], rtol=0, atol=1e-12))
        self.assertTrue(np.allclose(grids.width_grid_mm, [1.0, 2.0, 4.0, 8.0, 16.0], rtol=1e-12))
        self.assertTrue(np.array_equal(grids.lapse_values, [0.0, 0.1]))
        r50_grid, width_grid, lapse_values, floor_values = grids
        self.assertIs(r50_grid, grids.r50_grid_mm)

    def test_floor_values_are_read_when_the_section_has_them_and_are_a_single_zero_when_it_does_not(self):
        self.assertTrue(np.array_equal(grids_from_protocol(self.ANALYSIS).floor_values, [0.0]))
        with_floors = grids_from_protocol(self.ANALYSIS | {"floor_values": [0.0, 0.05, 0.10]})
        self.assertTrue(np.array_equal(with_floors.floor_values, [0.0, 0.05, 0.10]))
        posterior = fit_threshold([0.5, 1.5], [True, False], **with_floors._asdict())
        self.assertEqual(posterior.probabilities.shape, (5, 5, 2, 3))
        for bad in ([], [0.0, 0.95], [-0.1], "0.05"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                grids_from_protocol(self.ANALYSIS | {"floor_values": bad})

    def test_grids_made_before_the_floor_existed_have_none(self):
        three = psychometric.AnalysisGrids(np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([0.0]))
        self.assertTrue(np.array_equal(three.floor_values, [0.0]))
        with self.assertRaises(ValueError):
            three.floor_values[0] = 0.5  # the shared default must not be writable

    def test_the_repository_protocol_gives_grids_that_end_where_it_says(self):
        analysis = config.load_protocol()["analysis"]
        grids = grids_from_protocol(analysis)
        self.assertEqual(grids.r50_grid_mm[0], analysis["r50_grid_mm"]["min"])
        self.assertEqual(grids.r50_grid_mm[-1], analysis["r50_grid_mm"]["max"])
        self.assertTrue(np.allclose(np.diff(grids.r50_grid_mm), analysis["r50_grid_mm"]["step"], rtol=0, atol=1e-9))
        self.assertEqual(grids.width_grid_mm.size, analysis["width_grid_mm"]["count"])
        self.assertAlmostEqual(grids.width_grid_mm[0], analysis["width_grid_mm"]["min"], places=12)
        self.assertAlmostEqual(grids.width_grid_mm[-1], analysis["width_grid_mm"]["max"], places=12)
        ratios = grids.width_grid_mm[1:] / grids.width_grid_mm[:-1]
        self.assertTrue(np.allclose(ratios, ratios[0], rtol=1e-12))
        self.assertEqual(list(grids.lapse_values), analysis["lapse_values"])
        self.assertEqual(list(grids.floor_values), analysis.get("floor_values", [0.0]))

    def test_the_grids_go_straight_into_the_fit(self):
        posterior = fit_threshold([0.5, 1.5], [True, False], **grids_from_protocol(self.ANALYSIS)._asdict())
        self.assertEqual(posterior.probabilities.shape, (5, 5, 2, 1))

    def test_a_section_that_does_not_describe_grids_is_refused(self):
        bad_changes = (
            {"r50_grid_mm": {"min": 0.0, "max": 2.0, "step": 0.75}},  # the step does not land on the maximum
            {"r50_grid_mm": {"min": 0.0, "max": 2.0, "step": 0.0}},
            {"r50_grid_mm": {"min": 2.0, "max": 0.0, "step": 0.5}},
            {"r50_grid_mm": {"min": 0.0, "max": 2.0}},  # no step
            {"width_grid_mm": {"min": 0.0, "max": 16.0, "count": 5}},
            {"width_grid_mm": {"min": 1.0, "max": 16.0, "count": 1}},
            {"width_grid_mm": {"min": 4.0, "max": 1.0, "count": 5}},
            {"lapse_values": []},
        )
        for change in bad_changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                grids_from_protocol(self.ANALYSIS | change)
        with self.assertRaises(ValueError):
            grids_from_protocol({"interval_level": 0.95})


if __name__ == "__main__":
    unittest.main()
