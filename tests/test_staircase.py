"""The adaptive rule: where it starts, how it moves, when it shrinks its step, where it stops."""

import unittest

from tolerance_radius.staircase import StaircaseParams, next_level, reversal_indices

PARAMS = StaircaseParams(
    start_level_mm=10.0, n_trials=6, step_initial_mm=10.0, step_final_mm=5.0, min_level_mm=0.0, max_level_mm=50.0
)


def run(outcomes: list[bool], params: StaircaseParams = PARAMS) -> list[float]:
    """Levels the rule asks for when the trials come out as given."""
    history: list[tuple[float, bool]] = []
    levels: list[float] = []
    for success in outcomes:
        level = next_level(params, history)
        levels.append(level)
        history.append((level, success))
    return levels


class NextLevelTests(unittest.TestCase):
    def test_it_starts_at_the_starting_level(self):
        self.assertEqual(next_level(PARAMS, []), 10.0)

    def test_success_moves_farther_and_failure_moves_closer(self):
        self.assertEqual(next_level(PARAMS, [(10.0, True)]), 20.0)
        self.assertEqual(next_level(PARAMS, [(10.0, False)]), 0.0)

    def test_the_step_shrinks_after_the_first_reversal_and_stays_small(self):
        #                 10    20    30 (fail: first reversal)  25    20 (success: reversal)  25
        self.assertEqual(run([True, True, False, False, True, True]), [10.0, 20.0, 30.0, 25.0, 20.0, 25.0])

    def test_the_level_stays_within_its_limits(self):
        low = StaircaseParams(5.0, 5, 10.0, 5.0, 0.0, 50.0)
        self.assertEqual(run([False, False, False], low), [5.0, 0.0, 0.0])
        high = StaircaseParams(40.0, 5, 10.0, 5.0, 0.0, 50.0)
        self.assertEqual(run([True, True, True], high), [40.0, 50.0, 50.0])

    def test_it_ends_after_its_trials(self):
        history = [(10.0, True)] * PARAMS.n_trials
        self.assertIsNone(next_level(PARAMS, history))

    def test_the_same_history_always_gives_the_same_level(self):
        history = [(10.0, True), (20.0, False), (15.0, True)]
        self.assertEqual(next_level(PARAMS, history), next_level(PARAMS, list(history)))
        self.assertEqual(next_level(PARAMS, history), 20.0)


class ReversalTests(unittest.TestCase):
    def test_a_reversal_is_a_change_of_outcome(self):
        history = [(10.0, True), (20.0, True), (30.0, False), (25.0, False), (20.0, True)]
        self.assertEqual(reversal_indices(history), [2, 4])

    def test_no_reversal_in_a_run_of_equal_outcomes(self):
        self.assertEqual(reversal_indices([(10.0, True), (20.0, True)]), [])
        self.assertEqual(reversal_indices([]), [])


class ParamsTests(unittest.TestCase):
    def test_impossible_staircases_are_refused(self):
        for bad in (
            {"n_trials": 0},
            {"step_final_mm": 0.0},
            {"start_level_mm": 60.0},
        ):
            values = {
                "start_level_mm": 10.0,
                "n_trials": 6,
                "step_initial_mm": 10.0,
                "step_final_mm": 5.0,
                "min_level_mm": 0.0,
                "max_level_mm": 50.0,
            } | bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                StaircaseParams(**values)


if __name__ == "__main__":
    unittest.main()
