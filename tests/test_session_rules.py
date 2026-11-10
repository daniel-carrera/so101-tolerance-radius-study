"""What a trial session must satisfy before it may run: its execution settings and its blocks."""

import copy
import unittest

from fixtures_demos import PROTOCOL, audit
from tolerance_radius import session_rules

AUDITS = (*audit("clustered"), *audit("spread"))


def settings(**changes):
    """The protocol's own execution settings, with some changed."""
    return {**PROTOCOL["execution"], **changes}


class ExecutionSettingsTests(unittest.TestCase):
    def test_the_protocols_own_settings_have_no_problem(self):
        self.assertEqual(session_rules.execution_problems(PROTOCOL, settings()), [])

    def test_the_protocol_states_every_setting_the_rule_knows(self):
        self.assertEqual(set(PROTOCOL["execution"]), set(session_rules.EXECUTION_KEYS))

    def test_a_setting_that_differs_is_named_with_both_values(self):
        problems = session_rules.execution_problems(PROTOCOL, settings(clamp_deg=10.0))
        self.assertEqual(len(problems), 1)
        self.assertIn("clamp_deg", problems[0])
        self.assertIn("10", problems[0])
        self.assertIn("15", problems[0])

    def test_every_setting_is_checked(self):
        for key in session_rules.EXECUTION_KEYS:
            changed = settings(**{key: PROTOCOL["execution"][key] + 1})
            with self.subTest(key=key):
                problems = session_rules.execution_problems(PROTOCOL, changed)
                self.assertEqual(len(problems), 1)
                self.assertIn(key, problems[0])

    def test_a_whole_number_and_the_same_number_with_a_point_are_the_same_setting(self):
        self.assertEqual(session_rules.execution_problems(PROTOCOL, settings(control_hz=30.0, clamp_deg=15)), [])

    def test_a_tiny_difference_is_still_a_difference(self):
        self.assertEqual(len(session_rules.execution_problems(PROTOCOL, settings(temporal_ensemble_coeff=0.011))), 1)

    def test_a_setting_left_out_or_switched_off_is_a_problem(self):
        missing = settings()
        del missing["temporal_ensemble_coeff"]
        self.assertIn("temporal_ensemble_coeff", session_rules.execution_problems(PROTOCOL, missing)[0])
        # August's unnoticed fault was a setting that was simply not on.
        off = session_rules.execution_problems(PROTOCOL, settings(temporal_ensemble_coeff=None))
        self.assertEqual(len(off), 1)
        self.assertIn("temporal_ensemble_coeff", off[0])

    def test_a_setting_the_protocol_does_not_state_is_a_problem(self):
        problems = session_rules.execution_problems(PROTOCOL, settings(action_delay_steps=2))
        self.assertEqual(len(problems), 1)
        self.assertIn("action_delay_steps", problems[0])

    def test_a_protocol_without_the_section_is_a_problem_and_not_a_pass(self):
        protocol = copy.deepcopy(PROTOCOL)
        del protocol["execution"]
        problems = session_rules.execution_problems(protocol, settings())
        self.assertEqual(len(problems), 1)
        self.assertIn("[execution]", problems[0])

    def test_several_differences_are_all_reported_in_the_protocols_order(self):
        problems = session_rules.execution_problems(PROTOCOL, settings(stall_steps=0, control_hz=15))
        self.assertEqual(len(problems), 2)
        self.assertIn("control_hz", problems[0])
        self.assertIn("stall_steps", problems[1])


class ExecutionDefaultsTests(unittest.TestCase):
    def test_settings_not_given_are_taken_from_the_protocol(self):
        merged, given = session_rules.resolved_execution(PROTOCOL, {"clamp_deg": None, "control_hz": None})
        self.assertEqual(merged, dict(PROTOCOL["execution"]))
        self.assertEqual(given, [])

    def test_a_setting_given_by_hand_is_kept_and_reported(self):
        merged, given = session_rules.resolved_execution(PROTOCOL, {"clamp_deg": 10.0})
        self.assertEqual(merged["clamp_deg"], 10.0)
        self.assertEqual(merged["control_hz"], PROTOCOL["execution"]["control_hz"])
        self.assertEqual(given, ["clamp_deg"])

    def test_a_setting_given_by_hand_at_the_protocols_value_is_not_a_change(self):
        merged, given = session_rules.resolved_execution(PROTOCOL, {"clamp_deg": 15.0})
        self.assertEqual(given, [])
        self.assertEqual(session_rules.execution_problems(PROTOCOL, merged), [])

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(KeyError):
            session_rules.resolved_execution(PROTOCOL, {"clamp": 10.0})


class BlockProblemsTests(unittest.TestCase):
    def test_the_repositorys_blocks_can_all_be_run_against_the_planned_layouts(self):
        self.assertEqual(session_rules.block_problems(("T1a", "T1b", "T2", "T3"), PROTOCOL, AUDITS), [])

    def test_a_block_that_would_stop_midway_is_reported_with_its_name(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["blocks"]["T1a"]["max_level_mm"] = 95.0
        problems = session_rules.block_problems(("T1a", "T1b"), protocol, AUDITS)
        self.assertEqual(len(problems), 1)
        self.assertIn("T1a", problems[0])
        self.assertIn("95", problems[0])

    def test_an_unknown_block_is_reported_and_does_not_raise(self):
        problems = session_rules.block_problems(("T9",), PROTOCOL, AUDITS)
        self.assertEqual(len(problems), 1)
        self.assertIn("T9", problems[0])


if __name__ == "__main__":
    unittest.main()
