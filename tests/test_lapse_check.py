"""The check of the model's lapse values runs on the protocol and says what it found, the same way every time."""

import unittest

from tolerance_radius import config, lapse_check

PROTOCOL = config.load_protocol()


class LapseCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = lapse_check.run(PROTOCOL, replications=6, seed=3, lapses=(0.0, 0.35), radii=(20.0,), widths=(3.0,))

    def test_one_row_for_every_block_curve_and_list(self):
        blocks = {row["block"] for row in self.rows}
        self.assertEqual(blocks, {"T1a", "T1b", "T3"})
        self.assertEqual(len(self.rows), len(blocks) * 2 * len(lapse_check.FITS))
        for row in self.rows:
            self.assertGreaterEqual(row["rmse_mm"], abs(row["bias_mm"]) - 1e-9)
            self.assertTrue(0.0 <= row["contains_truth"] <= 1.0)

    def test_the_same_seed_gives_the_same_rows(self):
        again = lapse_check.run(PROTOCOL, replications=6, seed=3, lapses=(0.0, 0.35), radii=(20.0,), widths=(3.0,))
        self.assertEqual(again, self.rows)

    def test_the_wider_list_is_the_protocols_own_and_contains_its_main_values(self):
        wider = lapse_check.wider_list(PROTOCOL)
        self.assertEqual(list(wider), PROTOCOL["analysis"]["lapse_values_sensitivity"])
        self.assertTrue(set(PROTOCOL["analysis"]["lapse_values"]) <= set(wider))
        self.assertEqual(lapse_check.wider_list({"analysis": {}}), lapse_check.LAPSE_WIDER)

    def test_pooling_averages_the_rows_asked_for_and_refuses_an_empty_choice(self):
        pooled = lapse_check.pooled(self.rows, fit=lapse_check.FITS[0], true_lapse=0.0)
        chosen = [r for r in self.rows if r["fit"] == lapse_check.FITS[0] and r["true_lapse"] == 0.0]
        self.assertAlmostEqual(pooled["bias_mm"], sum(r["bias_mm"] for r in chosen) / len(chosen))
        with self.assertRaises(ValueError):
            lapse_check.pooled(self.rows, block="T9")

    def test_the_report_says_it_is_simulated_and_has_a_row_for_each_plateau(self):
        text = lapse_check.render_report(self.rows, replications=6, seed=3, protocol_sha="x" * 12,
                                         protocol_lapses=PROTOCOL["analysis"]["lapse_values"])
        self.assertIn("Nothing here is a measurement", text)
        self.assertIn("| 1.00 | as the protocol |", text)
        self.assertIn("| 0.65 | wider list |", text)


if __name__ == "__main__":
    unittest.main()
