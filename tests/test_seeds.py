"""Derived random generators: one stream per named choice, the same in every environment the project runs in."""

import unittest

import numpy as np

from tolerance_radius.seeds import derived_rng, pick_index, shuffled


class DerivedRngTests(unittest.TestCase):
    def test_the_same_seed_and_labels_give_the_same_stream(self):
        first = derived_rng(20261006, "spread", "positions").random(5)
        second = derived_rng(20261006, "spread", "positions").random(5)
        self.assertEqual(first.tolist(), second.tolist())

    def test_a_different_seed_or_label_gives_a_different_stream(self):
        base = derived_rng(7, "T1a", "order").random(4).tolist()
        self.assertNotEqual(derived_rng(8, "T1a", "order").random(4).tolist(), base)
        self.assertNotEqual(derived_rng(7, "T1b", "order").random(4).tolist(), base)
        self.assertNotEqual(derived_rng(7, "T1a", "order", 0).random(4).tolist(), base)

    def test_labels_are_not_run_together(self):
        self.assertNotEqual(derived_rng(1, "ab", "c").random(), derived_rng(1, "a", "bc").random())
        self.assertNotEqual(derived_rng(1, "a", 12).random(), derived_rng(1, "a1", 2).random())
        self.assertNotEqual(derived_rng(12, "a").random(), derived_rng(1, "2", "a").random())

    def test_it_returns_a_numpy_generator(self):
        self.assertIsInstance(derived_rng(1, "x"), np.random.Generator)

    def test_a_seed_must_be_a_whole_number(self):
        for bad in (1.5, "20261006", None, True):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                derived_rng(bad, "x")

    def test_the_stream_is_pinned_so_that_every_environment_draws_the_same_numbers(self):
        # Trials are planned with the rig's NumPy and analysed with another. This fails in whichever environment
        # stops producing these numbers.
        values = derived_rng(20261006, "pinned", 3).random(3).tolist()
        self.assertEqual(values, [0.4736987654276542, 0.6807840771030695, 0.8630621622417889])


class ShuffledTests(unittest.TestCase):
    def test_it_returns_the_same_items_in_a_seeded_order_and_leaves_the_input_alone(self):
        items = ["A2", "A3", "B1", "B3", "C1", "C2"]
        result = shuffled(items, derived_rng(3, "round"))
        self.assertEqual(sorted(result), items)
        self.assertEqual(items, ["A2", "A3", "B1", "B3", "C1", "C2"])
        self.assertEqual(result, shuffled(items, derived_rng(3, "round")))

    def test_different_streams_give_different_orders(self):
        items = list(range(20))
        self.assertNotEqual(shuffled(items, derived_rng(3, "a")), shuffled(items, derived_rng(3, "b")))

    def test_every_order_of_three_items_turns_up(self):
        rng = derived_rng(5, "orders")
        seen = {tuple(shuffled("abc", rng)) for _ in range(200)}
        self.assertEqual(len(seen), 6)

    def test_the_order_is_pinned(self):
        self.assertEqual(shuffled(list("ABCDEFGH"), derived_rng(20261006, "pinned")), list("GDFACBEH"))


class PickIndexTests(unittest.TestCase):
    def test_it_stays_in_range_and_reaches_every_index(self):
        rng = derived_rng(11, "pick")
        picks = [pick_index(4, rng) for _ in range(400)]
        self.assertEqual(set(picks), {0, 1, 2, 3})
        for index in range(4):
            self.assertGreater(picks.count(index), 60)

    def test_nothing_to_pick_from_is_an_error(self):
        with self.assertRaises(ValueError):
            pick_index(0, derived_rng(1, "x"))


if __name__ == "__main__":
    unittest.main()
