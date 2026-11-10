"""The two demonstration layouts: the same budget at the cell centres, or spread evenly over the trained cells."""

import copy
import itertools
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

from tolerance_radius import config, layouts
from tolerance_radius.config import ConfigError
from tolerance_radius.geometry import Grid
from tolerance_radius.layouts import LayoutError, PlannedEpisode
from tolerance_radius.schema import Point
from tolerance_radius.seeds import derived_rng

PROTOCOL = config.load_protocol()
GRID = Grid.from_protocol(PROTOCOL)
SEED = PROTOCOL["datasets"]["layout_seed"]
MIN_SPACING_MM = PROTOCOL["datasets"]["spread_min_spacing_mm"]
_CACHE: dict[tuple[str, int], list[PlannedEpisode]] = {}


def layout(dataset: str, seed: int = SEED) -> list[PlannedEpisode]:
    """A layout of the repository's protocol, computed once per seed."""
    if (dataset, seed) not in _CACHE:
        _CACHE[(dataset, seed)] = layouts.layout(dataset, PROTOCOL, seed)
    return _CACHE[(dataset, seed)]


def min_spacing(episodes: list[PlannedEpisode]) -> float:
    return min(a.target.distance_to(b.target) for a, b in itertools.combinations(episodes, 2))


class BothLayoutsTests(unittest.TestCase):
    def test_55_episodes_five_in_each_trained_cell_and_none_in_a_held_out_cell(self):
        for dataset in ("clustered", "spread"):
            with self.subTest(dataset=dataset):
                episodes = layout(dataset)
                self.assertEqual(len(episodes), 55)
                self.assertEqual({e.dataset for e in episodes}, {dataset})
                self.assertEqual(Counter(e.cell for e in episodes), dict.fromkeys(GRID.trained_cells, 5))
                for episode in episodes:
                    self.assertEqual(GRID.cell_of(episode.target), episode.cell)
                    self.assertTrue(GRID.is_trained(episode.cell))

    def test_every_round_visits_each_trained_cell_once(self):
        for dataset in ("clustered", "spread"):
            episodes = layout(dataset)
            for round_number in range(1, 6):
                cells = [e.cell for e in episodes if e.round == round_number]
                with self.subTest(dataset=dataset, round=round_number):
                    self.assertEqual(sorted(cells), sorted(GRID.trained_cells))

    def test_the_cells_of_a_round_are_shuffled_differently_from_round_to_round(self):
        for dataset in ("clustered", "spread"):
            episodes = layout(dataset)
            orders = {tuple(e.cell for e in episodes if e.round == r) for r in range(1, 6)}
            with self.subTest(dataset=dataset):
                self.assertEqual(len(orders), 5)
                self.assertNotIn(tuple(GRID.trained_cells), orders)

    def test_the_sittings_are_those_of_the_protocol(self):
        # clustered rounds 1-3, spread rounds 1-3, spread rounds 4-5, clustered rounds 4-5
        expected = {"clustered": {1: 1, 2: 1, 3: 1, 4: 4, 5: 4}, "spread": {1: 2, 2: 2, 3: 2, 4: 3, 5: 3}}
        for dataset in ("clustered", "spread"):
            for episode in layout(dataset):
                self.assertEqual(episode.sitting, expected[dataset][episode.round])

    def test_episode_indices_count_through_the_rounds_in_recording_order(self):
        for dataset in ("clustered", "spread"):
            episodes = layout(dataset)
            self.assertEqual([e.episode_index for e in episodes], list(range(55)))
            self.assertEqual([e.round for e in episodes], [index // 11 + 1 for index in range(55)])

    def test_episode_indices_follow_the_order_in_which_the_sittings_list_the_rounds(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["datasets"]["sittings"] = [
            {"dataset": "spread", "rounds": [4, 5]},
            {"dataset": "clustered", "rounds": [2, 1, 3]},
            {"dataset": "clustered", "rounds": [5, 4]},
            {"dataset": "spread", "rounds": [1, 2, 3]},
        ]
        clustered = layouts.clustered_layout(protocol)
        self.assertEqual([e.episode_index for e in clustered], list(range(55)))
        self.assertEqual([e.round for e in clustered[::11]], [2, 1, 3, 5, 4])
        self.assertEqual([e.sitting for e in clustered[::11]], [2, 2, 2, 3, 3])
        spread = layouts.spread_layout(protocol)
        self.assertEqual([e.round for e in spread[::11]], [4, 5, 1, 2, 3])
        self.assertEqual([e.sitting for e in spread[::11]], [1, 1, 4, 4, 4])

    def test_the_order_of_a_round_does_not_depend_on_how_the_sittings_are_arranged(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["datasets"]["sittings"] = [
            {"dataset": "clustered", "rounds": [5, 4, 3, 2, 1]},
            {"dataset": "spread", "rounds": [1, 2, 3, 4, 5]},
        ]
        rearranged = layouts.clustered_layout(protocol)
        for round_number in range(1, 6):
            self.assertEqual(
                [(e.cell, e.target) for e in rearranged if e.round == round_number],
                [(e.cell, e.target) for e in layout("clustered") if e.round == round_number],
            )

    def test_sittings_that_miss_or_repeat_a_round_are_refused(self):
        for rounds in ([1, 2, 3], [1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 6]):
            protocol = copy.deepcopy(PROTOCOL)
            protocol["datasets"]["sittings"] = [
                {"dataset": "clustered", "rounds": rounds},
                {"dataset": "spread", "rounds": [1, 2, 3, 4, 5]},
            ]
            with self.subTest(rounds=rounds), self.assertRaises(ConfigError):
                layouts.clustered_layout(protocol)

    def test_the_seed_defaults_to_the_one_in_the_protocol(self):
        self.assertEqual(layouts.clustered_layout(PROTOCOL), layout("clustered", SEED))
        self.assertEqual(layouts.spread_layout(PROTOCOL), layout("spread", SEED))

    def test_the_same_seed_gives_the_same_layout(self):
        for dataset in ("clustered", "spread"):
            self.assertEqual(layouts.layout(dataset, PROTOCOL, 7), layouts.layout(dataset, PROTOCOL, 7))

    def test_a_different_seed_gives_a_different_layout(self):
        self.assertNotEqual([e.cell for e in layout("clustered", 1)], [e.cell for e in layout("clustered", 2)])
        self.assertNotEqual([e.cell for e in layout("spread", 1)], [e.cell for e in layout("spread", 2)])
        self.assertNotEqual({e.target for e in layout("spread", 1)}, {e.target for e in layout("spread", 2)})

    def test_the_two_sets_do_not_share_their_round_orders(self):
        self.assertNotEqual([e.cell for e in layout("clustered")], [e.cell for e in layout("spread")])

    def test_an_unknown_layout_is_refused(self):
        with self.assertRaises(LayoutError):
            layouts.layout("scattered", PROTOCOL)


class ClusteredLayoutTests(unittest.TestCase):
    def test_every_episode_starts_at_the_centre_of_its_cell(self):
        for episode in layout("clustered"):
            self.assertEqual(episode.target, GRID.cell_centre(episode.cell))

    def test_only_the_order_depends_on_the_seed(self):
        self.assertEqual({e.target for e in layout("clustered", 1)}, {e.target for e in layout("clustered", 2)})


class SpreadLayoutTests(unittest.TestCase):
    SEEDS = (SEED, 1, 2, 3)

    def test_the_five_positions_of_a_cell_are_distinct(self):
        for seed in self.SEEDS:
            self.assertEqual(len({e.target for e in layout("spread", seed)}), 55)

    def test_no_two_positions_are_closer_than_the_minimum_spacing(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                self.assertGreaterEqual(min_spacing(layout("spread", seed)), MIN_SPACING_MM)

    def test_no_position_is_in_the_margin_at_the_outer_edge(self):
        for seed in self.SEEDS:
            for episode in layout("spread", seed):
                self.assertTrue(GRID.in_test_region(episode.target), episode)

    def test_a_position_stays_inside_its_cell_by_the_placement_tolerance(self):
        # A cube accepted within the tolerance of its target is then still measured in its own cell.
        tolerance = PROTOCOL["placement"]["tolerance_mm"]
        for seed in self.SEEDS:
            for episode in layout("spread", seed):
                rect = GRID.cell_rect(episode.cell)
                self.assertGreaterEqual(episode.target.x_mm - rect.x_min_mm, tolerance)
                self.assertGreaterEqual(rect.x_max_mm - episode.target.x_mm, tolerance)
                self.assertGreaterEqual(episode.target.y_mm - rect.y_min_mm, tolerance)
                self.assertGreaterEqual(rect.y_max_mm - episode.target.y_mm, tolerance)

    def test_positions_are_whole_millimetres(self):
        for episode in layout("spread"):
            self.assertEqual(episode.target.x_mm, round(episode.target.x_mm))
            self.assertEqual(episode.target.y_mm, round(episode.target.y_mm))

    def test_a_spacing_that_cannot_be_met_is_an_error(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["datasets"]["spread_min_spacing_mm"] = 40.0
        with self.assertRaises(LayoutError):
            layouts.spread_layout(protocol)



def as_episodes(positions: dict[str, list[Point]]) -> list[PlannedEpisode]:
    """Positions by cell as a list of episodes, for the quality figures."""
    pairs = [(cell, target) for cell in positions for target in positions[cell]]
    return [PlannedEpisode("spread", index, 1, 1, cell, target) for index, (cell, target) in enumerate(pairs)]


class SpreadConstraintTests(unittest.TestCase):
    """Small tables on which a rule is in the way of the most even layout, so that it has to hold against every step."""

    @staticmethod
    def positions(grid: Grid, per_cell: int, seed: int, min_spacing_mm: float, cell_inset_mm: float) -> dict:
        return layouts.spread_positions(grid, per_cell, min_spacing_mm, cell_inset_mm, derived_rng(seed, "t"))

    def test_the_minimum_spacing_holds_where_an_even_layout_would_undercut_it(self):
        # Two cells with one position each: the even layout puts them at the two centres, 50 mm apart.
        grid = Grid(cols=2, rows=1, cell_mm=50.0, held_out_cells=(), cube_edge_mm=30.0, test_region_margin_mm=5.0)
        for seed in range(6):
            positions = self.positions(grid, 1, seed, min_spacing_mm=50.5, cell_inset_mm=2.0)
            with self.subTest(seed=seed):
                self.assertGreaterEqual(positions["A1"][0].distance_to(positions["B1"][0]), 50.5)

    def test_the_margin_holds_where_an_even_layout_would_reach_into_it(self):
        # Four positions in one cell sit best 12.5 mm from its edges; a margin of 15 mm keeps them farther in.
        grid = Grid(cols=1, rows=1, cell_mm=50.0, held_out_cells=(), cube_edge_mm=30.0, test_region_margin_mm=15.0)
        for seed in range(6):
            positions = self.positions(grid, 4, seed, min_spacing_mm=5.0, cell_inset_mm=0.0)
            for target in positions["A1"]:
                with self.subTest(seed=seed, target=target):
                    self.assertTrue(15.0 <= target.x_mm <= 35.0 and 15.0 <= target.y_mm <= 35.0)

    def test_the_inset_from_the_cell_borders_holds_where_an_even_layout_would_come_closer(self):
        # Two positions in each of two cells sit best 12.5 mm from the border between the cells.
        grid = Grid(cols=2, rows=1, cell_mm=50.0, held_out_cells=(), cube_edge_mm=30.0, test_region_margin_mm=5.0)
        for seed in range(6):
            positions = self.positions(grid, 2, seed, min_spacing_mm=12.0, cell_inset_mm=15.0)
            with self.subTest(seed=seed):
                self.assertTrue(all(target.x_mm <= 35.0 for target in positions["A1"]))
                self.assertTrue(all(target.x_mm >= 65.0 for target in positions["B1"]))
                self.assertTrue(all(15.0 <= t.y_mm <= 35.0 for cell in positions for t in positions[cell]))

    def test_a_position_never_sits_on_the_border_with_the_next_cell(self):
        # With no inset and no margin a position could land on x = 50, which belongs to the next cell.
        grid = Grid(cols=2, rows=2, cell_mm=4.0, held_out_cells=(), cube_edge_mm=1.0, test_region_margin_mm=0.0)
        for seed in range(6):
            positions = self.positions(grid, 3, seed, min_spacing_mm=1.0, cell_inset_mm=0.0)
            for cell, targets in positions.items():
                for target in targets:
                    self.assertEqual(grid.cell_of(target), cell)


class SpreadRulesCheckTests(unittest.TestCase):
    """The check that every set of spread positions passes before it is returned."""

    TWO_CELLS = Grid(cols=2, rows=1, cell_mm=50.0, held_out_cells=(), cube_edge_mm=30.0, test_region_margin_mm=5.0)

    def check(self, a: Point, b: Point, **rules: float) -> None:
        rules = {"per_cell": 1, "min_spacing_mm": 12.0, "cell_inset_mm": 2.0} | rules
        layouts.check_spread_positions({"A1": [a], "B1": [b]}, self.TWO_CELLS, **rules)

    def test_positions_that_keep_every_rule_pass(self):
        self.check(Point(20.0, 25.0), Point(80.0, 25.0))
        self.check(Point(5.0, 5.0), Point(95.0, 45.0))  # on the edge of the margin: allowed
        self.check(Point(48.0, 25.0), Point(60.0, 25.0))  # exactly the inset and exactly the spacing: allowed

    def test_two_positions_closer_than_the_spacing_are_refused(self):
        with self.assertRaisesRegex(LayoutError, "apart"):
            self.check(Point(45.0, 25.0), Point(56.0, 25.0))

    def test_a_position_in_the_margin_is_refused(self):
        with self.assertRaisesRegex(LayoutError, "margin"):
            self.check(Point(4.0, 25.0), Point(80.0, 25.0))
        with self.assertRaisesRegex(LayoutError, "margin"):
            self.check(Point(20.0, 25.0), Point(80.0, 46.0))

    def test_a_position_in_another_cell_or_too_near_its_own_border_is_refused(self):
        with self.assertRaisesRegex(LayoutError, "A1"):
            self.check(Point(60.0, 25.0), Point(80.0, 25.0))
        with self.assertRaisesRegex(LayoutError, "A1"):
            self.check(Point(49.0, 25.0), Point(80.0, 25.0))

    def test_the_wrong_number_of_positions_in_a_cell_is_refused(self):
        with self.assertRaisesRegex(LayoutError, "B1"):
            self.check(Point(20.0, 25.0), Point(80.0, 25.0), per_cell=2)
        with self.assertRaises(LayoutError):
            layouts.check_spread_positions({"A1": [Point(20.0, 25.0)]}, self.TWO_CELLS, 1, 12.0, 2.0)

    def test_the_layouts_of_the_protocol_pass(self):
        for seed in SpreadLayoutTests.SEEDS:
            positions: dict[str, list[Point]] = {cell: [] for cell in GRID.trained_cells}
            for episode in layout("spread", seed):
                positions[episode.cell].append(episode.target)
            layouts.check_spread_positions(positions, GRID, per_cell=5, min_spacing_mm=12.0, cell_inset_mm=2.0)


class SpreadStepInvariantTests(unittest.TestCase):
    """The two steps that move positions, started by hand where the spacing rule is in their way."""

    def setUp(self):
        self.grid = SpreadRulesCheckTests.TWO_CELLS
        self.boxes = [layouts._allowed_box(self.grid, cell, 2.0) for cell in ("A1", "B1")]
        self.samples = layouts._sample_points(self.grid)

    def test_relaxation_stops_short_of_the_minimum_spacing(self):
        # Left alone the two positions would settle at the cell centres, about 50 mm apart.
        sites = np.array([[5, 25], [95, 25]], dtype=np.int64)
        layouts._relax(sites, [0, 1], self.boxes, 60**2, *self.samples, 40)
        self.assertGreaterEqual(math.dist(sites[0], sites[1]), 60.0)
        self.assertNotEqual(sites.tolist(), [[5, 25], [95, 25]])

    def test_relaxation_without_the_rule_in_the_way_reaches_the_cell_centres(self):
        sites = np.array([[5, 25], [95, 25]], dtype=np.int64)
        layouts._relax(sites, [0, 1], self.boxes, 12**2, *self.samples, 40)
        self.assertLessEqual(abs(int(sites[0, 0]) - 25), 1)
        self.assertLessEqual(abs(int(sites[1, 0]) - 75), 1)
        self.assertEqual(sites[:, 1].tolist(), [25, 25])

    def test_closing_gaps_does_not_buy_a_smaller_gap_with_a_broken_spacing(self):
        # The largest gap is midway between the two; closing it would bring them nearer than 60 mm.
        sites = np.array([[20, 25], [80, 25]], dtype=np.int64)
        layouts._close_gaps(sites, [0, 1], self.boxes, 60**2, *self.samples, 400)
        self.assertGreaterEqual(math.dist(sites[0], sites[1]), 60.0)

    def test_closing_gaps_without_the_rule_in_the_way_moves_towards_the_gap(self):
        sites = np.array([[20, 25], [80, 25]], dtype=np.int64)
        layouts._close_gaps(sites, [0, 1], self.boxes, 12**2, *self.samples, 400)
        self.assertLess(math.dist(sites[0], sites[1]), 56.0)

    def test_closing_gaps_leaves_a_layout_alone_when_no_move_helps(self):
        # One position in the middle of one cell: every move makes a corner of the cell farther.
        grid = Grid(cols=1, rows=1, cell_mm=50.0, held_out_cells=(), cube_edge_mm=30.0, test_region_margin_mm=5.0)
        for moves in (1, 400):
            sites = np.array([[25, 25]], dtype=np.int64)
            box = layouts._allowed_box(grid, "A1", 2.0)
            layouts._close_gaps(sites, [0], [box], 12**2, *layouts._sample_points(grid), moves)
            self.assertEqual(sites.tolist(), [[25, 25]])

    def test_a_step_that_went_wrong_cannot_hand_back_a_layout(self):
        def push_into_the_margin(sites, *_):
            sites[0] = (0, 0)

        with (
            mock.patch.object(layouts, "_close_gaps", push_into_the_margin),
            self.assertRaisesRegex(LayoutError, "margin"),
        ):
            layouts.spread_positions(self.grid, 1, min_spacing_mm=12.0, cell_inset_mm=2.0, rng=derived_rng(0, "t"))

    def test_the_nearest_and_second_nearest_positions_match_a_brute_force_search(self):
        rng = np.random.default_rng(11)
        sites = (rng.random((9, 2)) * [100.0, 50.0]).astype(np.int64)
        sample_x, sample_y = self.samples
        owner, nearest, second = layouts._nearest_two(sites, sample_x, sample_y)
        for index in range(0, len(sample_x), 37):
            x0, y0 = int(sample_x[index]), int(sample_y[index])
            squared = sorted(((x0 - x) ** 2 + (y0 - y) ** 2, i) for i, (x, y) in enumerate(sites.tolist()))
            self.assertEqual((int(nearest[index]), int(owner[index])), squared[0])
            self.assertEqual(int(second[index]), squared[1][0])


class SpreadStepsTests(unittest.TestCase):
    """What each step of the spread layout adds, by switching the later steps off."""

    @staticmethod
    def positions(seed: int, **steps: int) -> dict[str, list[Point]]:
        rng = derived_rng(seed, "spread", "positions")
        return layouts.spread_positions(GRID, per_cell=5, min_spacing_mm=12.0, cell_inset_mm=2.0, rng=rng, **steps)

    def quality(self, seed: int, **steps: int) -> layouts.LayoutQuality:
        return layouts.layout_quality(as_episodes(self.positions(seed, **steps)), GRID)

    def test_the_steps_run_by_default(self):
        for seed in (SEED, 1):
            by_cell = self.positions(seed)
            for cell in GRID.trained_cells:
                planned = {e.target for e in layout("spread", seed) if e.cell == cell}
                self.assertEqual(set(by_cell[cell]), planned)

    def test_relaxation_shrinks_the_holes_of_the_first_placement_and_parts_its_closest_positions(self):
        for seed in (SEED, 1, 2):
            first = self.quality(seed, relaxation_passes=0, gap_moves=0)
            relaxed = self.quality(seed, gap_moves=0)
            with self.subTest(seed=seed):
                self.assertLess(relaxed.largest_gap_mm, first.largest_gap_mm - 2.0)
                self.assertGreater(relaxed.min_spacing_mm, first.min_spacing_mm)

    def test_closing_gaps_makes_the_largest_gap_smaller(self):
        for seed in (SEED, 1, 2):
            relaxed = self.quality(seed, gap_moves=0)
            closed = self.quality(seed)
            with self.subTest(seed=seed):
                self.assertLess(closed.largest_gap_mm, relaxed.largest_gap_mm - 0.25)

    def test_every_stage_keeps_five_positions_per_cell_the_spacing_and_the_margin(self):
        for steps in ({"relaxation_passes": 0, "gap_moves": 0}, {"gap_moves": 0}, {"relaxation_passes": 0}):
            quality = self.quality(SEED, **steps)
            with self.subTest(steps=steps):
                self.assertEqual(quality.distinct_positions, 55)
                self.assertGreaterEqual(quality.min_spacing_mm, 12.0)


class QualityTests(unittest.TestCase):
    def test_the_clustered_layout_repeats_eleven_positions_and_leaves_the_cell_corners_35_mm_away(self):
        quality = layouts.layout_quality(layout("clustered"), GRID)
        self.assertEqual(quality.episodes, 55)
        self.assertEqual(quality.distinct_positions, 11)
        self.assertEqual(quality.min_spacing_mm, 0.0)
        self.assertEqual(quality.mean_nearest_neighbour_mm, 0.0)
        self.assertAlmostEqual(quality.largest_gap_mm, 25.0 * math.sqrt(2.0))
        self.assertAlmostEqual(quality.largest_gap_mm, 35.4, places=1)
        self.assertEqual(quality.largest_gap_at, Point(100.0, 50.0))  # a cell corner: of many, the most central

    def test_the_spread_layout_reports_its_spacings(self):
        episodes = layout("spread")
        quality = layouts.layout_quality(episodes, GRID)
        self.assertEqual(quality.distinct_positions, 55)
        self.assertAlmostEqual(quality.min_spacing_mm, min_spacing(episodes))
        nearest = [min(a.target.distance_to(b.target) for b in episodes if b is not a) for a in episodes]
        self.assertAlmostEqual(quality.mean_nearest_neighbour_mm, sum(nearest) / 55)
        self.assertGreater(quality.mean_nearest_neighbour_mm, quality.min_spacing_mm)

    def test_the_largest_gap_is_where_it_says_and_lies_in_a_trained_cell(self):
        episodes = layout("spread")
        quality = layouts.layout_quality(episodes, GRID)
        nearest = min(quality.largest_gap_at.distance_to(e.target) for e in episodes)
        self.assertAlmostEqual(nearest, quality.largest_gap_mm)
        self.assertTrue(GRID.in_trained_cells(quality.largest_gap_at))

    def test_the_spread_layout_leaves_a_far_smaller_gap_than_the_clustered_one(self):
        clustered = layouts.layout_quality(layout("clustered"), GRID).largest_gap_mm
        for seed in SpreadLayoutTests.SEEDS:
            with self.subTest(seed=seed):
                self.assertLess(layouts.layout_quality(layout("spread", seed), GRID).largest_gap_mm, 0.55 * clustered)

    def test_the_spread_layout_is_even_not_merely_random(self):
        # Five positions per cell drawn at random and independently leave clumps: two of them closer than the cube's
        # half width, and holes of about 30 mm. An even layout has neither.
        for seed in SpreadLayoutTests.SEEDS:
            quality = layouts.layout_quality(layout("spread", seed), GRID)
            with self.subTest(seed=seed):
                self.assertGreater(quality.min_spacing_mm, 14.0)
                self.assertGreater(quality.mean_nearest_neighbour_mm, 18.0)
                self.assertLess(quality.largest_gap_mm, 19.5)

    def test_quality_of_nothing_is_an_error(self):
        with self.assertRaises(LayoutError):
            layouts.layout_quality([], GRID)


class PlannedEpisodeTests(unittest.TestCase):
    def test_impossible_episodes_are_refused(self):
        good = {"dataset": "spread", "episode_index": 0, "round": 1, "sitting": 2, "cell": "C2"}
        good["target"] = Point(131.0, 77.0)
        PlannedEpisode(**good)
        for bad in (
            {"dataset": "scattered"},
            {"episode_index": -1},
            {"round": 0},
            {"sitting": 0},
            {"cell": ""},
            {"target": (131.0, 77.0)},
        ):
            with self.subTest(bad=bad), self.assertRaises(LayoutError):
                PlannedEpisode(**(good | bad))


class FileTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_a_layout_is_written_and_read_back(self):
        for dataset in ("clustered", "spread"):
            path = self.dir / "layouts" / f"{dataset}.json"
            layouts.write_layout(path, layout(dataset), seed=SEED, protocol_version="0.1")
            self.assertEqual(layouts.read_layout(path), layout(dataset))

    def test_the_file_says_which_set_seed_and_protocol_version_it_was_made_from(self):
        path = self.dir / "spread.json"
        layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset"], "spread")
        self.assertEqual(payload["seed"], SEED)
        self.assertEqual(payload["protocol_version"], "0.1")
        self.assertEqual(len(payload["episodes"]), 55)
        self.assertEqual(payload["episodes"][0]["episode_index"], 0)

    def test_writing_twice_gives_the_same_bytes_and_no_path_of_this_machine(self):
        first, second = self.dir / "a.json", self.dir / "b.json"
        layouts.write_layout(first, layout("spread"), seed=SEED, protocol_version="0.1")
        layouts.write_layout(second, layout("spread"), seed=SEED, protocol_version="0.1")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        text = first.read_text(encoding="utf-8")
        self.assertNotIn(str(Path.home()), text)
        self.assertNotIn(str(self.dir), text)
        self.assertTrue(text.endswith("\n"))

    def test_writing_the_same_layout_again_leaves_the_file_alone_and_says_so(self):
        path = self.dir / "spread.json"
        self.assertTrue(layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1"))
        before = path.read_bytes()
        self.assertFalse(layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1"))
        self.assertEqual(path.read_bytes(), before)

    def test_a_file_that_holds_another_layout_is_not_replaced_unless_asked(self):
        # Recording may have started from the file on disk: a changed protocol must not swap it unnoticed.
        path = self.dir / "spread.json"
        layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1")
        before = path.read_bytes()
        with self.assertRaisesRegex(LayoutError, "spread.json"):
            layouts.write_layout(path, layout("spread", 1), seed=1, protocol_version="0.1")
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(layouts.write_layout(path, layout("spread", 1), seed=1, protocol_version="0.1", replace=True))
        self.assertEqual(layouts.read_layout(path), layout("spread", 1))

    def test_the_same_episodes_under_a_later_protocol_version_leave_the_file_as_it_was_made(self):
        # The version in the file says which protocol the layout was made from. Freezing the protocol changes its
        # version and not the layout, and must not look like a different layout.
        path = self.dir / "spread.json"
        layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1")
        before = path.read_bytes()
        self.assertFalse(layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="1.0"))
        self.assertEqual(path.read_bytes(), before)

    def test_a_file_is_absent_or_holds_the_same_episodes_or_something_else(self):
        path = self.dir / "spread.json"
        self.assertEqual(layouts.layout_file_state(path, layout("spread")), "absent")
        layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1")
        self.assertEqual(layouts.layout_file_state(path, layout("spread")), "same")
        self.assertEqual(layouts.layout_file_state(path, layout("spread", 1)), "different")
        self.assertEqual(layouts.layout_file_state(path, layout("clustered")), "different")
        path.write_text("not a layout", encoding="utf-8")
        self.assertEqual(layouts.layout_file_state(path, layout("spread")), "different")

    def test_a_file_that_cannot_be_read_as_a_layout_is_not_replaced_unless_asked(self):
        path = self.dir / "spread.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(LayoutError):
            layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1")
        self.assertEqual(path.read_text(encoding="utf-8"), "{}")
        self.assertTrue(layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1", replace=True))

    def test_a_mixed_or_gapped_list_of_episodes_is_not_written(self):
        mixed = [*layout("spread")[:54], layout("clustered")[54]]
        with self.assertRaises(LayoutError):
            layouts.write_layout(self.dir / "mixed.json", mixed, seed=SEED, protocol_version="0.1")
        with self.assertRaises(LayoutError):
            layouts.write_layout(self.dir / "gapped.json", layout("spread")[1:], seed=SEED, protocol_version="0.1")
        with self.assertRaises(LayoutError):
            layouts.write_layout(self.dir / "empty.json", [], seed=SEED, protocol_version="0.1")
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_a_file_with_an_unknown_field_or_version_is_refused(self):
        path = self.dir / "spread.json"
        layouts.write_layout(path, layout("spread"), seed=SEED, protocol_version="0.1")
        good = json.loads(path.read_text(encoding="utf-8"))
        for change in (
            lambda p: p["episodes"][3].update(note="moved"),
            lambda p: p.update(schema_version=99),
            lambda p: p["episodes"][3].update(dataset="clustered"),
            lambda p: p["episodes"][3]["target"].pop("y_mm"),
            lambda p: p["episodes"].pop(3),
            lambda p: p.pop("episodes"),
        ):
            payload = copy.deepcopy(good)
            change(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(LayoutError):
                layouts.read_layout(path)

    def test_the_layouts_kept_in_the_repository_are_those_the_protocol_gives(self):
        # The operator records from these files. If the protocol, the code or NumPy came to give other layouts, this
        # fails in whichever environment sees the difference, before anyone records or analyses against the wrong one.
        for dataset in ("clustered", "spread"):
            path = layouts.layout_path(dataset)
            with self.subTest(dataset=dataset):
                self.assertEqual(layouts.read_layout(path), layout(dataset))
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["seed"], SEED)

    def test_the_layouts_have_their_place_in_the_repository(self):
        self.assertEqual(layouts.layout_path("spread"), config.REPO_ROOT / "config" / "layouts" / "spread.json")
        self.assertEqual(layouts.layout_path("clustered", self.dir), self.dir / "config" / "layouts" / "clustered.json")
        with self.assertRaises(LayoutError):
            layouts.layout_path("scattered")


class PlannedDemoStartTests(unittest.TestCase):
    def test_a_layout_can_stand_in_for_an_audit_in_a_rehearsal(self):
        starts = layouts.planned_demo_starts(layout("spread"))
        self.assertEqual(len(starts), 55)
        for start, episode in zip(starts, layout("spread"), strict=True):
            self.assertEqual((start.dataset, start.cell), ("spread", episode.cell))
            self.assertEqual(start.episode_index, episode.episode_index)
            self.assertEqual(start.measured, episode.target)
            self.assertEqual(start.planned, episode.target)
            self.assertTrue(start.trained)


if __name__ == "__main__":
    unittest.main()
