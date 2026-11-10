"""The trial planner: which evaluation trial comes next.

`next_trial` is a pure function of three things: the protocol (with its seeds), the audited start positions of the
demonstrations, and the trial log so far. It keeps no state and reads no clock, so after any interruption the same log
gives the same next trial. Pass it every trial recorded so far: trials of other blocks are ignored, and a block's own
trials are counted to number the next one.

Two kinds of block (`[blocks.*]` in the protocol)
-------------------------------------------------
*Threshold blocks* (one condition) are made of parts: anchors at distance 0 (the target is exactly a demonstration's
measured start), in some blocks far anchors at a fixed distance, and staircases whose next distance follows
`staircase.next_level` applied to that staircase's own scored trials. The parts are interleaved in an order fixed by
the block's seed: each part is spread evenly through the block, and which part comes next cannot be told from what
came before. The order does not depend on the outcomes.

* For a policy, a distance becomes a target through `targets.placement_at_distance`: a point of the test region whose
  nearest demonstration (of the policy's own set) is that far away.
* For the replay, one demonstration is played back, so the distance is to that demonstration:
  `targets.placement_around`. The replayed demonstrations are a fixed set, one from each of `replay_demos` cells
  (with as many as there are trained cells: one from every trained cell). They take turns: no demonstration is
  replayed twice in a part of the block before every other was replayed there as often, and the same holds over
  the whole block. A part with fewer trials than there are demonstrations therefore replays different ones, and the
  demonstrations it left out come first in the parts after it (`_replay_use`).

*The paired block* tests placements drawn evenly over a region (fixed by its `placement_seed`) once with each of two
policies, the two trials of a pair back to back; each policy goes first in half of the pairs, in a seeded order. With
a `regions` list in place of `region`, a stated number of placements is drawn from each of several regions and the
whole is shuffled.

What a trial is, and what does not count
----------------------------------------
Every place a block has for a scored trial is a *slot* with its own random stream, so its target depends on the slot
and not on how many trials were planned before it. A void trial fills no slot: the slot is asked again, with the same
target, under a new trial id and a new blind code.

Blind codes
-----------
The person who scores a trial is shown its `blind_code` in place of the condition. Codes are drawn without
replacement from one shuffled list shared by all blocks and are assigned by the trial's number alone, never by its
condition: knowing a code says nothing about the policy, and no two trials of the experiment share one.

A code is fixed by its block and its number, and a threshold block is one condition. So a pilot of a block, run
under the counted trials' codes, would tell for those numbers which code is a replay and which a policy, and the
counted session that interleaves the two blocks shows the same codes again. A pilot's trials therefore take their
codes from a second shuffle of the same list, with a P before them (`pilot=True`): nothing a pilot shows is a code
of a counted trial.

Seeds
-----
A block's choices are seeded by its `seed` entry if it has one, else its `placement_seed`, else
`[datasets].layout_seed` (the paired block's placements always use its `placement_seed` when it has one). The blind
codes are seeded by `[datasets].layout_seed`.

Standard library and NumPy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tolerance_radius import schema
from tolerance_radius.geometry import GeometryError, Grid, trained_demos
from tolerance_radius.schema import DemoStart, PlannedTrial, Point, TrialRecord
from tolerance_radius.seeds import derived_rng, pick_index, shuffled
from tolerance_radius.staircase import StaircaseParams, next_level
from tolerance_radius.targets import max_feasible_level, placement_around, placement_at_distance, uniform_placements

#: How many demonstrations a replay block plays back, unless the block says so with `replay_demos`. The
#: repository's protocol file says so (one from each trained cell): this is only what a file without the entry gets.
DEFAULT_REPLAY_DEMOS = 4

#: Blind codes are a letter and two digits, without the characters that are easily misread (I, O, 0, 1).
_CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_CODE_DIGITS = "23456789"

#: The two kinds of anchor in a threshold block. Its other parts are its staircases, named by their ids.
_ZERO, _FAR = "zero", "far"


class PlannerError(RuntimeError):
    """The planner cannot say what comes next: the protocol, the audit or the log does not allow it."""


# --------------------------------------------------------------------------------------------------------------------
# The two functions the rig calls
# --------------------------------------------------------------------------------------------------------------------


def next_trial(
    block: str, protocol: Mapping[str, Any], demo_starts: Sequence[DemoStart], log: Sequence[TrialRecord],
    *, pilot: bool = False,
) -> PlannedTrial | None:
    """The next trial of a block, or None when the block is complete.

    `demo_starts` is the start-state audit (both sets may be given together; the block's own set is picked out and
    episodes left out of training are ignored). `log` is every trial recorded so far, in order. Raises `PlannerError`
    when a requested distance cannot be placed on the table, or when the log does not fit the block. `pilot` gives
    the trial a pilot's blind code (see "Blind codes"); everything else of it is the same.
    """
    entry = _entry(block, protocol)
    records = [record for record in log if record.planned.block == block]
    number = _next_number(block, log)
    labels = {
        "trial_id": _trial_id(block, number),
        "block": block,
        "blind_code": _blind_code(block, number, protocol, pilot=pilot),
    }
    if _is_paired(entry):
        return _next_paired_trial(block, entry, protocol, records, labels)
    return _next_threshold_trial(block, entry, protocol, demo_starts, records, labels)


def block_status(block: str, protocol: Mapping[str, Any], log: Sequence[TrialRecord]) -> dict[str, Any]:
    """How far a block has come: counts of trials planned, scored, void and remaining, for the block and its parts.

    Only counts: no distance, target, condition or order, so it can be shown to whoever places and scores.
    """
    entry = _entry(block, protocol)
    records = [record for record in log if record.planned.block == block]
    status: dict[str, Any] = {"block": block}
    if _is_paired(entry):
        slots = _pair_slots(block, entry, protocol)
        done = _pair_slots_done(block, slots, records)
        remaining = sum((pair_id, condition) not in done for pair_id, condition, _ in slots)
        pairs = {pair_id for pair_id, _, _ in slots}
        complete = sum(all((pair_id, c) in done for c in entry["conditions"]) for pair_id in pairs)
        status["pairs"] = {"done": complete, "planned": len(pairs)}
        planned = len(slots)
    else:
        parts = _parts(block, protocol)
        filled = _scored_by_part(block, entry, parts, records)
        progress = {name: {"done": min(len(filled[name]), size), "planned": size} for name, size in parts.items()}
        status["anchors_zero"] = progress.pop(_ZERO)
        status["anchors_far"] = progress.pop(_FAR)
        status["staircases"] = progress
        planned = sum(parts.values())
        remaining = sum(size - min(len(filled[name]), size) for name, size in parts.items())
    status.update(
        complete=remaining == 0,
        trials_planned=planned,
        trials_scored=sum(record.scored for record in records),
        trials_void=sum(not record.scored for record in records),
        trials_remaining=remaining,
    )
    return status


# --------------------------------------------------------------------------------------------------------------------
# What the blocks are made of: also useful to the tools at the rig and to the analysis
# --------------------------------------------------------------------------------------------------------------------


def block_seed(block: str, protocol: Mapping[str, Any]) -> int:
    """The seed of a block's random choices: its `seed`, else its `placement_seed`, else `[datasets].layout_seed`."""
    entry = _entry(block, protocol)
    for key in ("seed", "placement_seed"):
        if key in entry:
            return entry[key]
    return protocol["datasets"]["layout_seed"]


def staircase_params(block: str, protocol: Mapping[str, Any]) -> dict[str, StaircaseParams]:
    """The staircases of a threshold block, by their full ids ("T1a-up"), in the protocol's order.

    Where a staircase starts and how long it runs come from the block's entry; its steps and lower limit from the
    `[staircase]` section; its upper limit is the block's `max_level_mm`.
    """
    entry = _entry(block, protocol)
    if "staircases" not in entry:
        raise PlannerError(f"block {block} has no staircases")
    rule = protocol["staircase"]
    if rule.get("rule") != "1up1down":
        raise PlannerError(f"the staircase rule {rule.get('rule')!r} is not one this planner knows (1up1down)")
    params: dict[str, StaircaseParams] = {}
    for staircase in entry["staircases"]:
        staircase_id = f"{block}-{staircase['id']}"
        if staircase_id in params:
            raise PlannerError(f"block {block} lists the staircase {staircase['id']!r} twice")
        try:
            params[staircase_id] = StaircaseParams(
                start_level_mm=float(staircase["start_level_mm"]),
                n_trials=int(staircase["trials"]),
                step_initial_mm=float(rule["step_initial_mm"]),
                step_final_mm=float(rule["step_final_mm"]),
                min_level_mm=float(rule["min_level_mm"]),
                max_level_mm=float(_need(entry, "max_level_mm", block)),
            )
        except ValueError as error:
            raise PlannerError(f"staircase {staircase_id}: {error}") from error
    return params


def replay_demos(block: str, protocol: Mapping[str, Any], demo_starts: Sequence[DemoStart]) -> list[DemoStart]:
    """The demonstrations a replay block plays back: a fixed set, one from each of several cells.

    Their number is the block's `replay_demos` (4 if it does not say); when it is the number of trained cells, the
    set is one demonstration from every trained cell. Which cells, and which demonstration of a cell, are drawn from
    the block's seed. Only trained demonstrations that start inside the test region are eligible: a trial at
    distance 0 puts the cube on the demonstration's own start, and a test placement stays in the test region.
    """
    entry = _entry(block, protocol)
    if entry.get("condition") != "replay":
        raise PlannerError(f"block {block} is not a replay block")
    by_cell = _eligible_by_cell(_block_demos(block, entry, demo_starts), Grid.from_protocol(protocol))
    wanted = entry.get("replay_demos", DEFAULT_REPLAY_DEMOS)
    if not 1 <= wanted <= len(by_cell):
        raise PlannerError(
            f"block {block} replays {wanted} demonstrations from distinct cells, "
            f"and {len(by_cell)} cells have a trained demonstration that starts in the test region"
        )
    rng = derived_rng(block_seed(block, protocol), block, "replay-set")
    cells = shuffled(sorted(by_cell), rng)[:wanted]
    return [by_cell[cell][pick_index(len(by_cell[cell]), rng)] for cell in cells]


def paired_placements(block: str, protocol: Mapping[str, Any]) -> list[Point]:
    """The placements of a paired block, in the order they are tested.

    They are drawn evenly over the block's `region`. A block with a `regions` list instead draws a stated number
    from each of several regions (so many in the trained cells, so many in the held-out cells) and then shuffles the
    whole, so that a placement's place in the order says nothing about its region.
    """
    entry = _entry(block, protocol)
    if not _is_paired(entry):
        raise PlannerError(f"block {block} is not a paired block")
    seed = entry.get("placement_seed", block_seed(block, protocol))
    grid = Grid.from_protocol(protocol)
    pairs = _need(entry, "pairs", block)
    if "regions" not in entry:
        region = entry.get("region", "trained_cells")
        return uniform_placements(grid, pairs, derived_rng(seed, block, "placements"), region)
    placements: list[Point] = []
    for region, count in _region_counts(block, entry):
        placements += uniform_placements(grid, count, derived_rng(seed, block, "placements", region), region)
    return shuffled(placements, derived_rng(seed, block, "placement-order"))


def _region_counts(block: str, entry: Mapping[str, Any]) -> list[tuple[str, int]]:
    """A paired block's `regions` list as `(region, how many placements)`, checked against its `pairs`."""
    if "region" in entry:
        raise PlannerError(f"[blocks.{block}] has both a region and a list of regions: keep one")
    counts: list[tuple[str, int]] = []
    for stratum in entry["regions"]:
        region, count = _need(stratum, "region", block), _need(stratum, "pairs", block)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise PlannerError(f"[blocks.{block}] region {region}: pairs must be a whole number above 0, got {count!r}")
        if region in [name for name, _ in counts]:
            raise PlannerError(f"[blocks.{block}] names region {region} twice")
        counts.append((region, count))
    total = sum(count for _, count in counts)
    if not counts or total != entry["pairs"]:
        raise PlannerError(
            f"[blocks.{block}] has {entry['pairs']} pairs, and its regions add up to {total}: the two must agree"
        )
    return counts


def check_block(block: str, protocol: Mapping[str, Any], demo_starts: Sequence[DemoStart]) -> list[str]:
    """What would stop a block before its end, as sentences; an empty list when nothing would.

    Meant to be run before a block's first trial, with the real audit. A staircase asks for whatever distance its
    rule gives, up to the block's limit; if the table has no point that far from the demonstrations, `next_trial`
    raises in the middle of the block. This reports such a limit beforehand, when it can still be changed cleanly.
    """
    try:
        entry = _entry(block, protocol)
        if _is_paired(entry):
            _pair_slots(block, entry, protocol)
            return []
        parts = _parts(block, protocol)
        grid = Grid.from_protocol(protocol)
        demos = _block_demos(block, entry, demo_starts)
        limits = {"max_level_mm": float(_need(entry, "max_level_mm", block))}
        if parts[_FAR]:
            limits["far_level_mm"] = float(_need(entry, "far_level_mm", block))
        problems: list[str] = []
        if entry["condition"] == "replay":
            for demo in replay_demos(block, protocol, demo_starts):
                for key, level in limits.items():
                    if placement_around(demo, level, grid, derived_rng(0, "check")) is None:
                        problems.append(
                            f"[blocks.{block}] {key} is {level:g} mm, and no target that far from the start of "
                            f"episode {demo.episode_index} stays in the test region"
                        )
        else:
            _anchor_demos(block, protocol, demos, grid, parts[_ZERO])
            largest = max_feasible_level(demos, grid)
            match = _match_mm(protocol)
            for key, level in limits.items():
                if level - match > largest:
                    problems.append(
                        f"[blocks.{block}] {key} is {level:g} mm, and no point of the test region is farther than "
                        f"{largest:.1f} mm from its nearest {demos[0].dataset} demonstration"
                    )
        return problems
    except (PlannerError, GeometryError) as error:
        return [str(error)]


# --------------------------------------------------------------------------------------------------------------------
# Labels: the trial's number, its id and its blind code
# --------------------------------------------------------------------------------------------------------------------


def _trial_id(block: str, number: int) -> str:
    return f"{block}-{number + 1:03d}"


def _next_number(block: str, log: Sequence[TrialRecord]) -> int:
    """The number of a block's next trial, from 0: its trials so far, void ones included."""
    used = {record.planned.trial_id for record in log}
    number = sum(record.planned.block == block for record in log)
    while _trial_id(block, number) in used:  # only in a log that was not written by this planner alone
        number += 1
    return number


def _blind_code(block: str, number: int, protocol: Mapping[str, Any], *, pilot: bool = False) -> str:
    """The code of a block's trial `number`: a pick of its own from one shuffled list that all blocks share. A
    pilot's code is the pick from a second shuffle of that list, with a P before it."""
    codes = [letter + first + second for letter in _CODE_LETTERS for first in _CODE_DIGITS for second in _CODE_DIGITS]
    share = len(codes) // len(schema.BLOCKS)
    if number >= share:
        raise PlannerError(f"block {block} has used the {share} blind codes it can have")
    codes = shuffled(codes, derived_rng(protocol["datasets"]["layout_seed"],
                                        "blind-codes-pilot" if pilot else "blind-codes"))
    code = codes[schema.BLOCKS.index(block) * share + number]
    return f"P{code}" if pilot else code


# --------------------------------------------------------------------------------------------------------------------
# Threshold blocks
# --------------------------------------------------------------------------------------------------------------------


def _next_threshold_trial(
    block: str,
    entry: Mapping[str, Any],
    protocol: Mapping[str, Any],
    demo_starts: Sequence[DemoStart],
    records: Sequence[TrialRecord],
    labels: Mapping[str, str],
) -> PlannedTrial | None:
    parts = _parts(block, protocol)
    filled = _scored_by_part(block, entry, parts, records)
    # The first slot, in the block's order, that no scored trial has filled yet.
    open_slots = ((name, index) for name, index in _order(block, protocol, parts) if index >= len(filled[name]))
    slot = next(open_slots, None)
    if slot is None:
        return None
    part, index = slot
    condition = entry["condition"]
    seed = block_seed(block, protocol)
    grid = Grid.from_protocol(protocol)
    demos = _block_demos(block, entry, demo_starts)
    rng = derived_rng(seed, block, "target", part, index)

    staircase_id = staircase_index = replay_episode = None
    if part == _ZERO:
        level, what = 0.0, f"anchor {index + 1} at distance 0"
    elif part == _FAR:
        level, what = float(_need(entry, "far_level_mm", block)), f"far anchor {index + 1}"
    else:
        history = [(record.planned.level_mm, record.success) for record in filled[part]]
        level = next_level(staircase_params(block, protocol)[part], history)
        staircase_id, staircase_index, what = part, index, f"staircase {part}, trial {index + 1}"
    where = f"{labels['trial_id']} ({what})"

    if condition == "replay":
        chosen = replay_demos(block, protocol, demo_starts)
        demo = chosen[_replay_use(block, protocol, parts, len(chosen))[part][index]]
        replay_episode = demo.episode_index
        target = placement_around(demo, level, grid, rng)
        if target is None:
            raise PlannerError(
                f"{where}: no target {level:g} mm from the start of episode {demo.episode_index} stays in the test "
                f"region. A limit of [blocks.{block}] is beyond what the table offers"
            )
    elif part == _ZERO:
        target = _anchor_demos(block, protocol, demos, grid, parts[_ZERO])[index].measured
    else:
        match = _match_mm(protocol)
        target = placement_at_distance(level, demos, grid, rng, match)
        if target is None:
            largest = max_feasible_level(demos, grid)
            raise PlannerError(
                f"{where}: no point of the test region is {level:g} mm, to within {match:g} mm, from its nearest "
                f"{demos[0].dataset} demonstration; the largest distance there is {largest:.1f} mm. "
                f"A limit of [blocks.{block}] is beyond what the table offers"
            )
    return PlannedTrial(
        condition=condition,
        target=target,
        level_mm=level,
        staircase_id=staircase_id,
        staircase_index=staircase_index,
        replay_episode=replay_episode,
        **labels,
    )


def _parts(block: str, protocol: Mapping[str, Any]) -> dict[str, int]:
    """The parts of a threshold block and how many scored trials each takes: the anchors, then each staircase."""
    entry = _entry(block, protocol)
    if "staircases" not in entry:
        raise PlannerError(f"block {block} is neither a threshold block (staircases) nor a paired block (pairs)")
    if _need(entry, "condition", block) not in schema.CONDITIONS:
        raise PlannerError(f"block {block} names the unknown condition {entry['condition']!r}")
    parts = {_ZERO: int(entry.get("anchors_zero", 0)), _FAR: int(entry.get("anchors_far", 0))}
    for staircase_id, params in staircase_params(block, protocol).items():
        parts[staircase_id] = params.n_trials
    return parts


def _order(block: str, protocol: Mapping[str, Any], parts: Mapping[str, int]) -> list[tuple[str, int]]:
    """The block's slots in the order they are run, as `(part, index within the part)`.

    Trial `i` of a part with `n` trials gets a place drawn evenly from the `i`-th `n`-th of the block, and the slots
    are run by place. So every part is spread through the block, the two staircases advance together, and within each
    stretch the order is left to chance.
    """
    rng = derived_rng(block_seed(block, protocol), block, "order")
    placed = []
    for rank, (name, size) in enumerate(parts.items()):
        jitter = rng.random(size).tolist()
        placed.extend(((index + jitter[index]) / size, rank, index, name) for index in range(size))
    return [(name, index) for _, _, index, name in sorted(placed)]


def _scored_by_part(
    block: str, entry: Mapping[str, Any], parts: Mapping[str, int], records: Sequence[TrialRecord]
) -> dict[str, list[TrialRecord]]:
    """The scored trials of each part of a threshold block, oldest first. A trial that fits no part is an error."""
    filled: dict[str, list[TrialRecord]] = {name: [] for name in parts}
    for record in records:
        planned = record.planned
        if planned.condition != entry["condition"]:
            raise PlannerError(
                f"{planned.trial_id}: block {block} runs {entry['condition']}, and the log has a trial of "
                f"{planned.condition} in it"
            )
        if planned.staircase_id is not None:
            part = planned.staircase_id
        elif planned.level_mm is None:
            raise PlannerError(f"{planned.trial_id}: a trial of block {block} asks for a distance, this one has none")
        else:
            part = _ZERO if planned.level_mm == 0 else _FAR
        if parts.get(part, 0) == 0:
            raise PlannerError(f"{planned.trial_id}: block {block} has no part {part!r} for this trial of the log")
        if record.scored:
            filled[part].append(record)
    return filled


def _block_demos(block: str, entry: Mapping[str, Any], demo_starts: Sequence[DemoStart]) -> list[DemoStart]:
    """The trained demonstrations of the set that a threshold block's distances are measured against."""
    dataset = schema.DATASET_OF_CONDITION[entry["condition"]]
    demos = trained_demos(demo_starts, dataset)
    if not demos:
        raise PlannerError(f"block {block} needs the audited starts of the {dataset} set, and none were given")
    return demos


def _eligible_by_cell(demos: Sequence[DemoStart], grid: Grid) -> dict[str, list[DemoStart]]:
    """The demonstrations a trial at distance 0 may sit on (those that start in the test region), by cell."""
    by_cell: dict[str, list[DemoStart]] = {}
    for demo in demos:
        if grid.in_test_region(demo.measured):
            by_cell.setdefault(demo.cell, []).append(demo)
    return by_cell


def _anchor_demos(
    block: str, protocol: Mapping[str, Any], demos: Sequence[DemoStart], grid: Grid, count: int
) -> list[DemoStart]:
    """The demonstrations whose starts are a policy block's anchors at distance 0: distinct cells while they last."""
    by_cell = _eligible_by_cell(demos, grid)
    if count > 0 and not by_cell:
        raise PlannerError(f"block {block}: no trained demonstration starts in the test region, so there is no anchor")
    rng = derived_rng(block_seed(block, protocol), block, "anchors")
    chosen: list[DemoStart] = []
    while len(chosen) < count:
        for cell in shuffled(sorted(by_cell), rng):
            chosen.append(by_cell[cell][pick_index(len(by_cell[cell]), rng)])
    return chosen[:count]


def _replay_use(
    block: str, protocol: Mapping[str, Any], parts: Mapping[str, int], count: int
) -> dict[str, list[int]]:
    """For each part of a replay block, which demonstration of the replay set (by position) each trial replays.

    Within a part the demonstrations take turns in shuffled rounds, so each is used as often as the others to within
    one. The trials left over after a part's whole rounds are handed out in one running order across the parts, so
    the same holds over the whole block.

    A part with fewer trials than there are demonstrations has no whole round: its trials replay different
    demonstrations, the next ones of the running order. So with 11 demonstrations, 4 anchors and two staircases of
    9 trials, the anchors are 4 different demonstrations, each staircase replays 9 different ones, and every
    demonstration is replayed exactly twice, in two different parts. None of this depends on an outcome.
    """
    rng = derived_rng(block_seed(block, protocol), block, "replay-use")
    left_over_turns = shuffled(range(count), rng)
    handed_out = 0
    use: dict[str, list[int]] = {}
    for name, size in parts.items():
        rounds, extra = divmod(size, count)
        sequence: list[int] = []
        for _ in range(rounds):
            sequence.extend(shuffled(range(count), rng))
        left_over = [left_over_turns[(handed_out + k) % count] for k in range(extra)]
        handed_out += extra
        sequence.extend(shuffled(left_over, rng))
        use[name] = sequence
    return use


# --------------------------------------------------------------------------------------------------------------------
# The paired block
# --------------------------------------------------------------------------------------------------------------------


def _next_paired_trial(
    block: str,
    entry: Mapping[str, Any],
    protocol: Mapping[str, Any],
    records: Sequence[TrialRecord],
    labels: Mapping[str, str],
) -> PlannedTrial | None:
    slots = _pair_slots(block, entry, protocol)
    done = _pair_slots_done(block, slots, records)
    for pair_id, condition, target in slots:
        if (pair_id, condition) not in done:
            return PlannedTrial(condition=condition, target=target, pair_id=pair_id, **labels)
    return None


def _pair_slots(block: str, entry: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[tuple[str, str, Point]]:
    """The slots of a paired block in the order they are run, as `(pair id, condition, target)`."""
    conditions = list(_need(entry, "conditions", block))
    policies = [condition for condition in schema.CONDITIONS if condition != "replay"]
    if len(conditions) != 2 or len(set(conditions)) != 2 or any(c not in policies for c in conditions):
        raise PlannerError(f"block {block} compares two different policies, got {conditions}")
    placements = paired_placements(block, protocol)
    # Each policy goes first in half of the pairs (to within one), in a shuffled order.
    rng = derived_rng(block_seed(block, protocol), block, "pair-order")
    first_listed_goes_first = shuffled([index % 2 == 0 for index in range(len(placements))], rng)
    slots = []
    for index, target in enumerate(placements):
        pair_id = f"{block}-p{index + 1:02d}"
        ordered = conditions if first_listed_goes_first[index] else conditions[::-1]
        slots.extend((pair_id, condition, target) for condition in ordered)
    return slots


def _pair_slots_done(
    block: str, slots: Sequence[tuple[str, str, Point]], records: Sequence[TrialRecord]
) -> set[tuple[str | None, str]]:
    """The slots of a paired block that a scored trial has filled. A trial that fits no slot is an error."""
    known = {(pair_id, condition) for pair_id, condition, _ in slots}
    done = set()
    for record in records:
        key = (record.planned.pair_id, record.planned.condition)
        if key not in known:
            raise PlannerError(
                f"{record.planned.trial_id}: block {block} plans no trial of pair {key[0]!r} with {key[1]}, "
                "and the log has one"
            )
        if record.scored:
            done.add(key)
    return done


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _entry(block: str, protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    blocks = protocol.get("blocks", {})
    if block not in blocks or block not in schema.BLOCKS:
        raise PlannerError(f"{block!r} is not a block of the protocol: it has {sorted(blocks)}")
    return blocks[block]


def _is_paired(entry: Mapping[str, Any]) -> bool:
    return "pairs" in entry


def _need(entry: Mapping[str, Any], key: str, block: str) -> Any:
    """An entry of a block that the planner cannot do without."""
    if key not in entry:
        raise PlannerError(f"[blocks.{block}] has no {key}, which the planner needs")
    return entry[key]


def _match_mm(protocol: Mapping[str, Any]) -> float:
    """How close a target's distance to its nearest demonstration must be to the distance requested."""
    placement = protocol.get("placement", {})
    if "level_match_mm" not in placement:
        raise PlannerError("[placement] has no level_match_mm, which the planner needs")
    return float(placement["level_match_mm"])
