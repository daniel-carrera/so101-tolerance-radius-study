#!/usr/bin/env python3
"""Write the paper's number macros: placeholders now, real values later, under the same names.

No result is typed into a section of the paper. Each result is a LaTeX macro with a fixed name, and this script owns
the list of names. Run with no arguments it defines every result as a visible placeholder (a bold "??"), so the paper
can be written and built before the experiment has run. Later the analysis hands over real values under the same
names (`--values FILE.json`); whatever it has not produced stays a visible "??".

It writes three files into `paper/generated/`:

    numbers.tex    the results of the experiment: one \\newcommand per name
    protocol.tex   the protocol's constants, read from config/protocol.toml (real values, never placeholders)
    MACROS.md      every name, what it holds, and which block of trials or which record produces it

A third kind of number, settings that a person copies from the rig or from the training configuration, lives in
`paper/numbers_by_hand.tex`. This script never writes that file; it only lists its macros in MACROS.md.

Use, from the repository's top folder (Python 3.11 or newer, standard library only):

    python3 paper/make_numbers_stub.py                    # placeholders; refuses if numbers.tex already holds values
    python3 paper/make_numbers_stub.py --values V.json    # real values for the names in V.json, "??" for the rest
    python3 paper/make_numbers_stub.py --force            # back to placeholders, discarding values
    python3 paper/make_numbers_stub.py --list             # print the result names, one per line, and write nothing

`V.json` maps a name (no backslash) to the text the paper prints, or to `{"value": text, "source": where from}`:

    {"radiusClustered": "23.5", "nPairs": {"value": "26", "source": "data/trials/t2.jsonl"}}

A name that is not in the list is refused, so a misspelt name cannot pass silently. From Python, the analysis can call
`write_generated(values)` instead.

How a value is written (the unit is typed in the section, after the macro):

    distances       millimetres with one decimal, no unit:   23.5
    counts          whole numbers:                            20
    shares          percent as a whole number, no sign:       85
    probabilities   two decimals, or a bound:                 0.93   <0.01   (the sections use them inside $...$)
    scale factor    three decimals:                           1.031
    camera's creep  millimetres with two decimals, no unit:   0.42   (one decimal could print it as its limit)

To add a number: add a line to RESULTS_TABLE (or an entry to PROTOCOL, if it is a parameter in config/protocol.toml),
run this script, then use the macro in a section. A name is letters only, starts lower-case and has a capital letter
in it, which is how `check_paper.py` tells a number macro from a LaTeX command.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parent
GENERATED_DIR = PAPER_DIR / "generated"
PROTOCOL_TOML = PAPER_DIR.parent / "config" / "protocol.toml"
BY_HAND_TEX = PAPER_DIR / "numbers_by_hand.tex"

PLACEHOLDER = r"\textbf{??}"
#: Letters only (TeX allows nothing else in a macro name), a lower-case start and at least one capital.
NAME_PATTERN = re.compile(r"^[a-z][A-Za-z]*[A-Z][A-Za-z]*$")
#: One definition per line, as this script writes them and as numbers_by_hand.tex must keep them.
DEFINITION_PATTERN = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\\xspace\}\s*(?:%\s?(.*))?$")

Config = Mapping[str, Any]

# ----------------------------------------------------------------------------------------------------------------------
# The results. A line "== block | title" opens a group, "<< ..." says what produces it, and every other line is
# "name | what it will hold". The order here is the order in numbers.tex and in MACROS.md.
# ----------------------------------------------------------------------------------------------------------------------
RESULTS_TABLE = """
== sim | Simulation of the procedure, before the protocol is frozen
<< the procedure simulation (analysis/simulation/procedure.md); no trials
simTruthRadius | tolerance radius assumed true in the reported simulation, mm
simTruthWidth | width of the success curve assumed true in the reported simulation, mm
simReplications | number of simulated blocks behind each simulation figure
simBias | mean of (estimated radius - true radius) for the clustered policy's block as planned, mm
simRmse | root-mean-square error of the estimated radius for that block, mm
simIntervalWidth | mean width of the 95 % interval of the radius for that block, mm
simCoverage | share of simulated blocks whose 95 % interval held the true radius, percent
simRmseConstant | the same error with the block's trials at fixed, equally spaced distances, mm
simIntervalWidthConstant | the same interval width with trials at fixed, equally spaced distances, mm
simPairedRateLow | lower of the two success rates assumed for the paired test's power, percent
simPairedRateHigh | higher of the two success rates assumed for the paired test's power, percent
simPairedPower | chance that the exact McNemar test tells those two rates apart at the planned pairs, percent
simWidthCliff | width of the true curve taken as a cliff in the simulated figures by block, mm
simWidthMiddling | width of the true curve taken as a middling width in the simulated figures by block, mm
simWidthSlope | width of the true curve taken as a slope in the simulated figures by block, mm
simRmseReplayCliff | simulated root-mean-square error of the radius for the replay's block as written and a cliff: the middle one of the simulated truths with that width, mm
simIntervalWidthReplayCliff | mean width of the 95 % interval for the replay's block as written and a cliff: the middle one of the simulated truths with that width, mm
simRmseReplayMiddling | simulated root-mean-square error of the radius for the replay's block as written and a middling width: the middle one of the simulated truths with that width, mm
simIntervalWidthReplayMiddling | mean width of the 95 % interval for the replay's block as written and a middling width: the middle one of the simulated truths with that width, mm
simRmseReplaySlope | simulated root-mean-square error of the radius for the replay's block as written and a slope: the middle one of the simulated truths with that width, mm
simIntervalWidthReplaySlope | mean width of the 95 % interval for the replay's block as written and a slope: the middle one of the simulated truths with that width, mm
simOpenBelowReplay | share of simulated blocks whose interval is open below, for the replay's block as written and truths already falling at d = 0 (radius under two widths), percent
simRmseClusteredCliff | simulated root-mean-square error of the radius for the clustered policy's block as written and a cliff: the middle one of the simulated truths with that width, mm
simIntervalWidthClusteredCliff | mean width of the 95 % interval for the clustered policy's block as written and a cliff: the middle one of the simulated truths with that width, mm
simRmseClusteredMiddling | simulated root-mean-square error of the radius for the clustered policy's block as written and a middling width: the middle one of the simulated truths with that width, mm
simIntervalWidthClusteredMiddling | mean width of the 95 % interval for the clustered policy's block as written and a middling width: the middle one of the simulated truths with that width, mm
simRmseClusteredSlope | simulated root-mean-square error of the radius for the clustered policy's block as written and a slope: the middle one of the simulated truths with that width, mm
simIntervalWidthClusteredSlope | mean width of the 95 % interval for the clustered policy's block as written and a slope: the middle one of the simulated truths with that width, mm
simOpenBelowClustered | share of simulated blocks whose interval is open below, for the clustered policy's block as written and truths already falling at d = 0 (radius under two widths), percent
simRmseSpreadCliff | simulated root-mean-square error of the radius for the spread policy's block as written and a cliff: the middle one of the simulated truths with that width, mm
simIntervalWidthSpreadCliff | mean width of the 95 % interval for the spread policy's block as written and a cliff: the middle one of the simulated truths with that width, mm
simRmseSpreadMiddling | simulated root-mean-square error of the radius for the spread policy's block as written and a middling width: the middle one of the simulated truths with that width, mm
simIntervalWidthSpreadMiddling | mean width of the 95 % interval for the spread policy's block as written and a middling width: the middle one of the simulated truths with that width, mm
simRmseSpreadSlope | simulated root-mean-square error of the radius for the spread policy's block as written and a slope: the middle one of the simulated truths with that width, mm
simIntervalWidthSpreadSlope | mean width of the 95 % interval for the spread policy's block as written and a slope: the middle one of the simulated truths with that width, mm
simOpenBelowSpread | share of simulated blocks whose interval is open below, for the spread policy's block as written and truths already falling at d = 0 (radius under two widths), percent
simOpenBelow | the same share over the three threshold blocks together, percent

== rule | Simulation of the comparison rule, before the protocol is frozen
<< analysis/simulation/comparison_rule.csv (python -m tolerance_radius.compare_power): the clustered policy's block against the replay's, as written; no trials
simRuleReplayRadius | replay's radius assumed true in every case of the simulated comparison rule, mm
simRulePairs | simulated pairs of blocks behind each figure of the comparison rule
simRuleLargerWidthOneDiffEqual | share of simulated pairs of blocks in which the rule says "larger" although the two radii are equal, both curves of width 1 mm, percent
simRuleLargerWidthOneDiffThree | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 3 mm beyond the replay's, both curves of width 1 mm, percent
simRuleLargerWidthOneDiffFive | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 5 mm beyond the replay's, both curves of width 1 mm, percent
simRuleLargerWidthOneDiffEight | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 8 mm beyond the replay's, both curves of width 1 mm, percent
simRuleLargerWidthOneDiffTen | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 10 mm beyond the replay's, both curves of width 1 mm, percent
simRuleLargerWidthOneDiffFifteen | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 15 mm beyond the replay's, both curves of width 1 mm, percent
simRuleSmallerWidthOneDiffEqual | share of simulated pairs of blocks in which the rule says "smaller" although the two radii are equal, both curves of width 1 mm, percent
simRuleLargerWidthThreeDiffEqual | share of simulated pairs of blocks in which the rule says "larger" although the two radii are equal, both curves of width 3 mm, percent
simRuleLargerWidthThreeDiffThree | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 3 mm beyond the replay's, both curves of width 3 mm, percent
simRuleLargerWidthThreeDiffFive | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 5 mm beyond the replay's, both curves of width 3 mm, percent
simRuleLargerWidthThreeDiffEight | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 8 mm beyond the replay's, both curves of width 3 mm, percent
simRuleLargerWidthThreeDiffTen | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 10 mm beyond the replay's, both curves of width 3 mm, percent
simRuleLargerWidthThreeDiffFifteen | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 15 mm beyond the replay's, both curves of width 3 mm, percent
simRuleSmallerWidthThreeDiffEqual | share of simulated pairs of blocks in which the rule says "smaller" although the two radii are equal, both curves of width 3 mm, percent
simRuleLargerWidthSixDiffEqual | share of simulated pairs of blocks in which the rule says "larger" although the two radii are equal, both curves of width 6 mm, percent
simRuleLargerWidthSixDiffThree | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 3 mm beyond the replay's, both curves of width 6 mm, percent
simRuleLargerWidthSixDiffFive | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 5 mm beyond the replay's, both curves of width 6 mm, percent
simRuleLargerWidthSixDiffEight | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 8 mm beyond the replay's, both curves of width 6 mm, percent
simRuleLargerWidthSixDiffTen | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 10 mm beyond the replay's, both curves of width 6 mm, percent
simRuleLargerWidthSixDiffFifteen | share of simulated pairs of blocks in which the rule says "larger" when the policy's radius is 15 mm beyond the replay's, both curves of width 6 mm, percent
simRuleSmallerWidthSixDiffEqual | share of simulated pairs of blocks in which the rule says "smaller" although the two radii are equal, both curves of width 6 mm, percent

== V0 | Instrument check against a printed scale (camera only)
<< data/instrument/ruler_*.csv through scripts/check_instrument.py; the camera checks of every sitting and block
nScaleRuns | number of scale runs (planned: 4, in x and in y at two places on the grid)
nScaleSteps | number of 10 mm steps measured over all scale runs
instrumentMeanAbsError | mean absolute error of a measured 10 mm step over all scale runs, mm
instrumentMaxStepError | largest error of a single 10 mm step, mm
scaleFactor | measured length of a true 50 mm run divided by 50, mean of the runs, three decimals
scaleFactorMin | smallest scale factor among the runs, three decimals
scaleFactorMax | largest scale factor among the runs, three decimals
nRepeatability | number of repeated placements on one target (planned: 10)
repeatabilityScatter | largest distance of a repeated placement from the mean of the repeats, mm
gridDriftMax | largest disagreement of a session's fresh camera-to-table solve with the stored one, mm

== V1 | Reset by eye (camera only): the cube placed from a reference picture, with no overlay
<< the by-eye placement log under data/instrument/
nByEye | number of by-eye placements measured (planned: 20)
nOperatorsByEye | number of people who placed the cube by eye
byEyeErrorMedian | median distance of a by-eye placement from its target, mm
byEyeErrorNinetyFive | 95th percentile of that distance, mm
byEyeErrorMax | largest by-eye error, mm

== data | The two datasets: recording and start-state audit
<< data/audit/clustered.json and data/audit/spread.json; the recorder's logs; the dataset checks before training
nDemosTrainedClustered | demonstrations of the clustered set that were trained on
nDemosTrainedSpread | demonstrations of the spread set that were trained on
nPracticeEpisodes | practice episodes recorded first and discarded
nRerecordedClustered | episodes of the clustered set that were recorded again
nRerecordedSpread | episodes of the spread set that were recorded again
demoPlacementErrorMedian | median distance of a demonstration's measured start from its planned target, both sets, mm
demoPlacementErrorMax | largest such distance, both sets, mm
clusteredScatterMax | largest distance of a clustered demonstration's start from the mean start of its point, mm
spreadSpacingMin | smallest distance between two measured starts of the spread set, mm
nTornFramesNew | torn video frames found in the two new datasets by the torn-frame scan
nMispacedEpisodesNew | episodes of the two new datasets whose frame pacing failed its check
nRecordingDays | days on which the demonstrations were recorded

== T1b | Replay: the hardware's own tolerance
<< data/trials/*.jsonl, block T1b (condition replay)
nTrialsReplay | scored replay trials
nVoidReplay | void replay trials
successZeroReplay | successes among the trials of block T1b at the replayed demonstration's own start (d = 0)
nZeroReplay | trials of block T1b at d = 0
radiusReplay | tolerance radius of the replay: posterior median of r50, mm
radiusReplayLow | lower end of its 95 % interval, mm
radiusReplayHigh | upper end of its 95 % interval, mm
radiusReplayWide | the replay's radius fitted with the longer list of lapse values (the plateau check), mm
radiusReplayWideLow | lower end of that fit's 95 % interval, mm
radiusReplayWideHigh | upper end of that fit's 95 % interval, mm
plateauShiftReplay | the shift of the replay's plateau check: its radius with the longer list of lapse values minus the reported one, mm
maxSuccessDistReplay | largest measured d with a success in block T1b, the replay's threshold block, mm
minFailureDistReplay | smallest measured d with a failure in block T1b, the replay's threshold block, mm
nReplayEpisodes | number of different demonstrations that were replayed

== T1a | Clustered policy: its tolerance radius
<< data/trials/*.jsonl, block T1a, and the clustered policy's trials of block T2 where a line says so
nTrialsClustered | scored trials of the clustered policy, all blocks (T1a and its half of T2)
nTrialsClusteredThreshold | scored trials of block T1a alone: what the threshold cost
nVoidClusteredThreshold | void trials of block T1a alone
nVoidClustered | void trials of the clustered policy, all blocks
successZeroClustered | successes of the clustered policy at d = 0, in its threshold block (T1a alone)
nZeroClustered | clustered-policy trials at d = 0, in its threshold block (T1a alone)
successFarClustered | successes of the clustered policy at the far anchor distance
nFarClustered | clustered-policy trials at the far anchor distance
radiusClustered | tolerance radius of the clustered policy from all its scored trials (T1a and T2), mm
radiusClusteredLow | lower end of its 95 % interval, mm
radiusClusteredHigh | upper end of its 95 % interval, mm
radiusClusteredWide | the clustered policy's radius from all its scored trials fitted with the longer list of lapse values (the plateau check), mm
radiusClusteredWideLow | lower end of that fit's 95 % interval, mm
radiusClusteredWideHigh | upper end of that fit's 95 % interval, mm
plateauShiftClustered | the shift of that plateau check: the radius from all its scored trials with the longer list of lapse values minus the reported one, mm
radiusClusteredThreshold | the same radius fitted without the T2 trials (the curve behind the lookup prediction), mm
radiusClusteredThresholdLow | lower end of that fit's 95 % interval, mm
radiusClusteredThresholdHigh | upper end of that fit's 95 % interval, mm
radiusClusteredThresholdWide | the radius of block T1a alone fitted with the longer list of lapse values (the plateau check), mm
radiusClusteredThresholdWideLow | lower end of that fit's 95 % interval, mm
radiusClusteredThresholdWideHigh | upper end of that fit's 95 % interval, mm
plateauShiftClusteredThreshold | the shift of that plateau check: the radius of block T1a alone with the longer list of lapse values minus the reported one, mm
maxSuccessDistClustered | largest measured d with a success of the clustered policy in its threshold block (T1a alone), mm
minFailureDistClustered | smallest measured d with a failure of the clustered policy in its threshold block (T1a alone), mm

== T2 | Workspace test: both policies at the same placements
<< data/trials/*.jsonl, block T2 (pairs sharing a pair_id)
nPairs | placements at which both policies have a scored trial
nTrialsWorkspace | scored trials of block T2, both policies
nVoidWorkspace | void trials in block T2
workspaceSuccessClustered | placements at which the clustered policy succeeded
workspaceClusteredLow | lower end of the Wilson 95 % interval of its success rate, percent
workspaceClusteredHigh | upper end of that interval, percent
workspaceSuccessSpread | placements at which the spread policy succeeded
workspaceSpreadLow | lower end of the Wilson 95 % interval of its success rate, percent
workspaceSpreadHigh | upper end of that interval, percent
pairsBoth | placements at which both policies succeeded
pairsNeither | placements at which neither succeeded
pairsOnlySpread | placements at which only the spread policy succeeded
pairsOnlyClustered | placements at which only the clustered policy succeeded
pairsDiscordant | placements at which the two outcomes differed
mcnemarP | two-sided p-value of the exact McNemar test on the discordant pairs
workspaceDistClusteredMedian | median d of the T2 placements to the nearest clustered demonstration, mm
workspaceDistClusteredMax | largest such d, mm
workspaceDistSpreadMedian | median d of the T2 placements to the nearest spread demonstration, mm
workspaceDistSpreadMax | largest such d, mm
lookupPredictedClustered | successes the lookup prediction expects of the clustered policy in T2, one decimal
lookupClusteredLow | lower end of the 95 % predictive interval of that count
lookupClusteredHigh | upper end of that predictive interval
lookupPredictedSpread | successes the lookup prediction expects of the spread policy in T2, one decimal
lookupSpreadLow | lower end of the 95 % predictive interval of that count
lookupSpreadHigh | upper end of that predictive interval
nPairsTrained | placements of block T2 whose measured position lies in a trained cell
nPairsHeldOut | placements of block T2 whose measured position lies in a held-out cell
workspaceSuccessClusteredTrained | placements in trained cells at which the clustered policy succeeded
workspaceSuccessClusteredHeldOut | placements in held-out cells at which the clustered policy succeeded
workspaceSuccessSpreadTrained | placements in trained cells at which the spread policy succeeded
workspaceSuccessSpreadHeldOut | placements in held-out cells at which the spread policy succeeded
lookupPredictedClusteredTrained | successes the lookup prediction expects of the clustered policy at the placements in trained cells, one decimal
lookupPredictedClusteredHeldOut | successes the lookup prediction expects of the clustered policy at the placements in held-out cells, one decimal
lookupPredictedSpreadTrained | successes the lookup prediction expects of the spread policy at the placements in trained cells, one decimal
lookupPredictedSpreadHeldOut | successes the lookup prediction expects of the spread policy at the placements in held-out cells, one decimal

== T3 | Spread policy: its tolerance radius (run only if time allows)
<< data/trials/*.jsonl, block T3, and the spread policy's trials of block T2
nTrialsSpread | scored trials of the spread policy, all blocks (T3 and its half of T2)
nVoidSpread | void trials of the spread policy, all blocks
nTrialsSpreadThreshold | scored trials of block T3 alone
nVoidSpreadThreshold | void trials of block T3 alone
successZeroSpread | successes of the spread policy at d = 0, in its threshold block (T3 alone)
nZeroSpread | spread-policy trials at d = 0, in its threshold block (T3 alone)
radiusSpread | tolerance radius of the spread policy from all its scored trials, mm
radiusSpreadLow | lower end of its 95 % interval, mm
radiusSpreadHigh | upper end of its 95 % interval, mm
radiusSpreadWide | the spread policy's radius from all its scored trials fitted with the longer list of lapse values (the plateau check), mm
radiusSpreadWideLow | lower end of that fit's 95 % interval, mm
radiusSpreadWideHigh | upper end of that fit's 95 % interval, mm
plateauShiftSpread | the shift of that plateau check: the radius from all its scored trials with the longer list of lapse values minus the reported one, mm
radiusSpreadThreshold | the same radius fitted from block T3 alone (the spread policy's block fit), mm
radiusSpreadThresholdLow | lower end of that fit's 95 % interval, mm
radiusSpreadThresholdHigh | upper end of that fit's 95 % interval, mm
radiusSpreadThresholdWide | the radius of block T3 alone fitted with the longer list of lapse values (the plateau check), mm
radiusSpreadThresholdWideLow | lower end of that fit's 95 % interval, mm
radiusSpreadThresholdWideHigh | upper end of that fit's 95 % interval, mm
maxSuccessDistSpread | largest measured d with a success of the spread policy in its threshold block (T3 alone), mm
minFailureDistSpread | smallest measured d with a failure of the spread policy in its threshold block (T3 alone), mm

== compare | Comparisons of radii, from independent posteriors
<< the fits of T1a, T1b and T3 (protocol section 10, comparison of radii)
radiusDiffClusteredReplay | clustered policy's radius minus the replay's: posterior median of the difference, mm
radiusDiffClusteredReplayLow | lower end of the 95 % interval of that difference, mm
radiusDiffClusteredReplayHigh | upper end of the 95 % interval of that difference, mm
probClusteredExceedsReplay | posterior probability that the clustered policy's radius exceeds the replay's
radiusDiffSpreadClustered | spread policy's radius minus the clustered policy's: posterior median, mm
radiusDiffSpreadClusteredLow | lower end of the 95 % interval of that difference, mm
radiusDiffSpreadClusteredHigh | upper end of the 95 % interval of that difference, mm
probSpreadExceedsClustered | posterior probability that the spread policy's radius exceeds the clustered policy's

== calc | Reset sensitivity: a calculation from the fitted curve, not a measurement
<< the clustered policy's fitted curve and the by-eye errors of block V1 (protocol section 10, reset sensitivity)
resetSensitivityPlaced | success the fitted curve gives at d = the placement errors of this study's resets, percent
resetSensitivityByEye | success the fitted curve gives at d = the by-eye errors of block V1, percent

== scene | The scene at each trial's start: the basket's place and the wrist camera's, read from the trials' own videos
<< data/audit/scene_<log>.json, written by scripts/rig/scene_audit.py; the outcomes from data/trials/*.jsonl
nTrialsBasketNamed | scored trials that began with the basket farther from its place than the protocol's 2 mm
nTrialsWristNamed | scored trials that began with the wrist camera farther from its place than the protocol's 2 px
nTrialsBasketMoved | scored trials in which the arm moved the basket
basketStartMax | largest distance of the basket from its place at a scored trial's start, across or down, mm
wristStartMax | largest distance of the wrist camera's picture from its place at a scored trial's start, across or down, in pixels, one decimal

== second | The second scoring from video: a part of the counted trials, scored again
<< data/second_scoring/main/agreement.json, written by scripts/second_scoring.py compare
nSecondScored | counted trials that were scored again from their videos
nSecondScoredAgree | of them, trials in which the second score is the first score
nSecondScoredCannotTell | of them, trials in which the second scorer could not tell from the video

== all | All trials, and what the trial records say about how they were run
<< data/trials/*.jsonl (every block): placements, voids, the execution and versions fields; docs/decisions.md
nTrialsTotal | scored trials, all blocks and conditions
nVoidTotal | void trials, all blocks and conditions
voidReasons | why trials were void, in words with a count each: setup fault (2), placement out of tolerance (1)
resetErrorMedian | median distance of an accepted placement from its target over all scored trials, mm
resetErrorMax | largest such distance, mm
viewCreptMedian | median distance of the front camera's picture from where its session solved the conversion, read just before each scored attempt, mm, two decimals
viewCreptMax | largest such distance before a scored attempt (a session makes no attempt beyond the protocol's limit), mm, two decimals
trialMinutesMedian | median time from one trial's start to the next one's, minutes, one decimal
nTrialDays | days on which evaluation trials ran
nOperators | people who placed the cube and scored the trials
nDeviations | deviations from the frozen protocol logged in docs/decisions.md
evalCheckpoint | training step of the checkpoints that were evaluated (the protocol plans 80,000)
execClamp | per-step clamp on a joint command during trials, degrees
execEnsembleCoeff | coefficient of the temporal ensembling used at rollout
execControlRate | control rate requested during trials, Hz
execControlRateAchieved | control rate achieved in scored policy trials, between the first step sent and the last: median over the trials, Hz, one decimal
execTimeLimit | time limit of one attempt, seconds
lerobotCommit | short commit id of the public LeRobot code that drove the arm
"""


@dataclass(frozen=True, slots=True)
class Result:
    """One number the experiment will produce."""

    name: str
    holds: str


@dataclass(frozen=True, slots=True)
class Group:
    """The results that one block of trials, or one kind of record, produces."""

    block: str
    title: str
    source: str
    results: tuple[Result, ...]


def parse_results(table: str) -> tuple[Group, ...]:
    """The groups of RESULTS_TABLE."""
    groups: list[Group] = []
    block = title = source = ""
    results: list[Result] = []

    def close() -> None:
        if block:
            groups.append(Group(block, title, source, tuple(results)))

    for line in (raw.strip() for raw in table.splitlines()):
        if not line:
            continue
        if line.startswith("== "):
            close()
            block, title = (part.strip() for part in line[3:].split("|", 1))
            source, results = "", []
        elif line.startswith("<< "):
            source = line[3:].strip()
        else:
            name, holds = (part.strip() for part in line.split("|", 1))
            results.append(Result(name, holds))
    close()
    return tuple(groups)


RESULTS: tuple[Group, ...] = parse_results(RESULTS_TABLE)


# ----------------------------------------------------------------------------------------------------------------------
# The protocol's constants, each read from config/protocol.toml.
# ----------------------------------------------------------------------------------------------------------------------
def _num(value: float) -> str:
    """A number as the paper prints it: whole numbers with a comma every three digits, others without padding."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


def _at(cfg: Config, path: str) -> Any:
    """The value at a dotted path of the parameter file."""
    node: Any = cfg
    for part in path.split("."):
        node = node[part]
    return node


def _trained_cells(cfg: Config) -> int:
    return int(_at(cfg, "table.cols")) * int(_at(cfg, "table.rows")) - len(_at(cfg, "table.held_out_cells"))


def _budget(cfg: Config) -> int:
    return _trained_cells(cfg) * int(_at(cfg, "datasets.episodes_per_cell"))


def _held_out_names(cfg: Config) -> str:
    """`A1, B2, D2 and E3`."""
    cells = list(_at(cfg, "table.held_out_cells"))
    return ", ".join(cells[:-1]) + " and " + cells[-1] if len(cells) > 1 else "".join(cells)


def _block_total(cfg: Config, block: str) -> int:
    spec = _at(cfg, f"blocks.{block}")
    if "pairs" in spec:
        return int(spec["pairs"]) * len(spec["conditions"])
    in_staircases = sum(int(s["trials"]) for s in spec["staircases"])
    return int(spec.get("anchors_zero", 0)) + int(spec.get("anchors_far", 0)) + in_staircases


def _staircase_trials(block: str) -> Callable[[Config], str]:
    """Trials per staircase of a block. The method section says "two staircases", so there must be two, alike."""

    def read(cfg: Config) -> str:
        staircases = _at(cfg, f"blocks.{block}.staircases")
        counts = {int(s["trials"]) for s in staircases}
        if len(staircases) != 2 or len(counts) != 1:
            raise ValueError(
                f"blocks.{block}.staircases is no longer two staircases of equal length: "
                "change the method section's text and this function together"
            )
        return _num(counts.pop())

    return read


def _staircase_start(block: str, staircase_id: str) -> Callable[[Config], str]:
    def read(cfg: Config) -> str:
        for staircase in _at(cfg, f"blocks.{block}.staircases"):
            if staircase["id"] == staircase_id:
                return _num(staircase["start_level_mm"])
        raise KeyError(f"blocks.{block}.staircases[{staircase_id}]")

    return read


def _region_pairs(region: str) -> Callable[[Config], str]:
    """Placements the workspace test draws over one kind of cell: from the block's list of regions, or from its
    single region. The method section says so many over the trained cells and so many over the held-out cells, so
    the block must draw over those two regions and no other."""

    def read(cfg: Config) -> str:
        spec = _at(cfg, "blocks.T2")
        if "regions" in spec:
            counts = {entry["region"]: int(entry["pairs"]) for entry in spec["regions"]}
        else:
            counts = {spec.get("region", "trained_cells"): int(spec["pairs"])}
        if not set(counts) <= {"trained_cells", "held_out_cells"} or sum(counts.values()) != int(spec["pairs"]):
            raise ValueError(
                "blocks.T2 no longer draws its placements over the trained cells and the held-out cells alone: "
                "change the method section's text and this function together"
            )
        return _num(counts.get(region, 0))

    return read


@dataclass(frozen=True, slots=True)
class Constant:
    """One parameter of the protocol. With no `value`, it is the number at `key` in the parameter file."""

    name: str
    key: str
    holds: str
    value: Callable[[Config], str] | None = None

    def text(self, cfg: Config) -> str:
        """What the paper prints for it."""
        return self.value(cfg) if self.value else _num(_at(cfg, self.key))


# One line per constant: "name | where it comes from in config/protocol.toml | what it holds".
# Where the middle column is a plain dotted key, the value is the number at that key. Otherwise COMPUTED says how.
PROTOCOL_TABLE = """
protocolVersion | protocol.version | version of the protocol the paper describes
gridCols | table.cols | columns of the grid
gridRows | table.rows | rows of the grid
cellSize | table.cell_mm | side of a grid cell, mm
cubeEdge | table.cube_edge_mm | edge of the cube, mm
nCellsHeldOut | table.held_out_cells (count) | cells in which no demonstration starts
heldOutCells | table.held_out_cells (names) | their names, in words
nCellsTrained | table.cols x table.rows - held-out cells | cells in which demonstrations start
testMargin | table.test_region_margin_mm | test placements stay this far inside the grid's edge, mm
placementTolerance | placement.tolerance_mm | a reset is accepted within this distance of its target, mm
levelMatch | placement.level_match_mm | a requested distance is met within this, mm
nDemosPerCell | datasets.episodes_per_cell | demonstrations per trained cell, in both layouts
nDemos | trained cells x datasets.episodes_per_cell | demonstrations in each dataset: the budget
nPointsClustered | trained cells (one point each) | distinct start points of the clustered layout
nPointsSpread | trained cells x datasets.episodes_per_cell (all distinct) | distinct starts of the spread layout
spreadMinSpacing | datasets.spread_min_spacing_mm | smallest planned distance between two spread starts, mm
clusteredMaxDistInCell | table.cell_mm / sqrt(2), rounded | farthest point of a trained cell from its centre, mm
nSittings | datasets.sittings (count) | recording sittings, alternating the two layouts
trainSteps | training.evaluation_checkpoint | training step at which both policies are evaluated
trainStepsFallback | training.fallback_min_checkpoint | smallest step the fallback rule allows
stepInitial | staircase.step_initial_mm | staircase step until the first reversal, mm
stepFinal | staircase.step_final_mm | staircase step after the first reversal, mm
plannedReplay | blocks.T1b (anchors + staircases) | replay trials planned
plannedClustered | blocks.T1a (anchors + staircases) | clustered-policy threshold trials planned
plannedWorkspace | blocks.T2.pairs x conditions | workspace-test trials planned
plannedSpread | blocks.T3 (anchors + staircases) | spread-policy threshold trials planned
plannedTotal | sum of the four blocks | scored trials planned in all
nPairsPlanned | blocks.T2.pairs | placements planned for the workspace test
nPairsTrainedPlanned | blocks.T2.regions[trained_cells].pairs | of them, placements drawn over the trained cells
nPairsHeldOutPlanned | blocks.T2.regions[held_out_cells].pairs | of them, placements drawn over the held-out cells
anchorsZeroReplay | blocks.T1b.anchors_zero | replay trials planned at d = 0
anchorsZeroClustered | blocks.T1a.anchors_zero | clustered-policy trials planned at d = 0
anchorsFarClustered | blocks.T1a.anchors_far | clustered-policy trials planned at the far distance
anchorsZeroSpread | blocks.T3.anchors_zero | spread-policy trials planned at d = 0
farLevel | blocks.T1a.far_level_mm | the far anchor distance, mm
stairTrialsReplay | blocks.T1b.staircases[].trials | trials in each of the replay's two staircases
stairTrialsClustered | blocks.T1a.staircases[].trials | trials in each of the clustered policy's two staircases
stairTrialsSpread | blocks.T3.staircases[].trials | trials in each of the spread policy's two staircases
stairStartNearReplay | blocks.T1b.staircases[up].start_level_mm | start of the replay's near staircase, mm
stairStartFarReplay | blocks.T1b.staircases[down].start_level_mm | start of the replay's far staircase, mm
stairStartNearClustered | blocks.T1a.staircases[up].start_level_mm | start of the clustered policy's near staircase, mm
stairStartFarClustered | blocks.T1a.staircases[down].start_level_mm | start of the clustered policy's far staircase, mm
stairStartNearSpread | blocks.T3.staircases[up].start_level_mm | start of the spread policy's near staircase, mm
stairStartFarSpread | blocks.T3.staircases[down].start_level_mm | start of the spread policy's far staircase, mm
maxLevelReplay | blocks.T1b.max_level_mm | largest distance a replay staircase may request, mm
maxLevelClustered | blocks.T1a.max_level_mm | largest distance a clustered-policy staircase may request, mm
maxLevelSpread | blocks.T3.max_level_mm | largest distance a spread-policy staircase may request, mm
gridRadiusMax | analysis.r50_grid_mm.max | upper end of the prior on the tolerance radius, mm
gridWidthMin | analysis.width_grid_mm.min | lower end of the prior on the curve's width, mm
gridWidthMax | analysis.width_grid_mm.max | upper end of the prior on the curve's width, mm
lapseValues | analysis.lapse_values | the lapse rates the model weighs, for use inside math braces
lapseValuesWide | analysis.lapse_values_sensitivity | the longer list of lapse rates of the plateau check, for use inside math braces
plateauShiftLimit | analysis.sensitivity_limit_mm | how far the radii of a fit and of its plateau check may lie apart before the plateau is in doubt, mm
floorValues | analysis.floor_values | the floors the model weighs (the chance of a success far from every demonstration), for use inside math braces
nReplayDemosPlanned | blocks.T1b.replay_demos | demonstrations the replay block plays back, each from another cell
handoffTolerance | execution.handoff_tolerance_deg | how far a policy's first command may be from the arm's pose, degrees per joint
handoffToleranceReplay | execution.replay_handoff_tolerance_deg | the same for a replay, degrees per joint
viewCreptLimit | execution.view_crept_limit_mm | no attempt once the front camera's picture stands this far from where its session solved the conversion, mm
intervalLevel | analysis.interval_level x 100 | level of every reported interval, percent
"""

COMPUTED: dict[str, Callable[[Config], str]] = {
    "protocolVersion": lambda c: str(_at(c, "protocol.version")),
    "nCellsHeldOut": lambda c: _num(len(_at(c, "table.held_out_cells"))),
    "heldOutCells": _held_out_names,
    "nCellsTrained": lambda c: _num(_trained_cells(c)),
    "nDemos": lambda c: _num(_budget(c)),
    "nPointsClustered": lambda c: _num(_trained_cells(c)),
    "nPointsSpread": lambda c: _num(_budget(c)),
    "clusteredMaxDistInCell": lambda c: _num(round(float(_at(c, "table.cell_mm")) / math.sqrt(2.0))),
    "nSittings": lambda c: _num(len(_at(c, "datasets.sittings"))),
    "plannedReplay": lambda c: _num(_block_total(c, "T1b")),
    "plannedClustered": lambda c: _num(_block_total(c, "T1a")),
    "plannedWorkspace": lambda c: _num(_block_total(c, "T2")),
    "plannedSpread": lambda c: _num(_block_total(c, "T3")),
    "plannedTotal": lambda c: _num(sum(_block_total(c, block) for block in ("T1b", "T1a", "T2", "T3"))),
    "nPairsTrainedPlanned": _region_pairs("trained_cells"),
    "nPairsHeldOutPlanned": _region_pairs("held_out_cells"),
    "stairTrialsReplay": _staircase_trials("T1b"),
    "stairTrialsClustered": _staircase_trials("T1a"),
    "stairTrialsSpread": _staircase_trials("T3"),
    "stairStartNearReplay": _staircase_start("T1b", "up"),
    "stairStartFarReplay": _staircase_start("T1b", "down"),
    "stairStartNearClustered": _staircase_start("T1a", "up"),
    "stairStartFarClustered": _staircase_start("T1a", "down"),
    "stairStartNearSpread": _staircase_start("T3", "up"),
    "stairStartFarSpread": _staircase_start("T3", "down"),
    "lapseValues": lambda c: ", ".join("0" if v == 0 else f"{v:.2f}" for v in _at(c, "analysis.lapse_values")),
    "lapseValuesWide": lambda c: ", ".join(
        "0" if v == 0 else f"{v:.2f}" for v in _at(c, "analysis.lapse_values_sensitivity")
    ),
    "floorValues": lambda c: ", ".join("0" if v == 0 else f"{v:.2f}" for v in _at(c, "analysis.floor_values")),
    "intervalLevel": lambda c: _num(round(float(_at(c, "analysis.interval_level")) * 100)),
}


def parse_protocol(table: str) -> tuple[Constant, ...]:
    """The constants of PROTOCOL_TABLE, each with its way of being computed if it has one."""
    constants = []
    for line in (raw.strip() for raw in table.splitlines()):
        if line:
            name, key, holds = (part.strip() for part in line.split("|"))
            constants.append(Constant(name, key, holds, COMPUTED.get(name)))
    unknown = sorted(set(COMPUTED) - {constant.name for constant in constants})
    if unknown:
        raise ValueError(f"COMPUTED names constants that PROTOCOL_TABLE does not have: {', '.join(unknown)}")
    return tuple(constants)


PROTOCOL: tuple[Constant, ...] = parse_protocol(PROTOCOL_TABLE)


# ----------------------------------------------------------------------------------------------------------------------
# Checks on the lists themselves
# ----------------------------------------------------------------------------------------------------------------------
def result_names() -> list[str]:
    """Every result name, in file order."""
    return [result.name for group in RESULTS for result in group.results]


def protocol_names() -> list[str]:
    """Every protocol-constant name, in file order."""
    return [constant.name for constant in PROTOCOL]


def check_lists() -> None:
    """Refuse names TeX cannot use, names without a capital, a name defined twice, and a result with no description."""
    names = result_names() + protocol_names()
    for name in names:
        if not NAME_PATTERN.match(name):
            raise ValueError(f"{name!r}: a macro name is letters only, starts lower-case and has a capital in it")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"defined more than once: {', '.join(repeated)}")
    for group in RESULTS:
        if not group.source or not group.results:
            raise ValueError(f"group {group.block!r} needs a '<<' line saying what produces it, and a result")
        undescribed = [result.name for result in group.results if not result.holds]
        if undescribed:
            raise ValueError(f"no description for: {', '.join(undescribed)}")


# ----------------------------------------------------------------------------------------------------------------------
# Values handed over by the analysis
# ----------------------------------------------------------------------------------------------------------------------
def checked_values(raw: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """`{name: (text, source)}` from what the analysis gave, refusing unknown names and text TeX would trip on."""
    unknown = sorted(set(raw) - set(result_names()))
    if unknown:
        raise ValueError(
            f"not in the list of results: {', '.join(unknown)}. Add the name to RESULTS_TABLE first, "
            "or correct its spelling (protocol constants come from config/protocol.toml, not from values)"
        )
    values: dict[str, tuple[str, str]] = {}
    for name, item in raw.items():
        text, source = (item.get("value"), item.get("source", "")) if isinstance(item, Mapping) else (item, "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{name}: the value must be non-empty text, as the paper prints it (got {text!r})")
        if not isinstance(source, str):
            raise ValueError(f"{name}: the source must be text")
        if "\n" in text + source or "\r" in text + source:
            raise ValueError(f"{name}: the value and its source must each fit on one line")
        if re.search(r"(?<!\\)[%#&_$]", text) or text.count("{") != text.count("}"):
            raise ValueError(f"{name}: {text!r} has a character TeX reads as a command; escape it (\\%, \\_) or reword")
        if "??" in text:
            raise ValueError(f"{name}: a value may not contain the placeholder mark '??'")
        values[name] = (text.strip(), source.strip())
    return values


# ----------------------------------------------------------------------------------------------------------------------
# What is written
# ----------------------------------------------------------------------------------------------------------------------
def _definition(name: str, body: str, comment: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{body}\\xspace}} % {comment}"


def render_numbers(values: Mapping[str, tuple[str, str]] | None = None) -> str:
    """The text of generated/numbers.tex: every result, as its value if one was given and as a placeholder if not."""
    values = values or {}
    total = len(result_names())
    lines = [
        "% GENERATED FILE: do not edit by hand. Written by paper/make_numbers_stub.py.",
        "% The results of the experiment, one macro each. generated/MACROS.md says what each holds.",
        f"% A macro that prints a bold ?? has no value yet: {total - len(values)} of {total} are placeholders.",
        "% After a macro write a space, a unit joined by a tie (~mm, ~\\%), punctuation, or {}. Never \\, or ].",
    ]
    for group in RESULTS:
        lines += ["", f"% --- {group.block}: {group.title}", f"%     produced from: {group.source}"]
        for result in group.results:
            if result.name in values:
                text, source = values[result.name]
                lines.append(_definition(result.name, text, result.holds + (f" | src: {source}" if source else "")))
            else:
                lines.append(_definition(result.name, PLACEHOLDER, result.holds))
    return "\n".join(lines) + "\n"


def load_protocol(path: Path = PROTOCOL_TOML) -> dict[str, Any]:
    """The protocol's parameter file."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def protocol_texts(cfg: Config) -> dict[str, str]:
    """`{name: text}` for every protocol constant, with a plain message if the parameter file lacks a key."""
    texts: dict[str, str] = {}
    for constant in PROTOCOL:
        try:
            texts[constant.name] = constant.text(cfg)
        except KeyError as missing:
            raise KeyError(
                f"config/protocol.toml has no {missing} (needed for \\{constant.name}, from {constant.key}). "
                "If the file's layout changed, change PROTOCOL in paper/make_numbers_stub.py to match"
            ) from None
    return texts


def render_protocol(cfg: Config) -> str:
    """The text of generated/protocol.tex: the protocol's constants as the parameter file has them now."""
    texts = protocol_texts(cfg)
    lines = [
        "% GENERATED FILE: do not edit by hand. Written by paper/make_numbers_stub.py from config/protocol.toml.",
        f"% protocol version: {_at(cfg, 'protocol.version')}",
        f"% protocol status: {_at(cfg, 'protocol.status')}",
        "% While the status is not frozen these values may still change: run the script again after any change.",
        "",
    ]
    for constant in PROTOCOL:
        comment = f"{constant.holds} | src: config/protocol.toml, {constant.key}"
        lines.append(_definition(constant.name, texts[constant.name], comment))
    return "\n".join(lines) + "\n"


def read_definitions(text: str) -> dict[str, tuple[str, str]]:
    """`{name: (body, comment)}` for every one-line definition in a macro file."""
    found: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        match = DEFINITION_PATTERN.match(line.strip())
        if match:
            found[match.group(1)] = (match.group(2), match.group(3) or "")
    return found


def _cell(text: str) -> str:
    """Text made safe for a cell of a Markdown table."""
    return text.replace("|", "\\|")


def render_macros_md(
    cfg: Config,
    values: Mapping[str, tuple[str, str]] | None = None,
    by_hand: Mapping[str, tuple[str, str]] | None = None,
) -> str:
    """The text of generated/MACROS.md."""
    values = values or {}
    by_hand = by_hand or {}
    texts = protocol_texts(cfg)
    total = len(result_names())
    out = [
        "# The paper's number macros",
        "",
        "Generated by `paper/make_numbers_stub.py`. Do not edit: change the lists in that script and run it again.",
        "",
        "Three files define every number the paper prints. A section never has a result typed into it.",
        "",
        "| File | Holds | Written by |",
        "|---|---|---|",
        "| `generated/numbers.tex` | The results of the experiment | The script: placeholders now, the analysis's "
        "values later (`--values`) |",
        "| `generated/protocol.tex` | The protocol's constants | The script, from `config/protocol.toml` |",
        "| `numbers_by_hand.tex` | Settings copied from the rig, the training configuration or the earlier project's "
        "records | The author, by hand, each with its source on the same line |",
        "",
        "How a value is written: distances in millimetres with one decimal and no unit (`23.5`); counts as whole",
        "numbers; shares as a whole percentage without the sign (`85`); probabilities and p-values with two decimals",
        "or as a bound (`0.93`, `<0.01`), which is why the sections use them inside `$...$`; the scale factor with",
        "three decimals. The unit is typed in the section, after the macro.",
        "",
        "## Results (`generated/numbers.tex`)",
        "",
        f"{len(values)} of {total} hold a value; the rest print a bold **??**.",
        "",
    ]
    for group in RESULTS:
        out += [
            f"### {group.block}: {group.title}",
            "",
            f"Produced from: {group.source}.",
            "",
            "| Macro | What it will hold | Now |",
            "|---|---|---|",
        ]
        for result in group.results:
            now = f"`{values[result.name][0]}`" if result.name in values else "??"
            out.append(f"| `\\{result.name}` | {_cell(result.holds)} | {_cell(now)} |")
        out.append("")
    out += [
        "## Protocol constants (`generated/protocol.tex`)",
        "",
        f"Read from `config/protocol.toml`, version {_at(cfg, 'protocol.version')}, status "
        f"**{_at(cfg, 'protocol.status')}**. These are real values, not placeholders. They change only when the",
        "parameter file changes: run the script again after any change. `check_paper.py` fails while the two disagree.",
        "",
        "| Macro | What it holds | Value now | Key in the parameter file |",
        "|---|---|---|---|",
    ]
    for constant in PROTOCOL:
        now, key = _cell(texts[constant.name]), _cell(constant.key)
        out.append(f"| `\\{constant.name}` | {_cell(constant.holds)} | {now} | `{key}` |")
    out += [
        "",
        "## Set by hand (`numbers_by_hand.tex`)",
        "",
        "Not produced by any block of trials. The author copies each one from the place its line names, and replaces",
        "the `??`. This table is read from that file.",
        "",
        "| Macro | What it holds, and where to copy it from | Now |",
        "|---|---|---|",
    ]
    for name, (body, comment) in by_hand.items():
        now = "??" if body.strip() == PLACEHOLDER else f"`{body.strip()}`"
        out.append(f"| `\\{name}` | {_cell(comment)} | {_cell(now)} |")
    if not by_hand:
        out.append("| (the file was not found) | | |")
    return "\n".join(out) + "\n"


def holds_values(numbers_tex: str) -> list[str]:
    """Names in an existing numbers.tex whose body is no longer the placeholder."""
    return [name for name, (body, _) in read_definitions(numbers_tex).items() if body.strip() != PLACEHOLDER]


def write_generated(
    values: Mapping[str, Any] | None = None,
    *,
    out_dir: Path = GENERATED_DIR,
    protocol_toml: Path = PROTOCOL_TOML,
    by_hand_tex: Path = BY_HAND_TEX,
) -> list[Path]:
    """Write numbers.tex, protocol.tex and MACROS.md, and return their paths. `values` is as in a `--values` file."""
    check_lists()
    checked = checked_values(values or {})
    cfg = load_protocol(protocol_toml)
    by_hand = read_definitions(by_hand_tex.read_text(encoding="utf-8")) if by_hand_tex.exists() else {}
    clash = sorted(set(by_hand) & set(result_names() + protocol_names()))
    if clash:
        raise ValueError(f"also defined in {by_hand_tex.name}: {', '.join(clash)}. A name belongs to one file only")
    files = {
        out_dir / "numbers.tex": render_numbers(checked),
        out_dir / "protocol.tex": render_protocol(cfg),
        out_dir / "MACROS.md": render_macros_md(cfg, checked, by_hand),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    return list(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the paper's number macros: placeholders, or given values.")
    parser.add_argument("--values", type=Path, help="JSON file: name -> text, or name -> {value, source}")
    parser.add_argument("--force", action="store_true", help="write placeholders even if numbers.tex holds values")
    parser.add_argument("--list", action="store_true", help="print the result names and write nothing")
    args = parser.parse_args(argv)

    try:
        check_lists()
        if args.list:
            print("\n".join(result_names()))
            return 0
        values: dict[str, Any] = {}
        if args.values:
            values = json.loads(args.values.read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError(f"{args.values}: expected a JSON object of name -> value")
        existing = GENERATED_DIR / "numbers.tex"
        if not args.values and not args.force and existing.exists():
            filled = holds_values(existing.read_text(encoding="utf-8"))
            if filled:
                print(
                    f"refused: generated/numbers.tex already holds {len(filled)} value(s), for example \\{filled[0]}. "
                    "Pass --values to write values, or --force to go back to placeholders.",
                    file=sys.stderr,
                )
                return 1
        written = write_generated(values)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as problem:
        print(f"make_numbers_stub: {problem}", file=sys.stderr)
        return 1

    total = len(result_names())
    for path in written:
        print(f"wrote {path.relative_to(PAPER_DIR.parent)}")
    print(f"results: {total} macros, {len(values)} with a value, {total - len(values)} placeholders")
    print(f"protocol constants: {len(PROTOCOL)} macros from config/protocol.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
