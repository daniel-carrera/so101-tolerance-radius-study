# The paper

A four-page workshop paper in IEEEtran conference mode, with references and an appendix beyond the four pages
(`docs/venue.md` has the call and its rules). This folder holds its sources and the checks that stand in for a
compiler.

**State on 2026-10-05: a skeleton, written before the experiment has run, and never compiled.** No TeX engine was
installed on the machine it was written on. The introduction, related work, method and limits are written in full.
The abstract, the results and the discussion are complete sentences with a macro wherever a number will go; every
such macro prints a bold **??** today. Build it once on Overleaf before relying on anything below.

**The text follows `protocol/protocol.md`, draft 0.3** (the decisions of 2026-10-05: a workspace test of 26 placements,
half of them in held-out cells; a replayed demonstration from every trained cell). When the protocol
changes again before its freeze, section 14 of that document says what changed, and the places to follow it are the
method, the results' IF lines, the limits and the appendix's section "The protocol".

## What is here

| File | What it is | Who writes it |
|---|---|---|
| `main.tex` | Preamble, title, the order of the sections | by hand |
| `sections/*.tex` | `abstract`, `intro`, `related`, `method`, `results`, `checklist`, `limits`, `appendix` | by hand |
| `references.bib` | The bibliography, verified entry by entry (`docs/reference/citations-verified.md`) | by hand, with its row in that file |
| `generated/numbers.tex` | The results of the experiment, one macro each | `make_numbers_stub.py`, then the analysis |
| `generated/protocol.tex` | The protocol's constants | `make_numbers_stub.py`, from `config/protocol.toml` |
| `generated/MACROS.md` | Every macro: what it holds and which block of trials produces it | `make_numbers_stub.py` |
| `numbers_by_hand.tex` | Settings copied from the rig, the training configuration and the earlier project's records | the author, each with its source on the line |
| `make_numbers_stub.py` | Owns the list of macro names; writes `generated/` | |
| `check_paper.py` | The checks below | |
| `forbidden.txt` | Generic strings that must not appear (paths, e-mail addresses, thanks) | shipped; the author's own strings go in a private list |
| `figures/README.md` | The four figures: what each shows and from which data. `scripts/make_figures.py` draws them; Figure 1 is drawn, the others wait for the audits and the results | |
| `submission.md` | Drafts for the OpenReview form, and the anonymity checklist | by hand |

## The rule about numbers

**No result is typed into a section.** A section may contain a number in three ways only:

1. **A result of the experiment** is a macro from `generated/numbers.tex` (`\radiusClustered`, `\nTrialsTotal`).
   Today each prints **??**.
2. **A parameter of the protocol** is a macro from `generated/protocol.tex` (`\nDemos`, `\stepInitial`). These hold
   real values, read from `config/protocol.toml`. When that file changes, run `make_numbers_stub.py` again;
   `check_paper.py` fails while the two disagree.
3. **A quantity that is already known** (the earlier evaluation's figures, a number quoted from a cited paper, a
   constant that lives only in `protocol/protocol.md`) is typed, with `% src: where it comes from` on the same line.
   `python3 paper/check_paper.py --numbers` lists every typed numeral that lacks one.

Settings that nobody measures in a trial (the camera's model, the batch size, the earlier project's audit counts)
are macros in `numbers_by_hand.tex`. The author copies each from the place its line names. Nothing there is filled
from memory.

### Which radius is which

The protocol fits each policy's success curve twice (`protocol/protocol.md`, section 10), and the paper says which fit a
number comes from wherever it prints a radius.

| Fit | Trials | Macros | Where it is printed |
|---|---|---|---|
| **Block fit**: the headline, and what every comparison and prediction uses | a condition's own threshold block | clustered policy: `\radiusClusteredThreshold` (`...Low`, `...High`); replay: `\radiusReplay` (its only fit); spread policy: **no macro yet** | abstract, results, Table I (a condition's own row), checklist, appendix |
| **Pooled fit**: a check | the threshold block and the policy's workspace-test trials | clustered policy: `\radiusClustered`; spread policy: `\radiusSpread` | results (one sentence) and Table I (rows marked "pooled"), always with the word "pooled" |

The method states the rule once ("every radius, comparison and prediction we report is or uses [a block fit]"), so a
radius without a label is a block fit. The differences and probabilities (`\radiusDiff...`, `\prob...Exceeds...`),
the lookup prediction and the reset sensitivity are made from block fits.

**An interval that is open on a side** is not printed as a radius with two ends. The analysis says for each fit
whether its interval is open below or above; the text then says "at most" the upper end or "at least" the lower end.
The abstract, the results and Table I hold the wording for each case in IF blocks or in a comment beside the cell.

### After a macro

Every number macro ends in `\xspace`, so a space after it is kept. Write one of these after a macro, and nothing else:

| Write | Example |
|---|---|
| a space | `\nDemos demonstrations` |
| a unit joined by a tie | `\radiusClustered~mm`, `\intervalLevel~\%` |
| punctuation | `\radiusClustered.` `(\radiusClusteredLow to \radiusClusteredHigh)` |
| `{}` before anything else | `\nTrialsReplay{} & ...` in a table |

Never `\,` and never `]` straight after a macro: `check_paper.py` fails on both, because the output would carry a
stray space. Probabilities and p-values go inside `$...$`, since a value such as `<0.01` prints wrongly outside math.

### How `generated/numbers.tex` is produced

```bash
python3 paper/make_numbers_stub.py                  # today: every result is a placeholder
python3 paper/make_numbers_stub.py --values V.json  # later: the analysis's values, ?? for whatever it lacks
```

The list of names is in `make_numbers_stub.py` (`RESULTS_TABLE`), one line each with what the number is. The analysis
(`scripts/make_report.py`, still to be written) computes the values from the logs and hands them over as JSON, or
calls `write_generated(values)`. A name that is not in the list is refused. Run without `--values`, the script
refuses to overwrite a file that already holds values, unless `--force` is given. To add a number: add its line to
the table, run the script, use the macro.

## What waits for the results

- **IF blocks.** Where the wording depends on which way a result falls, the section holds every version:

  ```latex
  % IF results-replay: that probability is at least 0.95, so the policy's radius is "larger" (protocol, section 9)
  \pending{IF policy radius larger}By our rule the policy's radius is the larger ...
  % ELIF the probability lies between 0.05 and 0.95: "not told apart"
  %| \pending{IF radii not told apart}By our rule these trials do not tell the two radii apart ...
  % ENDIF
  ```

  The first branch is live text; the others sit behind `%| `. To settle a block, keep the branch whose condition
  holds (remove its `%| `), delete the other branches, the four marks and the `\pending{...}` tag.
  `python3 paper/check_paper.py --branches` lists the blocks still open. The first branch is the protocol's
  prediction, not a finding: the bold tag in the PDF is there so that nobody reads it as one.
- **`% CONFIRM:` marks** sit on sentences that describe how the experiment is planned to run (one operator, the
  order of the blocks, the checkpoint). After the run, check each against `docs/journal.md` and `docs/decisions.md`,
  correct the sentence if the run differed, and delete the mark.
- **`\pending{NEEDS MACROS: ...}`** marks a number that has no macro yet (today: none; the spread policy's block
  fit and the simulation's figures by block, by width and for the comparison rule were given macros on 2026-10-05).
  The names are `make_numbers_stub.py`'s to give: add the lines there, run it, and put the macros where the tags are.
- **A rule that the protocol does not give.** The IF block "results-layout" reads the paired test at the level 0.05.
  `protocol/protocol.md` uses that level where it says what the test can show (section 8) and gives prediction 3 no rule
  of its own. The IF line says so. State the rule in the protocol before the freeze, or word the sentence without it.
  (The rule for "larger" in the comparison of two radii, which the skeleton had to propose, is now in the protocol,
  section 9, and the IF blocks follow it.)

## Checks

```bash
python3 paper/check_paper.py              # must pass before every upload
python3 paper/check_paper.py -v           # with the notes: unknown commands, caption lengths
python3 paper/check_paper.py --final --forbidden PRIVATE_LIST   # must pass before the submission
```

It fails on: a `\cite` key missing from the bibliography; a number macro that is not defined; a forbidden string;
a section more than 20 % over its word target; a field of `submission.md` over its character limit, or its
abstract out of step with `sections/abstract.tex`; and on what a
compiler would have caught or what would print wrongly (an open environment or brace, a `\ref` without its
`\label`, a missing `\input`, a character outside ASCII, a `<`, `>` or `|` in running text, a macro defined twice,
`generated/` out of step with its lists or with `config/protocol.toml`).

With `--final` it also fails on anything still open: a placeholder that the paper uses, an IF block, a CONFIRM or
TODO mark, a `\pending` tag, `\draftmarkstrue` in `main.tex`, a protocol that is not frozen.

A placeholder is open only where the paper would print it: its macro stands in `main.tex` or in a section as the text
is kept now, outside comments and outside the commented-out branches of IF blocks. A placeholder that nothing uses
(the numbers of a block of trials that was planned and not run can never get a value) is counted on a line of its
own, `unused`, and does not fail, with `--final` or without. `-v` lists both kinds by name.

**The forbidden strings.** `forbidden.txt` is committed and holds generic entries only. The author's own strings
(names, usernames, the institution, hub and repository names, the earlier project's dataset and policy names, the
exact task sentence) go in a private list: a file outside the repository passed with `--forbidden`, or
`paper/forbidden.local.txt`, which `paper/.gitignore` keeps out of git. Never upload that file. A hit from a private
list is reported by line number only.

It is not a compiler. It does not know every LaTeX command (`-v` lists the ones it does not recognise), and it cannot
see an overfull table or a figure that is too tall.

## Building

**On Overleaf.** New project, upload `main.tex`, `references.bib`, `numbers_by_hand.tex`, and the folders
`sections/`, `generated/` (the two `.tex` files) and `figures/` (the PDFs, once they exist). Compiler: pdfLaTeX.
IEEEtran and its bibliography style are part of the distribution. Do not upload a private forbidden list.

**Locally**, if a TeX engine is installed later, from this folder:

```bash
latexmk -pdf -outdir=build main.tex      # or: tectonic --outdir build main.tex
```

`build/` and LaTeX's by-products are ignored by git.

On the first build, look for: a `\newcommand` that clashes with a package's own name (the error names the macro:
rename it in `make_numbers_stub.py`); tables wider than a column (Table I and the appendix tables were sized by
arithmetic, not by eye); and the reference with a very long author list. For that last one, IEEEtran's own switch
shortens long lists: add a control entry to a second `.bib` file, as its documentation describes, rather than
editing `references.bib`.

## Page budget

The main text may fill four pages, and that is the limit that binds: the build writes the page the main text ends on
to its log, and `check_paper.py` prints today's word counts against the targets below (it fails a section more than
20 % over its target, to keep the sections in proportion).

| Section | Target until 2026-10-08 | Target now | |
|---|---|---|---|
| Abstract | 150 | 185 | |
| Introduction | 350 | 360 | with Figure 1 |
| Related work | 350 | 280 | |
| Method | 650 | 650 | with Figure 2 |
| Results | 550 | 740 | with Figure 3, Table I and the table of counts by distance (Figure 4 is in the appendix since the second blind pass) |
| Checklist and discussion | 250 | 280 | |
| Limitations | 120 | 150 | |
| **Total** | **2,420** | **2,645** | |

The targets were rebalanced on 2026-10-08 after five blind reads of the built paper: every reader wanted the counts
by distance, and what the policy's fit does and does not establish, in the results and not in the appendix; the
related work gave up one statement of novelty per paragraph for one at its end. Built with tectonic, the main text
then ended on page 4 (`docs/journal.md` has the entry).

If it runs over, cut in this order:

1. Move Figure 2 (the layouts) to the appendix: Figure 4 also shows both sets of demonstrations.
2. Move the "farthest success" and "nearest failure" columns of Table I to the appendix.
3. Shorten the captions of Figure 3, Figure 4 and Table I.
4. Drop the first paragraph of related work's "Layout" down to its last two sentences.

`main.tex` writes the page on which the main text ends to the build log (`MAIN TEXT ENDS ON PAGE`). Check it by eye
as well: the Limits section must end on page 4 or before.

## Before submitting

1. The protocol is frozen and `make_numbers_stub.py` has been run since.
2. The analysis has written every result (`--values`); `numbers_by_hand.tex` has no `??` left.
3. Every IF block is settled and every CONFIRM mark is checked and deleted.
4. `\draftmarksfalse` in `main.tex`.
5. `python3 paper/check_paper.py --final --forbidden PRIVATE_LIST` passes.
6. The PDF is four pages of main text or fewer, and has been read once from the first line to the last.
7. The anonymity checklist in `submission.md`, on the files that are uploaded.

## Known gaps

- **Settled on 2026-10-08, after the counted report** (`analysis/report/report.md`; `docs/journal.md` has the entry):
  every IF block, CONFIRM mark and `\pending` tag of the sections; the macros of the plateau check and of the second
  scoring, which the report now writes; and the scope of `\maxSuccessDist...`, `\minFailureDist...`, `\successZero...`
  and `\nZero...`: the threshold block alone (`generated/MACROS.md`), so Table I leaves those cells of a pooled row empty.
- **The spread policy's threshold block (T3) was not run**, so the generated macros of its block fit and of the
  comparison with the clustered policy's radius print `??` for good. Nothing in the paper uses them, and
  `check_paper.py` counts as open only a placeholder that the paper uses.
- **Typed with a `% src:`, for want of a committed record the report could read**: how many frames the torn-frame scan
  named in this study's two sets and how many of them are torn (appendix, "Faults"; the journal and the decision log
  of 2026-10-07 hold them), the episodes set aside before the kept recording, and what happened at each stop line.
- **The paired test's quoted case is no longer the one at the edge of what it can show.** The appendix prints the
  test's power at 40 against 90 % (`\simPairedRateLow`, `\simPairedRateHigh`, fixed in
  `report/bookkeeping.py`). With the 20 pairs of draft 0.2 that was the smallest difference from 40 % told apart four
  times in five; with 26 pairs it is 40 against 85 %, and 40 against 90 % is told apart in 96 % of cases
  (`analysis/simulation/procedure.md`, Tables 12 and 13). Either keep the case and its sentence, or move the case.
- **Four things the instrument's first evening taught, none of them in the text yet** (2026-10-05; `docs/journal.md`
  and `analysis/instrument/first_sitting.md` hold the figures). Each is the kind of thing the workshop asks about,
  and each needs words that the page budget does not have: choose on Thursday which, if any, buys its place in the
  checklist or the Limits, and against what.
  1. *How the fiducials' corners were found, at how many pixels per millimetre.* At 0.8 to 1.0 px per mm the tag
     detector reported corners 0.6 px inside the tags' outlines, and one corner of a small tag moved by 1.7 px when
     the picture changed by one grey level.
  2. *A check that could not fail.* The registration printed, as the instrument's accuracy, a figure that was the
     fit's own residual again by construction; an independent review found it, and found that the reference sheet
     was not flat by measuring printed lines that no fit used (0.35 mm rms, 1.2 mm at worst).
  3. *Camera controls that nobody set.* The vendor software's log showed exposure and white balance on automatic,
     and a digital zoom, pan and tilt held in the camera, not in any file of the project. The tag baseline cannot see
     an exposure change; the policy sees every pixel of it.
  4. *What a two-tag frame is worth.* One pixel on one corner of a floor tag moves the far side of the grid by up to
     2.5 mm, so the frame is fitted once through six tags and stored, and the two tags only watch over it.
- **`\rigHandoffTolerance` in `numbers_by_hand.tex` is no longer used**: the appendix takes both handoff tolerances
  from the protocol's constants (`\handoffTolerance`, `\handoffToleranceReplay`).
- **Typed from `config/protocol.toml` with a `% src:`, for want of a macro**: the step of the radius grid (0.5 mm)
  and the number of width values (30), both in the appendix. They do not follow the file when it changes.
- **`check_paper.py -v` reports `gamma` as an unknown command.** It is the floor in Eq. (1); the script's list of
  known commands lacks it. A note, not a failure.
- **Figure 3 and Figure 4 are captioned for block fits** (the trials of each threshold block, the curves of the two
  block fits, discs of the clustered policy's block-fit radius), and the figure code draws exactly that
  (`figures/README.md` agrees since 2026-10-05).
- **Two citations the text would like and the bibliography lacks**: the fiducial tags (AprilTag), and McNemar's
  test. Each needs an entry verified against its primary page and a row in `citations-verified.md` first. The text
  reads correctly without them.
- **One citation cut for room**: `kim2026robogate` (related work, "Thresholds"). The file's header says where it
  belongs if a line comes free.
- **The staircase rule's property** (a one-up-one-down rule gathers trials where success is one half) is stated
  without a citation, because neither staircase source was read (`citations-verified.md`, "Not verified", item 1).
- **What the run has to confirm**: every CONFIRM mark. Three wait for the author alone (the release of code and
  data; whether the earlier evaluation used the same physical arm; the earlier project's audit counts); the others
  wait for the recording, the freeze and the trials.
