SYNTHETIC. Nothing in this folder is a result: no trial was run to make it.

# An example of the analysis's output

`results.json` and `report.md` here are what `scripts/make_report.py` writes, made from a synthetic experiment:
audits built from the committed layouts, trials planned by the real planner and answered from invented success
curves (`tests/fixtures_experiment.py`). They show the shape of the two files with every block run, for whoever
draws a figure or writes a sentence before the counted trials exist. `protocol/data-contracts.md` explains the fields.

The workspace test of this experiment is the stratified one of the protocol (draft 0.3): 26 placements, 13 drawn
over the trained cells and 13 over the held-out cells, each tested with both policies.

The counted report will be in `analysis/report/`. Do not copy a number from here anywhere.

The numbers are those of the day the example was written: made from synthetic inputs with SHA-256
`5d0f27c8db4d85d156fbb2deebae79fe8f89db4c94b6473235bf4522f5269707`.
A test checks that the example has the shape the contract fixes today. When today's synthetic inputs are still the
ones named above, it also checks that the analysis gives these very bytes again; when the planner or the protocol
file has changed them, it does not. To write the example again:

    PYTHONPATH=src python tests/fixtures_experiment.py
