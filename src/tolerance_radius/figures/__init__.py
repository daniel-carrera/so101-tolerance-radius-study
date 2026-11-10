"""The paper's four figures, each drawn by a function from data files (`paper/figures/README.md` is the plan).

* `earlier_evaluation`: Figure 1, the earlier evaluation by distance, from the CSV of the August re-analysis;
* `layouts_on_grid`: Figure 2, the two demonstration layouts on the grid, from the start-state audits;
* `response_curves`: Figure 3, success against distance for the clustered policy and the replay, from the analysis
  results;
* `workspace`: Figure 4, the workspace test and the lookup prediction, from the analysis results.

`style` holds what they share (the page, the type, the colours, how a file is saved), `marks` the geometry of where
marks go, `gridmap` the drawing of the grid, `results` the reading of `analysis/report/results.json`, and `make` what
`scripts/make_figures.py` does with them.

Matplotlib is used here and nowhere else in the package, through its object interface: no window is opened and no
global setting is left changed.
"""

#: The file each figure is written to (without its folder), by the figure's number in the paper.
FILE_NAMES = {
    1: "fig1-earlier-evaluation.pdf",
    2: "fig2-layouts.pdf",
    3: "fig3-response-curves.pdf",
    4: "fig4-workspace.pdf",
}

#: The height of each figure in millimetres, as `paper/figures/README.md` fixes it. Every figure is 88 mm wide.
HEIGHTS_MM = {1: 42.0, 2: 32.0, 3: 46.0, 4: 40.0}
