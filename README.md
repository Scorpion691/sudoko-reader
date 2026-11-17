# P7 – Handwritten Sudoku Reader and Solver (Search + CSP + Neural Network)

A numpy MLP reads handwritten digits (MNIST) in a 9x9 mosaic. If a misread digit breaks the puzzle,
**search** (UCS / Greedy / A*) looks for the cheapest set of corrections (cost = how much less likely the
new digit is) and a **CSP** (Backtracking, Forward Checking, AC-3 – separately switchable) tests/solves each
corrected puzzle. If nothing works the system answers "unreadable" and lists the least confident cells.

## Setup
```
python -m venv .venv && .venv\Scripts\activate        # optional
pip install -r requirements.txt                        # numpy, matplotlib, pytest (no GPU / torch needed)
```
Run everything from this folder (`sudoku_reader/`).

## Run commands
```
python data/make_data.py --download --seed 0   # download MNIST + split 50k/10k (-> data/mnist19.npz)
python -m nn.train --epochs 10                 # MLP + logistic-regression baseline + temperature scaling (~20 s CPU)
python run_demo.py --seed 3 --misreads 2       # integrated demo (trace, corrections, solution, plots) + opens outputs/dashboard.html
python run_demo.py --no-browser                # same, without opening the browser
python -m pytest tests -q                      # 19 tests: normal / edge / impossible
python -m eval.experiments --seeds 5 --n 20    # the 3 experiments (~2 min) -> outputs/
```

## Layout
| Folder | Content |
|---|---|
| `data/` | `make_data.py` (reproducible preprocessing), `generator.py` (seeded puzzles, mosaics, hard-digit injection) |
| `nn/` | `mlp.py` (numpy MLP, Adam, temperature scaling, ECE), `train.py` |
| `csp/` | `sudoku.py`: BT, FC, AC-3 with counters (assignments, backtracks, AC-3 revisions, pruned values) |
| `search/` | `correction.py`: UCS, Greedy, A* (admissible heuristic) with counters (expanded, generated, CSP calls) |
| `integration/` | `pipeline.py`: NN -> conflict check -> search -> CSP |
| `eval/` | `experiments.py` |
| `tests/` | `test_all.py` |

Data dictionary: [DATA_DICTIONARY.md](DATA_DICTIONARY.md). Design: state = set of corrections (max 4 in the
experiments, 3 by default in the pipeline spec), cost = `log(p_read/p_new) >= 0`, goal = puzzle solvable
(`goal="unique"` = exactly one solution), candidate cells = 15 least confident, 2 alternatives each.
A* heuristic: each duplicate pair needs one of its two cells changed -> h = max over pairs of the cheapest change.

## Results (mean ± std over 5 seeds; 20 puzzles per seed per level; full tables in `outputs/`)
**Classifier** (test set): MLP 97.8 % vs logistic regression 92.3 %; ECE 0.0078 -> 0.0037 after temperature scaling (T = 1.35).

**Exp 1 – UCS vs Greedy vs A\*** (`outputs/exp1_search.md`): UCS is optimal by construction and A* (admissible h) gives identical recovery with fewer nodes; UCS expands the most nodes
(e.g. 3 misreads: 92 ± 37), A* ~13 ± 3 and Greedy ~9 ± 3 but Greedy makes about twice as many CSP calls (7.2 vs 3.5) and
recovers slightly less (0.64 vs 0.71).
Exact recovery: 0.93 / 0.73 / 0.71 / 0.42 for 1 / 2 / 3 / 4 misreads.

**Exp 2 – BT vs BT+FC vs BT+FC+AC-3** (`outputs/exp2_csp.md`, static variable order): at 30 givens assignments drop from
4332 ± 2519 (BT) to 1340 ± 837 (FC) to 623 ± 319 (FC+AC-3); at 25 givens plain BT hits the 200k cap on 8 % of puzzles.

**Exp 3 – NN-guided vs baselines** (`outputs/exp3_nn_vs_baselines.md`): exact recovery with 1 / 2 / 3 misreads –
NN-guided 0.96 / 0.77 / 0.55; no correction 0 / 0 / 0; random correction 0.23 / 0 / 0.

## Honest limitations
* Recovery is below 100 % mostly because the search stops at the *cheapest solvable* puzzle, which is sometimes a
  different valid puzzle. In a diagnostic run (60 puzzles, 2 misreads) all 17 failures (28 % of the puzzles) were of this kind and the
  `goal="unique"` extension raised recovery from 72 % to 88 %.
* Misreads are injected by a documented rule (hard test digits), and the first reading is forced to be unsolvable;
  results do not describe photographed real Sudoku.
* The 4-misread level needs 4 corrections, so the correction limit was raised from 3 to 4 in the experiments.
* Search is capped at 4000 expanded nodes; capped runs count as failures.
* Exp 2 uses 10 puzzles per seed per level (Exp 1/3: 20). Part of the report/presentation/contribution statement/git log
  deliverables (items 11-13) are not written yet.

## Desktop app (one button)
* `python app.py` or double-click `SudokuReader.bat` opens the app: pick a seed and the number of misreads, press **Run**
  (optionally also re-run the experiments) and the dashboard opens in the browser.
* `build_exe.bat` builds `dist\SudokuReader\SudokuReader.exe` (about 116 MB, no Python needed; copy the whole
  `dist\SudokuReader` folder). `SudokuReader.exe --selftest` runs the pipeline without the window and writes
  `_internal\outputs\selftest.txt`.

## Report, slides and contribution statement
* `docs/Report.pdf` (12 pages) and `docs/Slides.pdf` (15 slides, 10 min + 5 min live demo) are built from the real result tables:
  `python docs/build_docs.py` (needs Microsoft Edge or Chrome for the PDF). `docs/Slides.html` is the interactive deck
  (arrow keys, `N` = speaker notes, `F` = fullscreen); `docs/check_overflow.py` checks that no report page overflows.
* `CONTRIBUTION.md` is the individual contribution statement; `COMPLEXITY.md` derives the time complexity of every part
  (network, CSP, search, pipeline) with measured numbers.
