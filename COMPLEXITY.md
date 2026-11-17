# Time complexity of the whole project

Notation: **n = 81** cells, **d = 9** digits, **p = 20** peers of a cell, **e = n·p = 1,620** directed arcs,
**P = 109,120** weights of the network (784·128 + 128·64 + 64·9), **C = 15** candidate cells, **a = 2** alternatives per cell,
**k** = maximum number of corrections (3 by default, 4 in the experiments), **b = C·a = 30** actions per state,
**S** = number of search states, **T<sub>csp</sub>** = cost of one CSP call.

## 1. Neural network (`nn/mlp.py`)

| Operation | Time | Numbers |
|---|---|---|
| Inference for one cell | O(P) | 109,120 multiply-adds |
| Inference for one puzzle | O(n·P) | about 8.8 million multiply-adds |
| Training | O(E·N·P), with a factor of about 3 for the backward pass | 10 epochs · 45,053 images · 3 · 109,120 ≈ 1.5·10<sup>11</sup>, measured about 16 s on a CPU |
| Temperature scaling (grid search) | O(G·V·9) | 71 temperatures · 9,024 validation images ≈ 5.8·10<sup>6</sup> |
| Expected calibration error | O(N) | one pass over the test set |

## 2. Data generation (`data/generator.py`)

| Operation | Time |
|---|---|
| Full grid by randomised backtracking | exponential in the worst case; in practice fast (a complete 30-given puzzle with its uniqueness checks took about 0.13 s) |
| Puzzle with a unique solution | O(n·T<sub>csp</sub>): at most 81 uniqueness checks, each one a CSP call with `limit = 2` |
| Mosaic and blank-cell detection | O(n) |
| Hard-digit pools | O(M·9) over the M test images, once |

## 3. CSP (`csp/sudoku.py`)

| Part | Time | Notes |
|---|---|---|
| Conflict check | O(n·p) = O(1,620) | duplicate digits in a unit |
| AC-3 | O(n·d·p) = O(e·d) | each of at most n·d = 729 value removals re-queues at most p − 1 = 19 arcs, so at most 729·19 + 1,620 ≈ 15.5 thousand revisions. Every revision is O(1) because the constraint is "not equal" and domains are bit-masks (general AC-3 is O(e·d<sup>3</sup>)). Measured: 9,372 revisions on the classic puzzle. |
| Backtracking | O(d<sup>f</sup>) in the worst case, f = number of empty cells | exponential. Per node: O(p) to test a digit, O(n·p) when MRV recomputes legal digits without forward checking. Measured: 4,208 assignments on the classic puzzle, 4,332 on average at 30 givens. |
| Forward checking | O(n) per node | O(p) domain updates plus O(n) to copy the domains. Measured: 516 assignments (classic puzzle), 1,340 at 30 givens. |
| BT + FC + AC-3 | O(e·d) once + O(n) per node | measured 51 assignments (classic puzzle), 623 at 30 givens, about 5 ms per puzzle |

The worst case of the search stays exponential. The inference parts (FC, AC-3) do not change the worst case,
they shrink the number of nodes: at 30 givens by 86% (Experiment 2).

## 4. Search over corrections (`search/correction.py`)

* **Candidate selection:** O(n log n) to sort the cells by probability margin.
* **Number of states:** S = Σ<sub>j≤k</sub> C(C, j)·a<sup>j</sup> = 1 + 30 + 420 + 3,640 (+ 21,840 for k = 4) = **4,091** states for k = 3 and **25,931** for k = 4.
  States are sets, so order does not matter; this is O((C·a)<sup>k</sup> / k!), a constant here because C, a and k are fixed.
* **Per expansion:** O(b·(n·p + log S)) to generate the children (heuristic and heap insertion), plus the goal test O(n·p + T<sub>csp</sub>).
* **Total, UCS and A\*:** O(S·b·n·p + S·T<sub>csp</sub>) in the worst case. A cap of 4,000 expansions protects the experiments.
* **Greedy:** same bound, no optimality guarantee, but it usually expands fewer nodes.
* **Measured at 3 misreads:** 91.9 expansions for UCS, 12.7 for A\*, 9.2 for Greedy; 3.5 CSP calls for UCS and A\*, 7.2 for Greedy.
* **Memory:** O(S) for the open list and the best-cost table.

## 5. Whole pipeline for one puzzle (`integration/pipeline.py`)

```
T = O(n·P)                          read the grid with the network
  + O(n·p + T_csp)                  check the first reading
  + O(expanded · (b·n·p + T_csp))   search with a CSP call per goal test
```

The network cost is a fixed 8.8 million operations; the real work is `expanded × T_csp`. In practice: a few CSP calls
and a few milliseconds each, so one puzzle takes a few tens of milliseconds.

## 6. Experiments and tools

| Item | Time | Measured |
|---|---|---|
| Experiment 1 (5 seeds × 20 puzzles × 5 levels × 3 algorithms) | O(seeds · puzzles · levels · algorithms · T<sub>pipeline</sub>) | part of about 2 minutes for the three experiments |
| Experiment 2 | O(seeds · puzzles · levels · 3 · T<sub>csp</sub>), capped at 200,000 assignments per solve | part of the same 2 minutes |
| Experiment 3 | as Experiment 1, plus up to 50 random trials per puzzle | part of the same 2 minutes |
| Dashboard, 6 corrected cases | O(cases · T<sub>pipeline</sub>) | seconds (not benchmarked separately) |
| Tests (19) | | about 1 second |

All measurements are on one ordinary CPU, without a GPU.
