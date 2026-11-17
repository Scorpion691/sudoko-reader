"""NN -> AC-3 conflict check -> correction search -> CSP.  Optionally NN-only (no correction)."""
from dataclasses import dataclass

from csp.sudoku import ac3, find_conflicts, solve, ALL, Stats
from data.generator import read_grid
from search.correction import search


@dataclass
class Outcome:
    status: str          # solved | unreadable | no_solution
    read: list
    final: list = None   # the (corrected) puzzle that was solved
    solution: list = None
    corrections: list = None
    conflicts: list = None
    search: object = None
    message: str = ""


def solve_image(net, imgs, algo="ucs", use_correction=True, max_corr=3, goal="solvable",
                n_cells=15, n_alts=2, trace=False):
    grid, probs = read_grid(net, imgs)
    return solve_reading(grid, probs, algo, use_correction, max_corr, goal, n_cells, n_alts, trace)


def solve_reading(grid, probs, algo="ucs", use_correction=True, max_corr=3, goal="solvable",
                  n_cells=15, n_alts=2, trace=False):
    out = Outcome("unreadable", grid)
    dom = [(1 << g) if g else ALL for g in grid]
    out.conflicts = find_conflicts(grid)
    first = solve(grid, limit=2 if goal == "unique" else 1)
    ok_first = first.solved and (goal == "solvable" or len(first.solutions) == 1)
    if ok_first:
        out.status, out.final, out.solution, out.corrections = "solved", grid, first.solutions[0], []
        out.message = "solved without corrections"
        return out
    if not use_correction:
        out.status = "no_solution"
        out.message = "no solution: first reading is unsolvable (correction is switched off)"
        return out
    r = search(grid, probs, algo, max_corr, n_cells, n_alts, goal, trace)
    out.search = r
    out.corrections = r.corrections
    if r.status == "found":
        out.status, out.solution = "solved", r.solution
        out.final = list(grid)
        for c, _, d, _ in r.corrections:
            out.final[c] = d
    else:
        out.message = r.message
    out.message = out.message or r.message
    return out
