"""Search over misread-digit corrections: UCS, Greedy best-first and A*.

State  = frozenset of corrections {(cell, new_digit)} applied to the first NN reading.
Action = change one not-yet-corrected uncertain cell to one of its top alternatives.
Cost   = log(p_read / p_new) >= 0   (how much less likely the new digit is; read = argmax).
Goal   = the corrected puzzle is solvable by the CSP (goal='unique': exactly one solution).
Limit  = at most `max_corr` corrections.

A* heuristic (admissible): every duplicate-digit conflict needs at least one of its two cells
to change, so h = max over conflicts of the cheapest change available in that conflict.

Counters: expanded, generated, csp_calls (goal tests), csp_assignments (CSP effort).
"""
import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from csp.sudoku import find_conflicts, solve, Stats


@dataclass
class SearchResult:
    status: str                       # found | unreadable
    corrections: list = field(default_factory=list)   # [(cell, read, new, cost)]
    cost: float = 0.0
    solution: list = None
    expanded: int = 0
    generated: int = 0
    csp_calls: int = 0
    csp: Stats = field(default_factory=Stats)
    trace: list = field(default_factory=list)
    message: str = ""
    least_confident: list = field(default_factory=list)


def candidates(grid, probs, n_cells=15, n_alts=2):
    """The n_cells least-confident given cells (smallest margin top1-top2) and their alternatives.
    Returns {cell: [(digit, cost), ...]} cheapest first."""
    cells = [c for c in range(81) if grid[c]]
    margin = {}
    for c in cells:
        s = np.sort(probs[c])
        margin[c] = s[-1] - s[-2]
    cells.sort(key=lambda c: margin[c])
    out = {}
    for c in cells[:n_cells]:
        p = np.maximum(probs[c], 1e-9)
        read = grid[c]
        alts = [(int(d) + 1, float(math.log(p[read - 1] / p[d]))) for d in np.argsort(-p)
                if int(d) + 1 != read][:n_alts]
        out[c] = alts
    return out


def apply(grid, corr):
    g = list(grid)
    for c, d in corr:
        g[c] = d
    return g


def search(grid, probs, algo="ucs", max_corr=3, n_cells=15, n_alts=2, goal="solvable",
           trace=False, max_expanded=20000, csp_cfg=None):
    """algo: 'ucs' | 'greedy' | 'astar'."""
    csp_cfg = csp_cfg or dict(fc=True, ac3_on=True, mrv=True)
    cand = candidates(grid, probs, n_cells, n_alts)
    res = SearchResult("unreadable")
    least = sorted((c for c in range(81) if grid[c]), key=lambda c: np.sort(probs[c])[-1] - np.sort(probs[c])[-2])
    res.least_confident = [(c, grid[c], float(probs[c].max())) for c in least[:5]]

    def h(state):
        g = apply(grid, state)
        done = {c for c, _ in state}
        best = 0.0
        for a, b in find_conflicts(g):
            m = math.inf
            for c in (a, b):
                if c in cand and c not in done and cand[c]:
                    m = min(m, cand[c][0][1])
            if m == math.inf:
                return math.inf
            best = max(best, m)
        return best

    def goal_test(state):
        g = apply(grid, state)
        if find_conflicts(g):
            return None
        res.csp_calls += 1
        r = solve(g, limit=2 if goal == "unique" else 1, **csp_cfg)
        res.csp.add(r.stats)
        if r.solved and (goal == "solvable" or len(r.solutions) == 1):
            return r.solutions[0]
        return None

    def f(g, hh):
        return {"ucs": g, "greedy": hh, "astar": g + hh}[algo]

    tie = itertools.count()
    start = frozenset()
    h0 = h(start) if algo != "ucs" else 0.0
    openq = [(f(0.0, h0), next(tie), 0.0, start)]
    best_g = {start: 0.0}
    res.generated = 1
    while openq:
        fv, _, g, state = heapq.heappop(openq)
        if g > best_g.get(state, math.inf):
            continue
        res.expanded += 1
        if trace:
            res.trace.append((sorted(state), round(g, 4), round(fv, 4)))
        sol = goal_test(state)
        if sol is not None:
            res.status = "found"
            res.solution = sol
            res.cost = g
            corr = sorted(state)
            res.corrections = [(c, grid[c], d, next(k for dd, k in cand[c] if dd == d)) for c, d in corr]
            res.message = "solved after %d correction(s)" % len(corr)
            return res
        if res.expanded >= max_expanded:
            break
        if len(state) >= max_corr:
            continue
        done = {c for c, _ in state}
        for c, alts in cand.items():
            if c in done:
                continue
            for d, cost in alts:
                ns = state | {(c, d)}
                ng = g + cost
                if ng < best_g.get(ns, math.inf):
                    nh = h(ns) if algo != "ucs" else 0.0
                    if nh == math.inf:
                        continue
                    best_g[ns] = ng
                    res.generated += 1
                    heapq.heappush(openq, (f(ng, nh), next(tie), ng, ns))
    res.message = ("unreadable: no puzzle reachable with <= %d corrections; least confident cells: %s"
                   % (max_corr, [(c, d, round(p, 2)) for c, d, p in res.least_confident]))
    return res
