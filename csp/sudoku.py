"""Sudoku as a CSP: 81 variables, domain 1-9, 'all different' on rows/cols/boxes.

Three separately switchable inference parts (as required by the project):
  * Backtracking (BT)            - always on (the search skeleton)
  * Forward Checking (fc=True)   - removes the assigned digit from peers' domains
  * Arc Consistency (ac3=True)   - AC-3 run once before search; detects conflicts
                                   (two equal givens in a unit empty a domain)
Domains are stored as bitmasks: bit d set  <=>  digit d is still possible.
Counters (assignments, backtracks, ac3_revisions, pruned) are returned in Stats.
"""
from collections import deque
from dataclasses import dataclass, field

ALL = sum(1 << d for d in range(1, 10))  # bits 1..9


def _build():
    units = []
    for r in range(9):
        units.append([r * 9 + c for c in range(9)])
    for c in range(9):
        units.append([r * 9 + c for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)])
    peers = [set() for _ in range(81)]
    for u in units:
        for a in u:
            peers[a].update(x for x in u if x != a)
    return units, [sorted(p) for p in peers]


UNITS, PEERS = _build()


class CapReached(Exception):
    pass


@dataclass
class Stats:
    assignments: int = 0
    backtracks: int = 0
    ac3_revisions: int = 0
    pruned: int = 0  # domain values removed by FC / AC-3

    def add(self, o):
        self.assignments += o.assignments
        self.backtracks += o.backtracks
        self.ac3_revisions += o.ac3_revisions
        self.pruned += o.pruned


@dataclass
class Result:
    status: str                      # solved | no_solution | cap
    solutions: list = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)
    message: str = ""

    @property
    def solved(self):
        return self.status == "solved"


def parse(s):
    """81-char string ('.' or '0' = blank) -> list of 81 ints."""
    s = [c for c in s if not c.isspace()]
    assert len(s) == 81, "need 81 cells"
    return [0 if c in ".0" else int(c) for c in s]


def to_str(grid):
    return "".join(str(d) if d else "." for d in grid)


def find_conflicts(grid):
    """All pairs (a, b), a<b, of non-blank equal cells that share a unit."""
    out = []
    for a in range(81):
        if grid[a]:
            for b in PEERS[a]:
                if b > a and grid[b] == grid[a]:
                    out.append((a, b))
    return out


def ac3(dom, stats):
    """AC-3 for the 'not equal' constraints. Returns False if a domain empties."""
    q = deque((xi, xj) for xi in range(81) for xj in PEERS[xi])
    while q:
        xi, xj = q.popleft()
        stats.ac3_revisions += 1
        dj = dom[xj]
        if dj & (dj - 1) == 0 and dom[xi] & dj:  # xj fixed to one value -> remove from xi
            dom[xi] &= ~dj
            stats.pruned += 1
            if dom[xi] == 0:
                return False
            for xk in PEERS[xi]:
                if xk != xj:
                    q.append((xk, xi))
    return True


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def solve(grid, fc=True, ac3_on=True, mrv=True, limit=1, max_assignments=None):
    """Solve a Sudoku. limit = max number of solutions to look for (2 -> uniqueness test)."""
    stats = Stats()
    if find_conflicts(grid):  # cheap direct check; AC-3 would also find it
        if not ac3_on:
            return Result("no_solution", [], stats, "no solution: duplicate digits in a unit")
    dom = [(1 << g) if g else ALL for g in grid]
    assign = list(grid)
    if ac3_on:
        if not ac3(dom, stats):
            return Result("no_solution", [], stats, "no solution: AC-3 emptied a domain")
    elif fc:  # forward-check the givens once
        for c in range(81):
            if grid[c]:
                for p in PEERS[c]:
                    if not grid[p] and dom[p] >> grid[c] & 1:
                        dom[p] &= ~(1 << grid[c])
                        stats.pruned += 1
    sols = []

    def legal(c, dom):
        m = dom[c]
        if not fc:  # no forward checking: test against assigned peers on demand
            for p in PEERS[c]:
                if assign[p]:
                    m &= ~(1 << assign[p])
        return m

    def rec(dom):
        best, bm, bk = -1, 0, 10
        for c in range(81):
            if not assign[c]:
                m = legal(c, dom)
                k = m.bit_count()
                if k == 0:
                    stats.backtracks += 1
                    return
                if best < 0 or (mrv and k < bk):
                    best, bm, bk = c, m, k
                    if not mrv:
                        break
        if best < 0:
            sols.append(list(assign))
            return
        for d in _bits(bm):
            stats.assignments += 1
            if max_assignments is not None and stats.assignments > max_assignments:
                raise CapReached
            assign[best] = d
            if fc:
                nd = list(dom)
                nd[best] = 1 << d
                ok = True
                for p in PEERS[best]:
                    if not assign[p] and nd[p] >> d & 1:
                        nd[p] &= ~(1 << d)
                        stats.pruned += 1
                        if nd[p] == 0:
                            ok = False
                            break
                if ok:
                    rec(nd)
            else:
                rec(dom)
            assign[best] = 0
            if len(sols) >= limit:
                return
        stats.backtracks += 1

    # givens that are not 'assigned' inconsistently were handled above; for plain BT a
    # duplicate-given grid must still fail:
    if not fc and find_conflicts(grid):
        return Result("no_solution", [], stats, "no solution: duplicate digits in a unit")
    try:
        rec(dom)
    except CapReached:
        return Result("cap", sols, stats, "assignment cap reached")
    if sols:
        return Result("solved", sols, stats)
    return Result("no_solution", [], stats, "no solution: search exhausted")


def is_valid_solution(grid):
    return all(sorted(grid[i] for i in u) == list(range(1, 10)) for u in UNITS)
