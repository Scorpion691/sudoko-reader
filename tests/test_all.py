"""Tests: 5+ normal, 3+ edge, 2+ impossible cases (impossible ones give a clear 'no solution')."""
import itertools

import numpy as np
import pytest

from csp.sudoku import parse, solve, ac3, find_conflicts, is_valid_solution, to_str, Stats, ALL, PEERS
from data import generator as G
from nn.mlp import MLP, ece
from search.correction import search
from integration.pipeline import solve_reading

PUZ = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
SOL = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def probs_for(grid, wrong):
    """Fabricated NN output: confident everywhere, except `wrong` = {cell: (read, true)}."""
    P = np.full((81, 9), 0.001, np.float32)
    for c in range(81):
        if grid[c]:
            P[c, grid[c] - 1] = 0.99
    for c, (read, true) in wrong.items():
        P[c] = 0.001
        P[c, read - 1] = 0.52
        P[c, true - 1] = 0.45
    return P


# ---------------------------------------------------------------- normal (>=5)
@pytest.mark.parametrize("fc,ac", [(False, False), (False, True), (True, False), (True, True)])
def test_normal_all_csp_modes_give_known_answer(fc, ac):
    r = solve(parse(PUZ), fc=fc, ac3_on=ac, mrv=False)
    assert r.solved and to_str(r.solutions[0]) == SOL


def test_normal_inference_reduces_search_effort():
    p = parse(PUZ)
    bt = solve(p, fc=False, ac3_on=False, mrv=False).stats
    fc = solve(p, fc=True, ac3_on=False, mrv=False).stats
    assert fc.assignments < bt.assignments and fc.backtracks < bt.backtracks


def test_normal_ac3_prunes_domains_and_counts():
    dom = [(1 << g) if g else ALL for g in parse(PUZ)]
    s = Stats()
    assert ac3(dom, s) and s.pruned > 0 and s.ac3_revisions > 0
    assert any(d.bit_count() == 1 for d in dom)


def test_normal_generator_unique_and_seeded():
    a, sa = G.make_puzzle(np.random.default_rng(5), 32)
    b, sb = G.make_puzzle(np.random.default_rng(5), 32)
    assert a == b and sa == sb
    assert is_valid_solution(sa) and len(solve(a, limit=2).solutions) == 1


def test_normal_search_known_answer_two_misreads():
    puz, sol = parse(PUZ), parse(SOL)
    read = list(puz)
    givens = [c for c in range(81) if puz[c]]
    c1, c2 = givens[0], givens[20]  # corrupt two givens in a way that breaks the puzzle
    read[c1], read[c2] = puz[c1] % 9 + 1, puz[c2] % 9 + 1
    wrong = {c1: (read[c1], puz[c1]), c2: (read[c2], puz[c2])}
    P = probs_for(read, wrong)
    for algo in ("ucs", "greedy", "astar"):
        r = search(read, P, algo)
        assert r.status == "found"
        assert {c for c, *_ in r.corrections} == {c1, c2}
        assert r.solution == sol


def test_normal_ucs_astar_match_bruteforce_cost():
    """Search checked against brute force over all correction sets (<=2) on a small pool."""
    puz = parse(PUZ)
    read = list(puz)
    givens = [c for c in range(81) if puz[c]]
    cells = [givens[3], givens[20]]
    for c in cells:
        read[c] = puz[c] % 9 + 1
    P = probs_for(read, {c: (read[c], puz[c]) for c in cells})
    from search.correction import candidates
    cand = candidates(read, P, 15, 2)
    best = np.inf
    for k in range(0, 3):
        for cs in itertools.combinations(cand, k):
            for ds in itertools.product(*[cand[c] for c in cs]):
                g = list(read)
                cost = 0
                for c, (d, w) in zip(cs, ds):
                    g[c] = d
                    cost += w
                if solve(g).solved:
                    best = min(best, cost)
    for algo in ("ucs", "astar"):
        assert search(read, P, algo).cost == pytest.approx(best)


# ---------------------------------------------------------------- edge (>=3)
def test_edge_complete_grid_is_its_own_solution():
    r = solve(parse(SOL))
    assert r.solved and to_str(r.solutions[0]) == SOL and r.stats.assignments == 0


def test_edge_empty_grid_solves_and_is_valid():
    r = solve([0] * 81)
    assert r.solved and is_valid_solution(r.solutions[0])


def test_edge_underdetermined_puzzle_has_many_solutions():
    p = parse(PUZ)
    for c in range(81):
        if c % 2 == 0:
            p[c] = 0
    assert len(solve(p, limit=2).solutions) == 2


def test_edge_no_correction_needed_means_zero_search():
    p = parse(PUZ)
    o = solve_reading(p, probs_for(p, {}))
    assert o.status == "solved" and o.corrections == [] and o.search is None


def test_edge_correction_limit_is_respected():
    puz = parse(PUZ)
    read = list(puz)
    cells = [c for c in range(81) if puz[c]][:3]
    for c in cells:
        read[c] = puz[c] % 9 + 1
    P = probs_for(read, {c: (read[c], puz[c]) for c in cells})
    r = search(read, P, "ucs", max_corr=1)
    assert r.status == "unreadable"  # needs 3 corrections, only 1 allowed


# ---------------------------------------------------------------- impossible (>=2)
def test_impossible_duplicate_givens_all_modes():
    p = parse(PUZ)
    p[2] = 5  # row 0 already has a 5 at cell 0
    for fc, ac in itertools.product([False, True], repeat=2):
        r = solve(p, fc=fc, ac3_on=ac)
        assert r.status == "no_solution" and "no solution" in r.message


def test_impossible_empty_domain_without_duplicates():
    p = [0] * 81
    p[:8] = [1, 2, 3, 4, 5, 6, 7, 8]  # row 0 needs a 9 in cell 8 ...
    p[17] = 9                          # ... but column 8 already has a 9 (cell 17)
    assert not find_conflicts(p)
    r = solve(p)
    assert r.status == "no_solution" and "no solution" in r.message


def test_impossible_search_reports_unreadable_with_least_confident_cells():
    p = parse(PUZ)
    p[2] = 5
    P = probs_for(p, {})  # network is confident everywhere: no cheap alternative reaches a solution
    o = solve_reading(p, P, max_corr=1)
    assert o.status == "unreadable" and "unreadable" in o.message and "least confident" in o.message


# ---------------------------------------------------------------- misc
def test_blank_detection_rule():
    rng = np.random.default_rng(0)
    assert G.is_blank(G.noise_image(rng))
    assert not G.is_blank(np.ones((28, 28), np.float32))


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(9)[y]
    assert ece(perfect, y) == pytest.approx(0.0, abs=1e-9)
    wrong = np.eye(9)[[1, 1, 2, 0]]  # confident, 1 of 4 wrong
    assert ece(wrong, y) == pytest.approx(0.25)
