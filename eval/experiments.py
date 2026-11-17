"""The three experiments (>= 5 seeds, mean and std over seeds). Tables -> outputs/*.csv|md, plots -> outputs/*.png

  Exp1  UCS vs Greedy vs A*   : nodes expanded/generated + exact-recovery rate, 0-4 misreads
  Exp2  BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25 / 30 / 35 givens
  Exp3  NN-guided correction vs no correction vs random correction: exact-recovery rate

Usage (from sudoku_reader/):  python -m eval.experiments [--seeds 5] [--n 20] [--which 123]
"""
import argparse
import csv
import os
import time

import numpy as np

from csp.sudoku import solve
from data import generator as G
from integration.pipeline import solve_reading
from nn.mlp import MLP
from nn.train import load_data
from search.correction import candidates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
MAX_CORR = 4          # covers the 0-4 misread levels of Exp1 (the spec's default limit is 3)
MAX_EXPANDED = 4000   # safety cap on search nodes; a capped run counts as a failure


def setup():
    net = MLP.load(os.path.join(ROOT, "nn", "model_mlp.npz"))
    X, y = load_data()["test"]
    return net, X, y, G.build_pools(net, X, y)


def msd(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def write_table(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


def fmt(m, s, d=2):
    return f"{m:.{d}f} ± {s:.{d}f}"


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    return p


# ------------------------------------------------------------------ Exp 1
def exp1(env, seeds, n, levels=(0, 1, 2, 3, 4), givens=30):
    net, X, y, pools = env
    algos = ("ucs", "greedy", "astar")
    per = {(a, k): {"exp": [], "gen": [], "rec": [], "csp": []} for a in algos for k in levels}
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        puzzles = [G.make_puzzle(rng, givens) for _ in range(n)]
        for k in levels:
            insts = []
            for puz, sol in puzzles:
                imgs, _ = G.render(puz, X, y, pools, rng, misreads=k, net=net)
                insts.append((sol, G.read_grid(net, imgs)))
            for a in algos:
                e, g, ok, cc = [], [], [], []
                for sol, (grid, probs) in insts:
                    o = solve_reading(grid, probs, algo=a, max_corr=MAX_CORR)
                    r = o.search
                    e.append(r.expanded if r else 0)
                    g.append(r.generated if r else 0)
                    cc.append(r.csp_calls if r else 0)
                    ok.append(o.status == "solved" and o.solution == sol)
                for key, v in zip(("exp", "gen", "rec", "csp"), (e, g, ok, cc)):
                    per[(a, k)][key].append(float(np.mean(v)))
        print(f"  exp1 seed {s + 1}/{seeds} done")
    rows = []
    for k in levels:
        for a in algos:
            d = per[(a, k)]
            rows.append([k, a.upper(), fmt(*msd(d["exp"])), fmt(*msd(d["gen"])), fmt(*msd(d["csp"])),
                         fmt(*msd(d["rec"]), 3)])
    write_table("exp1_search", ["misreads", "algorithm", "nodes expanded", "nodes generated",
                                "CSP calls", "exact recovery rate"], rows)
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(10, 3.8))
    for a in algos:
        for j, key, title in ((0, "exp", "nodes expanded (mean over puzzles)"), (1, "rec", "exact recovery rate")):
            m = [msd(per[(a, k)][key]) for k in levels]
            ax[j].errorbar(levels, [x[0] for x in m], [x[1] for x in m], marker="o", capsize=3, label=a.upper())
            ax[j].set_title(title); ax[j].set_xlabel("injected misreads")
    ax[0].set_yscale("log"); ax[1].set_ylim(0, 1.05); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp1_search.png"), dpi=130); p.close(fig)


# ------------------------------------------------------------------ Exp 2
def exp2(seeds, n, givens_levels=(25, 30, 35), cap=200000, mrv=False):
    cfgs = {"BT": dict(fc=False, ac3_on=False), "BT+FC": dict(fc=True, ac3_on=False),
            "BT+FC+AC-3": dict(fc=True, ac3_on=True)}
    per = {(c, g): {"asg": [], "bt": [], "t": [], "solved": []} for c in cfgs for g in givens_levels}
    for s in range(seeds):
        rng = np.random.default_rng(2000 + s)
        for g in givens_levels:
            puzzles = [G.make_puzzle(rng, g)[0] for _ in range(n)]
            for c, cfg in cfgs.items():
                A, B, T, S = [], [], [], []
                for puz in puzzles:
                    t = time.perf_counter()
                    r = solve(puz, mrv=mrv, max_assignments=cap, **cfg)
                    T.append(time.perf_counter() - t)
                    A.append(r.stats.assignments); B.append(r.stats.backtracks); S.append(r.solved)
                for key, v in zip(("asg", "bt", "t", "solved"), (A, B, T, S)):
                    per[(c, g)][key].append(float(np.mean(v)))
        print(f"  exp2 seed {s + 1}/{seeds} done")
    rows = []
    for g in givens_levels:
        for c in cfgs:
            d = per[(c, g)]
            rows.append([g, c, fmt(*msd(d["asg"]), 0), fmt(*msd(d["bt"]), 0), fmt(*msd(d["t"]), 4),
                         fmt(*msd(d["solved"]), 2)])
    write_table("exp2_csp", ["givens", "method", "assignments", "backtracks", "time (s)",
                             f"solved within {cap} assignments"], rows)
    p = plt()
    fig, ax = p.subplots(1, 2, figsize=(10, 3.8))
    w = 0.27
    for i, c in enumerate(cfgs):
        for j, key, title in ((0, "asg", "assignments (log)"), (1, "bt", "backtracks (log)")):
            m = [msd(per[(c, g)][key]) for g in givens_levels]
            ax[j].bar(np.arange(3) + i * w, [x[0] + 1 for x in m], w, yerr=[x[1] for x in m], capsize=3, label=c)
            ax[j].set_xticks(np.arange(3) + w); ax[j].set_xticklabels(givens_levels)
            ax[j].set_yscale("log"); ax[j].set_title(title); ax[j].set_xlabel("number of givens")
    ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp2_csp.png"), dpi=130); p.close(fig)


# ------------------------------------------------------------------ Exp 3
def random_correction(grid, probs, rng, budget=50):
    """Baseline: try `budget` random sets of 1-3 corrections (random cells, random top-3 alternatives)."""
    cells = [c for c in range(81) if grid[c]]
    for _ in range(budget):
        j = int(rng.integers(1, 4))
        g = list(grid)
        for c in rng.choice(cells, size=j, replace=False):
            alts = [d + 1 for d in np.argsort(-probs[c])[:3] if d + 1 != grid[c]]
            g[int(c)] = int(rng.choice(alts))
        r = solve(g)
        if r.solved:
            return r.solutions[0]
    return None


def exp3(env, seeds, n, levels=(0, 1, 2, 3), givens=30):
    net, X, y, pools = env
    methods = ("NN-guided (UCS)", "no correction", "random correction")
    per = {(m, k): [] for m in methods for k in levels}
    for s in range(seeds):
        rng = np.random.default_rng(3000 + s)
        puzzles = [G.make_puzzle(rng, givens) for _ in range(n)]
        for k in levels:
            ok = {m: [] for m in methods}
            for puz, sol in puzzles:
                imgs, _ = G.render(puz, X, y, pools, rng, misreads=k, net=net)
                grid, probs = G.read_grid(net, imgs)
                o = solve_reading(grid, probs, algo="ucs", max_corr=MAX_CORR)
                ok[methods[0]].append(o.status == "solved" and o.solution == sol)
                o = solve_reading(grid, probs, use_correction=False)
                ok[methods[1]].append(o.status == "solved" and o.solution == sol)
                first = solve(grid)
                rs = first.solutions[0] if first.solved else random_correction(grid, probs, rng)
                ok[methods[2]].append(rs == sol)
            for m in methods:
                per[(m, k)].append(float(np.mean(ok[m])))
        print(f"  exp3 seed {s + 1}/{seeds} done")
    rows = [[k, m, fmt(*msd(per[(m, k)]), 3)] for k in levels for m in methods]
    write_table("exp3_nn_vs_baselines", ["misreads", "method", "exact recovery rate"], rows)
    p = plt()
    fig, ax = p.subplots(figsize=(6, 3.8))
    w = 0.27
    for i, m in enumerate(methods):
        ms = [msd(per[(m, k)]) for k in levels]
        ax.bar(np.arange(len(levels)) + i * w, [x[0] for x in ms], w, yerr=[x[1] for x in ms], capsize=3, label=m)
    ax.set_xticks(np.arange(len(levels)) + w); ax.set_xticklabels(levels)
    ax.set_xlabel("injected misreads"); ax.set_ylabel("exact recovery rate"); ax.set_ylim(0, 1.1); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp3_nn_vs_baselines.png"), dpi=130); p.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=20, help="puzzles per seed per level")
    ap.add_argument("--n2", type=int, default=10, help="puzzles per seed per level for Exp2")
    ap.add_argument("--which", default="123")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    env = setup() if set(a.which) & {"1", "3"} else None
    t0 = time.time()
    if "1" in a.which:
        exp1(env, a.seeds, a.n)
    if "2" in a.which:
        exp2(a.seeds, a.n2)
    if "3" in a.which:
        exp3(env, a.seeds, a.n)
    print("done in %.0fs" % (time.time() - t0))
