"""One integrated demo:  python run_demo.py [--seed 3] [--misreads 2] [--givens 30]

Shows: the handwritten input, what the NN read, the AC-3 conflict, the search trace,
the corrections log, the solution, how the CSP behaves with/without FC and AC-3,
and the system with the NN-guided correction ON vs OFF. Figures go to outputs/demo_*.png.
"""
import argparse
import os

import numpy as np

from csp.sudoku import solve, find_conflicts, to_str
from data import generator as G
from integration.pipeline import solve_image
from nn.mlp import MLP
from nn.train import load_data

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def show_grid(g):
    rows = []
    for r in range(9):
        cells = [str(g[r * 9 + c]) if g[r * 9 + c] else "." for c in range(9)]
        rows.append("  ".join(" ".join(cells[i:i + 3]) for i in (0, 3, 6)))
    return "\n".join(rows)


def main(seed, misreads, givens, open_browser=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    net = MLP.load(os.path.join(ROOT, "nn", "model_mlp.npz"))
    X, y = load_data()["test"]
    pools = G.build_pools(net, X, y)
    rng = np.random.default_rng(seed)
    puz, sol = G.make_puzzle(rng, givens)
    imgs, info = G.render(puz, X, y, pools, rng, misreads=misreads, net=net)
    read, probs = G.read_grid(net, imgs)

    print(f"== INPUT: handwritten 9x9 mosaic (seed {seed}), true puzzle has {sum(1 for c in puz if c)} givens")
    print(f"   injected hard digits at cells: {info['injected']}")
    print("\n== NN reading (first reading):\n" + show_grid(read))
    conf = find_conflicts(read)
    print(f"\n== AC-3 / conflict check: {len(conf)} duplicate pair(s) {conf}")
    print("   CSP on first reading:", solve(read).message or "solvable")

    # ---- NN ON: UCS / Greedy / A* with trace
    print("\n== SEARCH TRACE (UCS): [corrections so far], g=cost, f=priority")
    outs = {}
    for algo in ("ucs", "greedy", "astar"):
        outs[algo] = solve_image(net, imgs, algo=algo, trace=True)
    for st, g, f in outs["ucs"].search.trace[:12]:
        print(f"   expand {st}  g={g}  f={f}")
    print("   ..." if len(outs["ucs"].search.trace) > 12 else "")
    for algo, o in outs.items():
        s = o.search
        ok = o.solution == sol
        if s is None:
            print(f"   {algo:7s}: {o.message}")
        else:
            print(f"   {algo:7s}: {o.status}  expanded={s.expanded} generated={s.generated} csp_calls={s.csp_calls} "
                  f"cost={s.cost:.3f}  exact={ok}")
    o = outs["ucs"]
    print("\n== CORRECTION LOG (cell, read -> new, cost = log(p_read/p_new))")
    for c, a, d, cost in o.corrections or []:
        print(f"   cell {c} (r{c // 9 + 1},c{c % 9 + 1}): {a} -> {d}   cost {cost:.3f}")
    print("\n== SOLUTION:\n" + show_grid(o.solution) if o.solution else "\n== " + o.message)
    print("   matches ground truth:", o.solution == sol)

    # ---- NN OFF
    off = solve_image(net, imgs, use_correction=False)
    print(f"\n== SYSTEM WITH CORRECTION OFF: {off.status} - {off.message}")
    print(f"== SYSTEM WITH CORRECTION ON : {o.status}, exact match = {o.solution == sol}")

    # ---- CSP pruning with / without FC & AC-3 (on the corrected puzzle)
    final = o.final or read
    cfgs = {"BT": dict(fc=False, ac3_on=False), "BT+FC": dict(fc=True, ac3_on=False),
            "BT+FC+AC-3": dict(fc=True, ac3_on=True)}
    stats = {}
    print("\n== CSP PRUNING on the corrected puzzle (static variable order):")
    for name, cfg in cfgs.items():
        r = solve(final, mrv=False, max_assignments=300000, **cfg)
        stats[name] = r.stats
        print(f"   {name:11s} assignments={r.stats.assignments:7d} backtracks={r.stats.backtracks:7d} "
              f"pruned={r.stats.pruned:5d} ac3_revisions={r.stats.ac3_revisions}")

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(G.mosaic(imgs), cmap="gray"); ax[0].set_title("input: handwritten mosaic")
    ax[1].set_title("NN reading (red = misread, green = corrected)")
    ax[1].set_xlim(0, 9); ax[1].set_ylim(9, 0); ax[1].set_aspect("equal")
    corr = {c: d for c, _, d, _ in (o.corrections or [])}
    for c in range(81):
        r_, c_ = divmod(c, 9)
        if read[c]:
            wrong = read[c] != puz[c]
            ax[1].text(c_ + .5, r_ + .5, read[c], ha="center", va="center", color="red" if wrong else "k", fontsize=13)
        if c in corr:
            ax[1].text(c_ + .8, r_ + .25, corr[c], ha="center", va="center", color="green", fontsize=9)
    for i in range(10):
        ax[1].axhline(i, color="k", lw=2 if i % 3 == 0 else .4); ax[1].axvline(i, color="k", lw=2 if i % 3 == 0 else .4)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    names = list(stats)
    ax[2].bar(names, [stats[n].assignments + 1 for n in names], color=["#999", "#4c8", "#27a"])
    ax[2].set_yscale("log"); ax[2].set_title("CSP assignments (log): effect of FC / AC-3")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "demo_overview.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = ["correction OFF", "correction ON (UCS)"]
    vals = [float(off.solution == sol) if off.solution else 0.0, float(o.solution == sol)]
    ax.bar(labels, vals, color=["#c55", "#4a8"]); ax.set_ylim(0, 1.1); ax.set_title("exact solution recovered?")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "demo_nn_on_off.png"), dpi=130); plt.close(fig)
    print("\nfigures: outputs/demo_overview.png, outputs/demo_nn_on_off.png")

    # ---- dashboard (one HTML file, opens in the browser)
    from eval.dashboard import build
    demo = {"seed": seed, "misreads": misreads, "givens": givens, "status": o.status,
            "exact": o.solution == sol, "read": [int(v) for v in read], "final": [int(v) for v in final],
            "injected": [int(c) for c in info["injected"]], "conflicts": len(conf),
            "algos": [{"name": a.upper(), "expanded": r.search.expanded, "generated": r.search.generated,
                       "csp_calls": r.search.csp_calls, "cost": r.search.cost}
                      for a, r in outs.items() if r.search],
            "corrections": [[int(c), int(a), int(d), float(k)] for c, a, d, k in (o.corrections or [])],
            "csp": [{"name": n, "assignments": s.assignments, "backtracks": s.backtracks, "pruned": s.pruned}
                    for n, s in stats.items()]}
    build(demo, open_browser=open_browser)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--no-browser", action="store_true", help="write outputs/dashboard.html but do not open it")
    a = ap.parse_args()
    main(a.seed, a.misreads, a.givens, open_browser=not a.no_browser)
