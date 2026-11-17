"""Build outputs/dashboard.html (one self-contained file, no extra libraries) and open it in the browser.

The look lives in eval/dashboard_template.html. Data comes from the experiment tables (outputs/exp*.csv),
outputs/nn_results.json and, optionally, the data of the current demo run (passed in by run_demo.py).
Run alone:  python -m eval.dashboard
"""
import csv
import json
import os
import re
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")

STUDENT = {"name": "Abdelrhman Elhawarry", "id": "320210121",
           "project": "Handwritten Sudoku Reader and Solver", "code": "P7"}


def _ms(cell):
    m = re.match(r"\s*([-\d.]+)\s*±\s*([-\d.]+)", cell)
    return (float(m.group(1)), float(m.group(2))) if m else cell


def _table(name):
    p = os.path.join(OUT, name + ".csv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        r = list(csv.reader(f))
    head, rows = r[0], r[1:]
    return [{h: _ms(c) for h, c in zip(head, row)} for row in rows]


def _digits(color=(255, 255, 255)):
    """One real handwritten MNIST test image per digit 1-9 (white ink, transparent background) for the intro."""
    p = os.path.join(ROOT, "data", "mnist19.npz")
    if not os.path.exists(p):
        return None
    try:
        import base64
        import io

        import numpy as np
        from PIL import Image
        d = np.load(p)
        X, y = d["X_test"], d["y_test"]
        out = {}
        for dgt in range(1, 10):
            cand = np.where(y == dgt)[0][:300]
            means = X[cand].mean((1, 2))
            a = X[cand[int(np.argmin(np.abs(means - np.median(means))))]]
            rgba = np.zeros((28, 28, 4), np.uint8)
            rgba[..., :3] = color
            rgba[..., 3] = a
            buf = io.BytesIO()
            Image.fromarray(rgba, "RGBA").resize((84, 84), Image.LANCZOS).save(buf, "PNG")
            out[str(dgt)] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return out
    except Exception:  # the intro falls back to plain text digits
        return None


def _cases(specs=((11, 1), (12, 1), (21, 2), (22, 2), (31, 3), (32, 3)), givens=30):
    """Real corrected cases: each is a generated puzzle + handwritten mosaic with `misreads` injected hard digits,
    solved by the actual pipeline (NN -> conflict -> UCS correction -> CSP). Nothing is hand-made."""
    try:
        import numpy as np
        from data import generator as G
        from integration.pipeline import solve_reading
        from nn.mlp import MLP
        from nn.train import load_data
        net = MLP.load(os.path.join(ROOT, "nn", "model_mlp.npz"))
        X, y = load_data()["test"]
        pools = G.build_pools(net, X, y)
        out = []
        for seed, k in specs:
            rng = np.random.default_rng(seed)
            puz, sol = G.make_puzzle(rng, givens)
            imgs, info = G.render(puz, X, y, pools, rng, misreads=k, net=net)
            read, probs = G.read_grid(net, imgs)
            o = solve_reading(read, probs, algo="ucs", max_corr=4)
            s = o.search
            ok = o.status == "solved"
            out.append({"seed": seed, "misreads": k, "read": [int(v) for v in read],
                        "final": [int(v) for v in (o.final if ok else read)],
                        "status": o.status, "exact": bool(ok and o.solution == sol),
                        "cost": float(s.cost) if s else 0.0, "expanded": s.expanded if s else 0,
                        "generated": s.generated if s else 0, "csp_calls": s.csp_calls if s else 0,
                        "corrections": [[int(c), int(a), int(d), float(w)] for c, a, d, w in (o.corrections or [])]})
        return out
    except Exception as e:  # the dashboard simply hides the section
        print("cases skipped:", e)
        return []


def build(demo=None, open_browser=True):
    nn = None
    p = os.path.join(OUT, "nn_results.json")
    if os.path.exists(p):
        nn = json.load(open(p, encoding="utf-8"))
        nn["mlp"].pop("confusion_matrix", None)
    data = {"exp1": _table("exp1_search"), "exp2": _table("exp2_csp"), "exp3": _table("exp3_nn_vs_baselines"),
            "nn": nn, "demo": demo, "student": STUDENT, "digits": _digits(), "cases": _cases()}
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*DATA*/null", json.dumps(data))
    for k, v in STUDENT.items():
        html = html.replace("__%s__" % k.upper(), v)
    path = os.path.join(OUT, "dashboard.html")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("dashboard:", path)
    if open_browser:
        webbrowser.open("file:///" + path.replace("\\", "/"))
    return path


if __name__ == "__main__":
    build()
