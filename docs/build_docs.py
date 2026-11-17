"""Builds the report and the slides (HTML + PDF) in the same visual design as the dashboard.

    python docs/build_docs.py            -> docs/Report.html, docs/Report.pdf, docs/Slides.html, docs/Slides.pdf

All numbers in the text are filled from the real result tables in outputs/ (nothing is typed by hand),
the digit images are real MNIST test digits and the corrected cases come from the actual pipeline.
The PDFs are printed by Microsoft Edge / Chrome in headless mode (skipped if neither is installed).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from eval import dashboard as dash  # noqa: E402  (reuses the table loader, digit images and real cases)

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]


def load():
    nn = json.load(open(os.path.join(dash.OUT, "nn_results.json"), encoding="utf-8"))
    return {"exp1": dash._table("exp1_search"), "exp2": dash._table("exp2_csp"), "exp3": dash._table("exp3_nn_vs_baselines"),
            "nn": nn, "student": dash.STUDENT, "digits": dash._digits(), "cases": dash._cases()}


def numbers(D):
    e1, e2, e3, nn = D["exp1"], D["exp2"], D["exp3"], D["nn"]
    m = lambda x: x[0] if isinstance(x, (tuple, list)) else float(x)
    s = lambda x: x[1] if isinstance(x, (tuple, list)) else 0.0

    def row(t, **kw):
        return next(r for r in t if all(str(r[k]) == str(v) for k, v in kw.items()))
    N = {}
    N["acc_mlp"] = "%.1f" % (nn["mlp"]["test_acc_after"] * 100)
    N["acc_lr"] = "%.1f" % (nn["logreg"]["test_acc"] * 100)
    N["acc_gain"] = "%.1f" % ((nn["mlp"]["test_acc_after"] - nn["logreg"]["test_acc"]) * 100)
    N["ece_before"] = "%.4f" % nn["mlp"]["ece_before"]
    N["ece_after"] = "%.4f" % nn["mlp"]["ece_after_temperature_scaling"]
    N["ece_lr"] = "%.4f" % nn["logreg"]["ece_before"]
    N["temp"] = "%.2f" % nn["mlp"]["temperature"]
    for k in (1, 2, 3, 4):
        for a in ("UCS", "GREEDY", "ASTAR"):
            r = row(e1, misreads=k, algorithm=a)
            N["rec_%s_%d" % (a.lower(), k)] = "%.2f" % m(r["exact recovery rate"])
            N["recsd_%s_%d" % (a.lower(), k)] = "%.2f" % s(r["exact recovery rate"])
            N["exp_%s_%d" % (a.lower(), k)] = "%.1f" % m(r["nodes expanded"])
            N["expsd_%s_%d" % (a.lower(), k)] = "%.1f" % s(r["nodes expanded"])
            N["gen_%s_%d" % (a.lower(), k)] = "%.0f" % m(r["nodes generated"])
            N["csp_%s_%d" % (a.lower(), k)] = "%.1f" % m(r["CSP calls"])
    for g in (25, 30, 35):
        for key, name in (("bt", "BT"), ("fc", "BT+FC"), ("ac", "BT+FC+AC-3")):
            r = row(e2, givens=g, method=name)
            N["asg%d_%s" % (g, key)] = "{:,.0f}".format(m(r["assignments"]))
            N["asgsd%d_%s" % (g, key)] = "{:,.0f}".format(s(r["assignments"]))
            N["btk%d_%s" % (g, key)] = "{:,.0f}".format(m(r["backtracks"]))
            N["solved%d_%s" % (g, key)] = "%.0f" % (100 * m(r[[c for c in r if c.startswith("solved")][0]]))
        a_bt = m(row(e2, givens=g, method="BT")["assignments"])
        a_ac = m(row(e2, givens=g, method="BT+FC+AC-3")["assignments"])
        N["red%d" % g] = "%.0f" % (100 * (1 - a_ac / a_bt))
    for k in (0, 1, 2, 3):
        for key, name in (("nn", "NN-guided (UCS)"), ("none", "no correction"), ("rand", "random correction")):
            r = row(e3, misreads=k, method=name)
            N["x%d_%s" % (k, key)] = "%.2f" % m(r["exact recovery rate"])
            N["xsd%d_%s" % (k, key)] = "%.2f" % s(r["exact recovery rate"])
    cs = D["cases"]
    N["cases_n"] = str(len(cs))
    N["cases_exact"] = str(sum(1 for c in cs if c["exact"]))
    return N


def render(template, out, D, N):
    html = open(os.path.join(HERE, template), encoding="utf-8").read()
    html = html.replace("/*SHARED_CSS*/", open(os.path.join(HERE, "shared.css"), encoding="utf-8").read())
    html = html.replace("/*SHARED_JS*/", open(os.path.join(HERE, "shared.js"), encoding="utf-8").read())
    html = html.replace("/*DATA*/null", json.dumps(D))
    for k, v in N.items():
        html = html.replace("{{%s}}" % k, v)
    for k, v in D["student"].items():
        html = html.replace("__%s__" % k.upper(), v)
    open(os.path.join(HERE, out), "w", encoding="utf-8").write(html)
    return os.path.join(HERE, out)


def browser():
    return next((b for b in BROWSERS if os.path.exists(b)), None)


def to_pdf(html, pdf, query=""):
    b = browser()
    if not b:
        print("no Edge/Chrome found: PDF skipped for", html)
        return False
    uri = "file:///" + html.replace("\\", "/") + query
    subprocess.run([b, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    "--print-to-pdf=" + pdf, uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    return os.path.exists(pdf)


def pages(pdf):
    try:
        from pypdf import PdfReader
        return len(PdfReader(pdf).pages)
    except Exception:
        return None


if __name__ == "__main__":
    D = load()
    N = numbers(D)
    for tpl, name in (("report_template.html", "Report"), ("slides_template.html", "Slides")):
        if not os.path.exists(os.path.join(HERE, tpl)):
            continue
        h = render(tpl, name + ".html", D, N)
        ok = to_pdf(h, os.path.join(HERE, name + ".pdf"), "?print" if name == "Slides" else "")
        print(name, "html ok;", "pdf pages:" if ok else "pdf skipped", pages(os.path.join(HERE, name + ".pdf")) if ok else "")
