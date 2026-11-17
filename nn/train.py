"""Train the digit classifier + logistic-regression baseline, calibrate, write results.

Usage (from sudoku_reader/):  python -m nn.train [--epochs 10] [--seed 0]
Outputs: nn/model_mlp.npz, nn/model_logreg.npz, outputs/nn_results.json, outputs/confusion.png
"""
import argparse
import json
import os

import numpy as np

from nn.mlp import MLP, ece, confusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    d = np.load(os.path.join(ROOT, "data", "mnist19.npz"))
    return {k: (d["X_" + k].astype(np.float32) / 255.0, d["y_" + k].astype(int) - 1)
            for k in ("train", "val", "test")}


def main(epochs=10, seed=0):
    D = load_data()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = D["train"], D["val"], D["test"]
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    res = {}
    print("logistic regression baseline")
    lr = MLP([784, 9], seed).fit(Xtr, ytr, Xva, yva, epochs=epochs, lr=2e-3, seed=seed)
    print("MLP 784-128-64-9")
    nn = MLP([784, 128, 64, 9], seed).fit(Xtr, ytr, Xva, yva, epochs=epochs, seed=seed)
    for name, net in (("logreg", lr), ("mlp", nn)):
        p0 = net.predict_proba(Xte, T=1.0)
        r = {"test_acc": float((p0.argmax(1) == yte).mean()), "ece_before": ece(p0, yte)}
        if name == "mlp":
            r["temperature"] = net.fit_temperature(Xva, yva)
            p1 = net.predict_proba(Xte)
            r["ece_after_temperature_scaling"] = ece(p1, yte)
            r["test_acc_after"] = float((p1.argmax(1) == yte).mean())
            r["confusion_matrix"] = confusion(p1.argmax(1), yte).tolist()
        net.save(os.path.join(ROOT, "nn", f"model_{name}.npz"))
        res[name] = r
        print(name, {k: v for k, v in r.items() if k != "confusion_matrix"})
    with open(os.path.join(ROOT, "outputs", "nn_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array(res["mlp"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10))
    ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("MLP confusion matrix (test)")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6,
                    color="k" if M[i, j] < M.max() / 2 else "w")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "outputs", "confusion.png"), dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.epochs, a.seed)
