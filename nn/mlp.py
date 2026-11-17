"""Small numpy neural network (no deep-learning framework needed -> light on the CPU).

MLP([784,128,64,9]): ReLU hidden layers, softmax output, Adam, mini-batches.
MLP([784,9]) is multinomial logistic regression = the baseline.
Classes are digits 1..9 (index = digit-1). Temperature scaling calibrates probabilities.
"""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.T = 1.0  # temperature

    def logits(self, X):
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W + b, 0)
        return h @ self.W[-1] + self.b[-1]

    def predict_proba(self, X, T=None):
        X = np.asarray(X, np.float32).reshape(len(X), -1)
        return softmax(self.logits(X) / (self.T if T is None else T))

    def fit(self, X, y, Xv=None, yv=None, epochs=10, bs=128, lr=1e-3, seed=0, log=print):
        """X in [0,1], y in 0..8. Adam optimiser."""
        rng = np.random.default_rng(seed)
        X = X.reshape(len(X), -1).astype(np.float32)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        L = len(self.W)
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), bs):
                j = idx[i:i + bs]
                xb, yb = X[j], y[j]
                acts = [xb]
                h = xb
                for W, b in zip(self.W[:-1], self.b[:-1]):
                    h = np.maximum(h @ W + b, 0)
                    acts.append(h)
                d = softmax(h @ self.W[-1] + self.b[-1])
                d[np.arange(len(j)), yb] -= 1
                d /= len(j)
                gW, gb = [None] * L, [None] * L
                for k in range(L - 1, -1, -1):
                    gW[k] = acts[k].T @ d
                    gb[k] = d.sum(0)
                    if k:
                        d = (d @ self.W[k].T) * (acts[k] > 0)
                grads = gW + gb
                t += 1
                for q, (p_, g) in enumerate(zip(params, grads)):
                    m[q] = 0.9 * m[q] + 0.1 * g
                    v[q] = 0.999 * v[q] + 0.001 * g * g
                    p_ -= (lr * (m[q] / (1 - 0.9 ** t)) / (np.sqrt(v[q] / (1 - 0.999 ** t)) + 1e-8)).astype(np.float32)
            if Xv is not None:
                acc = (self.predict_proba(Xv, T=1.0).argmax(1) == yv).mean()
                log(f"  epoch {ep + 1}/{epochs}  val acc {acc:.4f}")
        return self

    def fit_temperature(self, Xv, yv):
        """Temperature scaling: pick T minimising validation NLL (grid search)."""
        z = self.logits(Xv.reshape(len(Xv), -1).astype(np.float32))
        best = (1e9, 1.0)
        for T in np.linspace(0.5, 4.0, 71):
            p = softmax(z / T)
            nll = -np.log(p[np.arange(len(yv)), yv] + 1e-12).mean()
            best = min(best, (nll, T))
        self.T = float(best[1])
        return self.T

    def save(self, path):
        np.savez(path, sizes=self.sizes, T=self.T,
                 **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(list(d["sizes"]))
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        net.T = float(d["T"])
        return net


def ece(probs, y, bins=15):
    """Expected calibration error (top-label confidence vs accuracy)."""
    conf, pred = probs.max(1), probs.argmax(1)
    ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(ok[s].mean() - conf[s].mean())
    return float(e)


def confusion(pred, y, k=9):
    M = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        M[a, b] += 1
    return M
