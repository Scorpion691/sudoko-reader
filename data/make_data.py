"""Reproducible MNIST preprocessing (fixed seed).

Reads the 4 raw gzip files in data/raw (download with --download), splits the 60k
training set 50,000 train / 10,000 validation (seeded), then drops digit 0 (Sudoku
uses 1-9). The official test set is untouched (except dropping 0).
Images stay uint8 (scaled to 0-1 when used). Labels are the digit itself (1..9).
Output: data/mnist19.npz
"""
import argparse
import gzip
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
MIRRORS = ["https://storage.googleapis.com/cvdf-datasets/mnist/",
           "http://yann.lecun.com/exdb/mnist/"]


def download():
    os.makedirs(RAW, exist_ok=True)
    for f in FILES:
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            continue
        for m in MIRRORS:
            try:
                urllib.request.urlretrieve(m + f, p)
                break
            except Exception as e:  # try next mirror
                print("mirror failed", m, e)
        else:
            raise SystemExit(f"cannot download {f}")


def _images(f):
    with gzip.open(os.path.join(RAW, f)) as fh:
        d = np.frombuffer(fh.read(), np.uint8, offset=16)
    return d.reshape(-1, 28, 28)


def _labels(f):
    with gzip.open(os.path.join(RAW, f)) as fh:
        return np.frombuffer(fh.read(), np.uint8, offset=8)


def main(seed=0, download_first=False):
    if download_first:
        download()
    X, y = _images(FILES[0]), _labels(FILES[1])
    Xt, yt = _images(FILES[2]), _labels(FILES[3])
    perm = np.random.default_rng(seed).permutation(len(X))
    tr, va = perm[:50000], perm[50000:]
    out = {}
    for name, (a, b) in {"train": (X[tr], y[tr]), "val": (X[va], y[va]), "test": (Xt, yt)}.items():
        keep = b != 0
        out["X_" + name], out["y_" + name] = a[keep], b[keep]
    np.savez_compressed(os.path.join(HERE, "mnist19.npz"), **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    main(a.seed, a.download)
