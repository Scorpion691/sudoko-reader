"""Seeded Sudoku puzzle + handwritten-mosaic generator, and the 'hard digit' injection rule.

Puzzle: full grid by randomised backtracking, then cells are removed (random order) while
the solution stays unique, until the wanted number of givens is reached.

Mosaic: each given cell gets a real MNIST test image of that digit; blank cells get a
dark noise image. Blank cells are later detected by a brightness rule (not by the NN).

Injection rule (documented, deterministic given the rng): a *hard digit* is a test image
whose true label is the cell's digit but that the network misreads, with the true digit
ranked 2nd or 3rd by the network. For k misreads we replace k given cells with such images
and (for k >= 1) re-draw until the first reading is NOT solvable (otherwise nothing is
'broken' and there is nothing to correct).
"""
import numpy as np

from csp.sudoku import PEERS, ALL, solve, _bits

BLANK_THRESHOLD = 0.5  # brightest pixel (0-1) below this => empty cell (noise stays < 0.15, digits reach ~1)


def random_full_grid(rng):
    assign = [0] * 81
    dom = [ALL] * 81

    def rec(dom):
        best, bm, bk = -1, 0, 10
        for c in range(81):
            if not assign[c]:
                k = dom[c].bit_count()
                if k == 0:
                    return False
                if k < bk:
                    best, bm, bk = c, dom[c], k
        if best < 0:
            return True
        digits = list(_bits(bm))
        rng.shuffle(digits)
        for d in digits:
            assign[best] = d
            nd = list(dom)
            nd[best] = 1 << d
            ok = True
            for p in PEERS[best]:
                if not assign[p] and nd[p] >> d & 1:
                    nd[p] &= ~(1 << d)
                    if nd[p] == 0:
                        ok = False
                        break
            if ok and rec(nd):
                return True
            assign[best] = 0
        return False

    assert rec(dom)
    return assign


def make_puzzle(rng, givens=30, tries=5):
    """Return (puzzle, solution) with a unique solution and (about) `givens` givens."""
    best = None
    for _ in range(tries):
        sol = random_full_grid(rng)
        puz = list(sol)
        order = list(rng.permutation(81))
        n = 81
        for c in order:
            if n <= givens:
                break
            keep = puz[c]
            puz[c] = 0
            if solve(puz, limit=2).solutions.__len__() == 1:
                n -= 1
            else:
                puz[c] = keep
        if best is None or n < best[2]:
            best = (puz, sol, n)
        if n <= givens:
            break
    return best[0], best[1]


def noise_image(rng):
    return np.clip(rng.normal(0, 0.02, (28, 28)), 0, 1).astype(np.float32)


def is_blank(img):
    return float(img.max()) < BLANK_THRESHOLD


def build_pools(net, X, y):
    """Per digit: indices of correctly-read images, and of 'hard' (misread, true in top-3) images."""
    P = net.predict_proba(X)
    order = np.argsort(-P, 1)
    pred = order[:, 0]
    rank_true = np.array([int(np.where(order[i] == y[i])[0][0]) for i in range(len(y))])
    good, hard = {}, {}
    for d in range(9):
        good[d + 1] = np.where((y == d) & (pred == d))[0]
        hard[d + 1] = np.where((y == d) & (pred != d) & (rank_true <= 2))[0]
    return good, hard


def render(puzzle, X, y, pools, rng, misreads=0, net=None, max_redraw=30):
    """Make the 81 cell images for `puzzle`. Returns (images[81,28,28], info dict).

    images are real MNIST test digits (blank = noise). `misreads` hard digits are injected.
    """
    good, hard = pools
    givens = [c for c in range(81) if puzzle[c]]
    for attempt in range(max_redraw):
        imgs = np.zeros((81, 28, 28), np.float32)
        used = {}
        for c in range(81):
            if puzzle[c]:
                used[c] = int(rng.choice(good[puzzle[c]]))
            else:
                imgs[c] = noise_image(rng)
        injected = []
        if misreads:
            cand = [c for c in givens if len(hard[puzzle[c]])]
            for c in rng.choice(cand, size=min(misreads, len(cand)), replace=False):
                used[int(c)] = int(rng.choice(hard[puzzle[int(c)]]))
                injected.append(int(c))
        for c, i in used.items():
            imgs[c] = X[i]
        info = {"injected": sorted(injected), "attempts": attempt + 1}
        if not misreads or net is None:
            return imgs, info
        read, _ = read_grid(net, imgs)
        if not solve(read).solved:  # first reading is broken -> something to correct
            return imgs, info
    return imgs, {**info, "note": "first reading stayed solvable"}


def read_grid(net, imgs):
    """NN reading of 81 cell images. Returns (grid[81] with 0 for blank, probs[81,9] or None)."""
    probs = np.zeros((81, 9), np.float32)
    grid = [0] * 81
    nonblank = [c for c in range(81) if not is_blank(imgs[c])]
    if nonblank:
        P = net.predict_proba(imgs[nonblank])
        for c, p in zip(nonblank, P):
            probs[c] = p
            grid[c] = int(p.argmax()) + 1
    return grid, probs


def mosaic(imgs):
    """81 cell images -> one 252x252 picture."""
    return imgs.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)
