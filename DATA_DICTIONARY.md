# Data dictionary

## Real data (MNIST, http://yann.lecun.com/exdb/mnist/; fetched from the cvdf-datasets mirror)
| Item | Type | Meaning |
|---|---|---|
| `X_train/X_val/X_test` (`data/mnist19.npz`) | real, uint8 28x28 | handwritten digit image, 0-255; scaled to 0-1 when used |
| `y_train/y_val/y_test` | real | digit label 1-9 (digit 0 removed; the NN uses index = digit-1) |

Split (seed 0): the official 60,000 training images -> 50,000 train / 10,000 validation (random),
then digit 0 is dropped -> 45,053 / 9,024. The official test set (9,020 images without 0) is
untouched and used for evaluation and for building the Sudoku mosaics.
MNIST licence: freely available for research use; keep a local copy in `data/raw/`.

## Generated data (everything about Sudoku is generated, seeded by `numpy.random.default_rng(seed)`)
| Item | How it is generated |
|---|---|
| Solution grid | randomised backtracking: MRV cell order, shuffled digit order (`data/generator.py::random_full_grid`) |
| Puzzle | cells removed in random order while the solution stays unique (checked with the CSP, `limit=2`) until the wanted number of givens (25/30/35) |
| 9x9 mosaic (252x252) | each given cell = a random real MNIST test image of that digit; blank cell = Gaussian noise (std 0.02, clipped to 0-1) |
| Blank-cell detection | brightness rule, not the NN: brightest pixel < 0.5 -> blank |
| Injected "hard digits" | for k misreads, k given cells are replaced by a test image of the same true digit that the NN misreads and for which the true digit is ranked 2nd or 3rd. For k >= 1 the draw is repeated (<= 30 times) until the first reading is unsolvable, so that there is something to correct |
| NN probabilities | softmax of the MLP logits divided by the fitted temperature T (temperature scaling on validation) |

This is **not** photographed real-world Sudoku: only the digit images are real.
