# Dataset

This project uses the **EMNIST Letters** split (26 handwritten English letter
classes, 28×28 grayscale, IDX-ubyte format).

Raw dataset files are **not** committed to this repository (see `.gitignore`)
because they are large. Download them yourself and place them in this folder:

- `emnist-letters-train-images-idx3-ubyte.gz`
- `emnist-letters-train-labels-idx1-ubyte.gz`
- `emnist-letters-test-images-idx3-ubyte.gz`
- `emnist-letters-test-labels-idx1-ubyte.gz`

## Where to get it

- Official NIST source: https://www.nist.gov/itl/products-and-services/emnist-dataset
- Mirror (commonly used): https://www.kaggle.com/datasets/crawford/emnist

## Known dataset shape (EMNIST Letters)

| File | Samples | Notes |
|---|---|---|
| train-images / train-labels | 124,800 | split 90/10 into train/val in the notebook |
| test-images / test-labels | 20,800 | held out, used only for final evaluation |

Labels in the raw files are **1–26** (not 0–25). The notebook and
`utils/preprocessing.py` convert them to zero-based class indices
(0=A ... 25=Z) — do not assume they are already zero-based.


