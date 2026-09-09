"""
Reusable IDX (ubyte) loader for MNIST/EMNIST-format datasets.
Handles both plain and gzip-compressed IDX files.
Verifies magic numbers before returning data (fails loudly on corruption/mismatch).
"""
import gzip
import struct
import numpy as np


def _open_maybe_gzip(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def load_idx_images(path):
    """Load an IDX3 image file (magic number 2051) into a (N, H, W) uint8 array."""
    with _open_maybe_gzip(path) as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(
                f"Invalid IDX3 magic number for {path}: got {magic}, expected 2051"
            )
        buf = f.read(n_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(n_images, rows, cols)
    return data


def load_idx_labels(path):
    """Load an IDX1 label file (magic number 2049) into a (N,) uint8 array."""
    with _open_maybe_gzip(path) as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(
                f"Invalid IDX1 magic number for {path}: got {magic}, expected 2049"
            )
        buf = f.read(n_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


if __name__ == "__main__":
    imgs = load_idx_images("/home/claude/proj/train-images.idx3-ubyte")
    labs = load_idx_labels("/home/claude/proj/train-labels.idx1-ubyte")
    test_labs = load_idx_labels("/home/claude/proj/test-labels.idx1-ubyte")

    print("Train images shape:", imgs.shape, imgs.dtype)
    print("Train labels shape:", labs.shape, labs.dtype)
    print("Test labels shape :", test_labs.shape, test_labs.dtype)
    print("Unique train labels:", sorted(np.unique(labs).tolist()))
    print("Unique test labels :", sorted(np.unique(test_labs).tolist()))
    print("Pixel min/max:", imgs.min(), imgs.max())
