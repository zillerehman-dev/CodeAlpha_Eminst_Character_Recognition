"""
Shared preprocessing pipeline for EMNIST-Letters.
MUST be used identically for training, validation, test, and Streamlit inference.

Verified empirically against the real emnist-letters-train-images/labels files:
raw IDX images are stored transposed relative to the human-readable orientation.
A simple 2D transpose (image.T) restores the correct upright orientation and was
confirmed by rendering images against their known labels (see orientation_check.png).
"""
import numpy as np

LABEL_OFFSET = 1  # EMNIST Letters labels are 1-26, not 0-25
NUM_CLASSES = 26


def raw_label_to_class_index(label):
    """Convert EMNIST Letters raw label (1-26) to zero-based class index (0-25)."""
    return np.asarray(label).astype(np.int64) - LABEL_OFFSET


def class_index_to_letter(class_index):
    return chr(ord("A") + int(class_index))


def fix_orientation(images):
    """
    Correct EMNIST's transposed storage orientation.
    Accepts (N, 28, 28) or (28, 28) uint8/float arrays and returns the same
    shape with each image transposed on its last two spatial axes.
    """
    images = np.asarray(images)
    if images.ndim == 2:
        return images.T
    if images.ndim == 3:
        return np.transpose(images, (0, 2, 1))
    raise ValueError(f"Expected 2D or 3D array, got shape {images.shape}")


def preprocess_images(images, already_oriented=False):
    """
    Full deterministic preprocessing pipeline:
      1. Fix orientation (unless already applied)
      2. Cast to float32
      3. Normalize [0,255] -> [0,1]
      4. Add channel dimension -> (..., 28, 28, 1)
    No statistics are fit from data (min/max normalization is a fixed constant),
    so this is safe to apply identically to train/val/test/inference data.
    """
    images = np.asarray(images)
    if not already_oriented:
        images = fix_orientation(images)
    images = images.astype("float32") / 255.0
    images = images[..., np.newaxis]
    return images


def preprocess_single_image_array(arr_28x28, already_oriented=False):
    """Preprocess a single (28,28) grayscale array (e.g. from Streamlit) into
    a (1, 28, 28, 1) batch ready for model.predict()."""
    arr = np.asarray(arr_28x28)
    if arr.shape != (28, 28):
        raise ValueError(f"Expected a (28,28) array, got {arr.shape}")
    x = preprocess_images(arr[np.newaxis, ...], already_oriented=already_oriented)
    return x
