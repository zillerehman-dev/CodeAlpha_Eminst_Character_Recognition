"""
Streamlit inference app for the EMNIST-Letters handwritten character recognizer.

IMPORTANT: this app performs INFERENCE ONLY. The model is trained offline
(in the companion notebook / Google Colab) and loaded here from disk.
Preprocessing here must match `utils/preprocessing.py` exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.preprocessing import preprocess_single_image_array  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "emnist_letters_cnn.keras"
LABELS_PATH = Path(__file__).resolve().parent.parent / "model" / "label_mapping.json"

st.set_page_config(page_title="Handwritten Character Recognition", page_icon="✍️", layout="centered")


@st.cache_resource
def load_model_and_labels():
    """Load the trained model and label mapping once per server process."""
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model in the notebook first "
            "and copy emnist_letters_cnn.keras into the model/ folder."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found at {LABELS_PATH}. Copy label_mapping.json into model/."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        raw_map = json.load(f)
    label_map = {int(k): v for k, v in raw_map.items()}
    return model, label_map


def to_emnist_style_28x28(pil_image: Image.Image) -> np.ndarray:
    """
    Convert an arbitrary uploaded image (RGB/RGBA/etc, any size) into a
    28x28 grayscale array matching EMNIST's pixel convention:
    white/bright strokes on a black background, already in the "raw"
    (un-oriented) EMNIST layout expected by preprocess_single_image_array.

    Real-world photos are usually black strokes on a white background,
    which is the OPPOSITE of EMNIST's convention, so we auto-detect and
    invert only when needed (based on mean brightness) rather than
    blindly inverting every image.
    """
    img = pil_image.convert("L")  # grayscale

    # Resize to 28x28 (EMNIST's native resolution) with high-quality resampling
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32")

    # EMNIST convention: background ~0 (black), strokes ~255 (white).
    # If the uploaded image looks like typical paper (bright background,
    # dark ink), the mean pixel value will be high -> invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    return arr.astype("uint8")


def main():
    st.title("Handwritten Character Recognition")
    st.write(
        "Upload an image of a handwritten English alphabet character and the CNN will "
        "predict the character."
    )

    try:
        model, label_map = load_model_and_labels()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception:
        st.error("The model could not be loaded. Please check that the model file is valid.")
        st.stop()

    uploaded = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Waiting for an image upload...")
        return

    try:
        pil_image = Image.open(uploaded)
        pil_image = ImageOps.exif_transpose(pil_image)  # respect camera orientation metadata
    except Exception:
        st.error("Could not read this file as an image. Please upload a valid PNG/JPG/JPEG.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        st.image(pil_image, use_container_width=True)

    try:
        raw_28x28 = to_emnist_style_28x28(pil_image)
        # already_oriented=True: our own 28x28 array is already upright;
        # it should NOT go through the EMNIST raw-file transpose fix.
        batch = preprocess_single_image_array(raw_28x28, already_oriented=True)
    except Exception:
        st.error("Failed to preprocess this image. Try a clearer, centered image of a single letter.")
        return

    with col2:
        st.subheader("What the model sees (28x28)")
        st.image(raw_28x28, use_container_width=True, clamp=True)

    try:
        probs = model.predict(batch, verbose=0)[0]
    except Exception:
        st.error("Model inference failed unexpectedly. Please try a different image.")
        return

    top_idx = np.argsort(probs)[::-1][:3]
    pred_class = int(top_idx[0])
    pred_letter = label_map.get(pred_class, "?")
    pred_conf = float(probs[pred_class]) * 100

    st.markdown("---")
    st.subheader("Prediction")
    st.markdown(f"## **{pred_letter}**")
    st.metric("Confidence", f"{pred_conf:.2f}%")

    st.subheader("Top 3 predictions")
    for idx in top_idx:
        letter = label_map.get(int(idx), "?")
        conf = float(probs[idx]) * 100
        st.write(f"**{letter}** — {conf:.2f}%")
        st.progress(min(max(conf / 100, 0.0), 1.0))


if __name__ == "__main__":
    main()
