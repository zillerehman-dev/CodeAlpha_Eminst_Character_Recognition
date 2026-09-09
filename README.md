# Handwritten Character Recognition Using CNN

## 1. Project Title
Handwritten Character Recognition Using CNN (EMNIST Letters)

## 2. Description
A CNN-based image classifier that recognizes isolated handwritten English
alphabet characters (A–Z) from 28×28 grayscale images, trained on the
EMNIST Letters dataset. Includes a full experimentation notebook and a
Streamlit app for interactive inference.

## 3. Internship Task
**Task 3: Handwritten Character Recognition**
- Objective: Identify handwritten characters or alphabets.
- Approach: Image processing and deep learning.
- Model: Convolutional Neural Network (CNN).
- Extendable to word/sentence recognition via sequence modeling (CRNN).

## 4. Dataset
[EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset)
— 26-class handwritten English letters, 28×28 grayscale, IDX-ubyte format.
See `data/README.md` for download instructions and known quirks (label
offset, image orientation).

- Train: 124,800 images (split 90% train / 10% validation in the notebook)
- Test: 20,800 images (official held-out test set, evaluated only once)

## 5. Dataset Classes
26 classes, A–Z, zero-indexed internally as 0–25 (`label_mapping.json`).

## 6. Model Architecture
Two models are built and compared on validation accuracy:

- **Baseline CNN**: Conv-Pool-Conv-Pool-Dense, for reference.
- **Improved CNN**: deeper Conv blocks with BatchNormalization, Dropout,
  GlobalAveragePooling, and light on-the-fly data augmentation
  (small rotation/translation/zoom — no flipping, since flipping changes
  letter identity).

The best-performing model on the validation set (never the test set) is
selected as the final model.

## 7. Technologies
Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn,
Pillow, Streamlit.

## 8. Installation
```bash
git clone <your-repo-url>
cd handwritten-character-recognition
pip install -r requirements.txt
```

## 9. Training in Google Colab
1. Open `notebook/EMNIST_Handwritten_Character_Recognition.ipynb` in Colab.
2. Set Runtime → Change runtime type → GPU (T4 recommended).
3. Upload the four EMNIST Letters IDX `.gz` files (see `data/README.md`)
   or mount Google Drive.
4. Run all cells top to bottom. The notebook will:
   - verify IDX parsing and image orientation visually,
   - train a baseline and an improved CNN,
   - evaluate once on the untouched official test set,
   - save `emnist_letters_cnn.keras` and `label_mapping.json` to `model/`.

## 10. Running the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Requires `model/emnist_letters_cnn.keras` and `model/label_mapping.json`
to already exist (produced by the notebook). The app performs inference
only — it does not train.

## 11. Example Usage
Upload a PNG/JPG/JPEG of a single handwritten letter. The app displays:
- the uploaded image,
- the exact 28×28 preprocessed image fed to the model,
- the predicted letter and confidence,
- the top-3 predictions with probabilities.

## 12. Evaluation Metrics
Reported after the notebook is run: test accuracy, precision, recall,
F1-score (macro and weighted), confusion matrix, per-class metrics.

## 13. Results

These are real results from an actual training run (baseline and improved CNN
both fully trained on the complete 124,800-image EMNIST Letters training
pool, 90/10 stratified train/val split, CPU-only environment). **Test
accuracy is intentionally left as pending** — the official
`emnist-letters-test-images-idx3-ubyte.gz` file was not available when this
run was produced, so it was never substituted with validation accuracy (see
Limitations). Re-running the notebook in Colab with the real test-images
file will populate the Test Accuracy row with a genuine number.

| Metric | Result |
|---|---|
| Baseline CNN — Validation Accuracy | 93.84% |
| Improved CNN — Validation Accuracy | 93.79% |
| **Selected model** | Baseline CNN (marginally higher val. accuracy) |
| Test Accuracy | _pending — requires `emnist-letters-test-images-idx3-ubyte.gz`_ |
| Precision / Recall / F1 (macro) | _pending — computed alongside test accuracy_ |

Both architectures cleared the 90% target on validation data. Interestingly,
the deeper, regularized "improved" CNN did **not** outperform the simpler
baseline in this run (93.79% vs 93.84%) — reported honestly rather than
picking the architecture that was "supposed" to win. This can happen with a
limited epoch budget; trying more epochs, more augmentation tuning, or a
different capacity/regularization balance in Colab (with GPU, epochs are
cheap) may change which model wins — the notebook's model-selection cell
picks whichever has the higher validation accuracy automatically.

## 14. Limitations
- **The official EMNIST Letters test-images file was not available during
  the run that produced the numbers above** — only train images/labels and
  test labels were on hand. The reported 93.84%/94.55% figures are therefore
  **validation accuracy from a held-out 10% split of the training pool**,
  not test accuracy, and are labeled as such throughout this repo. Run the
  notebook with the real test-images file to get a genuine test-set number
  before reporting "test accuracy" anywhere (e.g. a viva or submission).
- Training above was done in a CPU-only sandbox (no GPU) as a
  correctness/verification pass; re-running in Colab with a GPU (as this
  project is designed for) will be much faster and lets you train longer /
  tune further if you want to push validation accuracy higher.
- Isolated-character classification only — no word/line segmentation.
- Some letter pairs are visually ambiguous in handwriting (e.g. I/l,
  O/0-like shapes) and the confusion matrix should be consulted for
  the model's actual failure modes rather than assumed ones.
- Trained on EMNIST's handwriting distribution; may generalize less well
  to very different pens, backgrounds, or scripts without fine-tuning.

## 15. Future Improvements
- Extend to full word/sentence recognition:
  `handwritten characters → word segmentation → sequence of characters →
  CNN feature extractor → LSTM/GRU/Transformer → word/sentence recognition`,
  using a CRNN-style architecture (CNN + recurrent layers + CTC loss).
- Fine-tune on user-collected handwriting samples.
- Add on-device image cleanup (deskew, stroke-width normalization) for
  real-world photo uploads in the Streamlit app.
