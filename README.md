# Bean Disease Detector

End-to-end workflow for classifying bean leaf diseases from images. The project trains a MobileNetV2-based image classifier on four classes (`als`, `bean_rust`, `healthy`, `unknown`) and ships a Streamlit interface for easy, local predictions.

## Project Structure
- `Classification/` — training/validation/test image splits organized by class.
- `train_model.py` — CPU-friendly training pipeline (data aug, fine-tuning, reports).
- `models/` — saved weights (`bean_disease_final_model.h5`), class mapping, and training artifacts.
- `streamlit_app.py` — web UI to upload a leaf photo and view predictions.
- `SETUP.md` — quick environment notes.
- `bean_disease_classification_training.ipynb` — notebook version of the training flow.

## Prerequisites
- Python 3.8–3.12 (TensorFlow compatibility). On Windows, `py -3.10` or `py -3.11` works well.
- `pip` for dependency installs.

## Setup
1. (Recommended) create and activate a virtual environment.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify TensorFlow:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## Training the Classifier
The dataset is already organized under `Classification/`:
```
Classification/
  training/als|bean_rust|healthy|unknown/...
  validation/...
  test/...
```
Run the scripted training (generates checkpoints and reports):
```bash
python train_model.py
```
Artifacts land in `models/`:
- `bean_disease_best_model.h5` (best val. accuracy)
- `bean_disease_final_model.h5` (exported final model)
- `class_mapping.json` (class names, indices, image size)
- `training_history.png`, `confusion_matrix.png`

Notes:
- Defaults: `IMG_SIZE=224`, `BATCH_SIZE=16`, `EPOCHS=30` + 10 fine-tune.
- If memory is tight, lower `BATCH_SIZE` or `IMG_SIZE` inside `train_model.py`.
- GPU is optional; script is tuned for CPU.

## Running the Streamlit App
Ensure the following exist in `models/`: `bean_disease_final_model.h5`, `class_mapping.json`.
```bash
streamlit run streamlit_app.py
# or on Windows with a specific Python version
py -3.10 -m streamlit run streamlit_app.py
```
Usage:
- Upload a leaf photo (jpg/png) or drag one from `Classification/test/<class>/`.
- View predicted class, confidence, and class probability table.
- Adjust the sidebar confidence threshold to highlight low-confidence cases.

## Troubleshooting
- TensorFlow install issues: stick to Python 3.8–3.12; try `pip install tensorflow-cpu` or `pip install tensorflow==2.15.0`.
- Out-of-memory during training: reduce `BATCH_SIZE`, lower `IMG_SIZE`, or skip fine-tuning (Phase 2).
- Missing model files when launching the app: run `train_model.py` first or place the provided artifacts in `models/`.

## Extending
- Add new disease classes by introducing new subfolders under each `Classification/{training,validation,test}/` split, then retrain.
- The `Detection/` dataset is included for future object-detection experiments; current scripts focus on classification.


