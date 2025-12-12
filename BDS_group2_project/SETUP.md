# Setup Instructions for Bean Disease Classification

## Python Version Requirement

**Important**: TensorFlow currently supports Python 3.8-3.12. If you're using Python 3.14, you'll need to use Python 3.11 or 3.12.

### Check your Python version:
```bash
python --version
```

### If you need to install a compatible Python version:
1. Download Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/)
2. Or use a virtual environment with the correct Python version

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
```

### 3. Run Training

**Option A: Using the Python script (recommended)**
```bash
python train_model.py
```

**Option B: Using Jupyter Notebook**
```bash
jupyter notebook bean_disease_classification_training.ipynb
```

## Training Time

- **CPU Training**: Expect 2-6 hours depending on your CPU
- **Batch Size**: Set to 16 for CPU (can reduce to 8 if memory issues)
- **Epochs**: 30 for Phase 1, 10 for Phase 2

## Troubleshooting

### If TensorFlow installation fails:
1. Make sure you're using Python 3.8-3.12
2. Try: `pip install tensorflow-cpu` (CPU-only version)
3. Or: `pip install tensorflow==2.15.0` (specific version)

### If you run out of memory:
- Reduce `BATCH_SIZE` to 8 in the script
- Reduce `IMG_SIZE` to 192
- Skip fine-tuning phase (Phase 2)

## Streamlit Interface

Once training artifacts exist in `models/`, launch the interface:

```bash
streamlit run streamlit_app.py
```

If you rely on the Python 3.10 environment (for TensorFlow compatibility), use:

```bash
py -3.10 -m streamlit run streamlit_app.py
```

The app expects:
- `models/bean_disease_final_model.h5`
- `models/class_mapping.json`

Upload a bean leaf photo to view predictions and probability breakdowns.

