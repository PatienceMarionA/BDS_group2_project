"""
Streamlit interface for Bean Disease Classification.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Constants and cached loaders 
# ---------------------------------------------------------------------------

MODEL_PATH = Path("models/bean_disease_final_model.h5")
CLASS_MAP_PATH = Path("models/class_mapping.json")
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@st.cache_data(show_spinner=False)
def load_class_mapping(mapping_path: Path) -> Dict[str, Any]:
    """Load class mapping from JSON file.
    
    Args:
        mapping_path: Path to the JSON file containing class mappings.
        
    Returns:
        Dictionary containing class mapping information.
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping


@st.cache_resource(show_spinner=True)
def load_model(model_path: Path) -> tf.keras.Model:
    """Load the trained Keras model.
    
    Args:
        model_path: Path to the saved Keras model file.
        
    Returns:
        Loaded Keras model instance.
    """
    return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------------
# Code for Prediction utilities
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image, target_size: int) -> np.ndarray:
    """Preprocess image for model prediction.
    
    Args:
        image: PIL Image to preprocess.
        target_size: Target size for resizing (width and height).
        
    Returns:
        Preprocessed image array with shape (1, target_size, target_size, 3).
    """
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))
    array = np.array(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def predict(
    model: tf.keras.Model,
    image: Image.Image,
    class_names: List[str],
    img_size: int,
) -> Tuple[str, float, Dict[str, float]]:
    """Predict disease class from an image.
    
    Args:
        model: Trained Keras model for prediction.
        image: PIL Image to classify.
        class_names: List of class names in order.
        img_size: Target image size for preprocessing.
        
    Returns:
        Tuple containing predicted label, confidence score, and all class probabilities.
    """
    tensor = preprocess_image(image, img_size)
    probs = model.predict(tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_label = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    all_probs = {label: float(prob) for label, prob in zip(class_names, probs)}
    return predicted_label, confidence, all_probs


# ---------------------------------------------------------------------------
# Streamlit layout helpers
# ---------------------------------------------------------------------------

def show_sidebar_info(class_names: List[str]) -> None:
    """Display sidebar information including usage instructions and class descriptions.
    
    Args:
        class_names: List of class names to display.
    """
    st.sidebar.header("How to Use")
    st.sidebar.markdown(
        """
1. Train (or reuse) the model via `train_model.py`.
2. Ensure the following files exist in `models/`:
   - `bean_disease_final_model.h5`
   - `class_mapping.json`
3. Upload a bean leaf photo or pick one from the dataset preview.
4. Review the probability chart and confidence indicator.
        """
    )

    st.sidebar.header("Classes")
    descriptions = {
        "als": "Angular Leaf Spot",
        "bean_rust": "Bean Rust",
        "healthy": "Healthy Leaf",
        "unknown": "Unknown/Other issues",
    }
    for label in class_names:
        st.sidebar.markdown(f"- **{label}** — {descriptions.get(label, 'N/A')}")

    st.sidebar.header("Settings")
    st.sidebar.markdown(
        "Adjust the confidence threshold to highlight high-certainty predictions."
    )


def show_prediction_results(
    image: Image.Image,
    label: str,
    confidence: float,
    probabilities: Dict[str, float],
    threshold: float,
) -> None:
    """Display prediction results including image, label, and probability table.
    
    Args:
        image: The uploaded image.
        label: Predicted class label.
        confidence: Confidence score for the prediction.
        probabilities: Dictionary mapping class names to probabilities.
        threshold: Confidence threshold for highlighting predictions.
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Input image", use_column_width=True)

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted Class", label, f"{confidence*100:.2f}% confidence")
        if confidence < threshold:
            st.warning(
                "Confidence below threshold. Consider capturing another image or "
                "reviewing with an agronomist."
            )
        else:
            st.success("Confidence exceeds threshold.")

    st.subheader("Class Probabilities")
    prob_table = [
        {"Class": cls, "Probability (%)": f"{prob * 100:.2f}"}
        for cls, prob in probabilities.items()
    ]
    st.table(prob_table)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Bean Disease Detector",
        page_icon="🌱",
        layout="wide",
    )
    st.title("Bean Disease Detector")
    st.caption("Upload a bean leaf photo to diagnose ALS, bean rust, or other issues.")

    if not MODEL_PATH.exists() or not CLASS_MAP_PATH.exists():
        st.error(
            "Model artifacts not found. Please run `train_model.py` first to "
            "generate `models/bean_disease_final_model.h5` and `class_mapping.json`."
        )
        return

    class_mapping = load_class_mapping(CLASS_MAP_PATH)
    class_names = class_mapping["class_names"]
    img_size = int(class_mapping.get("img_size", 224))

    show_sidebar_info(class_names)

    with st.spinner("Loading model..."):
        model = load_model(MODEL_PATH)

    threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
    )

    st.header("Upload an Image")
    uploaded_file = st.file_uploader(
        "Upload a leaf photo (jpg/png)", type=["jpg", "jpeg", "png"]
    )

    example_zone = st.expander("Need a sample?", expanded=False)
    with example_zone:
        st.markdown(
            "Browse to `Classification/test/<class_name>/` and drag a sample file here."
        )

    if uploaded_file is None:
        st.info("Upload an image to start the diagnosis.")
        st.stop()

    image = Image.open(uploaded_file)
    predicted_label, confidence, probabilities = predict(
        model, image, class_names, img_size
    )

    show_prediction_results(
        image, predicted_label, confidence, probabilities, threshold
    )

    st.subheader("Raw Probabilities")
    st.json(probabilities, expanded=False)

    st.subheader("Notes")
    st.markdown(
        """
- Results are intended to guide scouting decisions, not replace expert diagnosis.
- Capture photos in good lighting and focus on the infected area for better accuracy.
- Consider retraining if new disease classes are added.
        """
    )


if __name__ == "__main__":
    main()

