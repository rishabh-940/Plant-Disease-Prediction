import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# Paths are relative to this file (app/main.py).
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model" / "plant_disease_prediction_model.h5"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"
IMAGE_SIZE = (224, 224)  # Matches the CNN training notebook.

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    # compile=False is suitable for inference and avoids requiring the
    # original optimizer/loss configuration to be restored.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    expected_input = (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    if tuple(model.input_shape) != expected_input:
        raise ValueError(
            f"Unexpected model input shape {model.input_shape}; "
            f"expected {expected_input}."
        )

    return model


@st.cache_data
def load_class_indices():
    if not CLASS_INDICES_PATH.is_file():
        raise FileNotFoundError(
            f"Class mapping file not found: {CLASS_INDICES_PATH}"
        )

    with CLASS_INDICES_PATH.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    # The notebook creates {integer_index: class_name}; JSON converts
    # integer keys to strings, e.g. {"0": "Apple___Apple_scab", ...}.
    try:
        mapping = {int(index): str(class_name)
                   for index, class_name in raw_mapping.items()}
    except (AttributeError, TypeError, ValueError):
        raise ValueError(
            "class_indices.json must map numeric class indices to class names."
        )

    if not mapping:
        raise ValueError("class_indices.json contains no classes.")

    return mapping


def preprocess_image(image):
    # flow_from_directory in the training notebook loads RGB images,
    # resizes to 224x224 using its default 'nearest' interpolation,
    # and rescales pixel values by 1/255.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image_class(model, image, class_indices):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(
            f"Unexpected model output shape: {predictions.shape}"
        )

    if predictions.shape[1] != len(class_indices):
        raise ValueError(
            f"Model returns {predictions.shape[1]} classes, but "
            f"class_indices.json contains {len(class_indices)} classes."
        )

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_indices.get(predicted_index)
