
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# ------------------------------------------
# 1. Project paths
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "trained_model"
    / "plant_disease_prediction_model.h5"
)

CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"

IMAGE_SIZE = (224, 224)


# ------------------------------------------
# 2. Streamlit page configuration
# ------------------------------------------

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)


# ------------------------------------------
# 3. Load TensorFlow model
# ------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ------------------------------------------
# 4. Load class indices
# ------------------------------------------

@st.cache_data
def load_class_indices():

    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Class indices file not found: {CLASS_INDICES_PATH}"
        )

    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as file:
        class_indices = json.load(file)

    return class_indices


# ------------------------------------------
# 5. Image preprocessing
# ------------------------------------------

def preprocess_image(image):

    # Correct image orientation
    image = ImageOps.exif_transpose(image)

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # Convert to NumPy array
    image_array = np.asarray(image, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ------------------------------------------
# 6. Prediction function
# ------------------------------------------

def predict_image_class(model, image, class_indices):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))

    # Support both index-to-class and class-to-index mappings
    if str(predicted_index) in class_indices:
        predicted_class = class_indices[str(predicted_index)]

    else:
        reverse_indices = {
            int(index): name
            for name, index in class_indices.items()
        }

        predicted_class = reverse_indices.get(
            predicted_index,
            f"Class {predicted_index}"
        )

    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_class, confidence


# ------------------------------------------
# 7. Main Streamlit application
# ------------------------------------------

def main():

    st.title("🌿 Plant Disease Classifier")

    st.write(
        "Upload a plant leaf image to identify "
        "the possible disease using a trained deep learning model."
    )

    st.divider()

    # Upload image
    uploaded_image = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        try:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(
                    image,
                    caption="Input Plant Image",
                    use_container_width=True
                )

            with col2:
                st.subheader("Classification")

                if st.button(
                    "Classify Image",
                    use_container_width=True
                ):

                    with st.spinner("Analyzing plant image..."):

                        model = load_model()
                        class_indices = load_class_indices()

                        prediction, confidence = predict_image_class(
                            model,
                            image,
                            class_indices
                        )

                    st.success(
                        f"Prediction: {prediction}"
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

        except Exception as e:

            st.error(
                f"An error occurred: {str(e)}"
            )


# ------------------------------------------
# 8. Run application
# ------------------------------------------

if __name__ == "__main__":
    main()
