import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# -------------------------------
# Download model from Google Drive (only once)
# -------------------------------
model_path = "face_mask_model.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1pWUbVHHN1DQ1NCw0MYVU-KjginRGrgPX"  # <-- replace this
    gdown.download(url, model_path, quiet=False, fuzzy=True)

# -------------------------------
# Load model
# -------------------------------
model = load_model(model_path)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("😷 Face Mask Detection")

st.write("Upload an image to check if the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Convert to numpy
    img = np.array(image)

    # Fix image channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize and normalize
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    # Prediction
    prediction = model.predict(img)

    # Show raw prediction (optional debug)
    st.write(f"Prediction Value: {prediction[0][0]:.4f}")

    # Classification (adjust if class mapping differs)
    if prediction[0][0] > 0.5:
        confidence = prediction[0][0] * 100
        st.error(f"❌ No Mask Detected ({confidence:.2f}%)")
    else:
        confidence = (1 - prediction[0][0]) * 100
        st.success(f"✅ Mask Detected ({confidence:.2f}%)")
