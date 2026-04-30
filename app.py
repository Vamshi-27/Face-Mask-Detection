import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("face_mask_model.h5")

# Title
st.title("😷 Face Mask Detection")

st.write("Upload an image to check if the person is wearing a mask.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    # Prediction
    prediction = model.predict(img)

    # Classification (adjust if your class mapping is different)
    if prediction[0][0] > 0.5:
        confidence = prediction[0][0] * 100
        st.error(f"❌ No Mask Detected ({confidence:.2f}%)")
    else:
        confidence = (1 - prediction[0][0]) * 100
        st.success(f"✅ Mask Detected ({confidence:.2f}%)")