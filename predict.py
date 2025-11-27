import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model once (cache to avoid reloading every time)
@st.cache_resource
def load_cattle_model():
    return load_model("cattle_buffalo_model.h5")

model = load_cattle_model()
class_names = ["Gir", "Holstein", "Sahiwal", "Murrah", "Jaffarabadi", "Surti"]

st.title("🐄 Cattle & Buffalo Breed Detector")

# File uploader
uploaded_file = st.file_uploader("Upload an image of cattle/buffalo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save and preprocess
    img = image.load_img(uploaded_file, target_size=(224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    pred = model.predict(x)
    breed = class_names[np.argmax(pred)]

    st.success(f"✅ Predicted Breed: **{breed}**")
