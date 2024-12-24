import tensorflow as tf
import numpy as np
import streamlit as st
import os
import pickle

# Path to the model file
model_path = 'Klasifikasi_gambar.sav'

# Check if the model file exists before loading
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error(f"Model file '{model_path}' not found. Please ensure it is in the correct directory.")

# List of categories
data_cat = ['apple', 'banana', 'bell pepper', 'cabbage', 'carrot', 'corn', 'cucumber', 
            'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 
            'mango', 'onion', 'orange', 'peas', 'pineapple', 'potato', 'spinach', 'sweetcorn', 
            'sweetpotato', 'tomato', 'watermelon']

# Image dimensions
img_height = 180
img_width = 180

# --- Streamlit UI ---
st.set_page_config(page_title="Image Classifier", page_icon="📸", layout="centered")

# Main title
st.title("📸 Pengklasifikasi Gambar Sayuran & Buah")
st.markdown("Unggah gambar atau ambil foto untuk mengklasifikasikan apakah itu sayuran atau buah!")

# Sidebar for image upload
st.sidebar.header("📂 Unggah Gambar")

# File uploader
uploaded_file = st.sidebar.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

# Camera control
camera_open = st.sidebar.checkbox("Buka Kamera")  # Checkbox to open/close camera
camera_input = None

if camera_open:
    camera_input = st.sidebar.camera_input("Ambil Foto")

# Header and Description
st.header("🌟 Hasil Prediksi")
st.markdown("---")

# Use uploaded file or camera input
if uploaded_file is not None:
    image = uploaded_file
elif camera_input is not None:
    image = camera_input
else:
    image = None

if image is not None:
    try:
        # Load and preprocess the image
        image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image)
        img_bat = tf.expand_dims(img_arr, axis=0)  # Add batch dimension

        # Prediction
        predictions = model.predict(img_bat)  # Get raw predictions
        label = data_cat[np.argmax(predictions)]  # Get predicted label
        confidence = np.max(predictions) * 100  # Calculate confidence

        # Display results in Streamlit
        st.image(image, caption="Input Image", use_column_width=True)
        st.success(f"**Prediction**: {label}")
        st.info(f"**Accuracy**: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah gambar atau ambil foto untuk melihat hasil prediksi.")

# Footer with correct GitHub link
st.markdown("---")
st.markdown("Developed by [Kelompok Sauqi, Ayub, Fulvian](https://github.com/your-profile) | Powered by TensorFlow & Streamlit")
