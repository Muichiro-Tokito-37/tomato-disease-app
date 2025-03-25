import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom CSS for gradient background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #d4145a, #feb47b); /* Gradient from orange to light orange */
        font-family: 'Pacifico', cursive;
        color: white; /* Default text color */
    }
    h1, h2, h3 {
        color: white; /* Headings color */
    }

    div.stFileUploader {
        background: linear-gradient(to bottom right, #78ffb6, #a7ff7a); /* Gradient background */
        border-radius: 15px; /* Rounded corners */
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); /* Soft shadow */
        font-family: 'Arial', sans-serif;
    }
    div.stFileUploader > label {
        color: black;
        font-size: 18px;
    }
    div.stFileUploader div div div button {
        background-color: ##274e13; /* Button background */
        color: #ff7e5f; /* Button text */
        border-radius: 10px;
        border: none;
        font-size: 16px;
        padding: 10px 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); /* Shadow on button */
        transition: all 0.3s ease-in-out;
    }
    div.stFileUploader div div div button:hover {
        background-color: #feb47b;
        color: white;
        transform: scale(1.05); /* Slight hover effect */
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved model (Update the path if needed)
MODEL_PATH = r"C:\a a Tomato-Project\training\tomato_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (Make sure they match your dataset)
class_names = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
               'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 
               'Tomato_mosaic_virus', 'Healthy']

# Function to preprocess and predict
def predict_image(image):
    img = image.convert("RGB")  # Convert to RGB to ensure 3 channels
    if image.mode == "RGBA":
        image = image.convert("RGB")  # Remove alpha channel
    elif image.mode == "L":
        image = image.convert("RGB")  # Convert grayscale to RGB
    img = image.resize((224, 224))  # Resize to match model's expected input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to match batch size
    
    predictions = model.predict(img_array)  # Make prediction
    predicted_class = class_names[np.argmax(predictions[0])]  # Get class label
    confidence = round(100 * np.max(predictions[0]), 2)  # Confidence score

    return predicted_class, confidence

# Streamlit UI
st.title("🍅 Tomato Leaf Disease Classification")
st.write("Upload an image of a tomato leaf, and the model will classify the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open image
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Show image
    
    st.write("Classifying...")
    predicted_class, confidence = predict_image(image)
    
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence}%")
