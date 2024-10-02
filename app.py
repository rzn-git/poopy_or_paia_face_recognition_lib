import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import pickle
import os

# Path to the saved face encodings
ENCODINGS_FILE = 'model/poopy-not-poopy.pkl'

def load_face_encodings(file_path=ENCODINGS_FILE):
    """
    Load known face encodings and names from a pickle file.
    
    :param file_path: Path to the pickle file containing face encodings and names.
    :return: Two lists - known face encodings and known face names.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    else:
        st.error("Model file not found. Please run the training script first.")
        return [], []

# Load known face encodings and names
known_face_encodings, known_face_names = load_face_encodings()

def preprocess_image(img):
    """
    Preprocess the uploaded image for face recognition.
    
    :param img: Uploaded image.
    :return: Processed image as a numpy array.
    """
    img = img.resize((150, 150))  # Resize image to a consistent size
    img_array = np.array(img)      # Convert the image to an array
    return img_array

# Streamlit app setup
st.title("Who are you madafakah!!! 🫵🏻")
st.title("Poopy or Not Poopy? 🤷🏻‍♀️")
st.write("Upload an image to recognize poopies")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    try:
        # Load the image and find face encodings
        unknown_image = face_recognition.load_image_file(uploaded_file)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)

        if unknown_face_encodings:
            # Compare the found face encodings with known encodings
            face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encodings[0])
            best_match_index = np.argmin(face_distances)
            name = known_face_names[best_match_index]
            confidence = (1 - face_distances[best_match_index]) * 100

            # Special case handling
            if name == 'Prapty':
                name = 'Bubblegum Babu 🐣'

            st.write(f"**Prediction:** {name}")
            st.write(f"**Confidence:** {confidence:.0f}%")
        else:
            st.write("No face detected in the image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload an image.")
