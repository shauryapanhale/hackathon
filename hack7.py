import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\yasha\OneDrive\Desktop\PBL_PROJECT\outfit_recommender.h5")

# Load data
@st.cache_data
def load_data():
    csv_path = r"C:\Users\yasha\OneDrive\Desktop\PBL_PROJECT\final_dataset.csv"
    image_folder = r"C:\Users\yasha\OneDrive\Desktop\PBL_PROJECT\datasets\train_images"
    data = pd.read_csv(csv_path)
    return data, image_folder

model = load_model()
data, image_folder = load_data()

# Debug: Display column names to identify issues
st.write("Dataset Columns:", data.columns.tolist())

# App title
st.title("Fashion Recommendation System")

# Quiz questions
st.header("Take the Quiz to Find Your Perfect Outfit")

age = st.slider("What's your age?", 10, 60, 25)
gender = st.radio("What's your gender?", ["Male", "Female"])
style = st.selectbox("Choose your preferred style:", data["PredictedStyle"].unique())
color = st.color_picker("Pick your favorite color:")

# Convert hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

selected_color_rgb = hex_to_rgb(color)

# Filter data based on user input
if st.button("Get Recommendations"):
    st.header("Your Recommendations")

    # Verify column names and adjust filtering logic
    gender_column = "gender"
    style_column = "PredictedStyle"
    image_column = "Filename"
    color_column = "Color"

    # Ensure columns exist
    if gender_column not in data.columns or style_column not in data.columns or image_column not in data.columns or color_column not in data.columns:
        st.error("Required columns are missing in the dataset. Please check the column names.")
    else:
        # Filter dataset by gender and style
        filtered_data = data[(data[gender_column] == gender) & (data[style_column] == style)]

        if not filtered_data.empty:
            # Calculate color similarity
            def calculate_color_distance(row_color):
                row_rgb = tuple(map(int, row_color.strip('()').split(',')))
                return np.linalg.norm(np.array(row_rgb) - np.array(selected_color_rgb))

            filtered_data["ColorDistance"] = filtered_data[color_column].apply(calculate_color_distance)
            filtered_data = filtered_data.sort_values(by="ColorDistance")

            # Get top 5 matches
            recommended_images = filtered_data.head(5)

            # Display images
            for _, row in recommended_images.iterrows():
                image_path = os.path.join(image_folder, row[image_column])
                image = Image.open(image_path)
                st.image(image, caption=f"Style: {row[style_column]} | Gender: {row[gender_column]}", use_column_width=True)
        else:
            st.warning("No recommendations found. Try adjusting your preferences.")