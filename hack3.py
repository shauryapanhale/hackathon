import os
import random
import streamlit as st
import torch
from PIL import Image

@st.cache_resource
def load_model():
    """Load the pre-trained PyTorch model."""
    model_path = r"C:\Users\yasha\OneDrive\Desktop\PBL_PROJECT\resnet18_scripted.pt"  # Update this path if needed
    if not os.path.exists(model_path):
        st.error(f"The model file '{model_path}' does not exist. Please check the path.")
        st.stop()
    model = torch.jit.load(model_path)
    model.eval()
    return model

def display_recommendations(title, image_folder, num_images=3):
    """Display random image recommendations."""
    st.subheader(title)
    try:
        images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(("jpg", "jpeg", "png"))]
        for i in range(num_images):
            img_path = random.choice(images)
            st.image(img_path, caption=f"Recommendation {i+1}", use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying images: {e}")

def main():
    st.title("Fashion Recommendation System")

    # Gender and Age Selection
    gender = st.radio("Select Gender", ["Male", "Female", "Other"], index=0)
    age = st.number_input("Enter Age", min_value=0, max_value=100, step=1)

    # Navigation Options
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose an Option", ["Home", "Get Recommendations", "Suggest Outfits"])

    if option == "Home":
        st.subheader("Welcome to the Fashion Recommendation System!")
        st.write("Use the sidebar to navigate.")
    elif option == "Get Recommendations":
        st.subheader("Get Recommendations Based on Your Preferences")

        # User Inputs
        color = st.text_input("Enter a Color (e.g., Red, Blue, Black)")
        event = st.selectbox("Select the Event Type", ["Casual", "Formal", "Party", "Sports", "Other"])
        style = st.selectbox("Select the Style", ["Trendy", "Classic", "Bohemian", "Athletic", "Other"])
        fit = st.selectbox("Select the Fit", ["Slim", "Regular", "Loose", "Other"])

        # Generate Recommendations
        if st.button("Get Recommendations"):
            st.write(f"Gender: {gender}, Age: {age}")
            st.write(f"Color: {color}, Event: {event}, Style: {style}, Fit: {fit}")
            display_recommendations("Recommendations Based on Your Preferences", r"C:\Users\yasha\OneDrive\Desktop\PBL_PROJECT\datasets\train_images")
    elif option == "Suggest Outfits":
        st.subheader("Suggest Outfits for an Existing Clothing Item")

        # User Upload
        uploaded_file = st.file_uploader("Upload an Image of Your Clothing Item", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            st.write(f"Gender: {gender}, Age: {age}")

            # Generate Suggestions
            if st.button("Suggest Outfits"):
                display_recommendations("Suggestions for Your Outfit", r"C:\Users\yasha\OneDrive\Desktop\PBL_PROJECT\datasets\train_images")

if __name__ == "__main__":
    main()
