import streamlit as st
import requests
from PIL import Image
import io

# Define your Lambda endpoint URL
lambda_endpoint = "YOUR_LAMBDA_ENDPOINT_URL_HERE"


# Function to make predictions using AWS Lambda
def predict_dog_breed(image):
    try:
        # Convert the image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Make a POST request to your Lambda function
        response = requests.post(lambda_endpoint, data=img_byte_arr)

        # Parse the response
        predicted_breed = response.json().get('predicted_breed')

        return predicted_breed

    except Exception as e:
        st.error(f"An error occurred: {e}")


# Set up the Streamlit app
st.title('Dog Breed Classifier')

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an image of a dog...",
                                 type=["jpg", "jpeg", "png"])

# Display the uploaded image and make predictions
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions using AWS Lambda
    predicted_breed = predict_dog_breed(image)
    if predicted_breed is not None:
        st.write('Predicted Dog Breed:', predicted_breed)
