import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import json

# Define your Lambda endpoint URL
# host = "rv4m3mgwb5.execute-api.eu-west-1.amazonaws.com/test"
host = "localhost:9000/2015-03-31/functions/function/invocations"
lambda_endpoint = f"http://{host}"


# Function to make predictions using AWS Lambda
def predict_dog_breed(image):
    try:
        # Convert the image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Encode image to base64
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        # Make a POST request to your Lambda function
        headers = {'Content-Type': 'application/json'}
        payload = {'image': img_base64}

        response = requests.post(lambda_endpoint,
                                 json=payload,
                                 headers=headers)
        # Check if the request was successful
        response.raise_for_status()

        # Parse the response
        predict_breed_list = response.json().get('image')
        predict_breed = json.loads(predict_breed_list)['predicted_breed']
        predict_breed_dict = dict(predict_breed)
        return predict_breed_dict

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except ValueError as e:
        st.error(f"Invalid response: {e}")


# Set up the Streamlit app
st.title('Dog Breed Classifier')

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an image of a dog...",
                                 type=["jpg", "jpeg", "png"])

# Display the uploaded image and make predictions
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image',
             use_column_width=True)

    # Make predictions using AWS Lambda
    predicted_breed = predict_dog_breed(image)
    if predicted_breed is not None:
        st.write('Predicted Dog Breed:', predicted_breed)
