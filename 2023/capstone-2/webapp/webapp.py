import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model_path = '../models_120_breeds/xception_v1_21_0.797.h5'


# Load your trained model
def load_model():
    # Replace 'model_path' with the path to your saved model
    model = tf.keras.models.load_model(model_path)
    return model


# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to the required input shape of your model
    image = image.resize((img_height, img_width))
    # Convert the image to a numpy array
    img_array = np.array(image)
    # Normalize the pixel values to be in the range [0, 1]
    img_array = img_array / 255.0
    # Expand the dimensions of the image to match
    # the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to make predictions
def predict_dog_breed(image, model):
    # Preprocess the uploaded image
    img_array = preprocess_image(image)
    # Perform inference using your model
    predictions = model.predict(img_array)
    # Get the index of the predicted class (dog breed)
    predicted_class_idx = np.argmax(predictions[0])
    # Map the predicted class index to the corresponding dog breed label
    predicted_breed = dog_breeds[predicted_class_idx]
    return predicted_breed


# Define the size of input images expected by your model
img_height, img_width = 224, 224

# List of dog breed labels
dog_breeds = ['Labrador Retriever', 'German Shepherd', 'Golden Retriever',
              'Bulldog', 'Beagle', 'Poodle', 'Rottweiler', 'Yorkshire Terrier',
              'Boxer', 'Dachshund']

# Load the model
model = load_model()

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

    # Make predictions
    predicted_breed = predict_dog_breed(image, model)
    st.write('Predicted Dog Breed:', predicted_breed)
