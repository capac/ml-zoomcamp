#!/usr/bin/env python

import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import altair as alt
from prediction_function import predict

# Set up the Streamlit app
st.title("Dog Breed Classifier")

# Add a file uploader widget
uploaded_file = st.file_uploader(
    "Choose an image of a dog...", type=["jpg", "jpeg", "png"]
)

# Display the uploaded image and make predictions
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)

    # Convert the image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Encode image to base64
    img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

    # Make predictions using 'predict' function from 'prediction_function'
    predicted_breed = predict(img_base64)
    if predicted_breed is not None:

        predicted_breed_df = pd.DataFrame(predicted_breed)
        predicted_breed_df.rename(
            columns={0: "Dog breed", 1: "Probability"}, inplace=True
        )
        y_axis = alt.Axis(offset=5)
        # Create the bar chart
        chart = (
            alt.Chart(predicted_breed_df)
            .mark_bar(color='#31B1E0')
            .encode(
                x=alt.X("Probability"),
                y=alt.Y("Dog breed:N", sort="-x").axis(y_axis),
            )
            .properties(title="Dog breed probability",
                        width="container")
            .configure_axis(labelPadding=20,
                            titlePadding=10,
                            labelLimit=300)
        )
        st.altair_chart(chart, use_container_width=True)
