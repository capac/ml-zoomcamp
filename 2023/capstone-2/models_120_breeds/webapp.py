import streamlit as st
from PIL import Image
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

    # Make predictions using AWS Lambda
    predicted_breed = predict(image)
    if predicted_breed is not None:

        predicted_breed_df = pd.Series(predicted_breed).reset_index()
        predicted_breed_df.rename(
            columns={0: "Probability", "index": "Dog breed"}, inplace=True
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
