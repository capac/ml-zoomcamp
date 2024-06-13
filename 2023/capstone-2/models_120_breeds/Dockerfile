FROM public.ecr.aws/lambda/python:3.11

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.14.0-cp311-cp311-linux_x86_64.whl

COPY ["prediction_function.py", "streamlit_app.py", "top_120_dog_breeds.tflite", "./"]

ENV MODEL_NAME='top_120_dog_breeds.tflite'
