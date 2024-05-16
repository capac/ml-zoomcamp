FROM public.ecr.aws/lambda/python:3.10

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

RUN pip install pillow

RUN pip install numpy

COPY ["lambda_function.py", "model_120_breeds/top_120_dog_breeds.tflite", "./"]

ENV MODEL_NAME='top_10_dog_breeds.tflite'

CMD [ "lambda_function.lambda_handler" ]
