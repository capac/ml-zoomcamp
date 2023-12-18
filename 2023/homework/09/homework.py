#!/usr/bin/env python

import os
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_NAME = os.getenv('MODEL_NAME', 'bees-wasps-v2.tflite')


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def prepare_input(x):
    return x / 255.0


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# photo_url = 'https://habrastorage.org/webt/rt/d9/dh/'\
#             'rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    return float(preds[0, 0])


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }

    return result