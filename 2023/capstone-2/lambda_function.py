#!/usr/bin/env python

import os
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite  # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_NAME = os.getenv('MODEL_NAME', 'top_10_dog_breeds.tflite')


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


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


names = ['afghan_hound', 'bernese_mountain_dog', 'great_pyrenees',
         'irish_wolfhound', 'leonberg', 'maltese_dog', 'pomeranian',
         'samoyed', 'scottish_deerhound', 'shih_tzu']


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    x = np.array(img, dtype='float32')
    X = np.array([x])

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return dict(zip(names, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }

    return result
