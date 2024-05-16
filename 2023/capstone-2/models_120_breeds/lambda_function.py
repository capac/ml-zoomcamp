#!/usr/bin/env python

import os
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite  # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_NAME = os.getenv('MODEL_NAME',
                       'model_120_breeds/top_120_dog_breeds.tflite')


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

names = ['silky_terrier', 'scottish_deerhound', 'chesapeake_bay_retriever',
         'ibizan_hound', 'wire_haired_fox_terrier', 'saluki', 'cocker_spaniel',
         'schipperke', 'borzoi', 'pembroke', 'komondor',
         'staffordshire_bullterrier', 'standard_poodle', 'eskimo_dog',
         'english_foxhound', 'golden_retriever', 'sealyham_terrier',
         'japanese_spaniel', 'miniature_schnauzer', 'malamute', 'malinois',
         'pekinese', 'giant_schnauzer', 'mexican_hairless', 'doberman',
         'standard_schnauzer', 'dhole', 'german_shepherd',
         'bouvier_des_flandres', 'siberian_husky', 'norwich_terrier',
         'irish_terrier', 'norfolk_terrier', 'saint_bernard', 'border_terrier',
         'briard', 'tibetan_mastiff', 'bull_mastiff', 'maltese_dog',
         'kerry_blue_terrier', 'kuvasz', 'greater_swiss_mountain_dog',
         'lakeland_terrier', 'blenheim_spaniel', 'basset',
         'west_highland_white_terrier', 'chihuahua', 'border_collie',
         'redbone', 'irish_wolfhound', 'bluetick', 'miniature_poodle',
         'cardigan', 'entlebucher', 'norwegian_elkhound',
         'german_short_haired_pointer', 'bernese_mountain_dog', 'papillon',
         'tibetan_terrier', 'gordon_setter', 'american_staffordshire_terrier',
         'vizsla', 'kelpie', 'weimaraner', 'miniature_pinscher', 'boxer',
         'chow', 'old_english_sheepdog', 'pug', 'rhodesian_ridgeback',
         'scotch_terrier', 'shih_tzu', 'affenpinscher', 'whippet',
         'sussex_spaniel', 'otterhound', 'flat_coated_retriever',
         'english_setter', 'italian_greyhound', 'labrador_retriever',
         'collie', 'cairn', 'rottweiler', 'australian_terrier', 'toy_terrier',
         'shetland_sheepdog', 'african_hunting_dog', 'newfoundland',
         'walker_hound', 'lhasa', 'beagle', 'samoyed', 'great_dane',
         'airedale', 'bloodhound', 'irish_setter', 'keeshond',
         'dandie_dinmont', 'basenji', 'bedlington_terrier', 'appenzeller',
         'clumber', 'toy_poodle', 'great_pyrenees', 'english_springer',
         'afghan_hound', 'brittany_spaniel', 'welsh_springer_spaniel',
         'boston_bull', 'dingo', 'soft_coated_wheaten_terrier',
         'curly_coated_retriever', 'french_bulldog', 'irish_water_spaniel',
         'pomeranian', 'brabancon_griffon', 'yorkshire_terrier',
         'groenendael', 'leonberg', 'black_and_tan_coonhound']


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
