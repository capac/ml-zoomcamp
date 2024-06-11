#!/usr/bin/env python

import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite  # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_NAME = os.getenv('MODEL_NAME', 'top_120_dog_breeds.tflite')

names = ['affenpinscher', 'afghan_hound', 'african_hunting_dog',
         'airedale', 'american_staffordshire_terrier', 'appenzeller',
         'australian_terrier', 'basenji', 'basset', 'beagle',
         'bedlington_terrier', 'bernese_mountain_dog',
         'black_and_tan_coonhound', 'blenheim_spaniel', 'bloodhound',
         'bluetick', 'border_collie', 'border_terrier', 'borzoi',
         'boston_bull', 'bouvier_des_flandres', 'boxer',
         'brabancon_griffon', 'briard', 'brittany_spaniel',
         'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever',
         'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie',
         'curly_coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
         'doberman', 'english_foxhound', 'english_setter', 'english_springer',
         'entlebucher', 'eskimo_dog', 'flat_coated_retriever',
         'french_bulldog', 'german_shepherd', 'german_short_haired_pointer',
         'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane',
         'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
         'ibizan_hound', 'irish_setter', 'irish_terrier',
         'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
         'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
         'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
         'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
         'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
         'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
         'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
         'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
         'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
         'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier',
         'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog',
         'shih_tzu', 'siberian_husky', 'silky_terrier',
         'soft_coated_wheaten_terrier', 'staffordshire_bullterrier',
         'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
         'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
         'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
         'west_highland_white_terrier', 'whippet', 'wire_haired_fox_terrier',
         'yorkshire_terrier']

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def _preprocess_image(image_data):
    # Decode the base64-encoded image
    image_data = base64.b64decode(image_data)
    # Prepare the image for sending
    image = Image.open(BytesIO(image_data))
    return image


def _prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def predict(image_data):
    img = _preprocess_image(image_data)
    img = _prepare_image(img, target_size=(150, 150))
    x = np.array(img, dtype='float32')
    X = np.array([x])

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    unsorted_dict = dict(zip(names, float_predictions))
    sorted_list = sorted([(val, key) for key, val in
                          unsorted_dict.items()], reverse=True)[:10]
    predict_breed_probs = [(key, np.round(1/(1 + np.exp(-float(val))), 4))
                           for val, key in dict(sorted_list).items()]
    return predict_breed_probs
