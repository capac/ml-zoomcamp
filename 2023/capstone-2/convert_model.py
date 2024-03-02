import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

original_model_filename = 'xception_v1_21_0.797.h5'
model = keras.models.load_model(original_model_filename, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_filename = 'top_120_dog_breeds.tflite'
with open(tflite_filename, 'wb') as f_out:
    f_out.write(tflite_model)

print(f'Saved {original_model_filename} to {tflite_filename}.')
