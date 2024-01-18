# ML Zoomcamp 2023 – Second Capstone Project

## Summary

This computer vision project aims to build a convolutional neural network for classification of 10 different dog breeds. The original dataset can be retrieved from the [ImageNet Dogs Dataset for Fine-grained Visual Categorization](http://vision.stanford.edu/aditya86/ImageNetDogs "http://vision.stanford.edu/aditya86/ImageNetDogs") from the [Stanford Vision and Learning Lab (SVL)](https://svl.stanford.edu/ "https://svl.stanford.edu/").

As mentioned on the website, the Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. The dataset can also be found on Kaggle at [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/ "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/").


## Data preparation

Use `make_dataset.py` to download the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs "http://vision.stanford.edu/aditya86/ImageNetDogs") and extract the images and annotations in the `data/raw/Image` and `data/raw/Annotation` subdirectories.

Use `data_selection.py` to pre-process the images by applying cropping that selects only the part of the image that contain the dog. Each dog breed folder is renamed by removing the alphanumeric prefix, which helps create a `tf.data.Dataset` object from image files in each directory. I've decided to only use the top 10 folders with the most dog images, and each image to 299 x 299 pixels. I've also checked for data corruption in the images, and removed those if they had 'JFIF' in the header. There were none found in the dataset.

## Data generation

I used the `keras.utils.image_dataset_from_directory` class in Keras to generate a training and validation dataset split, with a batch size of 32.
