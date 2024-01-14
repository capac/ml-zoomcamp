# ML Zoomcamp 2023 – Second Capstone Project

## Summary

This computer vision project aims to build a convolutional neural network for classification of 10 different dog breeds. The original dataset can be retrieved from the [ImageNet Dogs Dataset for Fine-grained Visual Categorization](http://vision.stanford.edu/aditya86/ImageNetDogs "http://vision.stanford.edu/aditya86/ImageNetDogs") from the [Stanford Vision and Learning Lab (SVL)](https://svl.stanford.edu/ "https://svl.stanford.edu/").

As mentioned on the website, the Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. The dataset can also be found on Kaggle at [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/ "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/").


## Data preparation

Use `make_dataset.py` to download the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs "http://vision.stanford.edu/aditya86/ImageNetDogs") and extract the images and annotations in the `data/raw/Image` and `data/raw/Annotation` subdirectories.
