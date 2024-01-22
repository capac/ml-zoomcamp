# ML Zoomcamp 2023 – Second Capstone Project

## Summary

This computer vision project aims to build a convolutional neural network (CNN) using transfer learning for classification of 10 different dog breeds. The original dataset can be retrieved from the [ImageNet Dogs Dataset for Fine-grained Visual Categorization](http://vision.stanford.edu/aditya86/ImageNetDogs "http://vision.stanford.edu/aditya86/ImageNetDogs") from the [Stanford Vision and Learning Lab (SVL)](https://svl.stanford.edu/ "https://svl.stanford.edu/").

As mentioned on the website, the Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. The dataset can also be found on Kaggle at [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/ "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/").

### Project files

* `make_dataset.py`
    * Downloads and extracts the Stanford Dogs dataset images and annotations.
* `exploratory_data_analysis.py`
    * Shows bar plot of the top 10 folders by number of unique dog breed images.
* `notebook.ipynb`
    * Crops dog images, and runs model training with hyperparameter fine-tuning, with plots of accuracy for the validation data, and saves the model with the best hyperparameters. IMPORTANT: run notebook only after running the `make_dataset.py` script.
* `training.py`
    * It is a standalone file that achieves the same result as the notebook without validation accuracy plots, and saves the model with best hyperparameters as an H5 file.
* `predict.py`
    * Run model on image test file in a local Docker container.
* `predict-test.py`
    * Run model on image test file in a local Docker container.
* `predict-cloud.py`
    * Run model on image test file in a remote Docker container on Render.

## Data preparation

First thing to do, before running `notebook.ipynb`, is to run `make_dataset.py` to download the complete [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs "http://vision.stanford.edu/aditya86/ImageNetDogs") and extract the images and annotations in the `data/raw/Image` and `data/raw/Annotation` subdirectories. The complete dataset contains 120 different dog breeds and will require a lot of compute resources and time. I decided to concentrate only on the top 10 folder with the highest number of dog images. The number of unique images in the top 10 folders is shown in the exploratory data analysis bar plot.

## Data generation

Once `make_dataset.py` has finished downloading and extracting the images and annotations, you may either run the `notebook.ipynb` Jupyter notebook, or run the `training.py` file. Both files accomplish the same goal: the images from the 10 most numerous subfolders of dog images are selected and cropped to 299 x 299 pixels to solely contain the image of the dog. Each dog breed folder is renamed by removing the alphanumeric prefix, which helps create a `tf.data.Dataset` object from image files in each directory. I've also checked for data corruption in the images, and removed those if they had 'JFIF' in the header. There were none found in the dataset.

# Model training

The model is generated from a pre-trained convolution neural network from ImageNet using transfer learning, and is accessed from the `tensorflow.keras.applications.xception.Xception` class in TensorFlow. I used the `keras.utils.image_dataset_from_directory` class in Keras to generate a training, validation and testing dataset split. 70% of the image dataset is used as the training set, while the remaining 30% is equally split between validation and testing datasets. Data augmentation was carried out on the data set, by horizontally flipping the images and by randomly adding a 10 degree rotation to the images. The dense, upper layers are built using the training dataset. Model hyperparameter fine-tuning was accomplished using different values for the learning rate, for different sizes of an additional internal layer, and for the dropout rate. The best values are saved and used to generate the final model, which is tested on the test set.

IMPORTANT: the model files haven't been saved on GitHub due to the large file constraint of over 100 MB.

## Model deployment
