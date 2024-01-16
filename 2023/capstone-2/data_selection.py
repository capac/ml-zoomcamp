#!/usr/bin/env python
# coding: utf-8

import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from time import time

images_dir = 'data/raw/Images'
annotation_dir = 'data/raw/Annotation'
processed_dir = 'data/processed/'
img_size = 300

breed_dir_list = [name for name in os.listdir(images_dir) if name not in ['.DS_Store']]

numeric_label_dict = {}
for index, breed_dir in enumerate(breed_dir_list):
    numeric_label_dict[breed_dir] = index

# dictionary that associates directory name with breed name
fullname_label_dict = {}
# dictionary with number of images of each breed in directory
breed_num_dict = {}
for breed_dir in breed_dir_list:
    breed_name = re.sub(r'(n[0-9]+)-(\w+\-?\_?)', r'\2', breed_dir)
    breed_name = re.sub(r'-', r'_', breed_name.lower())
    fullname_label_dict[breed_dir] = breed_name

    full_breed_dir = Path.cwd() / images_dir / breed_dir
    breed_num_dict[breed_name] = len(list(full_breed_dir.glob('**/*.jpg')))

# modified from https://www.kaggle.com/code/hengzheng/dog-breeds-classifier/notebook
if not Path(processed_dir).exists():
    Path.mkdir(Path.cwd() / processed_dir, exist_ok=True)
    for breed_dir in breed_dir_list:
        Path.mkdir(Path.cwd() / processed_dir / breed_dir, exist_ok=True)
    print(f'Created {len(os.listdir(processed_dir))} folders to store cropped images of the different breeds.')
else:
    print(f'Folder {processed_dir} already exists.')

breed_processed_dir_list = [name for name in os.listdir(processed_dir) if name not in ['.DS_Store']]

t0 = time()
print(f'Saving {img_size}x{img_size} image sizes.')
if not list(Path(processed_dir).glob('**/*.jpg')):
    for breed_dir in breed_processed_dir_list:
        counter = 0
        for breed_file in os.listdir(f'{annotation_dir}/{breed_dir}'):
            img = Image.open(f'{images_dir}/{breed_dir}/{breed_file}.jpg')
            tree = ET.parse(f'{annotation_dir}/{breed_dir}/{breed_file}')
            xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
            xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
            ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
            ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
            img = img.crop((xmin, ymin, xmax, ymax))
            img = img.convert('RGB')
            img = img.resize((img_size, img_size))
            img.save(processed_dir + breed_dir + '/' + breed_file + '.jpg')
            counter += 1
        print(f'Saved {counter} images in {breed_dir} folder.')
    print(f'Time elapsed: {round(time()-t0, 2)} seconds.')
else:
    print(f'Folder {processed_dir} with cropped images already exists.')


def image_paths_and_labels():
    paths = list()
    labels = list()
    for breed_dir in breed_dir_list:
        base_dir = f'{processed_dir}{breed_dir}'
        labels.append(fullname_label_dict[breed_dir])
        for img_file in os.listdir(base_dir):
            paths.append(f'{base_dir}/{img_file}')
    return paths, labels


paths, labels = image_paths_and_labels()
# print(f'Paths: {paths}')
print(f'Number of image paths: {len(paths)}')
# print(f'Labels: {labels}')
print(f'Number of labels: {len(labels)}')
