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
processed_dir = 'data/processed'
img_size = 300

# modified from https://www.kaggle.com/code/hengzheng/dog-breeds-classifier/notebook
breed_dir_list = [name for name in os.listdir(images_dir) if name not in ['.DS_Store']]
if not Path(processed_dir).exists():
    Path.mkdir(Path.cwd() / processed_dir, exist_ok=True)
    for breed_dir in breed_dir_list:
        breed_name = re.sub(r'(n[0-9]+)-(\w+\-?\_?)', r'\2', breed_dir)
        breed_name = re.sub(r'-', r'_', breed_name.lower())
        Path.mkdir(Path.cwd() / processed_dir / breed_name, exist_ok=True)
    print(f'''Created {len(os.listdir(processed_dir))} folders to store cropped images of the different breeds.''')
else:
    print(f'Folder {processed_dir} already exists.')

t0 = time()
print(f'Saving {img_size}x{img_size} image sizes.')
if not list(Path(processed_dir).glob('**/*.jpg')):
    for breed_dir in breed_dir_list:
        counter = 0
        for breed_prefix in os.listdir(f'{annotation_dir}/{breed_dir}'):
            img = Image.open(f'{images_dir}/{breed_dir}/{breed_prefix}.jpg')
            tree = ET.parse(f'{annotation_dir}/{breed_dir}/{breed_prefix}')
            xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
            xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
            ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
            ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
            img = img.crop((xmin, ymin, xmax, ymax))
            img = img.convert('RGB')
            img = img.resize((img_size, img_size))
            breed_name = re.sub(r'(n[0-9]+)-(\w+\-?\_?)', r'\2', breed_dir)
            breed_name = re.sub(r'-', r'_', breed_name.lower())
            img.save(processed_dir + '/' + breed_name + '/' + breed_prefix + '.jpg')
            counter += 1
        print(f'Saved {counter} images in {breed_name} folder.')
    print(f'Time elapsed: {round(time()-t0, 0)} seconds.')
else:
    print(f'Folder {processed_dir} with cropped images already exists.')
