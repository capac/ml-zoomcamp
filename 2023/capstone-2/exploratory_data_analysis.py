#!/usr/bin/env python
# coding: utf-8

import os
import re
from pandas import DataFrame
from pathlib import Path
import matplotlib.pyplot as plt


images_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
annotations_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar'
images_dir = 'data/raw/Images'
annotation_dir = 'data/raw/Annotation'

plt.style.use('barplot-style.mplstyle')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


breed_dir_list = [name for name in os.listdir(images_dir) if name not in ['.DS_Store']]
breed_dict = {}
for breed_dir in breed_dir_list:
    full_breed_dir = Path.cwd() / images_dir / breed_dir
    breed_name = re.sub(r'(n[0-9]+)-(\w+\-?\_?)', r'\2', breed_dir)
    breed_name = re.sub(r'-', r'_', breed_name.lower())
    breed_dict.setdefault('breed', []).append(breed_name)
    breed_dict.setdefault('num_photos', []).append(len(list(full_breed_dir.glob('**/*.jpg'))))
    breed_dict.setdefault('directory', []).append(breed_dir)

breed_df = DataFrame(breed_dict)
breed_df.sort_values(by='num_photos', ascending=False, inplace=True)
breed_df.reset_index(drop=True, inplace=True)
# print(breed_df.head(30))
# print(list(breed_df.directory.iloc[:30].values))

top_num = 10
top_index_names = breed_df['breed'].iloc[:top_num]
custom_labels = [' '.join(col.split('_')).capitalize() for col in top_index_names]

fig, ax = plt.subplots()
ax.bar(breed_df['breed'].iloc[:top_num], breed_df['num_photos'].iloc[:top_num], color=colors,)
plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor', rotation=45)
ax.set_xticks(range(top_num))
ax.set_xticklabels(custom_labels)
ax.set_ylim([195, 255])
ax.set_title(f'Top {top_num} most numerous dog images in data set')

Path.mkdir(Path.cwd() / 'plots', exist_ok=True)
plt.savefig('plots/eda.png')
