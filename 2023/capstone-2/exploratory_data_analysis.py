#!/usr/bin/env python
# coding: utf-8

import os
import re
from pandas import Series
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
    breed_dict[breed_name] = len(list(full_breed_dir.glob('**/*.jpg')))

breed_sr = Series(breed_dict)
breed_sr.sort_values(ascending=False, inplace=True)
breed_sr.head(20)

top_num = 10
top_index_names = breed_sr.index[:top_num]
custom_labels = [' '.join(col.split('_')).capitalize() for col in top_index_names]

fig, ax = plt.subplots()
ax.bar(breed_sr.index[:top_num], breed_sr.values[:top_num], color=colors,)
plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor', rotation=45)
ax.set_xticks(range(10))
ax.set_xticklabels(custom_labels)
ax.set_ylim([195, 255])
ax.set_title(f'Top {top_num} dogs in data set')

Path.mkdir(Path.cwd() / 'plots', exist_ok=True)
plt.savefig('plots/eda.png')
