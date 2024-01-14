#!/usr/bin/env python
# coding: utf-8

import tarfile
import urllib.request
import os
from pathlib import Path


def download_data(url):
    basename = os.path.basename(url)
    raw_data_dir = Path.cwd() / 'data/raw'
    raw_data_file = raw_data_dir / basename
    if not raw_data_file.is_file():
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {basename}...')
        urllib.request.urlretrieve(url, raw_data_file)
        with tarfile.open(raw_data_file) as data_tarball:
            data_tarball.extractall(path=raw_data_dir)
    Path.unlink(raw_data_file)
    print(f'Finished downloading and extracting {basename}.')


if __name__ == "__main__":
    images_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
    annotations_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar'
    download_data(images_url)
    download_data(annotations_url)
