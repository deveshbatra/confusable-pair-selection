import sys
import argparse
import os

import requests
import json
import wget

import pandas as pd
import numpy as np

from ImageApprover import ImageApprover

class ImageSelector:
    def __init__(self, storage_path):
        storage_path = self.storage_path

    def return_image_names(self):
        img_paths = [ f'{self.storage_path}{img}' for img in os.listdir(self.storage_path) if img.endswith((".jpg", ".jpeg"))]
        return img_paths

    def return_path_names(self):
        img_names = [ str(img).split("_")[0] for img in os.listdir(self.storage_path) if img.endswith((".jpg", ".jpeg"))]
        return img_names

    def return_paths_unselected_images(self, img_names, img_paths):
        set_img_names = list(set(img_names))
        not_selected_images = []
        for name in set_img_names:
            indices = []
            k = img_names.count(name)
            for kk in range(k):
                indices.append(img_names.index(name))
            same_img_paths = [ str(img_paths[i]) for i in indices ]
            assert len(indices) == k
            same_img_sizes = [ os.stat(img).st_size for img in same_img_paths ]
            selected_image = same_img_paths[same_img_sizes.index(max(same_img_sizes))]
            for img in same_img_paths:
                if(img != selected_image):
                    not_selected_images.append(img)
                    
        return not_selected_images

    def remove_unselected_images(self, not_selected_images):
        os.makedirs(str(self.storage_path).replace("base-images", "unselected-images"))
        for img in not_selected_images:
            os.rename(img, img.replace("base-images", "unselected-images"))
