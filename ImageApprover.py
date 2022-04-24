'''
This class removes images that are cropped, with non-white background, or
contain multiple objects. There are loads of such images that we obtain
from Shutterstock despite all precautionary rules in ImageDownloader class.

'''


import sys
import argparse
import os

import requests
import json
import wget

import pandas as pd
import numpy as np
from PIL import Image
from resizeimage import resizeimage

import cv2

from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import matplotlib.pyplot as plt


class ImageApprover:
    def __init__(self, path_to_downloaded_image, path_to_saved_image):
        pass

    def clip_bottoms(self, path_to_downloaded_image):
        try:
            if path_to_downloaded_image.endswith((".jpeg", ".jpg")):
                original_image = Image.open(path_to_downloaded_image)
                width, height = original_image.size   # Get dimensions

                left = 0
                top = 0
                right = width
                bottom = height - 19 # this number has been determined manually and is subject to change

                clipped_image = original_image.crop((left, top, right, bottom))
                return clipped_image
        except:
            print(f'{path_to_downloaded_image} can\'t be cropped.')

    def is_image_cropped_or_non_white_background(self, path_to_downloaded_image, path_to_saved_image):
        try:
            if path_to_downloaded_image.endswith((".jpeg", ".jpg")):
                np.random.seed(1)
                im = self.clip_bottoms(path_to_downloaded_image)
                im2 = im.convert('L')
                npim = np.asarray(im2, dtype=np.uint8)

                width, height = im2.size

                left = 0
                top = 0
                right = width-1 # this number has been determined manually and is subject to change
                bottom = height-3 # this number has been determined manually and is subject to change

                flag = 0

                for i in range(npim.shape[0]):
                    if(npim[i][right]<235):
                        print(i,npim[i][right])
                        print(str(path_to_downloaded_image)+" contains cropped objects + right")
                        im.save(path_to_saved_image)
                        os.remove(path_to_downloaded_image)
                        flag = 1
                        break

                for i in range(npim.shape[0]):
                    if(npim[i][left]<235):
                        print(i,npim[i][left])
                        print(str(path_to_downloaded_image)+" contains cropped objects + left")
                        im.save(path_to_saved_image)
                        os.remove(path_to_downloaded_image)
                        flag = 1
                        break

                for i in range(npim.shape[1]):
                    if(npim[top][i]<235):
                        print(i,npim[top][i])
                        print(str(path_to_downloaded_image)+" contains cropped objects + top")
                        im.save(path_to_saved_image)
                        os.remove(path_to_downloaded_image)
                        flag = 1
                        break

                for i in range(npim.shape[1]):
                    if(npim[bottom][i]<235):
                        print(i,npim[bottom][i])
                        print(str(path_to_downloaded_image)+" contains cropped objects + bottom")
                        im.save(path_to_saved_image)
                        os.remove(path_to_downloaded_image)
                        flag = 1
                        break

                if(flag == 0):
                    return im
                else:
                    pass
        except:
            print(f'{path_to_downloaded_image} can\'t be cropped.')
        # With the threshold as 235, 240 instead of 220, it removes anything not with white background

    def is_image_with_multiple_objects(self, path_to_downloaded_image, path_to_saved_image):
        if path_to_downloaded_image.endswith((".jpeg", ".jpg")):
            try:
                np.random.seed(1)
                try:
                    im = self.is_image_cropped_or_non_white_background(path_to_downloaded_image, path_to_saved_image)
                    im2 = im.convert('L')
                    npim = np.asarray(im2, dtype=np.uint8)

                    contours = measure.find_contours(npim, 220) #255

                    int_cont = [ cont.astype(int) for cont in contours ]

                    count = 0
                    for i in range(len(int_cont)):
                        if(cv2.contourArea(int_cont[i])>2000):
                            count = count + 1

                    if(count!=1):
                        im.save(path_to_saved_image)
                        os.remove(path_to_downloaded_image)

                except:
                    print(f'{path_to_downloaded_image} does not exist anymore.')

            except IOError:
                print(f'{path_to_downloaded_image} can\'t be cropped.')
