'''
This class works in tandem with ImageApprover and ImageSelector class to download, approve and clean images from Shutterstock.
A list of concept names can be used to download appropriate images from Shutterstock.
The result of this module is one image per concept.

'''

import sys
import argparse
import os

import requests
import json
import wget

import pandas as pd
import numpy as np

from ImageApprover import ImageApprover
# add sys.argv

class ImageDownloader:
    def __init__(self, name, number_of_images_to_download, storage_path, excluded_shutterstock_ids_file_path, excluded_shutterstock_ids_col_number):
        # rules
        self.QUERY_TEXT = f'{name} isolated'
        self.__CLIENT_ID = '0694ce17aa07eb227c6e'
        self.__CLIENT_SECRET = 'd45fea286b87eee92fde5f367891a3ef6635cc54'
        self.PEOPLE_NUMBER = 0
        self.IMAGE_TYPE = 'photo'
        self.VIEW = 'full'

        # terminal arguments
        self.name = name
        self.number_of_images_to_download = number_of_images_to_download
        self.storage_path = storage_path
        self.excluded_shutterstock_ids_file_path = excluded_shutterstock_ids_file_path
        self.excluded_shutterstock_ids_col_number = excluded_shutterstock_ids_col_number

    def create_image_download_link(self):
        return f'https://{self.__CLIENT_ID}:{self.__CLIENT_SECRET}@api.shutterstock.com/v2/images/search?page=1&per_page={self.number_of_images_to_download}&people_number={self.PEOPLE_NUMBER}&query={self.QUERY_TEXT}&view={self.VIEW}&image_type={self.IMAGE_TYPE}'
    
    def create_downloaded_image_name(self, shutterstock_id):
        return f'{self.storage_path}{self.name}_{shutterstock_id}.jpg'

    def download_image_json(self):
        try:
            r = requests.get(self.create_image_download_link())
            return r.json()
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print("Inside download_image_json")
            print(message)

    def excluded_shutterstock_ids(self):
        df = pd.read_csv(self.excluded_shutterstock_ids_file_path)
        excluded_ids = []
        if(self.excluded_shutterstock_ids_file_path):
            excluded_ids = [ df.iloc[:, self.excluded_shutterstock_ids_col_number] ]
        return excluded_ids


    def download_images(self):
        excluded_ids = []
        for k in range(self.number_of_images_to_download):
            try:
                data = self.download_image_json()
                if(str(data['data'][k]['id']) not in excluded_ids):
                    if(str(data['data'][k]['image_type'])==self.IMAGE_TYPE):
                        if(self.name.lower() in str(data['data'][k]['description']).lower()):
                            new_file_name = self.create_downloaded_image_name(data['data'][k]['id'])
                            wget.download(data['data'][k]['assets']['preview']['url'], new_file_name)
                            excluded_ids.append(str(data['data'][k]['id']))
                            IA = ImageApprover(new_file_name, new_file_name.replace("base-images", "saved-images"))
                            IA.is_image_with_multiple_objects(new_file_name, new_file_name.replace("base-images", "saved-images"))
                        else:
                            continue
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print("Inside download_images")
                print(message)
                continue

    # extension for wordnet

    def download_images_wordnet(self):
        # excluded_ids = self.excluded_shutterstock_ids()
        for k in range(self.number_of_images_to_download):
            try:
                data = self.download_image_json_wordnet()
                if(str(data['data'][k]['image_type'])==self.IMAGE_TYPE):
                    if(self.name.lower() in str(data['data'][k]['description']).lower()):
                        new_file_name = self.create_downloaded_image_name_wordnet(data['data'][k]['id'])
                        wget.download(data['data'][k]['assets']['preview']['url'], new_file_name)
                        IA = ImageApprover(new_file_name, new_file_name.replace("base-images", "saved-images"))
                        IA.is_image_with_multiple_objects(new_file_name, new_file_name.replace("base-images", "saved-images"))
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print("Inside download_images_wordnet")
                print(message)
                continue

    def create_downloaded_image_name_wordnet(self, shutterstock_id):
        print(f'{self.storage_path}{self.name}_{shutterstock_id}.jpg')
        return f'{self.storage_path}{self.name}_{shutterstock_id}.jpg'

    def create_image_download_link_wordnet(self):
        return f'https://{self.__CLIENT_ID}:{self.__CLIENT_SECRET}@api.shutterstock.com/v2/images/search?people_number={self.PEOPLE_NUMBER}&query={self.QUERY_TEXT}&image_type={self.IMAGE_TYPE}'

    def download_image_json_wordnet(self):
        try:
            r = requests.get(self.create_image_download_link())
            return r.json()
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print("Inside download_image_json_wordnet")
            print(message)
