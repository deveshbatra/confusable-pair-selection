'''
This class extracts features from the various deep-learning models being used in the pipeline.
Includes features from BNC (W2V), Inception-v3 (pool:3_0), but does not include features from
earlier layers.
Takes path of the image folder as input, returns its features - CNN, W2V or CNN-W2V.
'''


import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
# from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import pickle
import word2vec

class InceptionImageFeatures:
    def __init__(self, folder_path):
        self.model_dir = '/Users/devBat/Projects/gallery-game/models/tutorials/image/imagenet/'
        self.filepath_bnc = '/Users/devBat/Projects/gallery-game/pipeline/resources/vectors_nnet_500.txt'
        self.folder_path = folder_path

    def create_graph(self):
        with gfile.FastGFile(os.path.join(self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    # extract image features
    def extract_features_pool_3(self, list_images):
        features = np.empty((len(list_images), 2048))
        
        self.create_graph()

        with tf.Session() as sess:
            next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            for ind, image in enumerate(list_images):
                if not gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)
                image_data = gfile.FastGFile(image, 'rb').read()
                predictions = sess.run(next_to_last_tensor,
                {'DecodeJpeg/contents:0': image_data})
                features[ind,0:] = np.squeeze(predictions)

        return features


    def load_w2v_model_dict(self, filepath):
        print("Loading model")
        f = open(filepath,'r')
        model = {}
        for line in f:
            try:
                splitLine = line.split()
                word = str(splitLine[0]).lower()
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            except:
                print(line.split()[0])
        print("Done.",len(model)," words loaded!")
        
        return model


    def return_w2v_features(self, model, list_of_image_names, sub_cat_file_path):

        df_sub_cat = pd.read_csv(sub_cat_file_path, header=None)
        
        sub_category_dict = {}
        for i in range(len(df_sub_cat)):
            sub_category_dict[df.iloc[i,0]] = str(df.iloc[i,1])

        features = np.empty((len(list_of_image_names), int(list(model.values())[0])))
        for ind in range(len(list_of_image_names)):
            try:
                features[ind, 0:] = model[str(list_of_image_names[ind])]
            except:
                try:
                    features[ind, 0:] = model[str(dict_subcat[list_of_image_names[ind]])]
                except:
                    features[ind, 0:] = np.empty((1, int(list(model.values())[0])))
        
        return features


    def return_cnn_bnc_model_dict(self, sub_cat_file_path):
        list_of_image_paths = [ self.folder_name+str(img_name) for img_name in os.listdir(self.folder_name) if img_name.endswith((".jpg", "jpeg"))]
        list_of_concepts = [ str(img_name.split("_")[0]) for img_name in os.listdir(self.folder_name) if img_name.endswith((".jpg", "jpeg")) ]

        model_bnc = load_w2v_model_dict(filepath_bnc)

        cnn_features = self.extract_features_pool_3(list_of_image_paths)
        bnc_features = self.return_w2v_features(model_bnc, list_of_concepts, sub_cat_file_path)

        cnn_bnc_features = np.empty((len(cnn_features), 2548))

        for ind in range(len(cnn_features)):
            cnn_bnc_features[ind, 0:2048] = normalize(cnn_features[ind][:,np.newaxis], axis=0).ravel()
            cnn_bnc_features[ind, 2048:] = normalize(bnc_features[ind][:,np.newaxis], axis=0).ravel()

        return cnn_bnc_features