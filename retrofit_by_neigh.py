# original implementation by Faruqui et al. (2015) - forked from github.com/mfaruqui

import argparse
import gzip
import math
import numpy
import re
import sys
import os
import pandas as pd

from copy import deepcopy

import numpy as np
import pandas as pd

import os
import sys
sys.path.append("..")

import ImageFeatures
import word2vec
import ImagePairsFormer
import SimilarityMatrixCreator

folder_name = '/Users/$USER/images/'
list_of_paths = [ f'{folder_name}{img}' for img in os.listdir(folder_name) if img.endswith((".jpg", ".jpeg")) ]
list_of_im = [img for img in os.listdir(folder_name) if img.endswith((".jpg", ".jpeg"))]
list_of_images = [ str(img).split("_")[0].replace(" ","-") for img in os.listdir(folder_name) if img.endswith((".jpg", ".jpeg"))]

# calculate wordnet scores beforehand
wordnet_wup_scores = {}
df_ = pd.read_csv("/Users/$USER/similarities_wup_all_concepts.csv", header=None)
for i in range(len(list_of_images)):
    wordnet_wup_scores[list_of_images[i]] = []
    for j in range(len(list_of_images)):
        wordnet_wup_scores[list_of_images[i]].append({'value':df_.iloc[i,j], 'concept': list_of_images[j]})

# calculate bnc vectors
def loadBNCModelDict(filepath):
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

filepath_bnc = "/Users/$USER/vectors_nnet_500.txt"
m_bnc = loadBNCModelDict(filepath_bnc)

# BNC features
def return_w2v_features(model, list_of_image_names):
    features = np.empty((len(list_of_image_names), len(list(m_bnc.values())[5])))
    for ind in range(len(list_of_image_names)):
        try:
            features[ind, 0:] = model[str(list_of_image_names[ind])]
        except:
            print(list_of_image_names[ind])

    return features

# calculate normalised CNN-BNC vectors
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity

IF = ImageFeatures.InceptionImageFeatures(folder_name)

cnn_features = IF.extract_features_pool_3(list_of_paths)
bnc_features = return_w2v_features(m_bnc, list_of_images)

cnn_features_normalised_scaled = np.empty((len(cnn_features), 2048))
bnc_features_normalised_scaled = np.empty((len(bnc_features), 500))

for ind in range(len(cnn_features)):
    cnn_features_normalised_scaled[ind] = scale(normalize(cnn_features[ind][:,np.newaxis], axis=0).ravel())
    bnc_features_normalised_scaled[ind] = scale(normalize(bnc_features[ind][:,np.newaxis], axis=0).ravel())

cnn_bnc_features = np.empty((len(cnn_features), 2548))

for ind in range(len(cnn_features)):
    cnn_bnc_features[ind, 0:2048] = 0.30*cnn_features_normalised_scaled[ind]
    cnn_bnc_features[ind, 2048:] = 0.70*bnc_features_normalised_scaled[ind]

print("Finished obtaining features...")


isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs():
    wordVectors = {}
    for i in range(len(list_of_images)):
        word = list_of_images[i]
        wordVectors[list_of_images[i]] = np.array(cnn_bnc_features[i])

    sys.stderr.write("Vectors read")
    return wordVectors

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')
  for word, values in wordVectors.items():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')
  outFile.close()

''' Return ten nearest wordnet synsets and values '''
def nearest_wordnet_synsets(word, number_of_neighbours):
    list_neighbours = sorted(wordnet_wup_scores[word], key = lambda i: i['value'],reverse=True)[0:10]
    return list_neighbours

def retrofit_by_neighbors(wordVecs, number_of_neighbors):
    newWordVecs = deepcopy(wordVecs)
    for word_ in list(list_of_images):
        word = word_.replace(" ","-").replace("_","-")
        newVec = number_of_neighbors*wordVecs[word]
        count = 1
        for i in range(1, number_of_neighbors):
            concept_ = sorted(wordnet_wup_scores[word], key = lambda i: i['value'],reverse=True)[i]['concept']
            count += 1
            newVec += newWordVecs[concept_]
        newWordVecs[word] = newVec/(2*number_of_neighbors)

    return newWordVecs


if __name__=='__main__':
  ''' Enrich the word vectors using WordNet '''
  for i in range(1, 25):
      outFileName = f'/Users/$USER/cnn_bnc_{i}.txt'
      print_word_vecs(retrofit_by_neighbors(wordVecs, i), outFileName)
