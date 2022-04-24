import pandas as pd
import numpy as np
import csv
import os

class word2vec:
	def __init__(self, corpus):
		if(corpus == 'Wikipedia'):
			self.filepath = "/Users/devBat/Projects/gallery-game/pipeline/resources/wikipedia-global-vectors/model.txt"
		if(corpus == 'BNC'):
			self.filepath = "/Users/devBat/Projects/gallery-game/pipeline/resources/vectors_nnet_500.txt"


	def loadW2VModelDict(filepath):
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