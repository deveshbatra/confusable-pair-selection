import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import ImageFeatures as IF

class SimilarityMatrixCreator():
    def __init__(self):
        pass

    def obtain_feature_matrix(self, list_of_images):
        image_features = IF.InceptionImageFeatures()
        return image_features.extract_features_pool_3(list_of_images)

    def return_vanilla_similarity_matrix(self, list_of_images):
        matrix = self.obtain_feature_matrix(list_of_images)
        assert cosine_similarity(matrix).shape[0] == cosine_similarity(matrix).shape[1]
        return cosine_similarity(matrix)

    def return_diagonal_zero_similarity_matrix(self, list_of_images):
        return np.fill_diagonal(self.return_vanilla_similarity_matrix(list_of_images), 0)

    def return_diagonal_zero_similarity_matrix_upper(self, list_of_images):
        return np.triu(self.return_diagonal_zero_similarity_matrix(list_of_images), 0)
