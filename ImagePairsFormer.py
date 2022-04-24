'''
Takes path of image folder as input and creates a target-distractor pairs file.
It is important that the pairs file is returned, so that I have a record of 
target, distractor and CNN-BNC indexes.
'''



import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import SimilarityMatrixCreator
import ImageFeatures

class ImagePairsFormer:
    def __init__(self):
        pass

    def obtain_similarity_matrix(self, list_of_images, type_of_similarity_matrix):
        SMC = SimilarityMatrixCreator.SimilarityMatrixCreator()
        if(type_of_similarity_matrix == 'vanilla'):
            return SMC.return_vanilla_similarity_matrix(list_of_images)
        elif(type_of_similarity_matrix == 'diagonal_zero'):
            return SMC.return_diagonal_zero_similarity_matrix(list_of_images)
        elif(type_of_similarity_matrix == 'upper_triangular_matrix'):
            return SMC.return_diagonal_zero_similarity_matrix_upper(list_of_images)
    
    # returns max similarity pairs - only non-repeated, so half of database size
    # type_of_similarity - 'upper_triangular_matrix'
    def return_max_exclusive_image_pairs(self, list_of_images, minimum_index_similarity):
        similarity_matrix = np.array(self.obtain_similarity_matrix(list_of_images, 'upper_triangular_matrix'))
        all_simi_vals = []
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix[i])):
                all_simi_vals.append(similarity_matrix[i][j])

        sorted_all_simi_values = sorted(all_simi_vals, reverse=True)

        pair_indexes = []
        pair_similarity = []
        # the above two lists could have been one dict too, but I didn't want to get bothered with unique key values
        for k in range(800000):
            try:
                i, j = np.where(similarity_matrix == sorted_all_simi_values[k])
                print(i[0], i[1])
                pair_indexes.append((i[0], i[1]))
                pair_similarity.append(sorted_all_simi_values[k])
                similarity_matrix[i[0]] = 0
                similarity_matrix[:,i[1]] = 0
                if(sorted_all_simi_values[k] < minimum_index_similarity):
                    break
            except:
                print(k)
        
        return pair_indexes, pair_similarity

    # returns max similarity pairs, even if one of the images (target or distractor) is repeated
    def return_max_duplicate_image_pairs(self, list_of_images):
        similarity_matrix = np.array(self.obtain_similarity_matrix(list_of_images, 'vanilla'))
        pair_indexes = []
        pair_similarity = []
        for i in range(len(similarity_matrix)):
            max_ = max(list(similarity_matrix[i]))
            j = list(similarity_matrix[i]).index(max_)
            pair_indexes.append( (i,j) )
            pair_similarity.append(max_)
        
        return pair_indexes, pair_similarity


    # first munkres needs to be modified and then the duplicate pairs need to be removed
    def return_modified_munkres_pairs(self, folder_path, sub_cat_file_path):
        image_cnn_bnc_features = ImageFeatures(folder_path).return_cnn_bnc_model_dict(sub_cat_file_path)
        similarities = cosine_similarity(image_cnn_bnc_features)

        np.fill_diagonal(similarities, 0) # to avoid choosing the same image for target and distractor
        similarities = np.negative(similarities) # to choose maximum cost path instead of minimum

        row_ind, col_ind = linear_sum_assignment(similarities)
        sim_indices = [ similarities[row_ind[i], col_ind[i]] for i in range(len(row_ind)) ]
        
        np.savetxt(f'{folder_path}pairs.csv', np.c_[row_ind, col_ind, sim_indices], fmt = "%s", delimiter = ',')
        df_munkres = pd.read_csv(final_file_path, header=None)
        df_munkres_sorted = df_munkres.sort_values(by=[2])

        list_included_indices = []
        approved_cols = []
        approved_rows = []
        approved_sim = []
        for i in range(len(df_munkres_sorted)):
            if(int(df_munkres_sorted.iloc[i,0]) not in list_included_indices and int(df_munkres_sorted.iloc[i,1]) not in list_included_indices):
                approved_rows.append(int(df_munkres_sorted.iloc[i,0]))
                approved_cols.append(int(df_munkres_sorted.iloc[i,1]))
                approved_sim.append(float(df_munkres_sorted.iloc[i,2]))
                list_included_indices.append(int(df_munkres_sorted.iloc[i,0]))
                list_included_indices.append(int(df_munkres_sorted.iloc[i,1]))
        np.savetxt(f'{folder_path}pairs.csv', np.c_[approved_rows, approved_cols, approved_sim], fmt="%s", delimiter=",")

    def return_all_image_pairs_threshold_similarity(self, list_of_images, minimum_index_similarity):
        similarity_matrix = np.array(self.obtain_similarity_matrix(list_of_images, 'vanilla'))
        pair_indexes = []
        pair_similarity = []
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix[i])):
                if(similarity_matrix[i][j] > minimum_index_similarity):
                    pair_indexes.append( (i,j) )
                    pair_similarity.append(similarity_matrix[i][j])
        
        return pair_indexes, pair_similarity

    def return_top_n_image_pairs(self, list_of_images, top_n):
        similarity_matrix = np.array(self.obtain_similarity_matrix(list_of_images, 'vanilla'))
        top_n_image_pairs = {}
        for i in range(len(similarity_matrix)):
            sorted_list = sorted(similarity_matrix[i], reverse = True)
            top_n_image_pairs[i] = []
            for j in range(top_n):
                print(sorted_list[j])
                top_n_image_pairs[i].append(list(similarity_matrix[i]).index(sorted_list[j]))
                print(top_n_image_pairs[i])


    def return_image_pair_filenames(self, pair_indexes, pair_similarity, list_of_images, list_of_image_names):
        name_pairs = [ (list_of_image_names[pair_index[0]], list_of_image_names[pair_index[1]]) for pair_index in pair_indexes ]
        file_pairs = [ (list_of_images[pair_index[0]], list_of_images[pair_index[1]]) for pair_index in pair_indexes ]
        
        return name_pairs, file_pairs