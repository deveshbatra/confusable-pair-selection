from IPython.display import display, Image
import PIL.Image
import numpy as np

import ImagePairsFormer

class Eyeballer:
    def __init__(self):
        pass
    # def __init__(self, type_of_image_pairs, minimum_index_similarity):
    #     self.MINIMUM_INDEX_SIMILARITY = minimum_index_similarity
    #     self.TYPE_OF_IMAGE_PAIRS = type_of_image_pairs

    # def obtain_image_pairs(self, list_of_images):
    #     IPF = ImagePairsFormer.ImagePairsFormer()
    #     if(self.TYPE_OF_IMAGE_PAIRS == 'exclusive'):
    #         return IPF.return_max_exclusive_image_pairs(list_of_images, self.MINIMUM_INDEX_SIMILARITY)
    #     elif(self.TYPE_OF_IMAGE_PAIRS == 'duplicate'):
    #         return IPF.return_max_duplicate_image_pairs(list_of_images)
    #     elif(self.TYPE_OF_IMAGE_PAIRS == 'munkres'):
    #         return IPF.return_munkres_pairs(list_of_images)

    def create_eyeball_file_name(self, storage_folder, name_of_images):
        return

    def form_eyeballs(self, list_of_images):
        # try:
        #     stimulus_file = 
        return

    def forming_eyeballs(self, sim_value, storage_path, stimulus_image_files, distractor_image_files, stimulus_names, distractor_names):
        for i in range(len(stimulus_image_files)):
            try:
                stimulus_file = stimulus_image_files[i]
                distractor_file = distractor_image_files[i]

                eyeball_filename = str(storage_path)+str(sim_value[i])+"_"+str(stimulus_names[i].split("_")[0])+"_"+str(distractor_names[i].split("_")[0])
                list_im = [ stimulus_file, distractor_file ]
                imgs = [ PIL.Image.open(k) for k in list_im ]

        #         min_shape = sorted( [(np.sum(j.size), j.size) for j in imgs])[0][1]
        #         print(min_shape)
                imgs_comb = np.hstack( (np.asarray( j.resize((400,400)) ) for j in imgs ) )

                imgs_comb = PIL.Image.fromarray(imgs_comb)
                imgs_comb.save("%s.jpg" %eyeball_filename)
                print(eyeball_filename)
            except:
                print("Could not eyeball " + str(stimulus_file) + "_" + str(distractor_file))
            
    # forming_eyeballs(sim_value, stimulus_files, distractor_files, stimulus_names, distractor_names)