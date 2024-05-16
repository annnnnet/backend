import os
import pickle
import random


def getImagePath():
    dataset_root = r'D:\\ven\\images'
    return dataset_root


def get_test_list(num_images=5):
    dataset_root = r'D:\\ven\\images'

    image_names = os.listdir(dataset_root)
    random.shuffle(image_names)

    selected_image_names = image_names[:num_images]
    image_path_list = [os.path.join(dataset_root, name) for name in selected_image_names]

    return image_path_list



def get_pdefined_anchors():    
        with open('D:\\ven\Dataset\pdefined_anchor.pkl', 'rb') as f:
            pdefined_anchors = pickle.load(f, encoding="bytes")
        return pdefined_anchors