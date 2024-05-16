from torchvision import transforms
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def get_pdefined_anchors(pkl_file):
    return pkl.load(open(pkl_file))


def Get895Crops(image, pdefined_anchors):
    if isinstance(image, (str, pkl.UNICODE)):
        image = Image.open(image)

    w, h = image.size
    image_crops = []
    image_bboxes = []
    for s_anchor in pdefined_anchors:
        x1, y1, x2, y2 = int(s_anchor[0] * w), int(s_anchor[1] * h), int(s_anchor[2] * w), int(s_anchor[3] * h)
        image_crop = image.crop((x1, y1, x2, y2)).copy()
        image_crops.append(image_crop)
        image_bboxes.append([x1, y1, x2, y2])
    return image_crops, image_bboxes
