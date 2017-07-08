import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle

IMAGE_SIZE = 3072
LABEL = 841


def get_feed_data():
    directories = os.listdir('data')
    files = [file for directory in directories for file in os.listdir('data/' + directory)]
    data_len = len(files)

    img_matrix = np.array([np.array(Image.open('data/' + directory + '/' + file)).flatten() for directory in directories
                          for file in os.listdir('data/' + directory)], 'f')
    label = np.ones((data_len, ), dtype=int)

    # 0 : jeongyeon, 1 : momo, 2: nayeon
    label[:246] = 0
    label[246: 544] = 1
    label[547: 841] = 2
    print(len(label))
    data, label = shuffle(img_matrix, label, random_state=4)
    return data, label
