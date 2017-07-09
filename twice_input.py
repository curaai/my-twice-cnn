import numpy as np
import os
from scipy.misc import imread
from sklearn.utils import shuffle

IMAGE_SIZE  = 3072
CLASSES     = 3


def get_feed_data():
    directories = os.listdir('data')
    files = ['data/' + directory + '/' +file for directory in directories for file in os.listdir('data/' + directory)]

    images = list()
    labels = list()

    length = [len(os.listdir('data/' + directory)) for directory in directories]

    for index in range(len(files)):
        img = imread(files[index])
        img = img.flatten()

        label = np.zeros(CLASSES)
        if index < length[0]:
            label[0] = 1
        elif index < sum(length[:2]):
            label[1] = 1
        else:
            label[2] = 1

        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    images, labels = shuffle(images, labels, random_state=4)

    return images, labels


class DataSet:

    def __init__(self, images, labels):
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self.images = images
        self.labels = labels

        self.num_examples = images.shape[0]

        self.epoch_complete = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            self.epoch_complete += 1
            start = 0
            self.index_in_epoch = batch_size

        end = self.index_in_epoch

        return self.images[start:end], self.labels[start: end]


def read_data_sets(test=0):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels = get_feed_data()

    test_images = images[:test]
    test_labels = labels[:test]

    train_images = images[test:]
    train_labels = labels[test:]

    data_sets.test = DataSet(test_images, test_labels)
    data_sets.train = DataSet(train_images, train_labels)

    return data_sets