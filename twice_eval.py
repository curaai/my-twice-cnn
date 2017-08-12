import twice_input
import twice

from scipy.misc import imread
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # path = 'data.txt'
    # data = twice_input.read_data_sets(path, 100)

    BATCH_SIZE = 15
    before_epoch = 0

    with tf.Session() as sess:
        m = twice.Twice(sess, 0.001)
        m.saver.restore(sess, 'save/twice_third.ckpt')

        path = 'C:/Users/dsm2016/Desktop/faces/twice'
        list = os.listdir(path)
        pre = ''
        for x in list:
            img = imread(path + '/' + x).reshape(1, 32, 32, 3) / 255
            result = m.predict(img)
            if result == 0:
                pre = '정연'
            elif result == 1:
                pre = '모모'
            else:
                pre = '나연'
            print('name :', x, ', predict :', pre)
