from PIL import Image
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import twice_input
import twice

if __name__ == '__main__':
    data = twice_input.read_data_sets(100)

    BATCH_SIZE = 10

    with tf.Session() as sess:
        m = twice.Twice(sess, 1e-4)
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            x_batch, y_batch = data.train.next_batch(BATCH_SIZE)
            x_batch.reshape(BATCH_SIZE, twice_input.IMAGE_SIZE)

            c, _ = m.train(x_batch, y_batch)

            if i % 40 == 0:
                m.saver.save(sess, './model.ckpt')
                print(c)

        print('Learning Finish')