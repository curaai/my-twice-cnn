from PIL import Image
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import twice_input
import twice

if __name__ == '__main__':
    sess = tf.Session()
    m = twice.Twice(sess, 0.2)

    sess.run(tf.global_variables_initializer())

    print('start traning')

    data, label = twice_input.get_feed_data()

    avg_cost = 0
    for i in range(twice_input.LABEL):
        Y = np.zeros([3])
        Y[label[i]] = 1

        c, _ = m.train(data[i].reshape(1, twice_input.IMAGE_SIZE), Y.reshape(1, 3))
        # avg_cost += c / twice_input.LABEL

        print('Epoch:', '%04d' % (i + 1), 'cost =', '{:.9f}'.format(c))

    print('Learning Finished')

    img = Image.open('test/1.jpg')
    img = np.array(img.resize((32, 32))).flatten().reshape(1, 3072)
    print(m.predict(img))