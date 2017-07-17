import twice_input
import twice

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    path = 'dump.txt'
    data = twice_input.read_data_sets(True, path, 100)

    BATCH_SIZE = 15
    before_epoch = 0
    keep_prob = 0.5

    with tf.Session() as sess:
        m = twice.Twice(sess, 0.001)
        sess.run(tf.global_variables_initializer())
        print('Learning Start !!!')

        for i in range(1000):
            x_batch, y_batch = data.train.next_batch(BATCH_SIZE)

            c, p, _ = m.train(x_batch, y_batch, keep_prob)
            if i % 40 == 0:
                x, y = data.test.next_batch(BATCH_SIZE)
                print('Epoch : {}, Accuracy : {}, loss : {}'.format(data.train.epoch_complete,
                                                                    m.get_accuracy(x, y),
                                                                    c))

        m.saver.save(sess, 'save/another.ckpt')
        print('model was saved')
        print('Learning Finish')
