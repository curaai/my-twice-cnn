import twice_input
import twice

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    path = 'twice.txt'
    data = twice_input.read_data_sets(True, path, 75)

    BATCH_SIZE = 15
    before_epoch = 0
    keep_prob = 1

    with tf.Session() as sess:
        m = twice.Twice(sess, 0.001)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./board/twice_board', graph=sess.graph)
        print('Learning Start !!!')

        for i in range(1000):
            x_batch, y_batch = data.train.next_batch(BATCH_SIZE)

            c, p, s, _ = m.train(x_batch, y_batch, keep_prob)

            if i % 40 == 0:
                writer.add_summary(s, float(i))
                writer.flush()
                print("loss :", c, ", epoch :", data.train.epoch_complete)

        m.saver.save(sess, 'save/twice_third.ckpt')
        print('model was saved')
        print('Learning Finish')
