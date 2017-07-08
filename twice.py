import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import twice_input

IMAGE_SIZE = twice_input.IMAGE_SIZE


class Twice:
    def __init__(self, sess, learning_rate):
        self.learning_rate = learning_rate
        self.sess = sess

        self.inference()

    def inference(self):
        with tf.variable_scope('twice'):
            self.X = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
            X_img = tf.reshape(self.X, [-1, 32, 32, 3])
            self.label = tf.placeholder(tf.float32, [None , 3])
            self.keep_prob = tf.placeholder(tf.float32)
            # images = 32 x 32 x 3

            # convolution layer 1
            kernel1 = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=5e-2))
            bias1 = tf.Variable(tf.random_normal([64]))
            conv1 = tf.nn.conv2d(X_img, kernel1, [1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.bias_add(conv1, bias1)

            conv1 = tf.nn.relu(conv1, name='conv1')

            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            norm1 = tf.nn.lrn(pool1, 4, alpha=0.001 / 9.0, beta=0.75, name='norm1')

            # norm1 = tf.nn.dropout(norm1, keep_prob=self.keep_prob)
            '''
            conv1 result : 32 x 32 x 64
            pool1 result : 16 x 16 x 64
            '''

            # convolution layer 2
            kernel2 = tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=5e-2))
            bias2 = tf.Variable(tf.random_normal([64]))
            conv2 = tf.nn.conv2d(norm1, kernel2, [1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.bias_add(conv2, bias2)

            conv2 = tf.nn.relu(conv2, name='conv2')

            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            # pool2 = tf.nn.dropout(pool2, keep_prob=self.keep_prob)
            '''
            conv2 result : 16 x 16 x 64
            pool2 result : 8 x 8 x 64
            '''

            # local layer 3
            reshape = tf.reshape(pool2, [-1, 8 * 8 * 64])
            kernel3 = tf.get_variable("local3", shape=[8 * 8 * 64, IMAGE_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
            bias3 = tf.Variable(tf.random_normal([IMAGE_SIZE]))
            local3 = tf.matmul(reshape, kernel3)
            local3 = tf.nn.bias_add(local3, bias3)

            local3 = tf.nn.relu(local3, name='local3')

            # local3 = tf.nn.dropout(local3, keep_prob=self.keep_prob)

            '''
            reshape result : 4096
            kernel3 result : 3072
            '''

            # final layer 4
            final4 = tf.get_variable('final4', shape=[IMAGE_SIZE, 3],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.logit = tf.matmul(local3, final4)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logit, labels=self.label))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logit, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, image, label, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: image, self.label: label, self.keep_prob: keep_prop
        })

    def get_accuracy(self, image, label, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: image, self.label: label, self.keep_prob: keep_prop})

    def predict(self, image, keep_prop=1.0):
        return self.sess.run(self.logit, feed_dict={self.X: image, self.keep_prob: keep_prop})
