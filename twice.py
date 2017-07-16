import tensorflow as tf
import numpy as np
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
            self.X = tf.placeholder(tf.float32, [None, 96, 96, 3])
            self.label = tf.placeholder(tf.float32, [None, twice_input.CLASSES])
            self.keep_prob = tf.placeholder(tf.float32)
            # images = 96 x 96 x 3

            # convolution layer 1
            kernel1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=5e-2), name='kern1')
            bias1 = tf.Variable(tf.random_normal([32]), 'bisa1')
            conv1 = tf.nn.conv2d(self.X, kernel1, [1, 1, 1, 1], padding='SAME', )
            conv1 = tf.nn.bias_add(conv1, bias1)

            conv1 = tf.nn.relu(conv1, name='conv1')

            norm1 = tf.nn.lrn(conv1, 4, alpha=0.001 / 9.0, beta=0.75, name='norm1')

            pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1],
                                   strides=[1, 3, 3, 1], padding='SAME', name='pool2')

            # pool1 = tf.nn.dropout(pool1, keep_prob=self.keep_prob)

            '''
            conv1 result : 32 x 32 x 32
            '''

            # convolution layer 2
            kernel2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=5e-2), name='kern2')
            bias2 = tf.Variable(tf.random_normal([64]), name='bias2')
            conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.bias_add(conv2, bias2)

            conv2 = tf.nn.relu(conv2, name='conv2')

            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            # pool2 = tf.nn.dropout(pool2, keep_prob=self.keep_prob)

            '''
            conv2 result : 32 x 32 x 64
            pool2 result : 16 x 16 x 64
            '''

            # convolution layer 3
            kernel3 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=5e-2), name='kern3')
            bias3 = tf.Variable(tf.random_normal([128]), name='bias3')
            conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.bias_add(conv3, bias3)

            conv3 = tf.nn.relu(conv3, name='conv3')

            norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm3')
            pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool3')
            # pool3 = tf.nn.dropout(pool3, keep_prob=self.keep_prob)

            '''
            conv2 result : 16 x 16 x 128
            pool2 result : 8 x 8 x 128
            '''


            # local layer 4
            reshape = tf.reshape(pool3, [-1, 8 * 8 * 128])
            kernel4 = tf.get_variable("kern4", shape=[8 * 8 * 128, 2048],
                                      initializer=tf.contrib.layers.xavier_initializer())
            bias4 = tf.Variable(tf.random_normal([2048]), name='bias4')
            local4 = tf.matmul(reshape, kernel4)
            local4 = tf.nn.bias_add(local4, bias4)

            local4 = tf.nn.relu(local4, name='local4')

            local4 = tf.nn.dropout(local4, keep_prob=self.keep_prob)

            '''
            reshape result : 8192
            kernel3 result : 2048
            '''

            # local layer 5
            kernel5 = tf.get_variable('kern5', shape=[2048, 512],
                                      initializer=tf.contrib.layers.xavier_initializer())
            local5 = tf.nn.relu(tf.matmul(local4, kernel5))

            local5 = tf.nn.dropout(local5, keep_prob=self.keep_prob)

            # final layer 5
            final6 = tf.get_variable('final6', shape=[512, twice_input.CLASSES],
                                     initializer=tf.contrib.layers.xavier_initializer())
            logit = tf.matmul(local5, final6)

            self.saver = tf.train.Saver()
            self.logit = logit
            self.pred = tf.nn.softmax(self.logit)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logit, labels=self.label))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logit, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, image, label, keep_prob):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: image, self.label: label, self.keep_prob: keep_prob
        })

    def get_accuracy(self, image, label, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: image, self.label: label, self.keep_prob: keep_prob})

    def predict(self, image, keep_prob=1.0):
        return np.argmax(self.sess.run(self.pred, feed_dict={self.X: image, self.keep_prob: keep_prob}), axis=1)
