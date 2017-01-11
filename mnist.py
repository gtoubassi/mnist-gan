#!/usr/bin/env python
#
# From the MNIST tutorial: https://www.tensorflow.org/tutorials/mnist/pros/

import argparse
import tensorflow as tf
import numpy as np

class MNIST:
    
        def __init__(self):
            self.g = tf.Graph()
            with self.g.as_default():
                with tf.variable_scope("input"):
                    self.x = tf.placeholder(tf.float32, shape=[None, 784])
                    self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
                    x_image = tf.reshape(self.x, [-1,28,28,1])

                with tf.variable_scope("conv1"):
                    W_conv1 = self.weight_variable([5, 5, 1, 32])
                    b_conv1 = self.bias_variable([32])
            
                    h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
                    h_pool1 = self.max_pool_2x2(h_conv1)
    
                with tf.variable_scope("conv2"):
                    W_conv2 = self.weight_variable([5, 5, 32, 64])
                    b_conv2 = self.bias_variable([64])

                    h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
                    h_pool2 = self.max_pool_2x2(h_conv2)
    
                with tf.variable_scope("fc1"):
                    W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
                    b_fc1 = self.bias_variable([1024])

                    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
                    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
                    self.keep_prob = tf.placeholder(tf.float32)
                    h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
    
                with tf.variable_scope("fc2"):
                    W_fc2 = self.weight_variable([1024, 10])
                    b_fc2 = self.bias_variable([10])
        
                self.y_logit = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            
                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_logit, self.y_))
                self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
                self.correct_prediction = tf.equal(tf.argmax(self.y_logit, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
                self.sess = tf.Session(graph=self.g)
                self.sess.run(tf.global_variables_initializer())
    
        def train(self, mnist_data, num_steps=20000*50):
            for i in range(num_steps / 50):
                batch = mnist_data.train.next_batch(50)
                self.train_batch(batch[0], batch[1], i)
            
        def train_batch(self, batch_x, target_y, step):
            if step % 100 == 0:
                train_accuracy = self.eval_batch(batch_x, target_y)
                print("step %d, training accuracy %g" % (step, train_accuracy))
            self.sess.run([self.train_step], feed_dict={self.x: batch_x, self.y_: target_y, self.keep_prob: 0.5})

        def eval_batch(self, batch_x, target_y):
            accuracy = self.sess.run([self.accuracy], feed_dict={self.x: batch_x, self.y_: target_y, self.keep_prob: 1.0})
            return accuracy[0]
            
        def weight_variable(self, shape):
          return tf.get_variable('weights', shape, initializer=tf.contrib.layers.xavier_initializer())

        def bias_variable(self, shape):
          return tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.0))
    
        def conv2d(self, x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(self, x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", default='/tmp/mnist-data', help="Director where mnist downloaded dataset will be stored")
    parser.add_argument("--num-steps", default=20000*50, help="Number of total sample images to train on")
    args = parser.parse_args()
    
    mnist_data = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets(args.mnist_dir, one_hot=True)
    
    mnist = MNIST()
    mnist.train(mnist_data, args.num_steps)

    # Baseline: Test accuracy 0.9934
    test_accuracy = mnist.eval_batch(mnist_data.test.images, mnist_data.test.labels)
    print("Test accuracy %g" % test_accuracy)

if __name__ == "__main__":
    main()

