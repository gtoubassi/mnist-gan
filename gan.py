#!/usr/bin/env python
#
import argparse
import tensorflow as tf
import math
import random
import numpy as np
import png

def save_png(filename, array):
    pngfile = open(filename, 'wb')
    pngWriter = png.Writer(array.shape[1], array.shape[0], greyscale=True)
    pngWriter.write(pngfile, array)
    pngfile.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-data-dir", default='/tmp/mnist-data', help="Director where mnist downloaded dataset will be stored")
    args = parser.parse_args()    
    
    mnist = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets(args.mnist_data_dir, one_hot=True)
    
    # Get all the training '1' digits for our "real" data
    ones = []
    for image, label in zip(mnist.train.images, mnist.train.labels):
        # label is one hot encoded so if label[1] is on, its a one
        if label[1]:
            ones.append(image)

    sess = tf.Session()    
    
    # Generator network
    with tf.variable_scope('generator') as scope:
        g_x = tf.placeholder(tf.float32, shape=[None, 32], name='input')

        stdv = 1/math.sqrt(32)
        g_w1 = tf.Variable(tf.random_uniform([32, 1024], minval=-stdv, maxval=stdv))
        g_b1  = tf.Variable(tf.zeros([1024]))
        g_h1 = tf.nn.relu(tf.matmul(g_x, g_w1) + g_b1)
        
        stdv = 1/math.sqrt(1024)
        g_w2 = tf.Variable(tf.random_uniform([1024, 7*7*64], minval=-stdv, maxval=stdv))
        g_b2  = tf.Variable(tf.zeros([7*7*64]))
        g_h2 = tf.nn.relu(tf.matmul(g_h1, g_w2) + g_b2)
        g_h2_reshaped = tf.reshape(g_h2, [-1, 7, 7, 64])        
        
        g_w3 = tf.Variable(tf.random_uniform([5, 5, 32, 64], minval=-.02, maxval=.02))
        g_b3  = tf.Variable(tf.zeros([32]))
        g_deconv3 = tf.nn.conv2d_transpose(g_h2_reshaped, g_w3, output_shape=[32, 14, 14, 32], strides=[1, 2, 2, 1])
        g_h3 = tf.nn.relu(g_deconv3 + g_b3)
        
        g_w4 = tf.Variable(tf.random_uniform([5, 5, 1, 32], minval=-.02, maxval=.02))
        g_b4  = tf.Variable(tf.zeros(1))
        g_deconv4 = tf.nn.conv2d_transpose(g_h3, g_w4, output_shape=[32, 28, 28, 1], strides=[1, 2, 2, 1])

        g_y_image = tf.nn.sigmoid(g_deconv4 + g_b4)
        g_y = tf.reshape(g_y_image, [-1, 784])        
    
    def build_discriminator(x, keep_prob):
        def weight_variable(shape):
          return tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

        def bias_variable(shape):
          return tf.get_variable('biases', shape, initializer=tf.constant_initializer(0.0))
    
        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("input"):
            d_x_image = tf.reshape(x, [-1,28,28,1])

        with tf.variable_scope("conv1"):
            d_W_conv1 = weight_variable([5, 5, 1, 32])
            d_b_conv1 = bias_variable([32])
            
            d_h_conv1 = tf.nn.relu(conv2d(d_x_image, d_W_conv1) + d_b_conv1)
            d_h_pool1 = max_pool_2x2(d_h_conv1)
    
        with tf.variable_scope("conv2"):
            d_W_conv2 = weight_variable([5, 5, 32, 64])
            d_b_conv2 = bias_variable([64])

            d_h_conv2 = tf.nn.relu(conv2d(d_h_pool1, d_W_conv2) + d_b_conv2)
            d_h_pool2 = max_pool_2x2(d_h_conv2)
    
        with tf.variable_scope("fc1"):
            d_W_fc1 = weight_variable([7 * 7 * 64, 1024])
            d_b_fc1 = bias_variable([1024])

            d_h_pool2_flat = tf.reshape(d_h_pool2, [-1, 7*7*64])
            d_h_fc1 = tf.nn.relu(tf.matmul(d_h_pool2_flat, d_W_fc1) + d_b_fc1)
    
            d_h_fc1_drop = tf.nn.dropout(d_h_fc1, keep_prob)
    
        with tf.variable_scope("fc2"):
            d_W_fc2 = weight_variable([1024, 1])
            d_b_fc2 = bias_variable([1])

        d_y = tf.sigmoid(tf.matmul(d_h_fc1_drop, d_W_fc2) + d_b_fc2)
    
        d_correct_prediction = tf.equal(tf.round(d_y), tf.round(d_y_))
        d_eval = tf.reduce_mean(tf.cast(d_correct_prediction, tf.float32))
        
        return d_y, d_eval, d_keep_prob
    
    
    with tf.variable_scope('discriminator') as scope:

        d_x = tf.placeholder(tf.float32, shape=[None, 784])
        d_y_ = tf.placeholder(tf.float32, shape=[None, 1], name='d_y_')
        d_keep_prob = tf.placeholder(tf.float32, name='d_keep_prob')

        d_y, d_eval, d_keep_prob = build_discriminator(d_x, d_keep_prob)
        
        scope.reuse_variables()
        g_d_y, g_d_eval, g_d_keep_prob = build_discriminator(g_y, d_keep_prob)
    
    
    vars = tf.trainable_variables()
    d_loss = tf.reduce_mean(tf.square(d_y_ - d_y))
    d_training_vars = [v for v in vars if v.name.startswith('discriminator/')]
    d_optimizer = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_training_vars)

    # build loss function for training the generator
    g_d_loss = tf.reduce_mean(tf.square(1 - g_d_y))
    g_d_correct_prediction = tf.equal(tf.round(g_d_y), tf.constant(1.0))
    g_d_eval = tf.reduce_mean(tf.cast(g_d_correct_prediction, tf.float32))
    g_training_vars = [v for v in vars if v.name.startswith('generator/')]
    g_d_optimizer = tf.train.AdamOptimizer().minimize(g_d_loss, var_list=g_training_vars)

    sess.run(tf.global_variables_initializer())

    def genbatch(n):
        half_n = n/2
        result = sess.run([g_y], {g_x: np.random.uniform(-1, 1, size=(32,32))})
        full = result[0][:half_n]
        batch_g = [(full[i], [0]) for i in xrange(full.shape[0])]
        
        batch_mnist = zip(random.sample(ones, half_n), [[1]]*half_n)
    
        combined = batch_g + batch_mnist
        random.shuffle(combined)
    
        return zip(*combined)


    for epoch in range(20):

        #
        # Train the discriminator until it can beat the generator
        #

        for i in range(10000):

            if i % 2 == 0:
                batch_d_x, batch_d_y = genbatch(64)
                accuracy = sess.run([d_eval], {d_x: batch_d_x, d_y_: batch_d_y, d_keep_prob: 1.0})
                print "Epoch %d Discriminator Eval %d: %f" % (epoch, i, accuracy[0])
                if accuracy[0] > 62.0/64.0:
                    print "Discriminator network has achieved mastery over the current generator"
                    break;

            batch_d_x, batch_d_y = genbatch(32)    
            d_optimizer.run(feed_dict={d_x: batch_d_x, d_y_: batch_d_y, d_keep_prob: 0.5}, session=sess)

        #
        # Train the generator until it can beat the discriminator
        #
    
        for i in range(10000):

            if i % 2 == 0:
                accuracy = sess.run([g_d_eval], {g_x: np.random.uniform(-1, 1, size=(32,32)), d_keep_prob: 1.0})
                print "Epoch %d Generator Eval %d: %f" % (epoch, i, accuracy[0])
                if accuracy[0] > 30.0/32.0:
                    print "Generator network has achieved mastery over the current discriminator"
                    break;

            g_d_optimizer.run(feed_dict={g_x: np.random.uniform(-1, 1, size=(32,32)), d_keep_prob: 1.0}, session=sess)

        if epoch % 1 == 0:
            result = sess.run([g_y], {g_x: np.random.uniform(-1, 1, size=(32,32))})
            image = np.reshape(result[0], (32*28, 28)) * 255.0
            save_png('gen-0-epoch-%06d.png' % epoch, image)            

if __name__ == "__main__":
    main()
