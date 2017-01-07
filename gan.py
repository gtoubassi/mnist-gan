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
        if label[3]:
            ones.append(image)
    print(len(ones))
    sess = tf.Session()    
    
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    
    def leakyrelu(x):
        return tf.maximum(0.01*x,x)
        #return tf.nn.relu(x)

    def xbatch_norm(x):
        return x

    def batch_norm(x):
        return tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, is_training=is_training, updates_collections=None)

    def xbatch_norm(x):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        n_out = x.get_shape()[-1]
        with tf.variable_scope('bn'):
            beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]), trainable=True)
            gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]), trainable=True)
            moments_axes = list(range(x.get_shape().ndims - 1))
            batch_mean, batch_var = tf.nn.moments(x, moments_axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)

            ema_apply_op = ema.apply([batch_mean, batch_var])
            
            with tf.control_dependencies([ema_apply_op]):
                mean, var = tf.identity(batch_mean), tf.identity(batch_var)

#            def mean_var_with_update():
#                ema_apply_op = ema.apply([batch_mean, batch_var])
#                with tf.control_dependencies([ema_apply_op]):
#                    return tf.identity(batch_mean), tf.identity(batch_var)

#            mean, var = tf.cond(is_training,
#                                mean_var_with_update,
#                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
        
    # Generator network
    with tf.variable_scope('generator') as scope:
        g_x = tf.placeholder(tf.float32, shape=[None, 32], name='input')

        with tf.variable_scope("fc1"):
            g_w1 = tf.get_variable("g_w1", shape=[32, 1024], initializer=tf.contrib.layers.xavier_initializer())
            g_b1 = tf.get_variable("g_b1", initializer=tf.zeros([1024]))
            g_h1 = leakyrelu(batch_norm(tf.matmul(g_x, g_w1) + g_b1))
        
        with tf.variable_scope("fc2"):
            g_w2 = tf.get_variable("g_w2", shape=[1024, 7*7*64], initializer=tf.contrib.layers.xavier_initializer())
            g_b2 = tf.get_variable("g_b2", initializer=tf.zeros([7*7*64]))
            g_h2 = leakyrelu(batch_norm(tf.matmul(g_h1, g_w2) + g_b2))
            g_h2_reshaped = tf.reshape(g_h2, [-1, 7, 7, 64])        
        
        with tf.variable_scope("conv3"):
            g_w3 = tf.get_variable("g_w3", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
            g_b3 = tf.get_variable("g_b3", initializer=tf.zeros([32]))
            g_deconv3 = tf.nn.conv2d_transpose(g_h2_reshaped, g_w3, output_shape=[32, 14, 14, 32], strides=[1, 2, 2, 1])
            g_h3 = leakyrelu(batch_norm(g_deconv3 + g_b3))
        
        with tf.variable_scope("conv4"):
            g_w4 = tf.get_variable("g_w4", shape=[5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
            g_b4 = tf.get_variable("g_b4", initializer=tf.zeros([1]))
            g_deconv4 = tf.nn.conv2d_transpose(g_h3, g_w4, output_shape=[32, 28, 28, 1], strides=[1, 2, 2, 1])

        g_y_logits = tf.reshape(g_deconv4 + g_b4, [-1, 784])
        g_y = tf.nn.sigmoid(g_y_logits)
    
    def build_discriminator(x, keep_prob):
        def weight_variable(shape):
          return tf.get_variable('weights', shape, initializer=tf.contrib.layers.xavier_initializer())

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
            
            d_h_conv1 = leakyrelu(batch_norm(conv2d(d_x_image, d_W_conv1) + d_b_conv1))
            d_h_pool1 = max_pool_2x2(d_h_conv1)
    
        with tf.variable_scope("conv2"):
            d_W_conv2 = weight_variable([5, 5, 32, 64])
            d_b_conv2 = bias_variable([64])

            d_h_conv2 = leakyrelu(batch_norm(conv2d(d_h_pool1, d_W_conv2) + d_b_conv2))
            d_h_pool2 = max_pool_2x2(d_h_conv2)
    
        with tf.variable_scope("fc1"):
            d_W_fc1 = weight_variable([7 * 7 * 64, 1024])
            d_b_fc1 = bias_variable([1024])

            d_h_pool2_flat = tf.reshape(d_h_pool2, [-1, 7*7*64])
            d_h_fc1 = leakyrelu(batch_norm(tf.matmul(d_h_pool2_flat, d_W_fc1) + d_b_fc1))
    
            d_h_fc1_drop = tf.nn.dropout(d_h_fc1, keep_prob)
    
        with tf.variable_scope("fc2"):
            d_W_fc2 = weight_variable([1024, 1])
            d_b_fc2 = bias_variable([1])
        
        d_y_logit = tf.matmul(d_h_fc1_drop, d_W_fc2) + d_b_fc2
        d_y = tf.sigmoid(d_y_logit)
        
        return d_y, d_y_logit, d_keep_prob
    
    
    with tf.variable_scope('discriminator') as scope:

        d_x = tf.placeholder(tf.float32, shape=[None, 784])
        d_keep_prob = tf.placeholder(tf.float32, name='d_keep_prob')

        d_y, d_y_logit, d_keep_prob = build_discriminator(d_x, d_keep_prob)
        
        scope.reuse_variables()
        g_d_y, g_d_y_logit, g_d_keep_prob = build_discriminator(g_y, d_keep_prob)    
    
    vars = tf.trainable_variables()
    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(d_y_logit, tf.ones_like(d_y_logit))
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(g_d_y_logit, tf.zeros_like(g_d_y_logit))
    d_loss = d_loss_real + d_loss_fake
    d_training_vars = [v for v in vars if v.name.startswith('discriminator/')]
    d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=d_training_vars)

    # build loss function for training the generator
    #g_d_loss = tf.reduce_mean(tf.square(1 - g_d_y))
    
    g_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(g_d_y_logit, tf.ones_like(g_d_y_logit))
    g_d_correct_prediction = tf.equal(tf.round(g_d_y), tf.constant(1.0))
    g_d_eval = tf.reduce_mean(tf.cast(g_d_correct_prediction, tf.float32))
    g_training_vars = [v for v in vars if v.name.startswith('generator/')]
    g_d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(g_d_loss, var_list=g_training_vars)

    sess.run(tf.global_variables_initializer())

    for epoch in range(20000):

        #
        # Train the discriminator
        _, discriminator_loss = sess.run([d_optimizer, d_loss], feed_dict={is_training: True, d_x: random.sample(ones, 32), g_x: np.random.normal(size=(32,32)), d_keep_prob: 0.5})

        #
        # Train the generator
        z = np.random.normal(size=(32,32))
        _, generator_loss = sess.run([g_d_optimizer, g_d_loss], feed_dict={is_training: True, g_x: z, d_keep_prob: 1.0})

        if epoch % 10 == 0:

            print "Epoch %d Generator Eval: %f %f" % (epoch, discriminator_loss[0], generator_loss[0])

            result = sess.run([g_y], {is_training: False, g_x: np.random.normal(size=(32,32))})
            image = np.reshape(result[0], (32*28, 28)) * 255.0
            save_png('gen-0-epoch-%06d.png' % epoch, image)            

if __name__ == "__main__":
    main()
