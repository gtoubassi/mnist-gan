import tensorflow as tf
import numpy as np
import png
import random
import os
import code

class GAN:
    
    def __init__(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        self.g_x, self.g_y, self.g_y_logits = self.build_generator()

        with tf.variable_scope('discriminator') as scope:

            self.d_x = tf.placeholder(tf.float32, shape=[None, 784])
            self.d_y_ = tf.placeholder(tf.float32, shape=[None, 1])
            self.d_keep_prob = tf.placeholder(tf.float32, name='d_keep_prob')

            self.d_y, self.d_y_logit = self.build_discriminator(self.d_x, self.d_keep_prob)
        
            scope.reuse_variables()
            self.g_d_y, self.g_d_y_logit = self.build_discriminator(self.g_y, self.d_keep_prob)

        vars = tf.trainable_variables()
        # build loss function for discriminator
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(self.d_y_logit, tf.ones_like(self.d_y_logit))
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(self.g_d_y_logit, tf.zeros_like(self.g_d_y_logit))
        self.d_loss = d_loss_real + d_loss_fake
        d_training_vars = [v for v in vars if v.name.startswith('discriminator/')]
        self.d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=d_training_vars)

        self.d_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.d_y_logit), tf.round(self.d_y_)), tf.float32))


        # build loss function for training the generator
        self.g_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.g_d_y_logit, tf.ones_like(self.g_d_y_logit))
        g_training_vars = [v for v in vars if v.name.startswith('generator/')]
        self.g_d_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_d_loss, var_list=g_training_vars)
        

    def restore_session(self, path):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint(path))
        return sess

    def train_digit(self, mnist, digit, path):
    
        sess = tf.Session()    
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        
        # Get all the training '1' digits for our "real" data
        train_digits_of_interest = []
        for image, label in zip(mnist.train.images, mnist.train.labels):
            if label[digit]:
                train_digits_of_interest.append(image)
        test_digits_of_interest = []
        for image, label in zip(mnist.test.images, mnist.test.labels):
            if label[digit]:
                test_digits_of_interest.append(image)
    
        random.seed(12345)
        random.shuffle(train_digits_of_interest)
        random.shuffle(test_digits_of_interest)

        batch_size = 32
        for step in range(20000):
        
            batch_index = step * batch_size % len(train_digits_of_interest)
            batch_index = min(batch_index, len(train_digits_of_interest) - batch_size)
            batch = train_digits_of_interest[batch_index:(batch_index + batch_size)]

            #
            # Train the discriminator
            _, discriminator_loss = sess.run([self.d_optimizer, self.d_loss], feed_dict={self.is_training: True, self.d_x: batch, self.g_x: np.random.normal(size=(32,32)), self.d_keep_prob: 0.5})

            #
            # Train the generator
            z = np.random.normal(size=(32,32))
            _, generator_loss = sess.run([self.g_d_optimizer, self.g_d_loss], feed_dict={self.is_training: True, self.g_x: z, self.d_keep_prob: 1.0})

            if step % 100 == 0:
                print "Digit %d Step %d Eval: %f %f" % (digit, step, discriminator_loss[0], generator_loss[0])

            if step % 250 == 0:
                result = self.eval_generator(sess, 32)
                image = np.reshape(result, (32*28, 28)) * 255.0
                png.save_png('%s/digit-step-%06d.png' % (os.path.dirname(path), step), image)
                saver.save(sess, path, step)

                total_accuracy = 0
                total_samples = 0
                num_batches = 5
                for i in xrange(num_batches):
                    fake_samples = [(x, 0.0) for x in self.eval_generator(sess, 32)]
                    real_samples = [(x, 1.0) for x in random.sample(test_digits_of_interest, 32)]
                    samples = fake_samples + real_samples
                    random.shuffle(samples)
                    xs, ys = zip(*samples)
                    xs = np.asarray(xs)
                    ys = np.asarray(ys)
                    ys = np.reshape(ys, (64, 1))
                    accuracy = sess.run([self.d_accuracy], feed_dict={self.is_training: False, self.d_x: xs, self.d_y_: ys, self.d_keep_prob: 1.0})
                    total_accuracy += accuracy[0]
                    total_samples += len(samples)
                print("Discriminator eval accuracy %f%%" % (total_accuracy * 100.0 / total_samples))
                
        saver.save(sess, path, step)

    def eval_generator(self, sess, n_samples=1):
        result = sess.run([self.g_y], {self.is_training: False, self.g_x: np.random.normal(size=(n_samples,32))})
        return result[0]
    
    def leakyrelu(self, x):
        return tf.maximum(0.01*x,x)
        #return tf.nn.relu(x)

    def batch_norm(self, x):
        return tf.contrib.layers.batch_norm(x, decay=0.9, scale=True, is_training=self.is_training, updates_collections=None)

    def build_generator(self):
        with tf.variable_scope('generator') as scope:
            g_x = tf.placeholder(tf.float32, shape=[None, 32], name='input')

            with tf.variable_scope("fc1"):
                g_w1 = tf.get_variable("g_w1", shape=[32, 1024], initializer=tf.contrib.layers.xavier_initializer())
                g_b1 = tf.get_variable("g_b1", initializer=tf.zeros([1024]))
                g_h1 = self.leakyrelu(self.batch_norm(tf.matmul(g_x, g_w1) + g_b1))
        
            with tf.variable_scope("fc2"):
                g_w2 = tf.get_variable("g_w2", shape=[1024, 7*7*64], initializer=tf.contrib.layers.xavier_initializer())
                g_b2 = tf.get_variable("g_b2", initializer=tf.zeros([7*7*64]))
                g_h2 = self.leakyrelu(self.batch_norm(tf.matmul(g_h1, g_w2) + g_b2))
                g_h2_reshaped = tf.reshape(g_h2, [-1, 7, 7, 64])        
        
            with tf.variable_scope("conv3"):
                g_w3 = tf.get_variable("g_w3", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
                g_b3 = tf.get_variable("g_b3", initializer=tf.zeros([32]))
                g_deconv3 = tf.nn.conv2d_transpose(g_h2_reshaped, g_w3, output_shape=[32, 14, 14, 32], strides=[1, 2, 2, 1])
                g_h3 = self.leakyrelu(self.batch_norm(g_deconv3 + g_b3))
        
            with tf.variable_scope("conv4"):
                g_w4 = tf.get_variable("g_w4", shape=[5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
                g_b4 = tf.get_variable("g_b4", initializer=tf.zeros([1]))
                g_deconv4 = tf.nn.conv2d_transpose(g_h3, g_w4, output_shape=[32, 28, 28, 1], strides=[1, 2, 2, 1])

            g_y_logits = tf.reshape(g_deconv4 + g_b4, [-1, 784])
            g_y = tf.nn.sigmoid(g_y_logits)
        return g_x, g_y, g_y_logits

    def build_discriminator(self, x, keep_prob):
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
            
            d_h_conv1 = self.leakyrelu(self.batch_norm(conv2d(d_x_image, d_W_conv1) + d_b_conv1))
            d_h_pool1 = max_pool_2x2(d_h_conv1)
    
        with tf.variable_scope("conv2"):
            d_W_conv2 = weight_variable([5, 5, 32, 64])
            d_b_conv2 = bias_variable([64])

            d_h_conv2 = self.leakyrelu(self.batch_norm(conv2d(d_h_pool1, d_W_conv2) + d_b_conv2))
            d_h_pool2 = max_pool_2x2(d_h_conv2)
    
        with tf.variable_scope("fc1"):
            d_W_fc1 = weight_variable([7 * 7 * 64, 1024])
            d_b_fc1 = bias_variable([1024])

            d_h_pool2_flat = tf.reshape(d_h_pool2, [-1, 7*7*64])
            d_h_fc1 = self.leakyrelu(self.batch_norm(tf.matmul(d_h_pool2_flat, d_W_fc1) + d_b_fc1))
    
            d_h_fc1_drop = tf.nn.dropout(d_h_fc1, keep_prob)
    
        with tf.variable_scope("fc2"):
            d_W_fc2 = weight_variable([1024, 1])
            d_b_fc2 = bias_variable([1])
        
        d_y_logit = tf.matmul(d_h_fc1_drop, d_W_fc2) + d_b_fc2
        d_y = tf.sigmoid(d_y_logit)
        
        return d_y, d_y_logit
    
