#!/usr/bin/env python
#
import argparse
import tensorflow as tf
import math
import random
import numpy as np
import png
import os
from gan import GAN

def save_png(filename, array):
    pngfile = open(filename, 'wb')
    pngWriter = png.Writer(array.shape[1], array.shape[0], greyscale=True)
    pngWriter.write(pngfile, array)
    pngfile.close()

def train_digit(mnist, digit):
    
    model_path = "%s/digit-%d/model" % (args.model_dir, digit)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    # Get all the training '1' digits for our "real" data
    digits_of_interest = []
    for image, label in zip(mnist.train.images, mnist.train.labels):
        if label[digit]:
            digits_of_interest.append(image)
    
    gan = GAN()

    random.seed(12345)
    random.shuffle(digits_of_interest)

    batch_size = 32
    for step in range(20000):
        
        batch_index = step * batch_size % len(digits_of_interest)
        batch_index = min(batch_index, len(digits_of_interest) - batch_size)
        batch = digits_of_interest[batch_index:(batch_index + batch_size)]

        discriminator_loss, generator_loss = gan.train_step(batch)

        if step % 100 == 0:
            print "Digit %d Step %d Eval: %f %f" % (digit, step, discriminator_loss, generator_loss)

        if step % 250 == 0:
            result = gan.eval_generator(32)
            image = np.reshape(result, (32*28, 28)) * 255.0
            save_png('%s/digit-step-%06d.png' % (os.path.dirname(model_path), step), image)
            gan.save(model_path, step)
    gan.save(model_path, step)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", default='/tmp/mnist-data', help="Directory where mnist downloaded dataset will be stored")
    parser.add_argument("--model-dir", default='saved-models', help="Directory where models will be saved")
    global args
    
    args = parser.parse_args()    
    
    mnist = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets(args.mnist_dir, one_hot=True)
    
    for digit in range(10):
        train_digit(mnist, digit)

if __name__ == "__main__":
    main()
