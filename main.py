#!/usr/bin/env python
#
import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import code
import random
from gan import GAN
from mnist import MNIST

def gen_samples(gan, sessions):
    samples = []
    for i, s in enumerate(sessions):
        samples_for_digit = gan.eval_generator(s, 32)
        for sample in samples_for_digit:
            samples.append((sample, i))
    random.shuffle(samples)
    samples = zip(*samples)
    samples[0] = np.asarray(samples[0])
    samples[1] = tf.contrib.learn.python.learn.datasets.mnist.dense_to_one_hot(np.asarray(samples[1]), 10)
    return samples
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", default='/tmp/mnist-data', help="Directory where mnist downloaded dataset will be stored")
    parser.add_argument("--output-dir", default='output', help="Directory where models will be saved")
    parser.add_argument("--train-digits", help="Comma separated list of digits to train generators for (e.g. '1,2,3')")
    parser.add_argument("--train-mnist", action='store_true', help="If specified, train the mnist classifier based on generated digits from saved models")
    global args
    
    args = parser.parse_args()    
    
    mnist_data = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets(args.mnist_dir, one_hot=True)

    if args.train_digits:
        gan = GAN()
        for digit in map(int, args.train_digits.split(',')):
            path = "%s/digit-%d/model" % (args.output_dir, digit)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            gan.train_digit(mnist_data, digit, path)
    elif args.train_mnist:
        gan = GAN()
        print("Loading generator models...")
        sessions = [gan.restore_session("%s/digit-%d" % (args.output_dir, digit)) for digit in range(10)]
        print("Done")
        samples = [[], []]
        
        mnist = MNIST()
        for step in range(20000):
            if step % 2:
                if len(samples[0]) < 50:
                    samples = gen_samples(gan, sessions)
                xs = samples[0][:50]
                ys = samples[1][:50]
                samples[0] = samples[0][50:]
                samples[1] = samples[1][50:]
            else:
                batch = mnist_data.train.next_batch(50)
                xs = batch[0]
                ys = batch[1]
            mnist.train_batch(xs, ys, step)
        test_accuracy = mnist.eval_batch(mnist_data.test.images, mnist_data.test.labels)
        print("Test accuracy %g" % test_accuracy)

if __name__ == "__main__":
    main()
