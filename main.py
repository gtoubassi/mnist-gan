#!/usr/bin/env python
#
import argparse
import tensorflow as tf
import os
import random
from gan import GAN

def gen_samples(gan, sessions):
    samples = []
    for i, s in enumerate(sessions):
        samples_for_digit = gan.eval_generator(s, 32)
        for sample in samples_for_digit:
            samples.append((samples, i))
    random.shuffle(samples)
    samples = zip(*samples)
    samples[0] = list(samples[0])
    samples[1] = tf.contrib.learn.python.learn.datasets.mnist.dense_to_one_hot(list(samples[1]), 10)
    return samples
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", default='/tmp/mnist-data', help="Directory where mnist downloaded dataset will be stored")
    parser.add_argument("--output-dir", default='output', help="Directory where models will be saved")
    parser.add_argument("--train-digits", help="Comma separated list of digits to train generators for (e.g. '1,2,3')")
    parser.add_argument("--train-mnist", help="If specified, train the mnist classifier based on generated digits from saved models")
    global args
    
    args = parser.parse_args()    
    
    if args.train_digits:
        mnist_data = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets(args.mnist_dir, one_hot=True)
        gan = GAN()
        for digit in map(int, args.train_digits.split(',')):
            path = "%s/digit-%d/model" % (args.output_dir, digit)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            gan.train_digit(mnist_data, digit, path)
    elif args.train_mnist:
        gan = GAN()
        sessions = [gan.restore_session("%s/digit-%d/model" % (args.output_dir, digit)) for digit in range(10)]
        samples = []
        
        mnist = MNIST()
        for i in range(20000):
            if len(samples) < 50:
                samples = gen_samples(gan, sessions)
            xs = samples[0][:50]
            ys = samples[1][:50]
            samples[0] = samples[0][50:]
            samples[1] = samples[1][50:]
            self.train_batch(xs, ys, step)

if __name__ == "__main__":
    main()
