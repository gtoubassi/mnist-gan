#!/usr/bin/env python
#
import argparse
import tensorflow as tf
import os
from gan import GAN
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", default='/tmp/mnist-data', help="Directory where mnist downloaded dataset will be stored")
    parser.add_argument("--output-dir", default='output', help="Directory where models will be saved")
    parser.add_argument("--train-digits", help="Comma separated list of digits to train generators for (e.g. '1,2,3')")
    global args
    
    args = parser.parse_args()    
    
    if args.train_digits:
        mnist = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets(args.mnist_dir, one_hot=True)
        gan = GAN()
        for digit in map(int, args.train_digits.split(',')):
            path = "%s/digit-%d/model" % (args.output_dir, digit)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            gan.train_digit(mnist, digit, path)

if __name__ == "__main__":
    main()
