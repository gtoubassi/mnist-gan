# mnist-gan

mnist-gan is a simple GAN (Generative Adversarial Network) that learns how to generate images that look like mnist digits.  Separate networks are trained for each digit.  The idea is to train a "generator" network which when fed noise (in my case a 32 dimensional random vector) will generate an image that looks like an mnist style  "8" (for example).  In essence the network is finding a function of 32 variables that returns a matrix of pixels that look like an eight.  Mind bending!  Below is an image that shows how the output for the 8 digit evolves over time (from left to right).  You can see how the digit starts to take shape from the noise:

![digit-8](https://cloud.githubusercontent.com/assets/640134/22179791/8308a134-e012-11e6-9757-0f8290a83c64.png)

The interesting question (for me) is how well can you train an mnist classifier when only fed GAN generated images?  Meaning take real mnist sample data, train a GAN, then use the GAN to generate a bunch of synthetic images, train a classifier, and see how it does vs if you trained directly on the sample data.

For the classifier I used the CNN architecture described on the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/mnist/pros/).  When trained straight away for 1M "impressions" (20k batches of 50 each, training over the 50k training corpus) I get an accuracy of 99.34% (noice).  When training over 1M impressions on synthetic images I get 98.37%  This implies that the synthetic images aren't quite "good enough" to beat the real deal (even though we are sampling a far more diverse population: 1M unique images vs 50k images recycled 20 times each).  On the other hand its pretty impressive that we've found a manifold which generates digits so much like the real thing that a classifier trained on it exclusively does so well.

### How to run

To train a GAN for each digit 0-9: `./main.py --train-digits 0,1,2,3,4,5,6,7,8,9`.  Models and sample generated images will be saved to `./output`.

To train mnist on GAN generated models which are sitting in ./output: `./main.py --train-mnist` (you must have generated models for all 10 digits).

### Notes

One thing I learned was that batch normalization really helped training and in particular the diversity of the population generated.  Fortunately tf.contrib.layers makes this a 1 liner!

Another note is that for some reason my 1 digit training went off the tracks at the end.  Meaning while training the GAN it was consistently putting out nice looking 1 digits, and then at the very end started spitting out crap.  All other digits were fine.  Perhaps I have a subtle bug, or my GAN architecture is missing something which makes it less stable.

### Resources

[Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/) TensorFlow Tutorial walks you through how to train a 99+% 

[An Introduction to Generative Adversarial Networks](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/) A nice blog post showing a simple GAN attempting to learn a gaussian distribution with code in TensorFlow

[Image Completion with Deep Learning in TensorFlow](https://bamos.github.io/2016/08/09/deep-completion/)  A great blog post showing using GANs to generate images.  Also shows the vector math you can do on the generator input vectors to combine features.  i.e. imagine you have 3 vectors which when fed to your generator output the following: (1) a smiling man, (2) a straight-faced/neutral man, (3) a straight-faced/neutral woman, you can take V1-V2+V3 and feed that to the GAN and you will get a smiling woman.  V1-V2 effectively captures the "smiling" expression, which you can add to other vectors.

[NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160) A great overview from Ian Goodfellow, the inventor of GANs. ([slides](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf))
