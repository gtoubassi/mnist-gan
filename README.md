# mnist-gan

mnist-gan is a simple GAN (Generative Adversarial Network) that learns how to generate images that look like mnist digits.  Separate networks are trained for each digit.  The idea is to train a "generator" network which when fed noise (in my case a 32 dimensional random vector) will generate an image that looks like an mnist style  "8" (for example).  In essence the network is finding a function of 32 variables that returns a matrix of pixels that look like a one.  Mind bending!  Below is an image that shows how the output for the 8 digit evolves over time (from left to right).  You can see how the digit starts to take shape from the noise:

![digit-8](https://cloud.githubusercontent.com/assets/640134/22179791/8308a134-e012-11e6-9757-0f8290a83c64.png)

The interesting question (for me) is how well can you train an mnist classifier when only fed GAN generated images?  Meaning take real mnist sample data, train a GAN, then use the GAN to generate a bunch of synthetic images, train a classifier, and see how it does vs if you trained directly on the sample data.

For the classifier I used the CNN architecture described on the [TensorFlow tutorial](https://www.tensorflow.org/tutorials/mnist/pros/).  When trained straight away for 1M "impressions" (20k batches of 50 each, training over the 50k training corpus) I get an accuracy of 99.34% (noice).  When training over 1M impressions on synthetic images I get 9X.XX%  This implies that the synthetic images aren't quite "good enough" to beat the real deal (even though we are sampling a far more diverse population: 1M unique images vs 50k images recycled 20 times each).  Also note that I burned about 300 hrs of cpu time training these GANs (probably over did it), so computationally its a pretty expensive path for this particular application.

### Notes

One thing I learned was that batch normalization really helped training and in particular the diversity of the population generated.  Fortunately tf.contrib.layers makes this a 1 liner!

Another note is that for some reason my 1 digit training went off the tracks at the end.  Meaning while training the GAN it was consistently putting out nice looking 1 digits, and then at the very end started spitting out crap.  All other digits were fine.  Perhaps I have a subtle bug, or my GAN architecture is missing something which makes it less stable.

### Resources

[Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/) TensorFlow Tutorial walks you through how to train a 99+% 

[TensorFlow for Poets Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) - Shows how to leverage a pre-trained Inception CNN to classify your own image corpus.  It is a hands on tutorial for [retrain.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining)

[Course Videos for CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG) - An overview of neural networks for image recognition and an excellent discussion of convolutional neural netowkrs in lecture 7.

[Course Videos for CS224D: Deep Learning for Natural Language Processing](https://www.youtube.com/playlist?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG) - Richard Socher's lecture videos cover how to use neural networks in NLP tasks.  Word embeddings are covered as well as some nitty gritty backprop derivations.
