# An Analysis of Neural Language Modeling at Multiple Scales

This paper seeks to construct a language model that gives state of the art results on both __word__ and __character__
level modelling. A lot of previous models provide results using different hardwares and varying experimental conditions.
They provide good results on small datasets with small vocabularies but fail to replicate results on large 
vocabularies or character level modelling. 

To combat this variance, this paper aims to provide a strong baseline for
state of the art langauge modelling. It seeks to provide results on a single GPU in a manageble training time of hours or days.


# Model features

## QRNN vs LSTM

QRNNs are a variant on LSTM , they use convolutional networks to process the input instead of the sequentail mechanism of the LSTM.
The benefit of such an approach is that the QRNN can now parallelise the input sequence, this leads to a 16x gain in spped over a
CUDA optimised LSTM. Also LSTMs require far less epochs to reach the same accuracy, hence they are 2-4x faster to train.

## Longer BPTT Lengths

The time taken while backpropogating through a long sequence of text grows exponentially the longer the sequence is.
Hence, backpropogation is broken after a few steps of processing input. This is called Truncated BPTT. In LSTM's, BPTT 
windows of 50 for word level and 100 for character level modelling are used. Longer BPTT have the potential for an accuracy
improvement because they can theoretically capture longer term dependencies. The QRNN is perfect for using long BPTT windows.
This is because it does not feature the slow sequential hidden-to-hidden matrix multiplication at each timestep,instead it 
relies on a fast element-wise operation. The parallel nature of QRNN makes it perfect for improving GPU utilization.


## Adaptive Softmax and Weight Tying

For processing large vocabularies, one bottleneck in speeds is the large softmax multiplication. This is also called the softmax
bottleneck. To evade this, an adaptive softmax is used. The **adaptive softmax** breaks down the softmax vector into 2 levels.
The first level is called the short list, it contains the most frequent words in the vocabulary. The second level contains clusters
of rarely found words.Each of these clusters has a representative token on the short list. 

The intuition is, according to Ziph's Law, most words will require a softmax over the short list. Hence, that expensive softmax will only
be required rarely. Along with the softmax, another apprach known as **weight tying** is used to facilitate even more memory
optimisation.

