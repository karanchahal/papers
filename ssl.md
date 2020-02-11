# Momentum Contrast Paper

1. It is a technique for unsupervised visual representation learning. 
2. They look at contrastive learning as a dictionary lookup. They use a dynamic dictionary with a queue and a **moving averaged** encoder. The keys in this dictionary are sampled from data (images or patches) and these keys are represented by an encoder network.
3. This paper uses unsupervised learning to train the encoders to perform a dictionary look up. An encoded query should be similiar to it's matching key and dissimiliar to others. Here the encoded query is the image or patch input into the encoder. This encoded query and it's matching key should be similiar and disimilar to others. HOW IS MATCHING KEY GOTTEN ?
4. They use contrastive losses on a large and consistent dictionary. 
5. The method claims good results under the "common linear protocol" on Imagenet classficiation. 
6. More importantly, they beat supervised pre training counterparts in 7 detection tasks. 

6. One reason for why self supervised works better for language than images could be that language already has tokens in the form of words, sub word units etc. CV in contrast has to self supervise a signal from a continous high dimensional space which is not structured (like words).
7. MOCO is a mechanism for building dictionaries for contrastive learning. I can be used for **various** pretext tasks. The task used in the paper is:  *a query matches a key if they are encoded views (e.g., different crops) of the same image.* 



#  List of Loss Functions

### Traditional Losses

A common way of defining a loss function is to measure the difference between a model’s prediction and a fixed target.

#### Examples

1. L1, L2 Losses for reconstructing the input's pixels.
2. Classifying the input into pre defined categories: 
   1. Taking image patches out and classifying the positions the patches are in. this is mentioned in the paper "Unsupervised Visual Representation Learning by Context Prediction"
   2. Predicting the colour bins that a pixel is in. Mentioned in paper "Colorful Image Colorization"
3. This classification is done through cross entropy or margin based losses.

### Contrastive loss Papers

These losses measure the similarities of sample pairs in a representation space. 

"Dimensionality reduction involves mapping a set of high dimensional input points onto a low dimensional manifold so that “similar” points in input space are mapped to nearby points on the manifold."

# Papers in this space

#### Unsupervised Feature Learning via Non Parametric Instance Discrimination

Paper Link: https://arxiv.org/pdf/1805.01978v1.pdf

Main crux of this approach is:

![image-20200207213205548](/Users/karanchahal/Library/Application Support/typora-user-images/image-20200207213205548.png)

Their novel unsupervised feature learning approach is instance-level discrimination. They treat each image instance as a distinct class of its own and train a classifier to distinguish between individual instance classes



### Contrastive Predictive Coding (version 2)

Paper Link : https://openreview.net/pdf?id=rJerHlrYwH

They hypothesize that a pretrained model should be able to learn new categories of objects from a *handful* of examples. Hence, representations that can do that are "better". When provided only 1% of examples in Imagenet, the models with CPC's representations retains a strong classification performance (a increase of 28% over supervised networks and 14% over semi supervised networks).

Contrastive Predictive Coding learns representations by training newtworks to predict the representation of future observations from those of past ones. In the context of images, the past observations are the top part of the image and the future observations are the bottom half. This is done via a contrastive loss in which the network must classify the "future" representation among a set of unrelated "negative" representations. This loss is called the **InfoNCE** loss.

In short, first the image is divided into overlapping patches. These patches are then encoded through a neural hetwork to get a feature representation for each patch. A masked convolutional net is applied on this grid of feature vectors. The masks are such that the resulting context vector for all patches above a certain point in the image. Hence, we constrain the receptive field. The prediction task is: given a context vector i,j and length l = (1....k). We try to predict the correct feature representation for position i+k, l for all l.

This is done via the InfoNCE loss. 

Some thoughts:

1. What if the feature vector of the patch does not depend on the patches over it ?



### Learning Deep Representations By Mutual Information Estimation And Maximization (ICLR- Bengio et al)

Paper Link: https://arxiv.org/pdf/1808.06670.pdf

**Idea**: In this paper, they propose a new method for learning high level representations. They firsr encode an image using a convolutional netwrok generating a feature map Z of  M by M in shape. Then this M by M feature map is summer together to get a M size "global" vector. Then this global vector, Y is used with a M size vector from Z (assumed to be a representation of a image patch) to make a network predict the value "Real" if the M size vector is from Z and "Fake" if it is from the Z of some other image. 



### Contrastive MultiView Coding

Paper Link: https://openreview.net/pdf?id=BkgStySKPB

Humans view the world through all kinds of sensors: auditory, sensory, visual etc. The authors hypothesize that a powerful representation is one that models view invariant features. Hence a representation is learn't thay maximise mutual information between different views. These different views can be:

1. Different colour channels of an image
2. DIfferent modalities (depth images and normal image, audio with image)

Any contrastive loss can be used in this setting. This paper achieved 69% Top 1 accuracy on Imagenet using this approach. The converted the image into a Lab format and then seperated the L channel and the ab channels, hence getting 2 modalities. 



### Learning Representations by Maximizing Mutual Information Across Views

Paper Link: https://arxiv.org/pdf/1906.00910.pdf

Similiar idea as above paper. Both were released concurrently. They hypothesize that multiple views of an image could be different augmentations of the same image.

Overall, after a brief run through across these papers, I have the following intuitions:

1. There are 2 methods for self supervision: Contrastive methods and Predictive methods. Contrastive methods constrast and pick between two or more options. Predictive methods try to predict the pixels or fill in the gaps. Most papers claim contrastive methods are more powerful. 
2. Noise Contrastive Estimation and InfoMax objectives are abound in the loss functions. Useful notes on NCE are given [here](http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf).

3. Essentially all good contrastive methods contrast between different views:

    	1. Image augmentations
    	2. Predict future image patches
    	3. Contrast between different colour channels of an image.

4. Adverserial methods for feature learning are also promising. Big GANS show that we can generate a very life like images from noise. [Deepmind's](https://arxiv.org/pdf/1907.02544.pdf) work looks into modifying this for feature learning and although they do not achieve very convincing results (60% classification accuracy), the idea of combining contrastive methods and predictive methods is interesting. 

   

#### List of Pretext Tasks

1. Recovering input under some corruption: (colourization, denoising)
2. Form pseudo labels: (image augmentations, patch orderings ([Jigsaw](https://arxiv.org/abs/1603.09246)) etc)

Several contrastive tasks can be reconfigured to be pretext tasks too: Examples:

1. Instance discrimination -> NCE ?
2. CPC -> Context auto encoding ?
3. Contrastive Multiview coding -> Colorization ?

#### Evaluation metrics

1. On multiple tasks. A good representation should help along various axis. eg: object detection, image classification and semantic segmentation. 

2. Limited training data. (Train on only 1% of training dataset)

3. While finetuning on new task, what percentage of layers are frozen ?

   

# Momentum Contrast Deep Dive

### Dictionary and Momentum Update

We have a dictionary. It has queries (q) and keys (k). A query matches with some keys in the dictionary. 

We have two encoders, f_q and f_k, the two encoders for a query and a key.
$$
q = f_q(x^i)
$$
Here, q is the query and $x^i$ is the input sample. 
$$
k = f_k(x^k)
$$
Similarly, k is the keys by the $f_k$ encoder. The input, x_q and x_k can be images, patches or context from a set of patches. The networks f_q and f_k can be separate, partially shared or identical. 

The dictionary stores a large database of keys encoded by the f_k encoder. The f_k encoder is updated slowly to keep the keys in the dictionary consistent. This is acheived via a momentum update.
$$
\theta_k = m\theta_k + (1-m)\theta_q
$$
Here, m is a moentum coefficient between 0 and 1. Only the paramters of the query encoder are updated via backpropagation. This equation given above allows the params of the key encoder to be updated much more smoothly than the query encoder.

### Loss

The loss is a function whose value is low for a tuple (q, k+) for keys which query matches to and high for those keys for which query does not match to. This loss is called **InfoNCE** in the paper is given by:
$$
Lq = − log \frac{exp(q·k^+/τ)}{\sum_{i=0}^{K}{exp(q·k_i/τ)}}
$$
The similarity is expressed at a dot product between the query and the key vector. The sum is expressed over 1 positive and K negative samples. Intuitively, this loss is a log loss over a K+1 way softmax based classifier. 



## Pseudo Code

```python
# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (CxK)
# m: momentum
# t: temperature
f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
  x_q = aug(x) # a randomly augmented version
  x_k = aug(x) # another randomly augmented version
  q = f_q.forward(x_q) # queries: NxC
  k = f_k.forward(x_k) # keys: NxC
  k = k.detach() # no gradient to keys
  # positive logits: Nx1, similiarity: dot product of query and similar keys
  l_pos = bmm(q.view(N,1,C), k.view(N,C,1))
  # negative logits: NxK: similiarity: dot product of query and dissimiliar keys
  l_neg = mm(q.view(N,C), queue.view(C,K))
  # logits: Nx(1+K)
  logits = cat([l_pos, l_neg], dim=1)
  # contrastive loss, Eqn.(1)
  labels = zeros(N) # positives are the 0-th
  loss = CrossEntropyLoss(logits/t, labels)
  # SGD update: query network
  loss.backward()
  update(f_q.params)
  # momentum update: key network
  f_k.params = m*f_k.params+(1-m)*f_q.params
  # update dictionary
  enqueue(queue, k) # enqueue the current minibatch
  dequeue(queue) # dequeue the earliest minibatch
```

