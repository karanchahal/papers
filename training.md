# New Methods To Train Your Deep Learning Model

In this post, we shall talk about some new techniques on how to train your model. more specifically we shall talk about:

1. Finding your learning rate
2. Varying your learning Rate
3. Converging at global minima


## Finding Your Learning Rate

The learning rate parameter is a tricky thing to find. 
```
What learning rate should you start with?
```

It generally involves a lot of hit and trial.
This process can take anywhere from 2- 3 iterations of your model training procedure. 
Although,it is generally intuitive as to how one decides upon a learning rate. A large number of data scientists use a manual approach to this problem.

Generally what we do is, we look at the loss for a particular learning rate. if the loss stays the _same or goes up_ (__divergence__). 
We _bump down_ the learning rate by a factor (10 generallly). This process continues until our loss doesn't go down further even for very small learning rates.
At this point one has achieved _convergence_ for that network.

So the question we ask now is,
```Can this be automated ?```

There are two approaches to this problem.

1. Adaptive Learning Rates
2. Cyclic Learning Rates

We shall talk about Cyclic Learning Rates in this post. Comparisions with Adaptive Learning Rates will be made in a future post.


So coming back to our question. The answer is yes,we can automate this learning rate problem.
A researcher by the name of _Leslie Smith_ has been researching on making training __fast,efficient and accurate__. His work is generally unknown but 
the __Fast.ai__ videos have implemented his approach and have really popularised some techniques of his.

So the way the learning rate is found out is as follows.

### Steps

1. Start with a really small learning rate ~.000001.
2. Slowly increase the learning rate while training on a single epoch.
3. Stop increasing the learning rate once the loss starts to go up.

One can visualise this phenomenon via this graph


As we can see , there is an _inflection point_ in the grapoh where it starts to go up. We take the learning rate at this inflection point and start with this
as our initial learning rate to train our model.


