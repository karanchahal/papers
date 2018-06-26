# Gradients Of Deep Learning

This blog post talks about everything regarding the gradients and weights of deep learning. More specifically we will talk about these things.

1. Weight Initialisation

2. Batch Normalisation

3. Optimisation Algorithms for Tweaking Weights (Variants of Gradient Descent)


## Weight Initialisation


The weights to a neural network need to be initialised randomly  at first. If we are clever about this initialisation process, we can help the network reach _convergence quicker_, _avoid local minima_, _reduce chances of divergence_.

Here are some methods for initialising the weights of a neural network.

## Initilise the weights by 0

This isn't a great technique because of the following problem.

1. All the neurons will do the same thing , because they start at the same spot.

2. They might not be dead, but they're basically useless as they'll update in the same way.


## Initialise with small random values 

These numbers will be sampled from a normal distribution with a __zero__ mean and __1e-2__ standard deviation (small).

This works good for smaller networks but not so well for bigger networks.

This is because of the following problems:

1. If we take a 10 layer deep network and plot the mean and standard deviation of the layers.

2. We see that the mean of the layers remains close to zero but the standard deviation shrinks progressively leading to be a very peaky normal distribution. Hence, all of our activations become very close to _zero_.

3. When we do our backward pass now, the gradients will be very _close to zero_ . Leading to the __vanishing gradient__ problem.

## Initialise with larger random values 

These numbers will be sampled from a normal distribution with a __zero__ mean and __1.0__ standard deviation (larger).

The problem here is as follows:

1. Since our inputs are now all _big_, and we're passing them through a tanh linearity. The neuron becomes saturated. Either we'll get __-1__ or __+1__ . When tanh is saturated, the gradients turn to __zero__ and the gradients are __not updated__.

## Use Xavier Initialisation

These numbers will be sampled from a normal distribution with a __zero__ mean where we scale with the input we have. Intutively, if we have a small number of input, then we will get larger weights. If we have many inputs, we want smaller weights to get the same spread as the output.

In short, we want to maintain the same variance in the output and input.

Problems:

1. Assumes linear activation.

2. Breaks with Relu

3. We can counter this by dividing by 2 , taking into factor that we are nullifying half the sample space by doing max(0,x) via ReLU.


## Batch Normalisation

So what we want to do is to keep the input at each layer to be at a normal distribution.

```
So why don't we just do it ?
```

That's basically what Batch Normalisation does. It keeps the __input to the activations__ in the normal distribution form.(bell curve)

```
x = (x - mean(x))/Sqrt(Variance(x))
input  = x
```

This is differentiable. Here's how we implement it in practice.

1. We have input data as N training examples with dimension D in our current batch.

2. We will compute mean and variance across each example of the batch. We compute this mean and variance across our batch and then normalise by this over all examples. We do this for whatever output we get from a layer __before__ we plug into the activation function.

3. Usually inserted after Fully Connected or Convolutional networks before the activation functions.

4. Hence, we are modifying the input to be fed into the activation such  that the neuron doesn't saturate.

5. We scale the input and normalise it. For convolutional layers we will have one mean and one variance per activation map. So they obey the convolutional property where we want nearby data to be normalised the same way.

6. We want to be able to control how much saturation we want our neuron to have. Hence we add two parameters __A__ and __B__. Where A and B can effectively learn the variance and the mean respectively to recover the identity mapping.To control the saturation.

Hence, now the Batch Normalisation is given by

```
// input = prev batch norm output

final_input = A*input + B

```

## Advantages

1. Improves gradient flow through network.

2. Allows higher learning rates.

3. Removes strong interdependence on initialisation.

4. Acts as a form of regularisation. The output of each layer, each of these outputs is an output of both __input__ and __other examples in the batch___ (because of the mean and variance). Hence, the output is no longer deterministic, types all these examples together. This leads to a slight regularisation effect.

### Batch Norm During Testing

Batch Norm behaves differently during test time. We cannot take the mean and variance over a batch. This is no longer possible as during test time,we will have one input at a time. So during test time, we keep a emperical mean (running averages apprach). We use this mean during test time.

## Optimisation Algorithms for Tweaking Weights (Variants of Gradient Descent)

This part of the post deals with how gradients are updated.

We shall do into various optimisation algorithms that we use to tweak the weights.

### Stochastic Gradient Descent

```
weights += -step_size*weight_error

```

This relatively simple algorithm has a lot of problems.

1. What if the loss changes quickly in one direction and slowly in another ?

The gradient makes slow progress, with jitter along steep direction. This problem is much more common in high dimensional datasets.

Encountering:

1. Local Minima

2. Saddle Points

Gradient descent gets stuck at one point.  

### SGD + Momentum

It modifies SGD by adding a momentum.

It is given by:

```
vh  = 0
while True:
    vx = rho*vx + weight_error
    weight += step_size*vx
```

The vx denotes velocity and rho denotes friction. We update the weights with some velocity which is degraded with some friction.

This simple structure has a lot of good properties.

1. Even if the gradient is very small, the velocity allows the network to go ahead and potentially move free from a saddle point.

2. This structure also helps with local minima.

A more indepth study on various optimisation techniques will be examined in another blog post.

We shall also discuss the popular SGDR with Warm Restarts and Cyclic Learning Rates in a separate blog post.

## Decaying the Learning Rate

We decay the learning as we go along, this helps in converging the network. The intuition is that once the network has good weights a big learning rate, can destroy these weights.

## Regularisation

We require regularisation to reduce overfitting the neural network. We don't want the network to just work for the dataset but be more generalised.

There are several regularisation objectives.

1. L1 Regularisation

2. L2 Regularisation

3. Dropout

As covered before, Batch Normalisation also helps in regularisation.

### Dropout

In the forward pass, dropout randomly sets some neurons to zero. While setting these neurons to zero, this results in no contribution from them in computing the prediction.

The intuition behind Dropout is that during training the network learns __n number of subnetworks__ created while using the dropout masks. Granted, these models share weights so it's not a true ensemble. But, it gives the intuition that the final prediction is the _aggregation_ of these networks predictions.

Another intuition to dropout is that it forces the network to have a __redundant representation__.  Helps prevent co adaption of features. That means that to predict a cat, one neuron will not have the ability to fire at both the cat ear's and the cat tail. Those should be handled by their own neuron.  Dropout enabled that by forcing the network to learn these features seperately.


_Dropout makes the output random !_

Dropout at test time is handled by two methods.

1. __Inverted Dropout__. This modifies the forward pass of the network by dividing the neurons firing for that layer by the dropout probability __during training__

2. During __prediction__ , we multiply the outputs of the dropout layer by the dropout probability of that layer.

```
// p is dropout probability

H = activation(np.dot(W,x) + b) * p
```

 This leads to a scaling of the activations so that for each neuron so that _output at test time = expected output at training time_