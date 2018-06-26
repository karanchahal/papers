# The Different Activation Functions in Deep Learning

There is a very important function being run after a layer's neuron values are calculated. Each neuron undergoes an __activation__ function.

This function introduces the non linearity in neural networks that allow them to reason complex phenomenon.

If not for the activation function, all we would be able to model is _just_ a linear function.

But, our world is _non linear_!

Identifying a cat, the speed of a ball, recognising a langauge are all non linear functions! Activation Functions introduce this non linearity into our neural networks. This helps us model complex functions. Activation Functions are a very important piece of what makes a neural network a __universal function approximator__.

There are a few famous activations that we are going to go through. Each of them have their pros and cons. At the end of this post, you shall understand what activation to use under what circumstance.

## Sigmoid Function

This is the first popular activation function that worked well. It has several useful properties.

1. It is differentiable which helps in backpropogation
2. It squashes the input between 0 and 1. 0 signifies non activation and 1 activation.
3. It was popular, as the 1 represents a "firing" neuron. 

Formally , it is given by 
```

g(a) = 1/(1-e^(-a))
```

There are 2 problems with Sigmoid though:

1. The saturated neuron kills the gradient. If the input to activation function is either too high or too low. The gradient generated is 0. And hence, anything multiplied by 0 is 0. This effectively kills the gradient as it goes further upstream. This leads to no signal flowing through the neuron to update the weights. This is called the __gradient vanishing__ problem.

2. The values outputed aren't zero centered. This leads to output being always positive after coming from the sigmoid function. This leads to the gradients to always be either __all positive__ or __all negative__. Hence, the _directions_ where the graidents can update the function is _constrained_ to a fixed number of directions. This leads to a __longer time__ to converge as the _gradient needs to be updated in more directions to achieve convergence_ .

The second point is why we need __zero mean__ data so that gradients can be updated in both types of directions. with this the input is no longer only positive.

## Tanh Function

The Tanh function was introduced to solve this zero centered problem of the sigmoid function.

1. It squashes the input into -1 and +1. This leads to solving of the zero centered problem.

```

g(a) = 2/(1-e^(-2*a)) - 1
```

But, it still suffers with the vanishing gradient problem.

## Relu Function

The Relu function when introduced gave a 6x improvement over tanh.  It is quite popular. 
This is because it has the following properties.

```
g(a) = max(0,a)
```

1. Does not saturate in the _positive_ region. Unlike tanh and sigmoid.
2. Converges much faster. 6x Faster.
3. Very computationally efficient.

Problems

1. If the input is a negative value, no gradient will flow backwards as the gradient of zero is zero.

2. As we're killing the gradient for half the regime, we get about _20%_ of neurons to be __dead relu__ neurons. We get this phenomenon from several potential reasons. If the weights are adjusted in such a way that they will always get a 0 zero value from the relu, and hence a zero value for gradients. Hence, the weights will never be tweaked. This dead relu scenario can also happen because of large training updates, with a __huge learning rate__, the gradients gets knocked off the data manifold space. The other potential reason is __bad/unlucky initialisation__.

3. We still haven't resolved the zero centered problem as the gradients will only be all positive or all negative as the output is only positive or zero.

To fix this problem, a variant to the Relu was introduced.

## Leaky Relu

The RELu was modified to solve the zero centered problem.

It is given by

```
L = max(0.01x, x)
```

The advantages of this activation is as follows:

1. Has all the advantages of the RELU, no gradient saturation in the positive scenario.

2. It won't die when it encounters a negative value. There will be some gradient flow via that 0.01x parameter. Plus this also solves the non zero centered problem. Gradients can converge quicker.

3. Still can saturate for low negative values.

## Takeaways

In practice use

1. Relu -. pretty standard. Be carful about adjusting learning rates.

2. Try out Leaky Relu, Maxout, ELU 

3. try out tanh , but don't expect  much

4. Don't use Sigmoid.