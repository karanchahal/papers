# Berkley's Unsupervised Learning Course Notes

What are the real world problems:

1. Generating photo from text.
2. Generating video from script
3. Generate voice from text
4. Generate fake news ! :P
5. Generate UI code from images, stories, speeches, movies, ... anything !
6. Data compression ! (as compression is also about prediction, to compress data. It fits into generative models mould quite well)

## Likelihood Based Models

Estimate the distribution of data from some samples. For example, from the image net dataset, find the distribution of natural images data.
We have an empirical small sample of distributiuon (Imagenet) and from it's sample, estimate it's true distribution.

"How likely is this data point from the true distribution ?"

A liklihood based model is a model whihc is a joint distribution over data.

We have some data points from the true data distribution which are IID. We constrain the data distribution to a dataset (maybe Imagenet)

For this class, a distribution is a function that takes in input and spits out a prob whether it is generated from true data process or not.

In the first lecture we deal with only discrete data, will move on to continous data later.

Potential uses are anomaly detection fast.

Sampling: Generate random variable X that has the same distribution of the model.

Deep learning helps to estimate distributions of a complex high dimensional data. Hence this class is motivated by that, unlike older classical stastical techniques, which fail in this high dimensional scenario.

We want our model to be:

1. Small
2. Fast
3. Expressive
4. Generalisable

## Liklihood based generative models

We estimate p_data from samples.

TO do that, we have a function approximator to learn theta where theta apprxomates the real distribution.

1. How do we design function approximations to effectiely represent complex joint distributions over x, yet be easy to train.
2. There will be many choices, each with different tradeoffs. 

__Designing model and training procedure go hand in hand__

We also want

1. For loss function + search procedure to work for large dataset
2. yeild theta simialr to true data distribution, think of loss as distance between distributions. 
3. Note that the training procedure can only see empiral data distributions, should be able to generalise.


## Maximum Liklihood

Maximum Liklihood finds theta given dataset by solving optiisation problem. 

Also statistics say that if model is expressive enough and given enough data, then solving maximum liklihood problem will yield parameters that generate the data.

It is equivalent to mimimising KL divergence between true and approximate model.

## SGD 

We can solve Maximum Liklihood by using SGD to minimise expectations.

If f is differentiable function of theat , it solves arg_min(Expectation of f)

As maximum liklihood is an average, and SGD minimises averages !

With maximum liklihood, our optimisation problem is:

```
argmin_theta(For data in dataset (  we check log(prob(x))))
```

The noise is coming from sample of true dataset, in supervised setting we can think of this noise as mini batches over dataset. (Slightly connfusing)



