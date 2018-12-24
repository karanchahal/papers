# Model Free Prediction and Control

We don't know the MDP for this !
(more real world significance)

We dont have model, we use samples or experience in this. Model free experience based things are called Monte Carlo.

Bandits with context, (context wherein you take your action)

You get a state, action and then reward. Your action doesn't affect your next state, but we still want to get large expected reward. As an example assume different visitors to a website. We want to estimate the q value for a state and action, q could be a neural networ ad we could use a loss that minimises over the difference between predicted value and actual reward. This also works for large continous states (just like regression).

So core thought is: Don't use a model, get samples estimate reward and minimise over actual reward.

## Back to Tabular Land

## Monte Carlo Policy Evaluation

Learn value function, return is discounted and ends at some time T. The value functoin can be expected reward. Monte Carlo is called that because of sampling.

Markov doesnt care between two different episodes, TD sees this. So monte carlo is better for non Markov processes and TD is better for the inverse.

Model Free - Unknown MDP


## Monte Carlo Model Free Control