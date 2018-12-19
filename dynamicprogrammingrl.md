# Dynamic Programming for RL

Only used for perfect environments and MDPs. We try to evalue the policy given a vlaue function. Also called the prediction problem or policy evaluation.

There is a way to doing this, let's look at the first way


## Iterative Policy Evaluation

Start with a random value function, improve it iteratively until it stops getting improved. At that point you are at the optimal policy. Note that the value funtion for the terminal state is 0. We use the same bellman equation for getting value function with one small modification. Use the Old value function in the update rule with the reward to compute the new value function for a state.

We do this either by 2 ways
1. Keep 2 arraays , v old and v new and then swap for next step.
2. Or we do this insplace, this can lead to using new v values in the calculation of v value for a particular state. This has shown to increase convergence, it also matters in what order the states are processed.


## Policy Improvement

We check the value of q value functions with the old policy to see whether it improves over the existing value function. It is a greedy way to propose a new policy over the old one using an old value function. 


## GPI

Genralised Policy Iteration combines 2 both value and policy dependant on each other.