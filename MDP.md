# Markov Decision Processes
(Chapter 3 and 4 in Sutton Book)

##### Assumption in coming material that agent and environment states are the same. The environment is fully observable.
We can use MDPs to provide a mathematical framework for a fully observable environment. (You can make partially observable environment into a MDP by using the history of an agent)

Bandits are MDP's with only 1 state.

Markov Property: "The future is independent on the past given the present"

Acting in a MDP results in returns that is discounted (generally)
The return is a random variable that depends on the MDP and policy, the marginal value of a reward achived much later is discounted.

If y is very close to zero, we have a myopic agent, and if y is closer to 1 our agent is called far sighted.

A discount factor of 1 is used when we known that their are terminations (episodes). For example bandits are only  step long.

If we have a problem with infinte time steps, a discount should be used to make better long term decisions.

Discount factors that are low make problems "easier", changing discount factors lead to changing the problem in some sense and your solution might become different.

Discount makes return finite, no discount could lead to infinte rewards. (We could use averaging rewarding to prevent this !)

# Value Function

Expected return and can be written down recursively.
```
Sum over all the possible rewrds and state and policy probabilities to get the value functoin for a certain state !!
```
Replace sigma with integral if distribution is continous (eg: reward or even states might not be discrete)

We call value functions conditioned on the action as well as the state as "q values" or  "action values"

```
v = r + yPv

where v_i = v(s_i)
r_i = Expected(Reward given state s_i and action from policy pi)
P_i_j = Weighted_Average(Policy multiplied with probability of State j given action and s_i state)

```

Let's say actions are 3
States are 3
So P is 3 by 3

We can calculate v = r*(1 - yP)^(-1). We dont use this due to log(N^3) time.

## Iterative Methods

Methods like Dynamics Programming, Monte Carlo and/or Temporal/Difference Learning to solve this for v are used that are more tractable.

Our goal is to get to the optimal state and action value functions. To get to tht we need to be able to evaluate how good a policy is or isn't.  This is called policy evaluation or simply prediction. Estimating the optimal functions is sometimes called control, which can be used in policy optimisation.

We cannot have a policy better than the optimal policy. We can form a equation to convert to and from from state aand action values functions.

A policy with a higher value function is better than a policy with a lower value function. It's easy to get teh policy from the optimal state action values. 

## Solving Bellman Optimiality Equation
To get the optimal policy we have a max paramater over action state values.
Because of thee max parameter, belman optimal equation is non linear. 

We use dynamic programming for model based approaches.
1. Value Iteration
2. Policy Iteration

Or we can use samples
1. Monte Caro
2. Q learning
3. Sarsa

We focus on the first group model based appraches for now.

We want to get the optimal policy ! Remember this is for a case when we have a perfect view of the environment and a perfect model of it in the form of an MDP.

We have 2 parts in dynamic programming

1. Policy Improvement
2. Policy Evaluation

## Dynamic Programming


Hence we have two steps policy evaluation and policy improvement, until there is no change in policy. Hence Bellman equation holds true hence signifying optimal policy.

#### Policy Evaluation
How to evaluate a policy.(value functions)

Assume v_k as the predicted value function. At the start init v_k to all zeros

Calulate v_k+1 using same bellman equatoin but use v_k policy in discounted part.
THe theory goes that after a while, enough times of doing this you will arrive at the true value function.

This is becayse if we subtract vk+1 and v_optimal then we get a value that looks exactly the same but weighted by discount factor. Hence the intuition is bu doing this enough times, the discount factor will grow on and on until the difference is 0. Hence, we shall always arrive at the true distribution.

Hence, policy always converges if discount factor is less than 1. Hence, rate of convergence is determined by discount factor. Lower discount factor easier to solve, higher discount means harder problem to solve.

For finite horizon episode case, this is complex to show but holds true also.