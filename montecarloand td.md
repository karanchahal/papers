# Monte Carlo

MC methods try to learn the policy and value function by using experience of an agent and not the knwoledge of the MDP. These emthods are good for when we do not have all the information of the MDP or environment. The capability of learning from experience is really powerful and is the basis for solving intractable problems.

First Visit monte Carlo: Finding expected reward since the first state.
Multi visit monte carlo: Finding expected reward of a state from averaging returns from state in all the times it has been encountered.

Advantages of Monte arlo over Dp methods

1. Learn form experience in acula or simulated environment
2. The value of states is not dependant on value of other sttes i.e bootstrapping is not used in Monte Carlo (this is good if one wants to get info on only a few states)

## Sarsa (State action reward action)

Calculating action values that uses step size and adds error between q value and reward.

It explore in different directions with E- greedy. It takes greedy action with 1-e probability and e it takes random action.Leads to jittery (but good for exploration, very simple).

Sarsa is TD so adapts policy with every time step.

Sarsa is on policy


TD learning is used to just denote general value function error propogation
Sarsa uses q values instead of valure functions.
Q Learning is the same as Sarsa but maximises the next state q value by selecting the greedy action.

## On Policy

Learn on the job, estimate the policy and then improve it. Learn about policy from experience sampled from policy.

## Off Policy

Learn about any policy other than the policy being used. 

Behaviour Policy is the policy being followed now and pi is the policy is being learnt. 

Offf policy is important because:

1. Learn from observing other agents.
2. Reuse experience from old policies.
3. Learn about optimal policy while following exploratory policy, learn about multiple policies while following one policy.

So in Q learning we follow a greedy policy to learn the optimal policy.

Q learning will converge to the optimal action value function for infinte times.

Difference between on policy and off policy. Q learning will take optimal path in cliff walking examples because it doesnt use E policy as it follows a greedy policy. Sarsa will use E policy as inherent and show optimal path according to it.

Q learning might take a long time to learn for certain games.

Policy robustness = Sarsa.
Q learning = optimal policy.

Target Policy = greedy (for next state, action)
Behavior Policy =  E - greedy (for current state, action)

## Issues with Classical Q learning

Uses some values to select and evaluate. If values are estimated then we will be more likely to select overestimated value than underestimated values. Upward bias. We can solve this by using 2 policies one for selection, one for evaluation.

## Importance Sampling

When we are using different distribution, we can get the weighted sum for an action in the policy pi by using imporatance sampling metric.
 