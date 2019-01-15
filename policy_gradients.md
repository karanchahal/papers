# Policy Gradients

"Always try to learn a policy directly" - Take with a grain of salt

Model Based RL
'Easy to learn model'
Learns all there is to know from data
Objectives capture irrelevant info
Computation is non trivial

Value Based RL
Closer to true objective
Fairly well understood
Still not true objjective
Vauke function might be complex but policy won't.

Policy Based RL
Right objective
Ignores other learnable knowledge

Approximating policy results in model free reinforcement learning

value based : Learn value function (implicit policy)
Actor Critic: Learn both value function and policy, uses an adverserial approach.
We learn both simultaneosuly. but in some cases we can learn the policy and then get the value function.


Advantages:

Good convergence properties
Its very easy to plug in sam algo for continous data
Benefit to be able to learn __stochastic policies__ (alissased gridworld example, random motion in some indistinguisihable states) (e-greedy doesn't apply in some circumstances)
Sometimes policy are quite simple.

However,
Disadvantages:

It is suspestble to local optima
Does not always generalise well.
Is not always efficient.

## How do we learn policies ?

We want a policy that has a high value.

In episodic environments, I want a policy that gives high reward from start state.

In continuing envs we can use the average value.
Define a distribution over states,

Policy based reinforcement learning is an optimsation problem. Find a policy that maximises reward. We will focus on gradient ascent aand is easy to use for deep nets.
