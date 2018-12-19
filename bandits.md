# Reinforcement Learning Notes Multi Arm Bandits

Reinforcement Learning __evaluates__ the action taken, instead of __instructing__ on what actions to take. To study this evaluation problem,
a simple version of the n armed bandit problkem is studied.


## N-Armed Bandit Problem

The problem is given as follows.

You are given n actions, after executing an action you get a reward. Only one action can be executed at a time, the objective is to maximise 
reward over some number of time steps. We are given a probability distribution of the rewards that we shall get on selection of each action. 
These values are unceetain, in the sense that they have a high probability of recieving that reward.
THere are two methods to tackle this problem

1. The greedy way, select the action with the highest certainity. This is termed as exploitation
2. Select actions , that don't have the highest certainity, but may give a higher reward in the long term. This is termed as exploration.

There is always a constant tradeoff between exploration and exploitation in RL. 

Exploration and exploitation is balanced in a complex way depending on various paramteres which can include

1. Estimates
2. Uncertainties
3. Number of remaining steps among others...


A possible solution to this is:

### Sample Average Method 

The sample average method is used as follows:

1. The values for actions are calculated as the average of all the rewards received for those actions in previous time steps.
2. If the action hasn't been selected befroe, then it's value is init by a default value, say 0.
3. If there is a tie between values, a random action is picked out of the tying actions.


This greedy policy always picks the highest reward at that time, we introduce a small probablity E, of picking a random action. This gives us 
a way to explaore as well as exploit. These methods are called __E-greedy methods__. 

E-greedy methods show a higher final reward than just greedy methods for sample average. But note that this depends on the task at hand.
But in a non stationary environment, where the values for actions may change over time, exploration is needed to see whather a non greedy action
has not changed to become better than a greedy one. RL operated on a lot of non stationary environments, hence requiring a balance between exploration
and exploitation. 

The calculation of the averages of values for each time step can be done as follows:

```
Q = (R + (t-1)*Sigma(Old Q))/t
Q = old Q + (R - old Q)/k
```

where R is the Reward at this current step,t is the time step, Q old is the old value and Q is the new value being computed. 

we can also rework the above equation as 

```
NewEstimate ← OldEstimate + StepSize * [Target − OldEstimate]
```


### Tracking a Non Stationary Problem 

To track a problem who's action values can change over time, a greedy strategy is sub optimal. Therefore, in most cases it makes sense 
to weight recent rewards more heavily than long past ones. One of the ways to do this is to use a constant step size paramter a. a is betwwen 0 
and 1.

Now the equation for Q becomes becomes
```
Q = (1-a)^k * first Q + SigmaOverAll((a(1-a)^(total time steps - R_current)*R_current))
```

This is a weighted average because the sum of (1-a)^k + sigma(a(1-a)^k-i) is 1. ONe can also observe that the weighjt given to old rewardsa is less
than the weight given to newer rewards. (Beginnings of the discount factor ?), this is also called _expomnential, recency weighted average_

### Optimsitic Initial Values 

Sometimes the initial Q values are set to a high value, say +5. Recall that the values of Q are of mean centered at 0 and variance 1. An initial estimate of +5 is thus wildly optimistic..
 A high initial value has the follwoing benefits:

1. It will force a model to explore more, for example at the beginning the reward for a greedy value will not be able to be gretater than the 
initial values, hence requiring the model to explore more. 

This is a simple effective trick that can be quite effective in __stationary problems__. However, it is not well suited for non stationary problems 
because it's drive for exploration is inherently temporary. If the task changes, creating a new need for exploration, optimistic initial values 
will not help here. Any method that focuses on initial state is unlikely to help with the general non stationary case. 

### Upper Confidence-Bound Action Selection 

So now, we have a bunch of uncertain action values. Greedy methods that mnight look brest now might be suboptimal, e-greedy methods forces non greedy
methods to be tried.But they do this indiscrimately, with no preference. It would be better if we can select non greedy actions according to their potential
of actually being optimal by taking into account

1. How close they are to maximal
2. Taking into accounmt how uncertain these values are. 

One way of doing this is,

```
A_t = argmax_a[Q_T(a) + c* squareroot(ln(t)/N(a))]
```

First it was only argmax(Q_t(a)), now the extra term tries to take into account the above two factors.
c = degree of exploration, which is greater than 0. This can and should be varied at different points of training. 
ln(t) denotes natural log of t. 

The square root term is also denoted the amount of uncertainity or varinace in that action value pair. The more an action is selcted, the less the
variance of that is, also the more another action is selcted other than the current action pair, the more uncertain the value gets. This increase 
shall slow down over time but is still unbounded, meaning all actions will eventually be selected. 

UCB techniqyes often work well with the 10 armed mult bandit problem, but are more difficult to extend to more general RL problems. THere are problems 
like large state spaces, function approximation and more complexity needed for a amjority of RL problems. IN these advanced settings, there is no known
practical applications of UCB so far. 

