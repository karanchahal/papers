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
