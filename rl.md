# Reinforcement Learning 

Tracks progress of my learning from the Deepmind RL course by UCL.

## Introduction

An RL system consists of the following parts.

1. Policy: maps states to actions
2. Model: predicts next reward/next state given certain state/action
3. Value Function:  Gives how good being in a state is or expected reward at the end out of being in this state.
4. Rewards: The signals that the RL system gives to signify it's pleasure/displeasure of being in the current state.

An RL system generally has a single or multiple __agents__ interacting in an __environment__. The Agent takes an action depending on the state it is in and the environment responds with an observation of what happened and/or a reward depending on how good/bad the action was. The RL system can incorporate reward into it's observations.

The agent can be of two types.

1. Fully observable agent: An agent that has the full view of the environment.

```
If knowing the state is enough to get a fully judge of the agent's position in the environment. eg: Go board, Chess board
```

For a fully observed environment, we can use Markov Decision Processes.

2. Partially Observable Agent: hen an agent has a partial view of the enviroment.

eg: A first person view of a agent in a maze. 

We cannot use a MDP in a partially observed environent as the state is not enough to judge an agent's progress. Some form of history is needed to augment this state to provide a unique view. 

The problem is this history can grow unbounded, however we can use a fixed size of history. One solution that is being used are RNN's to maintain the concept of state in partially observable environments.

Note: Add small fixed history to make agent fully observable.

## Value Functions

Value functions give the expected total reward for an agent in a given state with a given policy. They can be calcuated by the belmaan equation like this

```
V = Expected_Reward(State,policy)
Now, we formally state like this

Vt = (Rt + Vt+1) for State S and Policy Pi
```

Q Values are only value functions but taking action as a parameter too along with the State. Optimal Value functions are value functions that gve the optimal route.

We use a discount factor to tell the agent to get on with it and get to it's goal fast.


## Categories Of Agents

1. Value Based Agents: Only value function used, policy is implicit.

2. Policy Based Agents: Only policy is used.

3. Actor Critic Agents: both policies and value functions are used, learning both. 

5. Model Free Agent: No Model and value and/or value function.

6. Mode Based: Model and optionally value and/or value function.

## Challenges with Reinforcement Learning

Two fundamental problems:

1. Learning: The environment is initially unknown. Thee agent interacts with environment
2. Planning: A model of environment is given, the agent plans in this model aka reasoning, thought, search etc.

Prediction: Evaluate the future.
Control: Optimize the future.

Sometimes supervised techniques are really good for Prediction!

These are strongly related, good policy if we have good prediction.

## Central Intuition

Each RL component can be represented as functions:

1. POlcy maps states to actions
2. Value Functoins map states to values
3. Model maps states to set of states/ reward
4. State updates maps states to observations to new states.

We can represent these functions as neural networks, however we violate some assumptions of supervised learning. Eg: Stationarity

Also, current neural networks not always the best tool.
