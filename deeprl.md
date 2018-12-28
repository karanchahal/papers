# Deep Reinforcement Learning

## Approximating Functions

The policy, value and model are all functions, we can approximate each with deep neural nets.

Reinforcement Learning can be used to solve large problems.

eg: Backgammon, Go, Helicopter and robot control

Some problems have an infinte continous space that Deep RL can approximate. How do we scale up our methods for prediction and control. SO far we have considered lookup table. But we can build large look up tables for Go. Also some environments are not fully observable. you an;t learn individual states, so we need to __generalise__

Worlds can't be fully observable (humans percieves through eyes).

## feature Vectors

We represent states as a feature vector (CNN for images ?)

eg: Distance of robot from landmarks
eg: Stock Market
eg: Chess configration of state of board