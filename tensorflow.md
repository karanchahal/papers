# Tensorflow Graph

The tensorflow graph has the following properties. Each __node__ has __zero or more inputs__ , and represents the __instantiation of an operation__.

Values that flow from egdes of the graph are known as __tensors__. These tensors undergo various transformations when they go through these __nodes__.

Tensors are __arbitrary dimensionality arrays__ , where the __underlying element type__ is inferred during graph construction time. This ii what allows Tensorflow
to be very fast , as it knows what operations ocuur in the future via this graph. Hence this knowldge allows for various compile time optimisations.

Special edges knows as Control Dependencies - No data flows through these edges, but they indicate that the __source node for control dependence__ must finish executing before __destination node can execute__

This property allows clients to enforce __happens before__ relationships. This is quite helpful in controlling peak memory usage for example.

## Operations and Kernels

An operation defines a computation:
Examples could be - 
 1. add
 2. matrix multiply

An operation can have __attributes__. One use case of attributes is to make operations __polymorphic__ (perform operations betwene elements of same data type)

A __kernel__ is defined as:
An __implementation of an operation__ that can be run on a __particular type of device__ (CPU,GPU,TPU) etc.

## Sessions

Clients interact with the Tensorflow system by creating a __Session__.

1.  The session interface has a method called __Extend__. This allows us to modify the computation graph with
additional __nodes and edges__. 

2. The session interface has another method __Run__. 

- This function computes the __transitive closure of all nodes__ that must be executed in order to compute the outputs
that were requested.

- It then __arranges the nodes in such an order__ which respects their dependencies

Generally , uses for Tensorflow are
1. Setting up a session with a graph once.
2. Running graph or distinct subgraphs thousands or millions of times via _run_

__Note__: The transitive closure of a graph is a matrix that defines reachibility between every node in a graph. This matix will be filled with 0 and 1. 0 defining
not reachable and 1 defining reachable

## Variables

A variable is a persistent tensor. Most tensors do not survive after a run operation. Variables survive after run operations. 
The use case for variables is in storing the parameters of the neural network. These variables are updated whene _Run_ is called on the graph.

