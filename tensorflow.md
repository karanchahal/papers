# Tensorflow Graph

This post talks about how tensorflow executes your machine learning models. We shall briefly overview the components of the tf graph , and then 
dvelve into how this graph is executed across single and multiple devices.

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

## Devices 

Workers handle one or more devices. These devices can be CPU cores, GPU's etc. They are identified by device name and device type. Eg of device name could be

```/job:localhost/device:cpu:0```

In a distributed setting, the job name refers to the job the device is performing. 
Each device object has two functions:

1. Allocating/Deallocating memory
2. Arranging for execution of kernels requested by higher level layers.

## Tensors

__Typed Multi Dimensional Arrays__ , these tensors are the base data type of Tensorflow. Tensors can be of various types ranging from:

1. 8 bits to 64 bits
2. IEEE float and double types
3. Complex number data type
4. String type (arbitrary byte array)

# Executing the Graph : An Implementation Perspective

## Overview

The main entitiy in Tensorflow in the client. the client contacts the __master__ and one or more __worker processes__. 

Worker processes handle comptation with devices such as GPUs or CPU cores.

There are tow setting in Tensorflow:

1. Local Setting - The client, master and workers are all in the same computer.

2. Distributed Setting - The client, master and workers can all be in different devices. In a distributed setting, we run these different components in containers.
These containers are scheduled via a cluster scheduling system like Kubernetes.

## Single Device Setting

The simplest scenario for running Tensorflow. 

1. A single worker process
2. Single device

Nodes are processed in a way that respects dependencies between nodes. More specifically

- Each node keeps a _count_ of how many dependent nodes need to be processed. This count is decremented everytime a dependency is executed.
- When _count_ is __0__ , the node is put into ready queue , where is processed subsequently.

Please note: _How the ready queue processes the nodes is not specified_

## Multiple Device Setting 

Once we have multiple devices. We have two things to worry about:

 1. Deciding which device to put the computation for each node
 2. Managing communication between these devices.


### Node placement
The node placement algorithm figures out what node to given to what device. The algorithm uses a __cost model__ to make decisions. As per the white paper, the node
placement algorithm uses __greedy heuristics__ , via the cost model among other parameters to decide which device to place the node in. This greedy heuristsc takes into account

1. Cost of executing the computation.
2. Cost of communicating inputs to this node from other devices.

The device where this computation will finish soonest is selected as the device. This process of placement is followed till the ndoes are placed. 

This algorithm might have changed by now, since the paper was written in 2016.

Once nodes are placed in the device, a communication protocol needs to be put in place between these devices.

## Inter device communication

Tensorflow removes the edges betwwen nodes in different devices and replaces them with a __send__ and __recieve__ call. At runtime, the send and recieve calls coordinate
to transfer data across devices.

This method has the following benefits:

1. Data is only sent once through the recieve call and memory is only allocated once for a single tensor. So all users of a tensor , will not need their seperate recieve/ send calls.

2. By handling communication is this manner, we let the scheduling of different nodes in a device to be decentralised into the workers. The master doesn't need to track this, since the 
_send_ and _recieve_ calls handle synchronisation between different workers and devices.

## Execution in a distributed setting

A distributed setting is very similiar to the multi-device setting. In that the send and recieve calls are implemented via TCP or RDMA calls to move data across machine boundaries.
Execution in distributed settings require __fault tolerance__.Failures are detected through two things:

1. Error in communication between send and recieve call.

2. Periodic health checks from master process to every worker process.

When a failure is detected, the entire graph execution is __aborted and started from scratch__.

The tensorflow systems however, supports __checkpointing and recovery after restart__.

The Variable values are checkpointed via something called __Save nodes__

These save nodes are __connected__ with variables. These nodes can be configured to be __executed periodically__. Say after every N iterations, or once after never N seconds.
Similiarly , these Variables are also connected with a restore Node, so that their value is restored after a restart.


# Techniques for Training on Multiple Devices

## Synchronous SGD

This SGD relies on a master server which keeps track fo the parameters of the model , and several worker threads which execute some computation.
These workers will then send data back to the master to update parameters. Once master recieves all the parameters from the workers, it __accumulates__ these gradients and then sends
each worker the copy of the fresh gradients so that the workers can process the next batch

## Asynchronous SGD

The above methodology is good, but we can do better. Asynchronous SGD simply means that the master upon recieving some paramters, performs the update and pushes 
gradients to all the workers. It doesn't wait for all the workers to complete the task.

## Model Parallel Training

Used to train Deep LSTMS.
This type of training is where __different portions of the model computation__ are done on __different computational devices simultaneously__ for the __same batch of examples__.

## Concurrent Steps for Model Computation Pipelining

Another common way to get better utilisation for training deep neural networks is to __pipeline__ the computation of the model within the __same device__. It is the exact same as 
ASGD, but instad of multiple devices, the same model is execiuted in the same device to better use the device ability to parallelise operations.



# Conclusion

In conclusion, Tensorflow is a system that supports

1. Training and inference on multiple devices, great for usage in a distributed setting.
2. Is designed in a manner that would make it great for future optimisations via the data flow graph structure.
3. Makes communication between devices simpler by suing compression techniques.
4. The placement algorithm is particularly intriguing, the authors have said that it could potentially be replaced by a deep learning algorithm.

