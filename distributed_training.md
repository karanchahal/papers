Communication- Computation Ratio

The research on training deep learning models faster is divided into two groups:

1.	Speeding up single machine training.
2.	Speeding up distributed training (multiple machines)

We shall look into both of these techniques in detail.

Single Machine Training

The innovations in single GPU training areas follows:
1.	Mixed precision training method
2.	Using superconvergence
3.	Various “faster” optimization algorithms

The Mixed Precision Training Approach

Write Here


Distributed Training 
Large scale deep learning systems try to speed up training by using larger mini-batch sizes. This leads to a reduction of the communication-computation ratio. In distributed deep learning training, the communication bottleneck should be decreased as much as possible. Large batch sizes result in less transfer of data through the network.

The side effect of using larger batch sizes are several

1.	Hurts generalization ability.
2.	Training instability, high probability of diverging

Hence, there are some algorithms in the community that allow us to train models fast without sacrificing on the above.


Distributed SGD is commonly used to train on multiple machines. Distributed SGD uses a large mini batch size, to take advantage of each machine.

A large batch size has a few problems.

1.	Results in a lower test accuracy, this means the model doesn’t generalizes well as when trained with smaller batch sizes. This makes sense, as the gradient updates are fewer, than with a small batch size.
2.	It is hard to achieve near-linear scalability on increasing the number of machines. This is because there is a substantial overhead in the form of the communication-computation ratio. 

Some arguments towards the case for large batch sizes are as follows:
1.	Larger batch size reduce the variance of gradients by taking the average over a larger set of examples. Thus, it allows the model to take a bigger step size and in turn make the optimization process faster. 



Solving Problem Number 1 

The test accuracy takes a hit because of the following reasons.

1.	 A smaller batch size, though slower to train arrives at a more general minima. This can maybe be solved by using cyclic learning rates.

Solving Problem Number 2


A throughput of a distributed system is measured as follows:
T = S.N.e
Where S = the throughput of a single machine, N =  Total number of machines and e = scaling efficiency which is generally between 1 and 0.

If e = 1, linear scaling is achieved. To increase e, more efficient bandwidth utilization is needed. To do that, we need more efficient communication primitives to handle a system of thousands of GPUs.

In this section, we shall talk only about increasing the value of e. 

Work being done so far
The first major work has been done by Goyal et al, who trained Imagenet in an hour using a large mini batch size (~8000). They used a linear scaling rule to adjust the learning rate as a function of mini batch size. They started with a warmup learning rate to overcome early epoch optimization problems. Further research has used larger number of GPU’s with increasing batch size (~32k). The major innovation being the usage of LARS to train the network for large mini batch sizes. But the research has two problems. Either the test accuracy suffers with the usage of more computation resources or the training fails to scale to more GPU’s or nodes. Some techniques that have been tried are listed as follows:

1.	RMSProp warm up
2.	Batch normalization without moving average
3.	Slow start learning rate schedule
4.	Increasing the mini batch size instead of decreasing the learning rate.
5.	Increasing batch size dynamically with learning rate decay

Advancements in Distributed Training

Tensorflow uses a centralized deployment mode. One bottleneck of this is the high communication cost on the central nodes (Parameter servers). The all reduce algorithm introduced by Baidu is a very important contribution to distributed deep learning. The ring all reduce algorithm greatly reduces the communication load when the number of nodes increase. The challenges of the all reduce are as follows:

1.	Efficient Bandwidth utilization with splitting tensors
To tackle this several methods have been tried
1. Gradient Fusion for reducing tensor fragmentation and increased bandwidth utilization
2. A DAG model for scheduling computation and communication tasks.


MIXED PRECISION TRAINING WITH LARS

What is LARS

Lars proposes using a different learning rate for each layer in the network. This local learning rate for eachb layer is calculated by taking the ratio between the L2 norm of weights and L2 norm of the gradients. This quanity is weighted with a LARS coefficient n. 

To use LARS with mixed precision training, we need to convert the weights and gradients back to FP32, apply LARS, then convert them back to FP16 format. 

All reduce algorithm

There are two ways of using SGD for training in a distributed setting- the asynchronous SGD and synchronous SGD.

Synchronous SGD

It is the simplest to understand, synchronous SGD does the following:

1. There are several replicas containing some subset of data and replicas of the machine learning model.
2. All replicas run the forward and backpropogation step.
3. All the replicas have their own set of gradients that are computed after the above step.
4. They send these gradients to the parameter server, where all the gradients and averaged. These gradients are then sent back to each replica.
5. Each replica then updates it's own set of weights with these global gradients.
6. The next bach of data is requested and the above process is repeated.

Please note, in a distributed SGD, the mini batch size is the size of the batch per node multipled by the total number of nodes. Hence, when the Distributed SGD takes places, the above steps are analogous to the processing of a mini batch on a single computer.

Using large batch sizes, enable us to 



