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

The Mixed Precision Training Approach with LARS

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

In this section, we shall talk only about increasing the value of e. To improve e, hybrid all reduce algorithms have shown to report great results. 


