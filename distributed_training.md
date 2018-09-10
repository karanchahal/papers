## Distributed Training

The research on training deep learning models faster is divided into two groups:

1.	Speeding up single machine training.
2.	Speeding up distributed training (multiple machines)

We shall look into both of these techniques in detail.

Single Machine Training

The innovations in single GPU training areas follows:
1.	Mixed precision training method
2.	Using superconvergence
3.	Various “faster” optimization algorithms

## The Mixed Precision Training Approach

Write Here

## Distributed Training 
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



#### Solving Problem Number 1 

The test accuracy takes a hit because of the following reasons.

1.	 A smaller batch size, though slower to train arrives at a more general minima. This can maybe be solved by using cyclic learning rates.

#### Solving Problem Number 2


A throughput of a distributed system is measured as follows:
T = S.N.e
Where S = the throughput of a single machine, N =  Total number of machines and e = scaling efficiency which is generally between 1 and 0.

If e = 1, linear scaling is achieved. To increase e, more efficient bandwidth utilization is needed. To do that, we need more efficient communication primitives to handle a system of thousands of GPUs.

In this section, we shall talk only about increasing the value of e. 

### Work being done so far
The first major work has been done by Goyal et al, who trained Imagenet in an hour using a large mini batch size (~8000). They used a linear scaling rule to adjust the learning rate as a function of mini batch size. They started with a warmup learning rate to overcome early epoch optimization problems. Further research has used larger number of GPU’s with increasing batch size (~32k). The major innovation being the usage of LARS to train the network for large mini batch sizes. But the research has two problems. Either the test accuracy suffers with the usage of more computation resources or the training fails to scale to more GPU’s or nodes. Some techniques that have been tried are listed as follows:

1.	RMSProp warm up
2.	Batch normalization without moving average
3.	Slow start learning rate schedule
4.	Increasing the mini batch size instead of decreasing the learning rate.
5.	Increasing batch size dynamically with learning rate decay

### Advancements in Distributed Training

Tensorflow uses a centralized deployment mode. One bottleneck of this is the high communication cost on the central nodes (Parameter servers). The all reduce algorithm introduced by Baidu is a very important contribution to distributed deep learning. The ring all reduce algorithm greatly reduces the communication load when the number of nodes increase. The challenges of the all reduce are as follows:

1.	Efficient Bandwidth utilization with splitting tensors
To tackle this several methods have been tried
1. Gradient Fusion for reducing tensor fragmentation and increased bandwidth utilization
2. A DAG model for scheduling computation and communication tasks.


## Mixed Precision Training With LARS

### What is LARS

Lars proposes using a different learning rate for each layer in the network. This local learning rate for eachb layer is calculated by taking the ratio between the L2 norm of weights and L2 norm of the gradients. This quanity is weighted with a LARS coefficient n. 

This was motivated by the observation that the ratio between the norm of tthe weights and gradiets varied widely between different layers during the start of training. This became better once traininng had been ongoing for 5 epochs. 


All reduce algorithm

There are two ways of using SGD for training in a distributed setting- the asynchronous SGD and synchronous SGD.

## Synchronous SGD

It is the simplest to understand, synchronous SGD does the following:

1. There are several replicas containing some subset of data and replicas of the machine learning model.
2. All replicas run the forward and backpropogation step.
3. All the replicas have their own set of gradients that are computed after the above step.
4. They send these gradients to the parameter server, where all the gradients and averaged. These gradients are then sent back to each replica.
5. Each replica then updates it's own set of weights with these global gradients.
6. The next batch of data is requested and the above process is repeated.

Please note, in distributed SGD, the mini batch size is the size of the batch per node multipled by the total number of nodes. Hence, a single step of Distributed SGD is analogous to the processing of a single mini batch on a single computer. 

Synchronous SGD works well in practice and is used widely. Some disadvantages of Synchronous SGD include

1. Every worker has to wait for every other worker to finish their processing, before they can start the next mini batch of data.
2. There might be slow machines (also known as stragglers) the slow down the entire training.
3. Using Distributed SGD naively for large batch sizes leads to divergence issues or increased generalisation error (low test/validation accuracy)

The problem of stragglers can be mitigted by using replicas, and starting a race for who finishes first among all these replicas while training for a batch of data. A possible solution for maintaining replicas of the workers can be done using algorithms like Paxos and Raft.

There has been a considerable amount of research into training models with large batch sizes. We shall explore these techniques in the coming sections.

### Linear Scaling Rule

The linear scaling rule was one of the first methodologies to tackle the large batch size problem. They follow a simple methodology.
```
Linear Scaling Rule: When the minibatch size is
multiplied by k, multiply the learning rate by k.
```

It is important to note that this rule didnt affect testing/validation accuracies and enables training networks to batches as large as 8,000 fo Imagenet. Interestingly, this rule fails for batch sizes larger than 8,000 while training for Imagenet. Using this rule, doesn't train the model. It is observed that the training diverges during the early epochs, the authors say that could be because the value of the gradients for a small batch size verses a large batch size widely differ during start of training.

The gradients start to become similar in later stages for training by using this rule. To tackle this early divergence problem, the authors propose using a warmup scheme.
```
Start with a learning rate n, and increase it gradually to k*n where k is the total number of machines over a period of 5 epochs.
```

Since batch normalization is used, the size of dataset on each machines(n) remains constant with only the number of machines is changed (k). This is because if n isn't constant, the batch normalisation equation becomes quite different ?? (explain better)

### Pitfalls of SGD

1. Scaling cross entropy loss is not equivalent to scaling learning rate. 
2. Momentum correction is added after changing learning rate.
3. Average the loss per worker by a factor of kn and then use all reduce, instead of averaging by a factor of n.
4. Use a single random shuffle over the entire data and then distribute to the k workers.

### Distribution Primitives

The paper uses k nuber of servers. Each server has 8 GPUs that talk to each other without going through the network.


The paper compared two algorithms:
1. The all reduce (need to research)
2. The doubling/halfing algorithm (need to research)

They decided to go with the doubling/halfing algorithm because of low time complexity. Also when the graident aggregatioun step is undergone, it is done in paralle to backprop. In other words, if the gradients for a layer has been computated. That gradient is scheduled for the gradient aggregation algorithm to distribute to every other node.

#### Gradient Aggregation

Gradients need to be aggregated once all workers have completed their training step for a batch of data. These gradients are collected and averaged to form the new gradients. These new gradients are sent to each of these workers. This entire process is called the gradient aggregation. 

Gradient aggregation is done using the all reduce algorithm.
The all reduce algorithm is an algorithm from the field of High Performace Computing. The all reduce algorithm sums the data present on each worker and stores the result on all the workers. This is done in a way that utilises bandwidth efficiently.

##### Interserver communication

1. Recursive Halfing and Doubling Algorithm

2. The bucket (ring algorithm)


Parallel Algorithms cost model

Parallel Algorithms are judged upon the following factors:

1. Latency - How much time it takes for a request from one process to another (a + nb) (a = startup  time, b = transfer time per byte, n = number of bytes)
2. Bandwidth Usage-  How much of the network capacity is being utilised at an instant.

There are two algorthms for all reduce we look into

1. Ring Algorithm
2. The Halfing and Doubling Algorithm

### Ring Algorithm

Size of data on each machine -> n
Total number of machines -> p
The data is broken down into chunks of size n/p, leading to a total  number of chunks to be p. 


In the first step, each process i sends data (first chunk) to the process i+1 and recieves data from the process i-1 (the processes are aligned in the shape of a ring). 
On receiving data, a reduction is performed (in this case addition). 

This process is performed p-1 times, where at every step j, the jth chunk is sent, the jth chunk where p is total number of processes. 

At the end of (p-1) time step, all the machines have the gradients accululated. 

The time complexity and bandwidth complexity is as follows:

t = logp(a + b + y)

This ring algorithm has several nice properties

1. It works for any number of processors.

But there are algorithms where we can decreae the (p-1)A step to a log(p)A using a concept of a tree.

### Halfing Doubling Algorithm

There are two types of messages. Short and Long messages.

For short messages (< 2 KB)

We use the halfing algorithm.

1. First we transfer all the data of process i to process i + 1 bidirectionally. 
2. After receiving the data, a local reduction is perforomed (sum of the data recieved with process data)
3. Then in the next step, each process sends data as before to a process i+2 away and the next step i + 4 away, i + 8 away. 
4. Hence, in logp steps all the data is reduced and stored at every process.

This is great for short messages, as there is no problem transferring this short amount of data.

t = logp(A + B + Y)

But the issue comes when the length of messages is greater than 2kb which is the case with most machine learning applications.

For long messages

A reduce scatter is used , which is then combined with a all gather.

#### Reduce Scatter

A Reduce-scatter is a variant of reduce in which the
result, instead of being stored at the root, is scattered
among all p.

1. Each process i, brekas up its data into parts, one for transferring, one for recieving. It then transfers and recieves data from process that is at a distance p/2 from it. 
2. Once data is received, it performs the local reduction. Now it breaks this recieved data into 2 parts again. One for transfer , one for recieving. It transfers/recieves data from a process that is p/4 distance from it.
3. This process continues until log(p) steps. Once it is complete, each process has a chunk of the final aggragated data.

The complexity of this algorithm is as follows:

t = alogp + (p-1)n*b/n + (p-1)*p*y/n

Now once the reduce scatter is complete, the all gather is used to gather all this scattered data to all the processes.

#### All Gather


The all gather uses a similiar algorithm as the above. But does it in reverse, and instad of the reduction step, it aggregates the data (conncatenates instead of summing).

THis leads to a similair time complexity, but without the Y step, which is thr term used for computation on a single byte

Hence, 
t = alogp + (p-1)*n*b/p


Hence total time complexity for reduce gather +  all gather is:

t = 2alogp + 2bn(p-1)/p + n*(p-1)*y/p


##### Note

This is descibed for when the number of machines are a power of 2.

### Todo 

To descibe these algorithms when processes are not a power of two.

#### Binary Blocks Algorithm




## Async vs Sync

Async has shown poorer convergence results according to Revisiting SGD paper.

Experiment Async-

40 worker on Inception- 18 layers

1. Test error increases with increased staleness
2. Restarting training led to better runs


Hence, SGD was revisited, the primary limitation of Su=ync SGD was stragglers (or the slowest machines which held up training)

Solutions:

1. Backup Workers: Add b extra workers. Take first N updates to gradients, the slowest b worker's gradients will be dropped when they arrive. They used various graphs to explain why they used around 96 workers and 4 backup workers to mitigate stragglers to achieve fastest training time.

### Ideas

Compare between Sync SGD and Async SGD
what is better in what situation
Compare between gradient aggregation algorithms. 
What is better where.
LARS.
Large batch sizes.
Compare using large batch sizes over small batch sizes.
Compare different ways of using learning rate. LARS, Warm up, linear scaling etc.


# Disadvantages of Sync SGD

1. The transfer of data among machines is quite fast because of ring all reduce primitives, but the issue is with the synchronisation barrier. All the other workers have to wait till the slowest worker to finish to start training the next batch of data.

# Async SGD

Each node asynchronously performs its own gradient updates and occasionally synchronises it's parameters with a central parameter store occasionally. 

Hogwild SGD and DownPour SGD addressed this first. The disadvatage with this is two fold

1. Central point of failure, the serverhas to deal with all these workers sending data to it. Communication throughput is limited by the finite link reception bandwidth of the server. 
2. Also, there is a possibility of introducing stale gradients. For example, if worker a finshed its batch early and sent new updates to the server, it could be operating on stale gradients. This stale gradients have found to lead to lmiited test accuracy and delays in convergence. 

To solve these problems partially, elastic averaging SGD was introduced. It modifies the stochastic gradient algorithm to achieve faster convergence. Elastic avergaing introduces an elastic force B, which is sued to modify the gradients computes by a small amount.

The intention is that, this elastic force allows the gradients to explore more local minimas, leading to faster convergence.
Elastic ASGD is experimentally shown to work faster than vanilla ASGD for the Cifar10 and Imagenet datasets.
Gossiping SGD, extends the all reduce algorthm by introducing some asyncronosity to it.

Overall the following results were observed:

1. If number of machines is upto 32. ASGD can converge faster than all reduce SGD when learning rate is large. But all reduce converges most consistentally
2. When machines are upto a scale of 100 nodes, all reduce SGD can consistently converge to a higher accuracy solution.

This establishes that the model of asyncrosity is not very compatible with deep learning, according to the paper "How to scale distributed learning ?"
## Introduction

Distributing training of neural networks can be approached in two ways- data parallelism and model parallelism. Data parallelism seeks to
divide the dataset equally onto the nodes of the system, each node will have a copy of the neural network along with it's local weights.
Each node operates on it's own set of data and updates it's local set of weights. These local weights are then shared across the network, where 
the weights of all the nodes are accumulated to generate a new global set of weights through an accumulation algorithm. These global
weights are then distributed to all the nodes from whereon the processing of the next batch of data commences. Model parallelism on the other hand 
seeks to distribute training by splitting up the architecture of the model onto seperate nodes. Alexnet was one of the first models which used
model parallelism by dividing the network among 2 GPU's to fit the model into memory. Model Parallelism is applicable when the model architecture 
is too big to fit on a single machine or the model has some parts that can be parallelized, e.g: an Object Detection Network has separate bounding 
and class prediction heads whose processing is independent of each other. Generally, most networks can fit on 2 GPU's which limits the amount of 
scalibility and time savings that can be achieved. Therefore, this paper primarily focuses on data parallelism.

Distributed training algorithms can be roughly classified into types- Asynchronous and Synchronous Stochastic Gradient Descent. 
Synchronous SGD aims to replicate the algorithm as is in a distributed setting thereby tightly coupling the nodes in the network.
On the other hand, Asynchronous SGD decouples the nodes from other worker nodes by decreasing their interdependence. In doing so
it brings fundamental changes to the vanilla version of the algorithm leading to lower quality results. Several modifications 
to Asynchronous SGD have been proposed to close the accuracy gap between it and Synchronous SGD.

Recent trends have gravitated towards scaling Synchronous SGD to promising results. More specifically, research has gravitated towards
training networks with large batch sizes. Large mini-batch sizes have a few benefits, the chief one being that SGD over 
large mini batches allow the model to take bigger steps towards the local minima, hence speeding up the optimisation procedure.
In practice however, training networks with large batch sizes lead to divergence problems or a "generalisation gap" i.e the
test accuracy of the network is at times much lower than on a single GPU model. Efforts have been made to enable model training (converge) over
large batches and have achieved some success. (Imagenet in an Hour) trained Imagenet in an hour by scaling up the batch size to 8,096. A 
technique called LARS was introduced that allowed the use of batches upto 32k and more recently with a combination of mixed precision training 
Imagnet was successfully trained in 4 minutes using a batch size of 64k. There have been problems removing the "generalisation gap" among models
that are trained with large batch sizes and methods like ___________ etc have helped mitigate it to an extent. It is however still an active topic of research. 

There is another important component to Distributed Training which is the communication of data to and fro between nodes, on which a lot of 
advanced research has already been done thanks to the work of GFS, Hadoop and a number of distributed file systems/databases. Collective 
Communication Primitives are particularly influential in bringing HPC to deep learning. They allow for a powerful backbone to 
trasfer gradients to a connected nodes effectively in optimal time. Modern deep learning frameworks like Tensorflow, Pytorch use a primitive 
called all reduce to share gradients. All reduce has several variants like the Ring All Reduce, Recursive Halfing/Doubling, Binary Blocks 
Algorithms among others are used in practice. In distributed training, the computation vs communication has to be kept optimal for efficient 
horizontal scaling. The training remain optimal if the communciation step is as efficient as possible while also synchronising the computation
of various machines such thast ideally computation is finished at roughly the same time. In slow network conditions, the communication between 
nodes proves to be the bottleneck. In addition to surveying distributed training we seek to provide a technique to train efficiently on
slow network conditions. Gradient compression and mixed precision training are promising techniques that can increase overall throughput
of the network. Recent work (superconvergence) has discovered that using cyclic learning rates can lead to a 10x reduction in the number
of epochs needed to achive network convergence thereby making it a promising research avenue in distributed training. This paper is roughly divided into four sections, the first section surveys the existing optimization training algorithms and the second focuses on handling communication across the nodes of the network.The third section explores techniques like gradient compression, mixed precision training and superconvergence for training under low powered devices and slow network conditions. Finally, a section compares the training approaches and selects the optimal training algorithm and communication primitive for different settings ending with future work and a conclusion.


### SGD 

Synchronous SGD is a distributed gradient descent algorithm, it is currently one of the most popular optimisers used to distribute training. 
Nodes in the network compute gradients on their local batch of data after which each node sends their 
gradients to a master server. After the master has recieved gradients from all the nodes the average of all these gradients is computed to form the 
final set of gradients for the weight update step.
The final gradients are used to update the weights using the same formula as the single machine SGD after which the updated weights are sent to
every node so that the nodes can start processing the next batch of data. This whole procedure is analogous to computing a forward pass and 
backpropogation step through a single mini batch of data on a single machine.
Hence, Synchronous SGD guarentees convergence. However, there are some disadvantages of using Synchronous SGD. 
These fallacies are described as follows:

1. Stragglers. In a distributed system, machines can take a long time to
return a response. Slow network conditions, failed network requests, machine crashes or even byzantine errors
are all possible failures that are common in a distributed network. In this unreliable network, Synchronous SGD due to its tightly coupled nature
can take a long time to converge. The machines which take a long time to respond are known as stragglers. (Revisitng Sync SGD) observes that 80% of the second last gradients arrive in under 2 seconds, whereas only 30% of the final gradients do.
Furthermore, the time to collect the final few gradients grows exponentially resulting in wasted idle resources and time expended in waiting
for the slowest gradients to arrive. A possible solution to this could be to decrease the number of machines. However, it has been observed 
that reducing the mini batch size increases the total number of iterations required for convergence. It is observed that there is nearly a 
linear increase in number of iterations required as the mini batch size is decreased. A popular approach to this problem is to introduce backup replicas
that perform the same processing as the worker nodes. The gradient aggregation completes when the gradients are received for the first N machines. The use of backup
replicas seeks to lower the probability of machine response delay. According to [x],  there is a tradeoff between the number of replicas and the 
time for convergence. It is observed for a 100 machine cluster, the optimal configuration is to have 96 workers and 4 backup replicas.

2. Another issue with Synchronous Gradient Descent is the synchronization barrier. The synchronization barrier is the amount of time spent
in waiting for all the workers to send their gradients before aggregating them. In practice, this can be quite a long time depending on the
machine state and network conditions. Training is only as fast as the slowest stragglers. This synchronization barrier can be mitigated to
some extent by introducing replicas. A promising area to look into to help allievate this is better communication primitives that utlilize 
network bandwidth more effectively. However, these are small hacks that don't solve the problem of the barrier due top the nature Sync SGD is 
modelled. Asynchronous SGD removes this synchronization barrier, however it brings along a new set of problems that will be explained in the next 
section.

3. Single Point of Failure. Due to a master slave setup of vanilla SGD, there is a single point of failure with the master. Worker nodes 
communicate with this master leading to a single point of failure problem. The single point of failure also lends itself to bandwidth problems
as a high number of machines try to communicate with a common machine at the same time. This can lead to network spikes requiring the master to be a powerful
machine. Dean et all try to address this by introducing parameter servers which act as the masters for a subset of worker nodes but a tree like 
hirerachy still lends itself to single point failures. Peer to peer communication mechanisms like the all reduce algorithm removes this single 
point of failure though it do not solve the synchronization barrier. All reduce has an added benefit of providing better utilisation of 
network bandwidth than the master slave edition. 

A fault tolerance approach in training with Synchronous SGD has not been addressed in literature as of now to the best of our knowledge.
Fault tolerance in production deep learning training is managed by systems like Docker, Kubernetes and Spark that use some form of state 
management and failure recovery although this has not been stated explicitly in the literature before. Currently, the the vanilla all reduce 
algorithm fails even if a single machine fails.

We propose a modification to the all reduce algorithm inspired by the Raft algorithm that allows it to operate in an unstable environment.


## Gradient Acculmulation Algorithms

Gradient Accumulation algorithms represent an important component of distributed training systems. These algorithms are responsible for 
accumulating the gradients from each worker node and after the updated weights are computed, distributing the updated gradients back to the
worker nodes. The all reduce algorithm makes for a very good fit for this functionality. The all reduce algorithm comes from the world of 
High Performance Computing (HPC), it offeres the following functionality. If there are n number of machines and each machine has some data with it,
the all reduce algorithm will perfrom an associative operation on the data from each machine and deposit the resultant value to all the machines in the network. 
This functionality is useful for SGD as the SGD procedure averages (an associative operation) all the gradients and deposits the updated gradients
to all the machines, hence making SGD a good candidate to integrate all reduce with. Baidu et al introduced the ring all reduce algorithm to deep learning
and much research on distributing deep learning training after that has used some form of the all reduce algorithm for gradient distribution
making it is a staple in deep learning. There are quite a few variants of the allreduce algorithm, these have been decribed in the coming sections.
This includes our proposed all reduce algorithm coined Tolerant All-Reduce, which is capable of providing fault tolerance in an unstable networking
environment.

### Ring All reduce 

The ring all reduce works by dividing its functinality into two parts.
The scatter-reduce and the all gather.
The scatter reduce process is repeated or p-1 steps where p is the number of machines.
1. Each machine at jth step sends i-j+1 chunk to process i+1 and recivies i- j -1 chunk from process i-1.
2. When machine gets the value, it performs it's reduction and keeps it as a store.
3. This process carries on with each machine sending it's reduced from and perfomring reduction with the recieved piece and the original peice to form the new reduced piece.

After the scatter reduce process ends, each machine has a chunk that is part of the final result. Now, th machine simply have to broadvcast their piece of the final chunk to all other machines.
This is done using the all gather, which is exactly same as the scatter gather, but instead of a reduction on recieving, the piece is simply stored as is, as it is the final result.

The all gather process is repeated or p-1 steps where p is the number of machines.
1. Each machine at jth step sends i-j+1 chunk to process i+1 and receives i- j -1 chunk from process i-1.
2. When machine gets the value, it performs it stores the value.
3. This process carries on with each machine sending it's stored value until p-1 steps.

Hence, the network latency of this ring all reduce algorithm is 2*(p-10) or log(P). The ring all reduce algorithm is quite popular and is in use in production systems like Tensorflow and Caffe. 
It's advantages are as follows:
1. Efficient Use of Network Bandwidth. Machine are always sending a chunnk of data from their machine to another machine. So no machines are idle.
2. Peer to peer approach ensures that there is no ingle point of failure. However, the ring algorithm does not take into account failure scenarios.

Disadvantages
1. The process takes O(N) time, the algorithms we will study later have log(N) complexity.
2. Not fault tolerant, if a single machine fails, the whole procedure will need to be started again.

## Recursive Distance Doubling and Vector Halfing Algorithm 

The recursive distance doubling and vector halfing algorithm works using 4 different primitives, that used in the algorithm. These are given as follows:

1. Recursive Vector Halfing - The vector is halfed at each time step
2. Recursive Vector Doubling - Small pieces of the vector scattered across processes are recursively gathered to form a large vector
3. Recusrsive Distance Halfing - The distance between machines is halfed with each communication iteration.
4. Recursive Distance Doubling - The distance between machines is doubled with each communication iteration.

Similar to the ring algorithm, the all reduce algorithm is made up of two procedures, the scatter-reduce and the all gather. The difference between this algorithm and the ring algorithm is in how these procedures
perfrom the operation. The scatter reduce for recursive distance doubling and vector halfing algorithm runs for log(P) steps, where P is the number of processors. 

Let's assume that P is a power of two.

1. Machine i communicates with machine i + p/b, where b = 2 in the first step and is multiplied by 2 on each new step. 
2. This communication between 2 machines happens as follows, machine i divides it's vector into two parts. One part is for sending and the other is for receiving and reducing on. For example, machine 1 could use the top half of the vector to send, and the bottom part to recieve and reduce on, then the second machine will use the opposite configuration.
3. Hence, after data is recieved from the counterpart process. The received data is used to reduce the original data, now, this new data is used for vector halfing in the next step. Hence, in the next step when 
distance between machines is p/4, the data thast is divided into half now is the reduced data from the previous step. Hence, the distance is doubled and the vector is halfed at each step.

If P is not a power of two, the algorithms is slightly modified by doing the following, the largest power of two less than P is calculated P`. 
We calculate r = P -P`, then we take the first 2r machines and do the following:

1. The even machine communicate with the odd machines, hence machine i (where i is even) commuincates with machine i+1. 
2. Both these machines, exchange data such that the even machines have the reduced vector of both the two machines.
3. Now, these even machines are used along with the last r machines in the recursive distance doubling and vector halfing algorithm given above with the odd machines in the first 2r machines not being used in the rest of the 
procedure.

The above algorithm makes sure that the recursive distance doubling and vector halfing algorithm operates on a power of two number machines because the even machines + r is always a power of two.

Once the reduce scatter process is complete, each machine has 1/pth sized chunk which is part of the final resultant vector. To bradcast these chunks on every machine the all gather collective proimitve is used.
The all gather primitive gathers/combines dasta from each machine and broadcasts the resultant vector to each machine. The all gather for recursive distance doubling and vector halfing algorithm runs for log(P) steps,
where P is the number of processors. It communicates in the exact opposite way as the scatter reduce

1. Machine i communicates with machine i + p/b, where b = 2^log(P) in the first step and is divided by 2 on each new step. 
2. This communication between 2 machines happens as follows, machine i divides it's vector into two parts. The reduced/final chunk is meant for sending and the data that is recieved is meant to replace the data that 
was there previously.

3. In the next step, the vector to be sent is the combination of the recieved chunk and the sent chunk, this is known as vector doubling, the vector doubles in size after every communication iteration.
4. This process continues on by doubling the vector asize and halfing the disgtance between machines until log(P) steps. It is the exact reverse of the scatter reduce process, and continues on until each machine has the final resultant vector. The time complexity for this algorithm is the same as the scatter-reduce.

After the end of the all gather, all machines have the resultant vector signaling the end of the all reduce process. The final complexity to the entire algorithm is 2alogP + 2nB where a is the startup time 
per message, where B is the transfer time per byte and n is the number of bytes transferred. We shall ignore the reduction procedure complexity as that is independent from communication between machines. Hence, the 
final complexity is 2alogP + 2nB. For non power of two processes, the time complexity is 2alogP + a + 3nB. If the number of machines are not a power of two, after the all reduce ends, the resultant vector needs
to be sent to the odd numbered machines which were not used in the process. This results in an overhead of a + nB.

Advantages
1. The complexity of this operation is reduced from 2aP + 2nB to 2alogP + 2nB, reducing the complexity of the algorithm from O(N) time to O(logN) time.
Disadvantages
1. When the number of machines are not a power of two, a substantial overhead can be introduced as the number of machines (the first 2r odd numbered machines) are left unused during the all reduce process,
hence the scalibility of the program with respect to the total number of machines is reduced. A binary blocks algorithm reduces this overhead.

## Binary Blocks Algorithm

The binary blocks algorithm is an extension to the recursive distance doubling and vector halfing algorithm as it seeks to lower the degree of load imbalance for when the number of machines are not a power of 
two. In the original algorithm for the non power of two case, a number of machines are set aside until the algorithm completes its execution, after which the resultant vector is transfered to these machines. 
This apprach leads to a large number of machines being left idle in some cases, for example, if there was a cluster of 600 machines, 86 machines would be left idle while the processing executes on the 512 machines.
Hence, there is a significant load imbalance encountered in the network using this approach. 

The binary blocks algorithm seeks to alliviate this problem by dividing the number of machines into blocks of power of twos. As an example, if we have a 600 machine cluster, there will be 4 groups having 2^9, 2^6, 2^4, 2^3
machines respectively. The binary blocks algorithm works as follows:

1. Each block executes the scatter-reduce procedure of the recursive distance doubling and vector halfing algorithm with the machines in it's block. After every block finishes its scatter reduce 
procedure, the machines in the smallest block send their reduced final chunk data to the machines of the block that is next in line in the size hierarchy. This data is reduced with the corresponding data
on the machines on the bigger block. 

2. This data transfer and reduction after the scatter reduce execution is continued up until the data has reached the biggest block. 
3. After the scatter reduce and transfer of data between all blocks has been completed, the reversal of the same process is started to distribute the final vector to all machines (the all gather procedure).
4. Starting from the biggest block, data is sent down to the smaller blocks, alongside the data transfer for the all gather procedure in their own block. Once a block gets data from the bigger block, it starts 
   it's own all gather procedure and data transfer to the block below .This process goes down the block hierarchy until the all gather process completes on all the blocks.

The time complexity of the binary block algoriothm is 2logP + 2nB, the load balance depends on the amount of data transfer between machine inter block. This algoroithm doesn't completely solve the load 
imbalance problem as there is a high tranfer of data between blocks whihc is imbalanced. However, it has been observed that the binary blocks algorithm workswell even for 8+4 and 16+8 making it a good alternate 
for clusters with non power of two number of machines.


