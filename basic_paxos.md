# Paxos

Welcome to the explanantion of one of the most important algorithms of distributed systems. Paxos is a protocol to gain consensus in distributed systems.

It was invented by Leslie Lamport in 1980 and continues to shape the distributed systems of today.

## What is Paxos

The goal of using Paxos is to create a replicated state machine. A state machine could be an application, database or a program. 

We want the same application to run concurrently on different servers.

To get a replicated state machine, we need a replicated log. 
A log tells the actions that the state machine to take to arrive at various states. It is obvious that if we have a replicated log ,the state machines at every server will run exactly the same by using this log. They will take as input the same actions,in the same order and arrive at the same result.

Each mahcine will have its own copy of the log.

Our goal is to replicate these logs in each machine and keep them replicated in the future.


The way these logs are replicated is through an algorithm called Paxos.

## Key Property Of Consensus

Key proprty of consensus. A consensus algorithm can work if a bare majority of its servers are still functioning. For example, if out of 5 servers , 3 are up, the cluster can continue to function.

Typically cluster sizes are small of the order of an odd number. 3, 5 or 7.

If the servers are up , they will run in a way tat make sense. There are no Bynatine errors.

Hence it is a _fail-stop_ model.

## Assumptions

We assume the following things in distributed systems

1. Machines can crash
2. Message may not always be sent, they may be delayed or dropped.
3. If a majority of the servers are up in a cluster, the cluster can still recover.
4. Partitions can occur. But when partition is restored , the whole system can recover.

## The Paxos Approach

There are two types.

1. Basic Paxos
  - One or more servers propose values
  - Systems must agree on a single value as chosen
  - Only one value is every chosen
2. Multi Paxos
  - Combine several instances of Basic Paxos to agree on series of values forming a log.


## Basic Paxos

### Requirements

1. Safety -> Algorithm must never do anything bad. Never choose second value to replace first one. A server never learns that a value has been chosen until it really has been chosen.

2. Liveliness -> As long as majority of servers are up and communicating with reasonable timelines. Some value is eventually chosen. If a value is chosen, servers eventually learn about it.

The term "consensus problem" typically refers to this single-value formulation.


### Components

1. Proposers -> They are Active. Proposers put forth particular values to be chosen. They handle client requests.

2. Acceptors -> They are Passive. Proposers respond to messages from proposers. Responses represent votes that form consensus. Store chosen value. State of the decision process. They want to know which value was chosen. 

Each Paxos server contains both components.

## Always keep in mind

Mahine can always crash, so it is important to form a quorum to make any decision. 
Quoram is a group of multiple acceptors. A value v is chosen , if accepted by majority of acceptors.

Hence, if one acceptor crashes, chosen value is still available.

``` Quorum is tricky !! ```

### Other failures

1. Suppose we decide each acceptor accepts the first value it sees.

  - Doesn't work as there will be one point where acceptors have accepted different values, with neither forming the majority.

  Hence this leads us to say that acceptors can change their mind...

2. Hence, now lets make the acceptors accept the latest value that comes about.

This again doesnt work as the acceptors could have accepted two values and have two chosen values. 

  - Hence, if a chosen value has been in the network, the proposer must choose/ propose that same value.

3. Hence we need to put a check on accpetors accepting any value. One way is that before proposing a new value, check and see is a value is not being proposed by anyone yet. If not , then propose your value.

  - This has a problem too, if there is some network delay and a proposer proposes a value. This could lead to two chosen values again. Hence the acceptor should be smart enough to reject values that come after it has already been accepted.

  This leads us to accept that consensus cannot be formed in a single pass. It could take multiple trips back and forth to come to a consensus.


## Three commandments Of Paxos

1. Acceptors can be open to letting go of their accepted values

2. Once a value has been chosen, future proposals must _propose/choose_ that same value. (2 phase protocol). First look around, then propose.

3. Must _order_ proposals. Reject old ones.


## Ordering the Proposals

1. Each proposal has a unique number.

  - Higher numbers take priority over lower numbers

  - Must be possible for a proposer to choose a new proposal number higher than anything it has seen/used before.


One way to do the second is to concantenate the ROUND NUMBER and Server Id

The ROUND NUMBER is the largest round number the server has seen so far.

To generate a new proposal number
  - Increment ROUND NUMBER
  - Concatenate with Server Id

2. Proposers must persist latest value of ROUND NUMBER on disk. Must not resuse proposal numbers after crash/restart.

## Basic Paxos Explained

Two Phase Approach.

1. Phase 1 : broadcast Prepare RPCs

  - Find out about any chosen values.
  - *Block older proposals* that have not yet been completed. So that they can't compete with us.

2. Phase 2: Broadcast Accept RPCs

  - Ask acceptors to accept a specific value.

Basic Paxos is all driven by proposers.


# Fallacies
1. ROUND NUMBER gets wiped, storage gets fucked up how to resolve this ?

## Steps

#### PHASE 1
```
1. Proposer 3 has a value X that it got from the client.

2. Proposer generates a number n Eg: 1.3. The 3 signifies the server id. 1 is the highest ROUND NUMBER it has encountered.

3. Proposer sends Phase 1 RPC. A prepare request.
  -The prepare request is of the type P 1.3 X.

4. It sends it to all the other servers. 

5. When a server gets a Prepare Request. The acceptor does the following:
  - if (n >= ROUND_NUMBER) ROUND_NUMBER = n
  - Sends back to proposer a packet with  (ACCEPTED_PROPOSAL,ACCEPTED_VALUE)
  - ACCEPTED_VALUE is a value that the acceptor must have previously accepted. ACCEPTED_PROPOSAL is the corresponding proposal number for ACCEPTED_VALUE

6. Back at the Proposer 3. It waits for a majority of responses.
Once that is gotten it:
  - If any ACCEPTED_VALUE come back:
  - _It chooses the accepted value of the highest ACCEPTED PROPOSAL number_ 
  - Hence VALUE = ACCEPTED_VALUE of highest ACCEPTED_PROPOSAL
```
#### PHASE 2
```
7. Now Proposer broadcasts Accept command to all servers with packet (ROUND NUMBER, value). A 1 AcceptedValue

8. The Acceptors receive this accept command. They do the following.
  - If n >= ROUND_NUMBER then ACCEPTED_PROPOSAL = ROUND_NUMBER = n; ACCEPTED_VALUE = value
  - return(ROUND_NUMBER)

9. If there is any result > ROUND NUMBER, go back to step 1.

10. Otherwise Value is chosen.
```

## Three Possibilities with this approach

1. Previous Value is already chosen.
  - New Proposer will find it and use it.

2. Previous Values not chosen, but new proposer sees it
  - New proposer will use existing value.
  - Both Proposers can succeed. As ROUND_NUMBER <= minNumber is faine!

3. Previous value has not been chosen, new proposer hasn't see it.
  - New proposer chooses it won value.
  - Older proposal blocked. Bak to square 1 for old proposer.

## Problems with Basic Paxos

1. Basic Paxos is Safe. no matter how many competing vaue there are , only one value will be chosen.

2. Competing proposers can livelock. The accept phase can be sabatoged by a prepare phase. It can form a live lock that goes indeterminedly.

  - Solution 1 -Some randomised Delay Before Restarting (Raft !)
  - Solution 2 - Multi Paxos Leader Election

3. The server needs to do a Paxos Protocol to get the Chosen Value. Onlly the proposer knows the chosen Value.

## Important Note Here

Acceptors and Proposers work independently !!!

# Multi Paxos

Now that you have understood Basic Paxos. We come to Multi Paxos, which is a way to solve this livliness problem and get hte chosen value to all servers quicker.

Multi Paxos is explained when I blog next. This is tiring.

Thanks !
