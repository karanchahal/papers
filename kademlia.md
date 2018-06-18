# Kademlia

This is a description of a Distributed Hash Table. Kademlia is a protocol to mainatian a distributed hash table. It is quite popular and has been in use in Peer to peer applications. Several Torrent clients use Kademlia as their DHT implementation.

The need for a DHT arose in Torrent Applications to find out where a certain file was kept. The DHT found out the location of the file and then hence the file transfer could initiate.

## Purpose

The purpose of a distributed hash table especially in a peer to peer file sharing system is two fold.

1. To figure out where a certain file is in this labyrnith of nodes/machines.

2. Update the location of these files continously and confirming node liveliness.

These two functionalities must be done in the most efficient manner possible. 

Kademlia manages this node finding operation in log(n) time,given that n is the toal number of nodes in the system.

Kademlia also has the added benefit of refreshing stale nodes.

Along with that, with kademlia a node will know more information of nodes similiar to it than nodes further from it.

How? Well, read on.

## Kademlia Explained

### Setup

Every computer in this Kademlia network has an id. This id represents that node in the network. This id is a 256 bit numbere in most cases. Let's assume that it is 256 bit and let's call it node id.

Now, each computer has a hash table of size 256 stored. The indexes of the hash table represent the position of the bits of the node id. The value for each entry of the hash table is a list of some size. This size is 20 in the original paper.

The lists store information of nodes. Information like IP Addresses, node id etc.

### Operations

This Hash Table supports two operations.

1. Find Nodes closest to key k.

2. Store key,value

Lets go into detail how the hash table does these operations.

### Find Nodes

The FIND_NODES call.

The DHT recieves a 256 bit key. The requirement for this functionality is to find what are the closest nodes to that key.

Before invoking this call, the application checks if it has the data requested or not through a FIND_VALUE call. If not , we come to this FIND_NODES call to get a list of nodes more likely to know about this key.

The key represents the node which has the data needed.

1. Now, moving from the most significant bit in the key to the least significant bit. At each step ,the DHT compares the values of the key with the machine's node id.

2. If the bit of the key and node id are the same, the DHT goes to the next bit.

3. If not, it stops there and looks at the corresponding hash table _list_ for _that_ bit index.

4. It collects a collection of k nodes by traversing this list and the subsequent list until it gets k nodes. It returns these k nodes to the client.

5. After the client gets these k nodes, it calls the FIND_NODE call on each of these k nodes.

6. It continues this process again and again until there are no nodes closer than the previous best results.

7. This list is ordered by similiairty of node with key and the top k nodes are returned to the client. When the iterations stop, the best k nodes in the result list are the ones closest to the desired key.

8. The intuition is that soon the client will receive nodes that are closer and closer to the key until it finally chances over the correct node.

One more note about calculating the similiarity of a key with a node id. You might have realised that we are doing the XOR operation to figure out what list to get the nodes from.

XOR has some nice properties that help define distance and are integral part of Kademlia. t doesn't have any geographical constraints, so a machine in Australia and a machine in India can be very close is their XOR reveals a small number.

###  Store (Key, Value)

The Store key value call is very similiar to the operation above. We find the list to put in the node information with the same procedure as above. Going from most significant bit to least significant bit. Once we arrive on out list, we check if the list has space availiable.

1. If yes, we simple append the node information to the list.

2. If not, we Ping the last node in the list to check if its still alive. If not, we replace this node, if yes we reject the key , value pair and don't store it in our DHT.

This methodology leads to two  very important things:

1. The DHT keeps old reliable nodes more than new uncertain nodes. The network needs reliable users to be a good file sharing application.

2. The second property is that by going from most significant bit to least significant bit, we are ensuring that our DHT stores more nodes nearer to it than those farther to it.

The PING operation checks if the node is still in the network or not. Think of it as a heartbeat message.


## Food For Thought

```

As nodes are encountered on the network, they are added to the lists. This includes store and retrieval operations
and even helping other nodes to find a key. Every node encountered will be considered for inclusion in the lists. Therefore,
the knowledge that a node has of the network is very dynamic. This keeps the network constantly updated and adds resilience
to failures or attacks.

```

We shall discuss more about Kademlia in later posts. Maybe even dvele into its implementation and constructing an in-memory persistent data store with snapshotting and a read only transactional log.


Cheers
