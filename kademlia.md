# Kademlia

This is a description of a Distributed Hash Table. Kademlia is a protocol to maintain a distributed hash table. It is quite popular and has been in use in Peer to peer applications. Several Torrent clients use Kademlia as their DHT implementation.

The need for a DHT arose in Torrent Applications to find out where a certain file was kept. The DHT discovered the location of the file after which the file transfer process could initiate.

## Purpose

The purpose of a distributed hash table especially in a peer to peer file sharing system is two fold.

1. To figure out the location of a certain file in this maze of nodes/machines.

2. To refresh the location of these files with continous confirmation of node liveliness. To keep a group of the freshest nodes for a file.

These two functionalities must be processed in the most efficient way possible.

Kademlia manages this node finding operation in log(n) time, where n is the toal number of nodes in the system along with refresheshing stale nodes. Kademlia also has the property of , nodes will have more information of nodes similiar to it in it's Hash Table.

How? Well, read on.

## Kademlia Explained

This explanation will be broken down into three parts

1. Operations Supported 
2. Setup
3. Distribution Of Information

### Operations Supported

The DHT recieves a 256 bit key from the client. This function discovers the closest nodes to that key.

There are three checks before this call.

1. Checks if the node has information for the key. This could be done by hitting some database or data source, we will not go into detail of how that is implemented.It is abstracted away by the FIND_VALUE call.

2. If the FIND_VALUE call returns data, we return the data to the client with a FOUND_DATA message.

3. If the FIND_VALUE call returns null / no data , we want to return a list of nodes to the client that have a better chance of having the required data. Enter the FIND_NODES call.

Some intuition must be forming in your mind, that the client keeps requesting various nodes in a recursive fashion until required data is found.

Now, let's head on to the setup that helps support these calls.

### Setup

Every computer in the Kademlia network has an id. This id represents the node in the network and is called *node_id*. The *node_id* is a 256 bit number in most cases. Let's assume that it is a 256 bit number for now.

Each computer has an array called *NODE_ARRAY* of the capacity of the number of bits of the node_id, 256 in this case. The indexes of the *NODE_ARRAY* represent the position of the bits of the node id. The value for each entry of the hash table is a list of some size. This size is 20 in the original paper. These lists store information of nodes and their location. Information includes IP Addresses, node id, port number among others.

The *NODE_ARRAY* supports two operations.

1. Find Nodes closest to key k. The FIND_NODES call.

2. Store key,value. The STORE call.

All of these *NODE_ARRYAS* in each node across the entire network make up the Distributed Hash Table. Clients hit these tables asking for information for a certain key and recieve a response.

Other Operations supported by a node are

1. PING. Checks liveliness of a certain node. It checks if the node is still in the network or not. Think of it as a heartbeat message.

2. FIND_VALUES. Returns data from Data Source if exists, else return null.

Let us go into detail into how the hash table implements the FIND_NODES and STORE call. We won't be going into how the PING and FIND_VALUES call work.

#### Find Nodes Call

The key in the argument of this call represents the node which has the data needed.

The call works as follows:

1. Moving from the most significant bit in the key to the least significant bit. At each step ,the DHT compares the values of the key with the machine's node id.

2. If the bit of the key and node id are the same, we go to the next bit.

3. If not, we consult the list of the *NODE_ARRAY* at the bit index where the node id and the key differ.

4. We traverse this list and the other lists ahead until we get k nodes. We returns these k nodes to the client.

This details the FIND NODES call. What happens when this list is sent to the client ?  
Read on.

##### Client Operations

1. After the client recieves these k nodes, it calls the FIND_NODE call on each of these k nodes.

2. It continues this process again and again until there are no nodes closer than the previous best results.

The ordering of lists is done by examining the distance between the node id with the key. If the distance is less, it is likely that the machine has the information the key is looking for. Hence, keys and node ids are very tightly correlated. We can thnk that the machine is named according to the data it posseses or vice versa. We calculate this distance by computing the XOR of the node with the key. THe XOR distance metric has several useful properties. It supports the triangle inequality theorum and doesn't have any geographical constraints, so a machine in Australia and a machine in India can be very close if their XOR reveals a small number. Even the FIND NODES procedure is simply doing the XOR call by comparing bits, it stops the XOR procedure when it encounters a difference in bits. Anyway, I digress. Back to the client operations call.

3. When the iterations stop, the best k nodes in the result list are the ones closest to the desired key.

The intuition with this process is that recursively the client will receive nodes that are closer and closer to the key until it finally chances over the correct node.

The question now , that may come to mind is how exactly is it taking just log(N) hops to find the right node. This is cleared in how the STORE call works. 

####  Store (Key, Value)

The STORE operation stores a value to a list in the *NODE_ARRAY*. 

The STORE procedure is called _whenever a new key is encountered in the node_ . Whenever the *NODE_ARRAY* encounters a new node in a FIND_NODE call, it's information is updated into the table via the STORE call. This results in each node keeping a very dynamic list of nodes in its table. Implying that the DHT is always changing and adjusting to the requirements of the data and the users.

The value is inserted into the list according to the key, similiar to how the searching operation in FIND_NODES works. Once the list is identified, we ask the following question.

Is space available in the list?

1. If yes, we simple append the node information to the list.

2. If not, we Ping the last node in the list to check if it's still alive. If not, we replace this node, if yes we reject the key , value pair and don't change the DHT at all.

## Distribution Of Information

he Kademlia algorithm favours old active nodes compared to new nodes. This does not mean that there can be a condition where the key value is not stored in any node as in the process of a FIND_NODE call, we shall come across nodes which have space. This is because of how the NODE_ARRAY table stores elements. The NODE_ARRAY has the following properties.

1. The farther in you go inside the NODE_ARRAY table, the sparser the lists get.
2. Also, the NODE ARRAY is structured in such a way that it knows more about the nodes closer to it. This is obvious as the sample space reduces by 1/2 by going one bit in. 1/4 by going two bits in. 1/8 by going 3 bits in. 
3. We choose the 256 bit node id as this gives us a sample space of 2^256, which is more than the number of atoms in the universe.

As each bit we go ahead we reduce the smaple space by log(n)

This methodology leads to two very important things:

1. The DHT keeps old reliable nodes over new uncertain nodes. The network needs reliable users to be a good file sharing application.

2. The second property we get is that by going from most significant bit to least significant bit, we ensure that our DHT stores more nodes nearer to it than those farther to it.




On the dynamic nature of the DHT

```

As nodes are encountered on the network, they are added to the lists. This includes store and retrieval operations
and even helping other nodes to find a key. Every node encountered will be considered for inclusion in the lists. Therefore,
the knowledge that a node has of the network is very dynamic. This keeps the network constantly updated and adds resilience
to failures or attacks.

```

We shall discuss more about Kademlia in later posts. Maybe even dvele into its implementation and constructing an in-memory persistent data store with snapshotting and a read only transactional log. The things that I haven't talked about are things involving security (DHT's are particularly suspectible to Sybill Attacks) and a concrete explanation of the performance of a DHT.

DHT's are finding new found research interest with the advent of the block chain and hopefully new research will leads us to a efficient , secure decentralised application. Fun fact about Blockchain, Ethereum uses Kademlia.


Cheers
