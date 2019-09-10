# Data Structures & Algo

Office Hours, (11:15- 1:00 Clock)

#### Arrays

Sequence of locations a[1], a[2]...a[n], array of size n. 
Can update, read the array value in 1 step if given i. 

Problem: Given array of size n, find maximum element in the array.

#### Pseduo Code
```
Let M = a[1]
for i = 2,3...n {
if a[i] > m then M= a[i]
}
Output M
```
Now Prove Correctness and Running Time:
Ans: Self evident ?!

In the exam, for complex problems, write in plain text instead of pseudo code. 

### Binary Search
Binary Search: Given sorted array, Given b, decide whether b appears in the array. 

O(logn) time. -> worst case. 
T(n)  <= C + T(n/2) (we get a recurrance relation.)

HOw do we solve recurrance relations.

T(n) <= O(logn)

#### Linked List

Sequence of size n, but the cost of accessing index i, it takes time i.

Problem: 2 lists each of size n, sorted. Problem is to combine them into 1 list. 

Worst Case time: Naively O(n^2) time, but 
more cleverly we can merge in O(n) time.

This algo is also called Merge List.

#### Merge Sort. 

Proof Of correctness: "Obvious": Mumble some sentences to be safe. 
Running Time: THe algoritgms has log n lebvels or steps. If looked at the tree perspective.
And each level, we perform merge list, which is O(n) time. 

# Heaps (Priority Queues)
Data strruct allows
1. Insertion (mostly ints)
2. Find Minimum (or Maximum)
3. Delete a value (customary to delete the min value)
4. All these ops are done in O(logn) time where n: number of elements in the data struct. 

Note => O(nlogn) sorting. 

## Trees

trees are graphs with no cycles, n nodes, n-1 edges. 
Rooted trees define a directed tree. edges have direction. 

Binary Trees: Every node has at most 2 children. 

Tress have *levels* and *heights*

Complete binary tree has full levels, each internal node has 2 children. 

If height is h for a complete binary tree. 
start at level=0;
Nodes = 1+ 2+ ....2^h
= 2^(h+1) - 1


Almost Complete Binary tree (ACBT)

Just oprefix or full internal nodes. 

CBT except only a prefix of nodes are present. 
#nodes >= 2^(h)

## Heap

Def: ACBT with value at each node st value(v) <= value(u ), where u is child of v. 

Min: Read the root. 
Insert: Insert at next available location. in ACBT move up as needed. 
Delete Minimum: 
1. Delete "last" node of ACBT
2. Change value in the root to x
3. Move x down carefully. 

To delete an arbitary node, 

1. One way to do this: Decrease key to -inf and then delete.

you can simulate a heap in an array
children of node in pos n, is (2*n) and (2*n + 1)

## Basic Techniques

1. Divide and conquer
2. Greedy
3. Dynamic Programming
4. Amortisized Analysis (tool to analyse data structures)

# Divide and Conquer

High level recipe to design algos.

To solvea a problem of size n, we divide the problem into 2 sub problems.
And then continue down that path. Until the input is of a managable size. 

- Then combine their solutions into solution of the original problem. 

Recursive Formula for running time 

Let T(n) = running time

So calc running time of merge sort = O(n) (get 2 sublists) + T(n/2) + T(n/2) + O(n) (merge sorted list

T(n) <= 2*T(n/2) + Cn (recursive formula or recurrance relation) for some constant C

- T(1) <= C.

NOw onece we get recurrance relation the T(n) = nlogn

Proof ?

Guess and Verify is one approach. 

T(n) <= Cnlogn

### Proof by induction
base case = T(1) <= C

T(n) <= 2.T(n/2) + c.n

<= 2.C(n/2)log(n/2) + Cn
<= 2.C(n/2)(logn - 1) + Cn
<= C.nlogn , hence proved.

This is induction, but what if you can't guess ?

There is another way,
### Unrolling Recursion
T(n) <= 2.T(n/2) + c.n
 <= 2.(2.T(n/4) + c.n/2) + c.n
 <= 4T(n/4) + 2cn
 .... (now do the same thing again
 ....
 ....
 <= 2^(i)T(n/2(^(i)) + i.Cn
 till i = logn
 so then:
 <= nT(1) + Cnlogn
 <= Cnlogn time. 
 
 
 ### You can also see the and calulate the recurrsion by veiwing it in a tree and adding the time spent. 
 
 
 Exercise: T(n) < = T(n/3) + T(2n/3) + n, what is ther solution to this recurrance, as long as two sizes are 
 comparable. 
 
