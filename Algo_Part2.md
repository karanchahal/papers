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
 
 
 ## Fourier Transform Divide and Conquer
 
 let n = 2^k
 theta = 2pi/n
 w = e^(i2(pi)/n) (complex n^(th) root of unity)
 w^(n) = 1 ( = e^(i2pi) = 1)
 1, w, w^2, ....w^(n-1), 1, .......
 
The above is a circle. 

WHat is fourier tranforms, takes a seq of n numbers, get new seq of n numbers.

How does new differ from old seq.

What is the use of fourier algos ? just know it is important. It is so useful that the whole algo is actually implemented in hardware. 

#### Defination:
A = (a0, a1, a2.... a(n-1) ), its F.T (fourier transform)
is a seq B = FT(A) = (b0, b1, b2 .... b(n-1))

B = (Some matrix) * A
entry of matrix i,j (row, col) = w^(ji) : (w = omega)

Hence, b0 = a0 + a1+ a2..... + a(n-1)
b1 = a0 + a1w + a2w^(2) + ..... + a(n-1)w^(n-1)
b2 = a0 + a1w^(2) + a2w^(4) + .... + an-1w(2(n-1))


Can also be eval in a polynomial eqn:

Pa(x) = a0 + a1x + a2x^(2) + ....a(n-1)x^(n-1)

FT(A) = Pa(1) + Pa(w) + Pa(w^2) +..... Pa(w^(n-1))

Hence FT is finding out this polynomial of a set of points.


#### O(nlogn) Algo: 
A = (a0, ..... a (2n -1))

B = (a0, a2, a4,....a(2n-2)) = b...
C = (a1, a3, ....., a(2n-1)) = c...

FT(A)j = a0 + a1w^(j) + a2w^(2j).... (a2n-1)w^((2n-1)j)
split into odd and even terms
 = b0 + b1w^(2j) + ... + w^(j) + w^(j)(c0 + c1w^(2j))
 = 
 FT(A)j = FT(B)(jmodn) + w^(jmodn)FT(C): (j mod n is minor point as ft cycles around) ("merging procedure")
 
 This is called Fast Fourier Transform.

Application fo Fast Fourier transform:
1. Detecting periodicity. 
2. Compression


Polynomial Multiplication in O(nlogn) time.
P(x) = a0 + a1x + a2x^(2)...an-1x(n-1) +... 0
Q(x) = b0 + b1x + ....b(n-1)x^(n-1) + .... 0
R(x) = P(x)* Q(x) = r0 + r1 +.... r2n-1x^(2n-1)


Fact: A poly of degree <= 2n-1 is *uniquely* determined by its values at 2n distinct points.

F.T of sequence of length 2n 
Using F.T evaluate (P(1), P(w)...., P(w^(2n-1)))
Using F.T evaluate (Q(1), Q(w)...., Q(w^(2n-1)))

Since R = P * Q;
S  = R(1), R(w)...... R(w^(2n-1))


S = FT(r0, r1, ...r2n-1)
R = FT_inverse(S)

FT is same as FT_inverse, but in the latter w is replaced by w_bar.

Exercise: Verify the follwoing, product of 2 matrices
[...wij...] * [....w_bar_ij...] = n*[identity matrix]

Cool thing is we can multiply polynomials in O(nlogn) time !


Theorum: Any (comparsion based) sorting algorithms must make Omega(nlogn):

Whatever algo we have, make a "decision tree". eventually we end up at a leaf to get sorted array. 
leaves are n!, approx 2^(nlogn)
Decision tree is binary.
Hence as num of leaves is X = 2^height,

Hence, height of decision tree is nlogn.
These are called lower bound results.

Its generally extremely difficult to prove lower bounds.
