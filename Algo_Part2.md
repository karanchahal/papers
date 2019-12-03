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


# Greedy Algorithms

- Seq of choices
- Each choice is (what appears to be) the "best" at that step
- It works sometimes. Not on every problem
- The emphasis is on **Proof Of Correctness**
- Polynomial Running Time

#### Example with toy examples
Given n items with values, b1,b2,b_n. 
Find a subset of S of k items with max value. 

Algo: "without loss of generality" we can assume after sortingb1> b2>...bn
: outpouts set S = {b1,,....bk}

### Proof: "Exchange Argument"
To prve that the choice we make is the correct choice, 
say for example we pick b1, and say b1 is the correct choice, and we can make a precise statement by saying
that an optimal set contains b1. 
Let ``` S` ``` be the optimal "hypothetical" set.
We must show thst b1 belongs to ``` S` ```. 
If not, then bi belongs to ``` S` ```, i > 1 be an arbitrary element.
Then S = S* \ {b1} U {b1} witll have value > value(S) which is a contradiction.
If b1 is not in the optimal set, and something is say b17, then we just swap it and the condition still holds. 

### More Complicated Example. 

Same question but this time coins have weights, so thief csn only carry only so much. 

Check whether items are divisible or not divisible. (name self obvious)

##### Case of divisible items


**Be greedy according to the criteria !**
Choosing right criterion is part of designing 

Here, be greedy with value per unit of weight = (value/weight)

##### Case of indivisible items
It like a gold cup/silver spoon, we can not turn it into powder and take powder to it. 
This is an NP hard problem, meaning that there is no polynomial time algorithm for this
We don't expect to find one, although it is an innocent looking problem. 
It is not easy to say what has fast/slow algorithms.
Thinkn aout greedy according to what criteria. 
There is always a brute force algorithms



## Interval Scheduling

-------- -------- -------- --------
  --------    --------
Maximum intervals we can pick that arr paisrwise disjoint (non intersecting)
- Given n internvals i_th: [start, finish]
- Goal is to find maximum sized subset of non intersecting intervals. 
- Solution need not be unique.

Shortest job first
-------- --------
       ----
Nope

Earliest start time first
------------------------
 ----- ----- ------ Nope
 
 Minimum Conflicts First: See counter example in [KT]
 This is also not good, thinking of counter example is still non trivial
 
 Algorithm:
 - Earliest finish time first
 - After sorting, let's assume that
    f1 <= f2 <= f3... <= fn 
 - At each step - pick interval with earliest/least finish time
 - Delete all intervals that overlap with.
 - Stop when no intervals remain.
 
 Proof: What intervakls we are choosing are "correct"
 
 Usually done with induction and exchange argument. 
 
 Lets prove that this algo makes correct choice in first step. 
 That is let m = maximum # of disjoint intervals possible.
 We want to prove that there is a set S of m disjoint intervals
 
 such that [s1, f1] is part of this set S. that f1 is part of this set.
 "this is all hypothetical !"
 Always be sure your first step is not shooting yourself in the foot !
 
 Proof: let O (optimal) = {j1,...j_m} be indices of m disjoint intervals. 
 without loss of generality: --- --- --- --- ...  ---
 then we can say that if the first element does not have finsih time less than f1,
 hence we can simply replace it with f1. 
 
 Now we have proved that the choice was good for only the first choice. 
 Hence, we need to make the actual claim that m is the maximum number.
 
 BUt we need some care to write a statement, to see if this first proof carries forward.
 
 Claim: Let j1,j2,...jl be indices of intervals chosen by algo (i1 = 1)
 Prove l == m which we cant do 
 
 For every k between 1 and l:
     So we prove hat there exists a set S of m disjoint intervals such that this holds:
     - Up to every step, the algorithm was right
     - the intervals {i1,....ik} is contained in S. 
 
 Proof: (Inductive)
 - We will assume that the above is the case for step upto k.
 - Prove that this holds for k+1. 
 - There is a set O of m disjoint intervals such that the first k ones are contained in O.
 So we do thge same ting to get k+1 as before
 
 Let S = {i1...im} is set of disjoint intervals that include i1, i2 ... i_m. 
 
 Follows: hence l=m, because algo terminated in l steps.


# Greedy Algorithms

## Minimum Spanning Tree
## Graph
- We are talking abot Undirected G(V, E). 
Simple path: No reptition of vertices, walk allows repitition
Cycle - start at vertex and come back to the vertex, that wouls be a cycle.

Connectedness
Induced subgraph: Set of vertices, take all the edges of those vertices.

## Tree
Tree: Graph which is connected and has no cycles. T(V,E)

Theroum: If tree has n vertices. The number of edges is exactly n-1 edges. 

Spanning Tree: Given a graph G(V,E) it's spanning tree is a subgraph T(V,E') 
such that T is a tree. Given tree and graph are undirected.

There can be many spanning trees.

## Minimum Spanning Tree
Goal is to build a subgraph that is a spanning tree, which minimises the toal cost of the sum of edges given we have a cost for each edge. 

Problem: Given G(V,E) cost Ce >= 0  (each edge cost is positive) for each e belongs to E, e = (u,v)

To find a spanning tree T so as to minimise cost(T) = Sum_of(cost of all e)

# Algo to find minimum spanning tree (undirected graph, positive edge costs)

Algorithm:
1. Begin by sorting edges in increasing order. e1,.....em
2. Assume that these edges are arranged in increasing order by their costs c1,.....cm
3. Start with graph G', same vertices as G but with no edges.
4. For i = 1,2..m:
    if e_i does not intrduce a cycle in G'
    add e_i to G', other wise throw out this edge.
5. Final output will be graph G_prime. and sum of all edgs: cost of that edge.

Theorum G' is a MST. This might not be unique

Proof of Correctness:
How do we show ,

In greedy algos, we have to show that the choices up to a certain point are correct.


There is a MST T such that e1 belongs to T
### Proof:

"The Exchange Argument": 
If edge e1 doesn't belong in T, 
then maybe let's try to add e1 and see what the picture looks like.

Let T' be MST (a hypothetical one)
- If edge e belongs to T', done T=T'
- Else edge doesn't belong to T'

Then draw dotted line to add edge to T'.
Then since we have a cycle now, we will need to drop an edge,

now we can remove an edge, and it will something not equal to e1 andf the overall cost of T' will be even smaller than below.

Hence, that is the proof.

Then T = has (T' \ e_j) U {ei} is a spanning tree. 

cost(T) = cost(T') - c_j +ci <= cost(T') . c_j >= ci

Therefore T is also MST and ei belongs to T. 

(The non uniqueness of a MST comes from edges having a different cost)

Claim: Let A_i denote frist i edges e1,,,e_i }

and let Si = set of edges among {e1...ei} that the algo chose to include
0<= i <= m.

Then there exists a minimum spanning tree T such that T intersect A_i = S_i

## Proof by Induction

by induction i = 0 t intersect empty = empty.

assume there exists a MST T' such that T' intersect Ai = Si

Now for i+1, we need to show that our choice is consistent with T'.

Lets look at edge e_(i+1)

**Case1**: e(i+1) Union S_i has a cycle. hence e(i+1) does not belong to S_i
e(i+1) doesn't belong to T' either, as since it intruduces cuycle with S_i, it will introduce cycle with T' too.

Hence T' intersectoin = S(1+1)
take T=T', T interests A(i+1) = S(i+1)


MOre tirckeier case is **case 2**

ei+1 Union S_i doesnt introduce cycle. 

Sub case 1: e(i+1) belongs to T' done (easy case)
            T = T' ,  T union A(i+1) = S(i+1)
Sub case 2 e(i+1) doesnt belong to T' (tricky case)
- hence T' Union ei+1 contains a cycle C. 

- This cycle is not poart of S_i+1 but is is part of the Tree T'. 
Hence Cycle must contains an edge e_j that is part of the tree T'

THE PUNCH LINES:
Therefore this edge is an edge whose index is more than i+1. This is an edge which the algo hasnt seen so far
this is because this tree T' is completely consistent with first i decisions. If algo had seen i before and thrown it out, then tree T' wont contain it either.

There j > i+1.
So t' intersect Ai = S_i

Hence C_j >= C_(i+1) . hence, T = (T' \ ej) Union e(i+1) is a MST

such that T INtersect Ai+1 = Si+1.

Consider i =m,, There exists MST T = Sm (what algo outputs.)

This si Kruskul's Algorithm
# Minimum Spanning Tree

G(V,E), |V| = n, |E = m

## Runtime:
Sort : mlogn (as m is between n and n^2), hence it's mostly the same thing ?

After sorting, take edges one by one, hence making a "forest" of edges.
n components, then these components will merge into each other. 

We need a dats structure to maintain connected components.
And the ops for this data structure are :
Given pair of vertices, the data structure should be able to tell you if these two components fall into distinct or same
components. 
" And if they belong to the different components, can they be merged together ? "

This data structure can take logn time per step.

hence this 2nd step takes O(mlogn)

Hence totoal running time: O(mlogn)

## Union Find Algorithm
(cover on your own)
Instead of logn it takes O(log*n), you start with n, the number of time we reduce n to get constant time. ??

Inverse_Akerman_Function(n)

## Interval Partitioning

It's a scheduling problem. Scheduling jobs on machine is one analogy. 
- Given jobs Intervals.. start time and finsih time per interval. 
Goal is to schedule these jobs on to k machines, so that job scheduled on any machine are disjoint.
Goal is to minimise k.

Have d = depth, maximum number of jobs alive at any time instant. 

### Algorithms
Sort intervals according to start time, such that s1, s2 are in increasing order. 

Take d machines and
for j equals 1,2...n:
schedule the job Ij on any machine that is available. 

### Proof Of Correctness
Its a proof by contradiction,

if machine gets stuck, then it contradicts the assumption that d was the depth.
Hence that is is the proof. 

Suppose on the contrary that Ij cannot be scheduled. 
____________ Ij

1
2
3
.
.
.
d _______________


morover, the jobs already there, their start time must be less than Ij.
Ik1...Ikd start earlier than Ij and this employs that the depth is >= d+1,
which is a contradiction. 

### Running Time. 
nlogn, sort intervals:


## Dynamic Programming

- Recurrqnce Relation + "memoization" - (keeping in memory or remembering)

Or one can think of it making a list of subproblems, and solves them all (systematically).

## fibonacci NUmbers

current value of Vi is addition of Vi-1 and Vi-2

Compute nth Fibonacci number. 

Easy efficent algo: Compute F0,F1... place them, them in order, and solve them left to right. 
Difficult Recursive Inefficient Algo: 
Fib(n) {
 if n == 1: return 1
 if n == 0: return 1
 
 a = Fib(n-1)
 b = Fib(n-2)
 
 return a+b
}

This is very ineffient as it does repeat stuff, it is exponential in time. ( (1 + root(n))/2 )^n


- Only n+1 distinct subproblems F0,,,,,Fn
- Store solution to every sub problem which was solved, use that solution whenever needed. 


It is important to think in recursion.


## Dynamic Programming Strategy

3 step process

1. Identify a list of problems, that we want to solve. Keep careful to keep this list to polynomial size. That includes the original problem. 
2. Identify order among these subproblems. Smaller to larger, left to right...
3. Finally, identify a "recurrance relation" or way that computes solution to a sub problem given solution to its own subproblems.


Algo is iterative, compute solution to all sub problems in order -->.

Emphasis on getting polynomial time running algorithm. 


## Subset Sum (1st Dynamic Programming Problem)
Subset Sum with Bounded Integers

### Problem

Given positive integers a1,....an where each ai <= W , where W is polynomial so think of W = n^2.  (if this is not the case, problem is NP Hard.)
Given 1 <= b <= nW.

Find if there's a subset of these integers which sum exactly to b. 


Sub problems:

SUBSETSUM({aK,....aN}, B): 1<= k <= n ; i <= B <= nW

= { True if there is some suffix ak..an such that suim ==B
= { False, otherwise.

num of subproblems = n*nW = n^4, so it's polynomial, we're good. 

Identify order of sub problems:

SUBSETSUM({Ak,...An}, B) = SUBSETSUM({Ak+1,..An}, B) | SUBSETSUM({Ak+1,....An}, B- aK)

### problem 

Chain Matrix Multiplication
Compute Time  = O(pqr)


If we have Apq, Bqr , Crs matrices.
Matric Multiplication is associative. 

The compute time is different if we multiply ABC like A(BC) or (AB)C.

(A1q Bq1 C1q) = (AB)C = A(BC) but compute time of first is better than second. 


Given matrices A1,....An where sizes match for multiplication. Goal to find the product with least compute time.


Flip problem around, we're thinking of first step, but try to think about last step.

Think about where we want to cut in tyhis sequqence, where we need to multiply the left and right resultant matrices.
to get the final matrix, hence we are done.
ONce we get this , we can recur on left and right, and do the same problem. 

Hence the num of subsequences are O(n2) as we have a left and right end point and we have consecutive subsequences.

cost(Ai....Aj) = Minimal cost of multiplying this consective subsequences. 
- original problem i =1, j=n;
- cost(Ai) = 0
- cost(Ai*Ai+1) = PiPi+1Pi+2.


Recursion:
min(cost(Ai...Ak) + cost(Ak+1,,,,Aj) + PiPk+1Pj+1) {i <=k < j}

Another Toy Example

## Finding Longest Common Subsequence. 
X = AGTCGCAG
Y = GATACCA

So at any instance, take one char from X and one from Y. 
Then, uif two are equal, we can remove both and add to length of subsequnce
or remove any one of X and Y and move forward.

Hence as they are prefixes, then pick the maxmim of these three possiblities, and move forward. 

## Weighted INterval Scheduling Example

Given intervals, and each interval has weight, fnd disjoint intervals such that total value is max. 

The recurrance formula is 

given sorted intervals according to start time,

then opt(t) (optimal time(t)) = max{ .opt(t+1), vt + opt(k), where k is minimum finish < start }


## Shortest Paths in Directed Graphs

Directed graph
- v denotes vertices
- e denotes directed edges

There are notions of paths and cycles. 

Edges have costs >= 0
S: source, t: target

Observation:
n = number of vertices, m = number of edges

1. Least code s to t path has at most n-1 edges. 

2. if there si a shortest path s to y with y, z in between, then it must be the shortest path from s to y and s to w. 

subproblems:

Subproblem of every vertex u belongs to V
0<=i<=n-1
opt(s,u,i) = lenght of shortest s to u path using at most i edges. 

opt(s,u,i) = min( w : (w,u) belongs to edges E (opt(s,w,i-1) + cwu))

Ordering: order according to increasing value of i.

opt(s,u,0) = {INFINITY if u different from s, or 0 otherwise}.

Same algo works if there are no negative cost cycles.


## Finding Independent Sets in Graphs

NO two adcant vertices constitute an independent set, 

Given a tree, find maximum sized independent set. Let's assume it is a rooted tree with no loss in generality. 

Choose a set of vertices which have no edges in between them. 

max_sum(r) = { 1 + sum(max(sumgrandchildofr)), max(sum(childrenofr)) }

terminating condition, if we come on leaf then just take it.

order of problems: according to decreasing r ?




## Amortized Analysis

- Data structures - technique to evaluate data structures. 

- Data structures: add , delete, min, update

n operations:

Let T = total running Time over n operations. 

- A-A shows is that example, T(n) <= nlogn, but without necessarily showing that each op <= logn

### Stack

- push(x)
- pop(k)


running_time = 1 unit of time.
pop(k) = k units of time. 

- sequence of n push, pop operation.

- total runtime <= C.n 

cannot be more than 2 time n. 

## Potential Function Method

Di = State of data struct after ith op
Do = initial state
- potential function si: {d0, d1...} -> R where R is mostly a positive value.

Ci = cost of ith op. | push(x, pop(k))

usually phi is non negative function, phi(d0)= 0

C_hati = actual cost + change in potential
= ci + deltaphi = ci + phi(di) - phi(di-1)


Total amortized cost: of all ops:

Sigma(c_hati) = sigma(ci) + phi(dn) - phi(d0)

Sigma(c_i) = sigma(c_hati) + phi(d0) - phi(dn)

if phi(d0) = 0, phi(.) >= 0,

sigma(i=1, n) <= sigma(i=1,n)c_hati
<= n.t(n)

### now seeing with stack

phi = num of element son the stack. 

phi(d0) = 0
phi(.) >= 0

*for push operation*

amortizied cost will be =  actual cost + change in potential
= 1 + 1
= 2

*pop operation*

amortizied cost . = actual cost + change in potential
 = k + (-k)
 = 0
 
 
 n-ops T(n) <= n.2 + phi(d0) - phi(dn) <= 2n
 
 ## Binary Counter
 
 onl;y op : Increment by 1
 
 worst case time = k (when going from 011111+1 = 10000000 )
 but will not happen very frequently. 
 
Total Time <= n + n/2 + n/4 +.... <= 2.n

- Potential Function: = number of 1s in the counter. 

Am.C = ActC + delta(phi)
= i+1 + (-i) + 1
= 2

## Aside:  Heaps
min = o(1)
insert, delete min, delete , delete key = o(logn)
Union of 2 heaps = O(nlogn)

Where we can't do better than this time. 

## Binomial Heaps

We can do union in O(logn)

In fibonacci heaps, we can do everything in O(1) except the ops that require deletion. 

 ## Bionomial Tree
 
 B0, B1, B2.... and so on. 
 
 B0 = .
 b1 = . .
 b2 = . .-. .
 b3 = . .-. .-. .-. .
 
 so bk = bk-1 bk-1 connect both sub trees. 
 
 Facts: Bk has 2k nodes, height= k. 
 bk has 2 k nodes, height = k
 
 bk = bk-1 - (bk-2,,,b0) all connected hence inductively it is equiuvalent to older bk description.
 
 Nodes at depth i is k choose i for i = 0,1....k.
 
 
 Always assume 
 - integers at nodes,
 
 - heap order.
 
 Defination H = {Bi1, Bi2....Bip}
 
 
 # Directed Graphs
 
 G(V,E) 
 Defination: G is called acyclic if it has **no** directed cycles.
 Fact: Place all vertices on a line, and all edges went only forward. 
 
 Think of this as the given graph. This is an acyclic graph. 
 
 Every acyclic graph is exactly of this type. It is always pssible to take a acyclic graph and get it in this format.
 
 
 Theorum: This is called topological ordering. Find topological ordering in O(m+n) or polynomial time too. 
 
 ## Strongly Connectedness
 
 G(V,E) is strongly connected if for all u an v belongs to V , there is u to v path. 
 
 in polynomial time you can check if a graph is strongly connected or not.
 For each vertex, check every other vertex connects or not. 
 
 # Directed Graphs
 
 
 If we can put the vertices on a line , the edges only go forward. It's a acyclic graph.
 All vertices c an be numbered 1, ...n sucha that all edges (i,j): i < j. Topological ordergin, O(m+n) time using adjacency list.
 
 Strongly connected: If for all u,v belong to V then there is u to v path. 
 
 THeorum: Check in O(m+n) we can decide if G is strongly connected.
 
 Algo:
 
 - B.F.S also works on dir graph. 
 - O(m+n) time given s belongs to V, decide if all vertices reachable from S. 
 - Hence, if we have some node s, then check s to every else, check if opath, and from everywhere else to s is a path. 
 
 Hence, we can prove that the graph is strongly connected. 
 
 Decide if s ->>> V  (through BFS)
 and also decide V ---> s. (do BFS on reversed graph G): change edges to point opposite and then do BFS from s to all V ??
 
 
 # Structure Theorum for Directed Graphs. 
 
 Given a Graph(V,E) 
 
 We cut graph into pieces of vertices, where each parition is strongly connected.
 
 So C1,c2...ci such that:
 
 - For every i, take the graph and restrict it to G| ci, then this is strongly connected. 
 - All other edges go from some component ci to cj with i < j. 
 
 Every directed graph consists of it's strongly connected components plus an acyclic graph on top of them.
 
 This is a decomposition of a graph, and it takes O(m+n) time to get this decomposition into strongly connected components. 
 
 17 problems on this topic kept by faculty member. 
 
 ## IMPORTANT !!!!!!!! Will come in some way in the exams
 
 # How to use this theorum
 Assume graph is acyclic, then it becomes much easier to design the algorithm, then you say in general graph is not acyclic. After this is done, We find the decomposition above, and then generalise the algo for cyclic graphs so that now it works in the general case.  
 
# Dijikstra's Algorithm

- G(V,E), adj list
- for all u, v belongs to E, cost/weight wt(u,v) >= 0
s belong to V, source :
GOal- to find shortest s->u path for all u in V. 

Runs in time O(m + nlogn) time using FIbonaaci HEaps: where m is the number of edges. 
Fibonaaci Heap is O(1) amortized. 
m = Decrease Key and n - delete min- each O(logn) amortized tune, 

## Algorithm
 High Level: Maintain d[v] for all V
 It stores length of s to v path found so far. 
 - Initially d[s] = 0, d[v] = inf for all s.
 - dist(s, u) = weight of shortest s->u path. 
 
 Always dist(s, u) <= d[u] => Note : d[u] can only decrease
 ## Idea
 Relaxing an edge (u, v) : 
 The label for any vertex v (d[v]), you can replace it by min(d[u] + wt(u,v)). 
 

One non efficient solution:
- Relax e1, e2....em
- Do this n times. 
O(mn) time. 
Why does it work ? 

Fix any vertex u, there is a hypothetical shortest path which can at most be of n edges. 

And this being he shortest path, it is also shortest path for every intermediate vertex in this path.


## Efficient Way
NOw we shall find someordering so that we dont have to see all edges n times. 

Relax(u): For every v such that there is an edge from u to v belongs to E
D[V] = MIN{D[V], D[U] + WT(U,V)}

Dijikstra's Algorithm:
- Relax(s), Relax(u2).....Relax(uN)in some order

- At each step, u_i is the vertex with minimum value of d[u_i] among {ui,ui+1,...un}

## Claim

u be outside the cluster, s is in the cluster, hypothetical path jumps out of cluster and then comes to u. 
Ley x->y be first edge outside of T

# Max Flows in Networks

Very important and deep algorithm !!!

Given a directed graph, has source and destination or source and sink.

Each edge has a capacity.

We want to ship maximum amount of data from source to sink. 


### Formal Definition

A flow network is :
- Given a dir graph G
- Every edge has a capacity - C_ee > 0 for each edge e belongs to E. 
- Source s, sink t. 
- No edges into s, no edges out of t. 


What do we want to do:

Send maximum amt of flow from source to sink. 

What is flow ? - The capcities of each edge when we send data on 2 conditions-
1. Can't ship more than capacity
2. Convervation of flow. Apart from s and t, the amount of flow flowing in is equal to amount of flow going out.

Measure of flow is done by seeing the amount of flow going out of source node.
Defination:

An assignment / function f: Edges -> Non negative number (real number)
Such that 
1. Capacity constraint- for all edges little e the flow that we send f_e is at most equal to capacity of e. 
2. Flow conservation constraint.

Problem: value(f) = sum of(flows of all e out of s)

Algorithm: Find a flow with maximum value. 

So basic idea, is start with all 0 flows and try to send flow 1 by 1, increasing it at every step. 

#### Fact
The value of flow out of s = sum of(flow) into the t. 

Proof:

Sum of flows of all edges = sum of flows of all edges

Sum of all vertices, all edges going out of e = Sum of all vertices, all edges going into e

cancel all vertices terms for v not equal to s and t

flow of all edges going out of source = flow of all edges going out of sink

## CUts

A cut X, X_bar is a partition V = X union X_bar

such that: s belongs to X
t belongs to X_bar.

capacity(X, X_bar) = sum of all capacities c_E such that e is from a, b with a belinging to X and b belonging to X_bar.

("sum of capcities of all forward edges")
#### Claim:
The flow value is amount of flow in the forward eges - flow value in backward edges

Let X, X_bar be a cut: X is the source side and X_bar is in sink side.
value(f) = sum over all edges f_e, where e is of type a,b where a is in X and b is in X_bar - sum of flows in backward edges (edge in c . and c, where c is in X_bar and d is in X)
<= sum over all edges f_e, where e is of type a,b where a is in X and b is in X_bar
<= cap(X, X_bar)

Max flow <= min cut capacity. 


##### Now prove that max flow = min cut capacity.

Value(f) = sum of all f(e) out of s + (sum of flow out of v - sum of flow out of v for all v in X) (will be zero so hence we put it int)

 = sum of flow of e forward -  sum of flow of e backward

### Algorithm -  Find paths and send flow along it (and repeat)

If we find a bad oath, we are kind of stuck. If we cant do better and some nodes are unutilized. 

Send flow in reverse, we we can interpret it as decreasing the flow of a valid edge.

### Ramsey Numbers
TODO

### Contention Resolution

Controller,  Database, processes.
Send requets , if exactly 1 prcess gets the access or nobody gets it.
How to design protocol where each process can get request served.
Processes cannot talk to each other.

How will you design protocol ?
If we allow randomisation, we can achieve everything we hope to achieve. 

### Protocol
p = sends request with probability : 1/n.
For each round t. 
Ai,t = event that sends Pi sends req Pr[Ai,t intersec Aj,t]

pr[pi succeds inb round t] = p(1-p)^(n-1)
= 1/n(1-1/n)^(n-1)
= 1/n[1/e,1/2]
T rounds t = 1,2,...T:   T = 2enlogn
Pr[Pi does not suceed even once] <= (1-(1/ne))^T
= ((1-(1/ne))^ne)^2logn
<= (1/e)^2logn
= (1/n^2)

Pr[some process Pi does not suceed even once] <= (union bound) n*(1/n^2) = 1/n


## Expectation

Random varibale is a varibale for a thing we are interested in, assign number o every entity.

Expectation of the random variable is average of a random variable. 
Indicator random varibales -  boolean of a state, whether student is cience major or not types. 
omega = {head,tails}10:
Random varibale can be how many heads did you get.

Given a random varibale, undertsand the distribution of this random variable. 
FOr example, we are interested in how heights are distributed. 
Expecteation is average

Formal Deifnition:
X: Map of omega to real numbers, 
Expectation of X: weighted avergage with he weights being the probabilities of that occurance. 
It is a weighted average of the value of the random variable.

### Note 
If (omega, P) is uniform, E[X] = E(X(w)/|omega|

Problem
: FInd number Expectatiopn of num heads in n tosses of coin
SIgma from 0 to n(pr[X=j]*j
= sigma from o to n(j*(nCj)/2^n = n/2

# Linearity of Expectation
If we have 2 random varibale, and we wantthe average of sum of two random varibnales

Then it is the sum of averages of 2 random variables::=>
E[X+y] = E[x] + E[y]
Expec1+ Expec2

Very simple but very powerful idea. 

Hence use this prob of number if heada

Xi = {1 if ith toss is heads, 0 otherwise}
then E[X] = sum from 1 to n E[Xi] = n*1/2


Fact  If Z is indicator {0,1}- valued
then E[Z] = Pr[Z=1]

# Finding max cut in graphs (randomised algo)

Definition: A cut in a graph G is just a a partition of the vertices V = S union S_

e(S,S_): set of all deges with vertices i and j where i is in set i and j is in set j.

If in a graoh, find me a cut whise size is maximum, size meaning number of edges cut. 

Size = look at edges which are cut, give their number.

Theorum: There exists S, S_ such that number of edges cut is half of the number of edges. n/2. Moreover, we can find this cut in polynomial time with a randomised algorithm. 

Basic Idea- Average should be high but the MAXIMUM shold not be that high. 

# Hashing

Universe of hashing functions;
A hashing function maps a universe of keys to n values.
hash function family = pairwise independendent or 2- universal if : for all x,y in the universe which are distinct for all indices, i and j. if we pick hash function h from H, the probabilt that h(x) = i and h(y) isj =  1/n^2. 

Fix i where 0 < i < n-1

Let L(i) = number of {x belongs to S | h(x) = i}
Claim Expecttatoin L(i) = 1

Let Xa = { 1 if h(a) = i, 0 otherwise }
l(i) = sigma(Xa) for all a belongs to S.
Expectattion L(i) = n*1/n = 1

By Markov: Pr[L(i) >= 50] <= 1/50; pr[X >= t.E[x]] <= 1/t

## Variance
Definition: X is a random varibale. Then it's varinace let U = E[x].
var(x) = E[|X - u|^2]
Variance of x is expected value of x^2 - u^2.

Second moment of random variable = E[x^2]

Proof of Variance of x is expected value of x^2 - u^2.:
Open up soltion and do something so thst renmove scalers. etc. 

Ex: X = {1 p, 0 1-p.   E[x] = p, E[x^2] = p : var(x) = p(1-p) (p - p^2)



Chebychev's Inequality: 

Pr[X >= t.E[X] ] <= E[X^2]/(E[x]^2*t^2)

Proof:
Pr[X >= E[x]] = Pr[ X^2 >= t^2E[X]^2] <= E[x^2] / (t^2*E[x]^2

pr[|X-u| >= T = Pr[X-u|^2 >= T^2] <= var(X)/T^2

Now back to proof of hashing:
= E[L(i)^2] = E[(sigma(Xa)^2 for all a belongs to S]: { Xa = 1 if h(a) = 1 else 0 }
= E[sigmaXa^2] + E[sigma a not equal to b XaXb]
= n*1/n + n(n-1).1/n^2 <= 2

## By Chebychev's Inequality

Pr[L[i] >= t] <= E[Li(i)^2]/t^2 = 2/t^2
Pr[L(i) >= 50] <= 1/1250


## Constructing pairwise indep hash family

Goal: H Pr[h(x) = i intersect h(y) = j] = 1/n^2
Assume for now that \U\ = n, U = {0,1,...n-1}, n prime, h: {0,1,...n-1}-> {0,1,...n-1}

For any two numbers a,b in 0 to n-1 let ha,b(x) = ax + b (mod n);

## NP Completness

You will have one problem on NP completness in the exam. 
P subspace of problems - P stands for polynomial algorithms. 

problem Name = MST; Instance = Graph with V,E and cost in each edge}


NP Problems = 3SAT, Vertex Cover, Hamiltonion Cycle, Subset Sum etc. 
NP is not NOT POLYNOMIAL, fron theoretical view.


NP Problems can be solved using Non Deterministic Turing Machine in polynomial time. 
Can this device exist for real ?


NP = Non Deterministic Polynomial Time. 

4 terms in NP:
- Problem
- instance
- Solution
- Candidate Solution


# Hamiltonian Cycle
instance: G(v,e), |v| = n
Solution: hamiltonian cycle in G(V,E)

Candidate Solution:  Sequence of vertices, V1,V2....Vn.

Checking a solution: for all i between 1 and n are all distinct. 
- V(i, Vi+1) belongs to E, for 1 < i < n-1, (Vn,V1) belongs to E. 

Checking for solution n is not always easy (polynomial)

# Computational Problems, Computing, Algorithms, P, NP

Alphabet Sigma = {0,1}
= {a,z}
= {a,z,A-Z, 0-9,!@#$...}

STring over alphabet = Sigma star = Set of all finite length strings over sigma
= {enumerate all strings in increasing order of length}
Epsilon . is empty string. 

## Language
L is set of string but not set of all strings. 
L is a set of STRINGS. 
L is subset of Sigma star. 

Leven = set of all even length strings. 
L english = Set of all valid words in dictonary. 
and so on...

Computational Problem is basically membership problem of a language !!

For every language L, there is a decision problem Pl

" Given input x belongs to sigma star, is x belonging to L ? "

Note: For every Language L, there is a decision membership problem Pl
"Given input x belongs to Sigma star, is x belongs to L ? "
- Size of input n = abs x. 

Note: every decision problem can be cast/ translated as a membership problem for an apprx language L. 

Eery object can be encoded into a string say for example a graph: make a string of adjacency matrix. Sigma: {0,1,#}
0011#0101:01010 etc. 

More examples of computational problems:

Lgraphs = {<G> G is a undirected graph } ; here diagonal should be zero, Mij = Mji. 

Lbipartite = {<G> | G is a bipartite graph}: Plbipartite :  Given graph, is it bipartite. 
 
L half_clique = {Set of all graphs G, G has 2n vertices and has a clique of size n }.
 
L clique = {<G,k> |  G has clique has size >= K}
 
L3sat = {All formulas phi, such that phi has a satisfying assignment}


# Computational Device

TURING MACHINES !!!!

It solves a computational problem. 


Takes input x outputs yes or no whether x belongs to lang, no if x doesnt belong to lang. 
T(n) = worst case.


## Turing Machine

It has 2 componoents, one is called the input tape.
 A tape is simply an infinte sequence in one direction of cells. Each cell contains a symbol. 
 Given A alphabet sigma = {0, 1}, each cell would contain 0 or 1.
 
 Given input x we write it down as initial prefix in input tape. 
 
 
 There will be a special token, we mention where input has ended.  We use the blank symbol to portray this. 
 
 So we can think of input tape inital version to be input x and infinte series of Blanks. 
 
 gamma = sigma union infinite series of blanks. 
 
 Second compinent in turing machine is control. 
 
 We have Q:  finite set of "states" . S1, s2,...sk. , S start, Saccept, Sreject.
  Machine or control starts from S start. called start state. 
  
 k is independent of n, length of input. The description of machine is of constant size. 
 
 
 Program is a bunch of instructions of following type:
 
Example Of Instruction :=> If current state equals S6. symbol read = '0', then change state to S8, chamge symbol at pointer to '1', move pointer by one place to right. 
 
 
 Set of instructions 
 Gamma: Q x P-> Q x P x {left, right, or stay put}
 
 Gamma is set of symbols pointed by pointer. 
 You can iagena table: we have 10 alphabets, 36 set of states. so we will have 360 gamma. 
 
 
 M = {(Q, sigma, P, gamma, s_start, s_accept, s_reject)}
 
 Execution COmputing
 
 Initial config pointer or contrl is at S_start:
 
 
 AT some point we have to conclude if x is in language or not. 
 
 In T(n): The machine enters accept or reject. 
 
 
 One would want to design a turing machine that for some T(n) on input of length n,  the machine executes at most T
(n) steps and enters either the state Saccept or Sreject. 


Algorithm for problem L that runs in time T(n). 

WHat if machine goes on indefinitely, its the theory of undecidable problems. 

Exercies: Consider Language is palindromes, {all x such that x = reverse x}
Sigma = {0,1}

Claim: n^2 algorithms: 01100100BBB
States incorprate memory:

Time complexity of Turning Machine problem,  will be polynomial to T(n) if T(n) is a computer running time. 


# Deterministic Turing Machine

DTIME = T(n), defines all class of languages that can be solved by a determinitic turing machine in time O(T(n)).
Example: 
Language of palindromes - belongs to DTIME(n^2).

P = union of these languages that are of DTIME(n^k), where k is a constant.

Problems not known in P:

- 3SAT
- Vertex Cover
- TSP

Now we can use Non determinitic turing machines to compute problems not in P quickly. 

Some ideas:
1. Polynomial algos is coinsidered good 
as historically we have lessened n20 to n2. 
- Results in good theory (group theory)
- composition of polynomial algorithms is polynomial. 

# Non Determinitic Turing Machine

Superpower to win the lottery (randomly get good results)

### What is it ?

Look similiar to turing machines, same input x and followed by series of blanks, start state. Pointer.

In deterministic case, at every point, machine has only 1 move.
But in ND, machine can take more than 1 possible choices. Without loss in generality, we assume that we have exactly 2 choices. 
a: QxP -> QxPx{L,R} for determinisitc turing machine
a: QxP -> P(QxPx{L,R})
P means for certain state, we could have 2 or 17 possibkle moves. or any thing. 


At each state, we have a posiibkle next step, form tree of steps, at the leaves we have a accept or rehect

we have 2^(t(n)) leaves, then we can do 

AND/OR(NP), MAJ(PP), XOR(+P)

+P and PP are much greater in number than NP languages. 

Definations : L belong to NTIME(t(n)) if there is a NTM that runs in time t(n) and

x belongs to L => M has at least one accepting computation. 
X doesnt belong to L -> all computations reject. 

NP = Union of NTIME(n^k)

P is a subset of NP. 

every determinitic machine is also a non determinisitc machines, thinkn of determinitic turning machine to aslo have 2 choices, one the right path, second the reject. 

## Fact:

A NTM M, running in time t(n) can w.l.og be assumed to be of type 

M = "on input X, \X\ = n, non det selects y belongs to {0,1}t(n).

Run a det m/c M* on (x,y).


## Example
Problem Lang 3Sat
instance phi .   phi (X1,...Xm)
Candidate soln : y .   {X1, Xm} -> {0 or 1}
Checking / Verification . M* .  acc/rej depending on whether candidate soln satisfies instance.
Correct soln: y : M*(x,y) = acceot : candiate solumn : is satisfying assignment. 

3SAT = {phi | phi has a satisying assignment} (phi is "satisfiable")

3SAT belong sot NP: Following polytime NTM M

M = "Given phi with n vars, non det select assignment phi belongs to {0,1}^n".

Accept if sigma(cand soln) satisfies phi(instancde of problem). Reject otherwise. 


phi beings to 3 sat -> phi does have a sat assighn -> M on choice sigma star accept. 
phi doesnt belong to 3 sat -> phi has no sat assign => M always rejects.

Therefore M accepts 3SAT. 


FOr hamiltonian cycle: Lang : all graohs, G such that g has a hamiltonion cycle. 

The following NDM, M will non determinitically select a sequence of vertices (v1,v2...vn) then accept if this sequence forms a hamiltonion cycle.


Accept if v1 and v2 i an edges, and vn connect back to v1 and they are all distince and number of vertices are at least k. 

Reject otherwise.

Note: Sequence of vertices witll be of size n*log(n) as vertices is numbered form 1 to n and can be denoted as logn. 


# Reduction using Turing Machines:

A,B are languages. A <=p B if 

there is polytime det Turing Machine M that on input x produces output M(x)

Note: Modify turing machine to have output tape, and machine will keep writing to output tape, write symbol to output tape and moves pointer ahead to output tape. 

Lang A reduced to B if there is a reduction that on input x it produces output M(x) such that 

x belong to A <=> M(x) belong to B

Instance of A can be reduced to Instance of B. 
in polynomial time. 

Excercie: A reduced to B, and B reduces to C then A reduces to C. Think about composibility of polynomial algorithms. 

2. A reduced to B, B belongs to P, then A belongs to P. If input of size n of A , goes to n^5 in B, then running time of A will be (n^3)^5


Defination: A language L is N.P complete 
- if L belongs to NP.
- For all A belongs to NP, A reduces to L.

