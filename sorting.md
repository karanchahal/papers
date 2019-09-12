### Find mendian in O(n) time. 

Algo to find median i.e ai such that |{a_i | ai <= a_i0}| = [n/2] in O(n) time.
Given n number, divide into n/5 groups of size 5 each.

- Sort each group. 
Say a1,a11, b1, c1, c11 (sorted)
a2,a21, b2, c2, c21 (sorted)
.....
.....   bn/10
.................
.................
.................


- Find median of medians called B. recursively !

- X = 3n/10, Y = 3n/10 ??
Every element in x < every element in y. 

- If hypothetical mean m satisfies X< m < Y
then m would remain the median of the whole set, and we can drop X and Y
{a1,,,,an} \ XUY.

NOTE: THIS IS FALSE^: Because median could be part of X or Y. 

- Find median of remaining set recursively. 

There is a small error in this, fix it and then you can do this algo. 

Running Time: T(n) <= T(n/5) + T(4n/10) + O(n);

Proove that this is O(n).

Guess and Verify !
Fact T(n) <= T(an) =T(bn) + Cn where a+b < 1

Proof: Suppose T(n) <= alpha*n (this is standard) (alpha will be set in hindsight)

T(n) <= alpha*a*n + alpha*b*n +C*n <= alpha*n

holds if alpha = c/(1-(a+b))


We get a greater assumption: algo find kth smallest element in sorted order. 

_____________________________________________________________________________
Counting Inversions:

a1....aN
Inversion is a pair i, j such tha i < j but A[i] > A[j]. 

Count # inversions. in ) O(nlogn) time
Applications ? -- How sorted the sequence is. 

Divide and Conquer Strategy - given 2n numbers. 
partion it as A: a1..aN and B: b1...bN

Basically merge sort:
- sove recursively A
for each swap, count 1 inversion. 
- solve recusively on B
for each swap count 1 inversion.

Count num of cross inversions (in merge list procedure) O(n) time.
|{(i, j)| a_i > b_j}|





