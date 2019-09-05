# Data Structures & Algo

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
