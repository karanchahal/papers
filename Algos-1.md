# Hons Algorithms Course 
#### Lecture 1 by Prof Subhash Khot

## Introduction
Office 416
Room 101 ?

#### Exam
6 questions, 4 hours. 
6th question - about NP completeness.

#### Hints
Struggle with the problem.

## Intro

Algorithms- Systematic method to solve a computational problem.

Turing Machine- Captures essence of a real computer. Makes precise the notion of 
problem, computing, solving, algorithm. 

- Makes one ask can every problem be solved ?- no

- Even if we have infinite amount of time, some problems can't be solved. 

-eg: Polynomial question x2 + y2 -xy +3 = 5 , does it have an int solution for x any y ?
Ans: Turns out no. No algorithm. This problem is undecidable. 

- How fast can we solve it ? Computational complexity.

- This is the thoery of Comp Sci.
------------------------------------------------------------------------------

Basic Algo Techniques
1. Divide & Conquer
2. Greedy
3. Dynamic
4. Amortized Analysis 
------------------------------------------------------------------------------

Specific / Advanced ALgos
1. Shortest Paths
2. Max Flows
3. Randomized Algos (allow for some possibility of error)
4. Couple more if time allows. 
5. NP Completeness (define turning machine and formalise notions of what an algo is)

------------------------------------------------------------------------------
Pre Reqs: 
Practice proofs. 
Prove upper bound on running time. 
Proof by induction, contradiction, basic math. 



Exam Date: Thursday, December 19th 10-2. 
H/W - Deadlines. (1st due in 2 weeks)
------------------------------------------------------------------------------
## Asymptotic Running Time

- Problem vs INstance size.
eg: Problem- Sorting
Instance- 1,5,4,3,5
Size - 5 (denoted by n)

- How large is n ? 
Ans- It should be unbounded, as large as you want. We dont care about small n (not aqpplicable in real world so much) 

- "Algo should work for all n"

T(n) - Running Time. 
Running time measures number of "steps".

- "constants are not imp" theoretically. Makes theory nicer. 
Makes theory nicer, asymptotic, "turing m/c step" is not defined.
 
- if we start worrying about constants, we need to define what a step is.

------------------------------------------------------------------------------
Big O notation.
T(n) = running time of algo, on input of size n.
f(n) = n, n^2, nlogn (by default base 2 but doesnt really matter).

Defn: T(n) = O(f(n)) if there exists a constant C> 0 and integer n0 st n >= n0
T(n) <= C.f(n)

Functoin "grows" more rapidly than other.

------------------------------------------------------------------------------

EXERCISES
Proof of:
n^3 is asympotoitically small than O(2^n)
n^50 is asympotoitically small than O(1.01^n)

Exponential functions explode.
------------------------------------------------------------------------------


n^2 + 5n + 100 <= n^2 + 5n^2 + 100n^2 = 106n^2.

Theorum:
There is a sorting algorithm that runs in time O(nlogn). (Comparision Based Sorting)

Omega Notation:

T(n) = Omega(f(n)) if T(n) >= c.f(n). (get a Lower Bound)
FOR c >0, n , such that for all n >= n0


You can prove lower bounds:

Example: An algo for C.B sorting must take Omega(nLogn) time. 
------------------------------------------------------------------------------
Theta Notation

T(n) = Theta(f(n)) if both T(n) = O(f(n)) and Omega(f(n)) hold true.

Rapid increase to the right.
logn << log2n << sqrt(n) << n<< n^2 << 1.01^n<< 2^n<< 3^n <<....n! (or 2^(nlogn)) << 2^(2^n)

------------------------------------------------------------------------------
