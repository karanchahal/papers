# Log Loss Explained

We applied "log loss" and trained until convergence

I have seen this line in countless papers and never hasa paper AFAIK explained what a log loss is. Well maybe it's easily googleable.
But I never really understood how it really worked until now.

I like to think of it as a loss that has two terms.

Assume p to be the predicted probability. 

## First Term 
p

## Second Term

1 - p

## Explanation

p denotes the probability of the object is that object is really 1 . 
1-p denotes the probability of the object if that object wants to be 0.

Hence think of p and 1-p as the probabilities of the object wanting to be what is really is.

Now the log loss term comes from the fact that these probabilities enclosed by a log.

Hence a = log(p) and b = log(1-p)

Now , if you look at the graph of a log, it is negative for values between 0 and 1. It skews to -infinity at 0 and arrives at 0 at 1.


Hence, we want a high loss, if probability of an object wanting to be what is really is , is low.

Take a minute, digest this statement.

```
we want a high loss, if probability of an object wanting to be what is really is , is low.
```

Hence to get a high loss , we need to flip the negative sign.

1. a = -a
2. b = -b

Now , we have these 2 losses, what we finally need is a way to weight these losses.

It s a simple methodology, works with simply toggling them.
Let y = actual probability ( 1 or 0 )

log_loss = y*a + (1-y)*b

or 

log_loss = -y*log(p) - (1-y)*log(1-p)

TADA !!


