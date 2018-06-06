# Word Level Semantics Notes

Word Vectors -> repreentation of words as a vector of numbers.

These word vectors once correctly learnt, exhibit very useful properties.

Some properties observed are that similiar words are found to be clustered together.

Also basic arithmatic functions like addition and subtraction lead us to right answers.

For example king - man reveals a vector that is very similiar to queen.

Word Vectors were used to represent words as creating a one hot vector of the entire vocabuluary is very memory intensive.


# Old Techniques

Classical IR used to compute the most similiar document for a list of query word vectors.


# Motto of Word Vectors

You shall know a word by the company it keeps.

There are three main approches to calculate word vectors.

1. Count Based Methods
2. Predictive Methods
3. Task Based Methods

## Count Based Methods

1. Define a basis vocabulary C of *context words*.

2. Define window size of w. A limit of the _company_ the word keeps.

3. Count the words belonging to C occuring in the window w around the target word.

4. Form a vector representation of the target word based on these counts.

5. A vector is formed for each word , the vector being the size of the C. With 1 marking the context words, everything else being zero.

6. Similiarity between words can be calculated by using cosine distance.

7. Cosine is a norm invariant metric.

### Disadvantages

Need to distinguish counts which are high, than others that are just independently frequent. Techniques like TF-IDF are used to solve this.

Count Based Methods get good results using a neural embedding model.

The task is to generate a embedding matrix E. 

The embedding matrix outputs a word vector for a word.

## Neural Embedding Model

General idea to create this model is as follows.

1. Get instances of words w and get their corresponding context words.
c(ti), where ti is the instance of word t . 

2. Define some kind of scoring function score(ti,c(ti))
with upper bound on output. Eg output can't be more than 1.

3. Loss is defined. 

4. We obviously take the E that outputs the minimum amount of loss.

5. Use E as your embedding matrix.

### Scoring Function

1. It is very important. Should output how well context words ft the target word or vice versa. 

2. It embeds a word with Embedding Matrix.

3. word = context words and vica versa more than any other word.

4. Produces a loss which is differentiable, so model can be learnt.


## Popular Models

### C&W Model ( NLP From Scratch )

1. Embed context words.

2. Undergo shallow convolution over these embeddings to generate one vector.

3. Undergo MLP on this one vector

4. Output scalar score.

Score should be high if these context words are sampled and low if some other context words are sampled. 

Hence we use a hinge loss.

Hinge loss has two terms , one good one bad.

Te good one comes by getting score by taking training input. The bad one comes from getting score from corrupted sentence.

Hence ,

L = max(0, 1 - (good - bad))
Loss will be low if score is low for bad and high for good.

Using these ideas, a new Model was introduced.

## Word2Vec Bag Of Words Model

1. Similiar to previous model, instead of convolution. Word embeddings are simply added.

2. A MLP is used with a softmax upon it.

3. THe output is a softmax over all the words in the vocabulary.

4. Loss used is Negative Log Liklihood.

5. After model is trained. Use the Embedding Matrix to get word vectors.

## Word2Vec Skip Gram Model

Exactly the same as above model, but target and input are reversed.

Hence , word predicts context words.

This is faster, but its a tradeoff between effeciency and a more structured representation.

# Some thoughts on count based methods

It is theoretically known that count based and objective based models have the same general idea.


# Benefits

1. Easy To Learn

2. Highly Parallel Problem.

3. Can get other dependencies POS tags , harder to do with count based methods.

4. Can use images and speech too into its reasoning.


## Task Based Embedding Models

Embedding Matrix can be learnt from scratch based on the task being learnt.

Hence contributing to a better accuracy.

Bag Of Words has shallow semantics as the words are just added up.

More words , more noise.

Simple objectives can yield better grounding of word representations.

## Bilingual Features

The way of getting representaion of two sentences in two different languages.

We have 2 different embedding matrices. And we want the similiar sentences to have a low loss with each other.

Simple way of constructing representation would be bag of words or to enforce some kind of word order like below: 

```
lang1 = SIGMA(tanH(xi + xj))
```
where xi and xj are words one after another.

The loss function would be 
 a hinge loss, with a distrator sentence.

This task based model learns better word embedding models as it says that the aligned sentences share high level meaning, so embeddings should reflect a high level meaning in order to minimise loss.

Bag of words models don't learn such a grounded representation.

Hence , thus this wraps up our exploration of word vector models.
