# Dynamic Coattention Networks For Question Answering

This paper aims to provide a model for QA. It encorporates a novel co attention system between the question and the passage. The basic gist of this paper is that attention is usually calculated for a single entity, or vector.
In this paper , they devised a mechanism to create attention vectors for two vectors/modalities, the passage and the question.



## Author 
Salesforce 

## Paper Link
https://arxiv.org/abs/1611.01604

# Nomenclature

```
@ -> Matrix Multiplication
* -> Hammard's Product
```

## Model

The model contains of 3 parts

# The Encoder

1. Passage and Questions are converted into word vectors. 
2. Glove embeddings used. Link is http://nlp.stanford.edu/data/glove.840B.300d.zip 
3. Encodes the question and passage word vectors into an lstm/gru as Q and P 

## Dimensions

len question = n
len passage = m

For passage 

```
1. if word vector length is 300
2. Input into lstm is passage_len x 300
3. Output of dimension passage_len x hidden_size
```

They add a sentinel vector at the end ,which allows the model not attend over one particular word. SO 
final output size is (passage_len+1) x hidden_size

Let's say output is P

For question

Same as above but the output of lstm is fed into a linear neural layer followed by a tanh activation.
This is done to allow variation b/w "question space and passage space"

Let's say output is Q

hence final dimensions are

```
1. P => (passage_len+1) x hidden_size
2. Q => (question_len+1) x hidden_size
```

# The Co-Attention Encoder


Gets co-fused attention of both question and passage
```
1. Generates affinity matrix . L = P @ Q
2. Generate Attention of question AQ = Softmax(L)   
3. Generate Attention of passage AP = Softmax( transpose(L) ) 
4. Compute attention contexts of question CQ = P@AQ
5. Compute Co-Attention C = [Q ; CQ]@AP
6. Finally after getting this attention , we encode it into a bi lstm , taking P and C as input getting
7. U = BiLSTM([P;C])
```

The bi lstm has two hidden states that are initialised by zero (assumption)
Then they run the lstm through the passage inputs , one by one, taking last hidden state and next hidden state of the bi-lstm. Doubts on how exactly one does that.

So finally we get U which 
``` which provides a foundation for selecting which span may be the best possible answer, as the coattention encoding```

## Dimensions
```
1. L => (passage_len+1) x (question_len + 1)
2. AQ => (passage_len+1) x (question_len + 1) 
3. AP => (question_len+1) x (passage_len + 1)
4. CQ => (hidden_size) x (question_len + 1)
5. Q:CQ => 2*( (question_len+1) x hidden_size) )
6. C => 2*( hidden_size x (passage_len+1) ) 
7. P:C => 3*(hidden_size x passage_len+1)
8. U => 2*(hidden_size, passage_len) as it's a bi lstm
```
# Dynamic Pointer Network

A pointer network predicts points in a document by indexes, for example it predcist what is the starting index and ending index of the answer to a question in the passage. As the SQUAD dataset has answers given in this format,pointer networks are currently one of the best ways to predict that.

```
Sample passage : blah blah blah, tom is in Kenya, blah blah blah
Question: Where is Tome
Answer : Index 31 - Index 34
```

The authors take an iterative approach to predicting start and end pointers, because sometimes there may be multiple answers to a question. And these answers may represent a local minima.
To escape this local minima, prediction is done iteratively first predicting the start index, then the end index, then the start index , then the end index...
Until the start and end index stop changing and/or the pre defined iterations comes to an end.

Think of decoder as a state machine, state is stored in LSTM.
```
So for i iterations

1. hidden state  = LSTM(previous_hidden_state, [U_previous_start; U_previous_end] )
```

alpha and beta are computed to denote the probabilites for start and end states.
The max alpha and the max beta probability denote the index of the passage , the model has predicted.
```
s = argmax(alpha)
e = argmax(beta)
```
Assumption

```
Initial start and end index can be 0 . Need to ask/find more about this
```

The alphas and betas are calculated through a highway maxout network. A high way network is such which has skip connections and max out networks are such that they if we have a output vectos of size m x n. The maxout procedure will just take the max(of each inner tensor)

So output will be of size m, each value being the maximum value in that interior vector of size n.

Two different highway maxout networks are used (HMN) for start and finish to predict alpha and beta.

```
HMN is as follows:
1. R = tanh(Linear([hidden:U_previous_start:U_previous_end]))
2. m1 = max( Linear( [U:R] ) + bias )
3. m2 = max( Linear( m1 ) + bias )
4. HMN = max( Linear( [m1;m2] ) + bias )
```
HMN is run for all U to geenrate alpha of same size as passage_len

Then the current start/end state index is calcualated by arg max

```
1. s_new = argmax(alpha)
2. e_new = argmax(beta)
```

And this process continues until iterations are complete or start/end stop varying.

## Dimensions

Todo







