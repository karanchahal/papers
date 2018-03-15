# Dynamic Coattention Networks For Question Answering

This paper aims to provide a model for QA. It encorporates a novel co attention system between the question and the passage. 

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


