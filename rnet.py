# R-NET: Machine Reading Comprehension with Self-matching Networks

This paper aims to provide a model for QA. It encorporates a self matching attention system of the passage with itself to capture outlier infomation.

## Author 
Microsoft 

## Paper Link
https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

# Nomenclature

```
@ -> Matrix Multiplication
* -> Hammard's Product
```
# Introduction

The R-Net was the state of the art most recently on the Squad leaderboard. It is a model that incorporates various lstm's each with its attention mechanism.
A lot of vector multiplications and a lot of linear layers. The intuition for this paper is that first the question and the passage is encoded. 
Second the passage undergos a tailered on the question. Think of it like the passage read with the question in mind. After this representation, the passage is matched with itself. Think of this step , as skimming the pasage again to make sure , we didnt miss anything.
Lastly with this latest self matched vector, a two step lstm predicts the start index and the end index of the answer in the passage. 

# Model

The model contains of 4 parts. Given as below.

## The Encoder
0. First GRU inilialise hidden state to zeros
1. Passage and Question indexes are run through an embedding layer , or through pretrained Glove word vectors.
2. Then embedded Passage and Question is encoded by a  bidrectional GRU consisting of 3 layers , let's call them u_q and u_p.

### Dimensions

1. P => (1,passage_len)
2. Q => (1,question_len)
3. Passage, Question after embedding are of size (1,passage_len/question_len, 75)
4. Hidden State of GRU is H => (6,1, 75) [6 as GRU is biredectional, hence 3*2 = 6 ]

## The Gated Attention Recurrent Network

Aim is to build question aware passage representation. 
1. For all passage words, these steps apply
 1. Get last hidden layer of gated recurrent network, H_G
 2. Get h, p and q by running H_G , Passage word and Question through linear layers. (Matrix Multiplication)
 3. a = sigmoid(tanh(h + p + q))
 4. C = sum(a*q) on the last dimension
 5. final_input = concat(passage_word, C)
 6. G_I  => sigmoid(linear_layer(final_input)) ( Gated input (Hence the name gated recurrent networks) )
 7. gated_input => final_input*G_I
 8. hidden_layer = GRU(gated_input)
 9. return hidden_layer
2. Concat all hidden layers at each passage word time passage into v


### Dimensions
0. passage_word, question => (1,150), (1,13,150)
1. H_G = 1,150 of last layer 
2. h,p,q => (1,75)
3. a => (1, 13, 75)
4. C => (1, 75)
5. final_input => (1,225)
6. gated_input => (1,225)
7. hidden_layer => (1,150) (concatenate bidirectional hidden layers)
8. v => (1, passage_len, 150)

## Self Matching Networks

## Pointer Networks
