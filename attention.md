
The RNN reduces the language modeling problem to modeling for the probability for the next word given the preceding words.

Conditional MOdels are similiar to the above approach. Just now a variable x is used as the history for predicting the first word.

Hence , x is the condition to the language model.

The applictions of this model are amazing.

Eg: 
Input | Output
Document | Transaltion
Document | Summmary

# Interpretation

BLEU, ROUGE are used to do task specific evaluation.

Cross Entropy and Perxplexity are used to evaluate Language Models.

# Encoder Decoder Architecture

# Encoder

Convolutional Layers, 
Recurrent Layers
Bag Of Words 

There are many ways to encode a input.

Bag Of Words = Fast
Convolution = Dependencies can be learnt by stacking layers. One con is need to get more dynamic for different size sentences.

# Decoder

Need to add condition to every step. So maybe concat condition with hidden state at every step.

# Tricks

Bi-directional. Training for reverse string too. +4 BLEU
Use ensemble of different models. +4 BLEU for 5 models

# Word On Decoding

In general we want to find the most probable output given the input.

This for RNN is a hard problem.

Hence we do a greedy search. We do an approximation with Beam Search. +1 BLEU


# Problem

You can't cram th meaning of a whole @#@$@#%@ sentence in a single vector !

We are compressing a lot of information in a finite sized vector. 
Gradients still have a long way to travel.( Even LSTMs forget !)


# Solution

Represent source as matrix.
Generate target from matrix.

Matrix : 

Fixed number of rows.
Variable number of columns.

Possible Method 1
Make a matrix of input. Run it through some convolutional layers. Attain context dependant input. Don't do pooling.

# BiDirectional RNNs Method

Get one column per word.
Each column has two representations concaternated together. Left and Right. via bidirectional RNNs.

# New Areas To Explore

Try multi phrase expressions instead of one word. 

Convolutions are particularly interesting and underexplored.

# Attention

Columns are words.

Multiply hidden state vector with matrix above.

Do a weighted sum across the rows.

Do a weighting of all the words. Sum them together to get the attention energy vector.

Softmax to exponentiate and noramlise to 1.

Now use this attentioun vector to mulitply dot product with Source matrix along columns now.

THe resultant vecotor needs to be concatednated with input and fed into the network.

Adding Attention = +11 BLEU !

This si soft attention.

Hard attention is Reinforcement Learning.

Early Binding and Late binding
