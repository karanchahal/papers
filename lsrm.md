## LSTM

Intuition for the LSTM.

Input gate: Allows how much influence we want from the current input

Forget Gate: Toggles by how much we want to forget the previous computation

Output gate: The gate that decides what we want to output to the hidden state.

# Scaling for Large Vocabularies


Negative Sampling or factorisation

Much of the computational cost of a neural LM is a function of size of the vocabulary and is dominated by calculating:

probability = softmax(Wh + b)

activation of neuron is always positive

so what sign will the graidents be.
a = input to neuron
d = sigmoid(aw)
we need to find error of d.

partial differenatisation.
you have a loss coming in called z.

s gradient/error of d is 

m = x*y

d(m)/dx = y*z;

d(m)/dy = x*z;

gradient for x d(x*y)/dy


gradient for y d(x*y)/dx

d(z)/d(d)