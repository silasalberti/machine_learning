# 2 layer neural network
# Gebaut parallel zum Stream von Siraj "How to Make a Neural Network (LIVE)"

import numpy as np
import time

# Variables
n_hidden = 10
n_in = 10
n_out = 10 # Outputs
n_sample = 300 # Ssample data

# Hyperparameters
learning_rate = 0.01
momentum = 0.9

# So that we get the same random numbers every time for debugging purposes
np.random.seed(0)

# Activation functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of tanh(x)
def tanh_prime(x):
    return 1 - np.tanh(x)**2


# Training function
# x - Training Data
# t - Transpose
# V - Layer 1
# W - Layer 2
# bv - bias for Layer 1
# bw - bias for Layer 2
def train(x, t, V, W, bv, bw):
    
    # Forward propagration -- matrix multiplication + biases
    A = np.dot(x,V) + bv
    Z = np.tanh(A)
    
    B = np.dot(Z, W) + bw
    Y = sigmoid(B)
    
    # Backward propagration
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)