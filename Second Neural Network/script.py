# 2 layer neural network
# Gebaut parallel zum Stream von Siraj "How to Make a Neural Network (LIVE)"

import numpy as np
import time

# Variables
n_hidden = 10
n_in = 10
n_out = 10 # Outputs
n_samples = 300 # Ssample data

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
    
    # Predict our loss
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)
    
    # Cross-entropy-function
    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))
    
    return loss, (dV, dW, Ev, Ew)


def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A),W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Create layers
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))


bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

# Generate our data
X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1

# TRAINING TIME
for epoch in range(100):
    err = []
    upd = [0]*len(params)
    
    t0 = time.clock()
    # For each data point, update our weights
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)
        # Update loss
        for j in range(len(params)):
            params[j] -= upd[j]
            
        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]
        
        err.append(loss)
        
    print('Epoch: %d, Loss: %.8f, Time: %.4fs'%(epoch, np.mean(err), 
                                              time.clock() - t0))
    

# Try to predict something
x = np.random.binomial(1, 0.5, n_in)
print('XOR prediction')
print(x)
print(predict(x, *params))
