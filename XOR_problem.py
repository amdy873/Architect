#XOR problem solver using a neural network

import sys
sys.path.append("lib\architect")


from lib.architect.Neural_Network import Network
from lib.architect.dense import *
from lib.architect.activation_functions import *
import numpy as np


X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
Y = np.reshape([[0],[1],[1],[0]], (4,1,1))

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
    ]

epochs = 20000
learning_rate = 0.1

brain = Network(network)
brain.train(epochs,learning_rate,X,Y, False)


while(True):
    x = [[int(input("First value: "))]]
    x.append([int(input("Second value: "))])
    X = np.reshape(x,(1,2,1))
    

    guess = brain.test(X[0],True)

    if(guess > 0.5):
        print("True")
    if(guess <= 0.5):
        print("False")


