#MNIST data set solver
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from Neural_Network import Network
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activation_functions import Sigmoid
from loss_functions import *

def preprocess_data(x,y,limit):
    z_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    all_indices = np.hstack((z_index, one_index, two_index))
    all_indices = np.random.permutation(all_indices)
    #x, y = x[:], y[:]
    x = x.reshape(len(x), 1, 28, 28)
    #normalizes the input
    x = x.astype("float32")/255
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)


network = [
    Convolutional((1,28,28),3,5),
    Sigmoid(),
    Reshape((5,26,26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
    ]

#epochs and learning rate
epochs = int(input("Number of epochs to train: "))
learning_rate = 0.1

print("Training")
brain = Network(network, "bce")
brain.train(epochs, learning_rate, x_train, y_train, False)

print("Testing")
for i in range(len(x_test)):
    output = brain.test(x_test[i]) 
    index = output.argmax(0)
    print("Model's guess", index[0])
    print("True value: ", y_test[i].argmax(0)[0])
    input("Press Enter to Continue")

