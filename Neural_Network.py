#Neural Network
from dense import *
from activation import *
from activation_functions import *
from loss_functions import *
import numpy as np

class Network():
    #loss functions supported: MSE
    def __init__(self, network, loss_function):
        self.network = network
        self.loss_function = loss_function
        

    def train(self, epochs, learning_rate, X, Y, show_output=True):
        for e in range(epochs):

            error = 0

            #feed forward algorithm
            for x, y in zip(X,Y):
                
                output = self.test(x)

                #error calculation
                error += self.error(y,output)

                #backward
                grad = self.error_prime(y,output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad,learning_rate)

            
            error /= len(X)
            if(show_output):
                print('%d/%d, error=%f' % (e+1, epochs, error))

    def test(self, x, show_output=False):
        
        output = x        
        for layer in self.network:
            output = layer.forward(output)

        if(show_output):
            print(output)

        return output


    def error(self,y,output):
        if(self.loss_function == "mse"):
            return mse(y,output)

    def error_prime(self,y,output):
        if(self.loss_function == "mse"):
            return mse_prime(y,output)
