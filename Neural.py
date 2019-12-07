# Nerual Netwok class
import numpy as np


#TODO
# How to implemnet backword propogation

def activation_function(Z):
       return 1 / (1 + np.exp(-Z)) #sigmoid formula


def sigmoid_derivative(Z):
    return (np.exp(-Z))/((1+np.exp(-Z))**2)  #sigmoid derivative formula


class Neural(object):

	# Testing with only on hidden layer hidden layer
    def __init__(self, input_size, output_size):
        self.__input_size = input_size
        self.__output_size = output_size
        self.__bias = 0.01  # or learing rate
        self.w1 = self.__initweight(int(self.__input_size/3), self.__input_size)
        self.w2 = self.__initweight(10, int(self.__input_size/3))

    def __initweight(self, output_size, input_size) :
    # Inital weighting = np.random.randn(output layer,input layer) * np.sqrt(2/100) (ReLU) or Rectifier
        return np.random.randn(output_size, input_size) * np.sqrt(2/input_size)

    def __feedforword(self, train):
        self.output_vect_1 = np.dot(train, self.w1)
        self.hidden_layer1 = activation_function(self.output_vect_1)
        self.output_hidden = np.dot(self.hidden_layer1,self.w2)
        output = activation_function(self.output_hidden)
        return output

    def train(self, train, target):
        train = np.array(train, ndmin=2).T #Testing but not working
        target = np.array(target, ndmin=2).T #Testing but not working
        output = self.__feedforword(train)
        self.__backwordpropogation(train, target, output)

    def __loss_function(self, target, output):
        return target-output

    def __backwordpropogation(self,train,target,output): 
	# Wrong implementation figuring out how to do it right ;(
	#
	#

        output_error = target-output
        dw1 = output_error * output \
              * (1.0 - output) 
        print(dw1.shape)
        print(self.hidden_layer1.shape)
        dw1 = self.__bias  * np.dot(dw1,self.hidden_layer1)
        exit(0) #Debugging

    def evaluate(self, test, label):
        correct, wrong = 0, 0
        for i in range(len(test)):
            res = self.__feedforword(test[i])
            res_max = res.argmax() #givies the index from the 10 output
            if res_max == label[i]:
                correct += 1
            else:
                wrongs += 1
        return corrects,wrong
