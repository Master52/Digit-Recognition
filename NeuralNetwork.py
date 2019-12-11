from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot as plt

class NeuralNetwork(object):
    '''
    A Feed Forword Neural Network
    It consist of 3 layers
    1) 1st layer takes input of 784 pixels and produce 512 ouput(or neuorons)
    2) 2nd layer takes input of 512 and produce 512 output
    3) 3rd layer takes input of 512 and produce 256 output

    We have use ReLU activation function and output activation function is softmax

    '''
    def __init__(self,input_shape,output_shape):
        self.model = Sequential()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model.add(Dense(512,activation='relu',input_shape=(self.input_shape,)))
        self.model.add(Dropout(0.5)) # To avoid overfitting
        self.model.add(Dense(512,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256,activation='relu'))
        self.model.add(Dense(self.output_shape,activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self,train,label,batch_size = 256,epochs = 20, verbose = 1):
        self.history = self.model.fit(train,label,batch_size = batch_size,epochs = epochs,
                                verbose = verbose)

    def evaluate(self,test):
        return self.model.predict(test),self.model.predict_classes(test)

