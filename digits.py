import numpy as np
import idx2numpy
import cv2 as cv
from Neural import Neural
from sklearn.model_selection import train_test_split

def get_train_label(p1,p2):
	return np.copy(idx2numpy.convert_from_file(p1)),\
		np.copy(idx2numpy.convert_from_file(p2))
	

def convet(train,labels):
    #converting train and label data so that it fits our neural netowork
    # train contatin the value between [0.01,1] for each cell
    train = train*(0.9/255)+0.01 #0.01 is just padding so that output can be 0 or 1

    # for converting label we are using one hot encoding 
    one_hot_label = np.zeros((labels.size,10))
    one_hot_label[np.arange(labels.size),labels] = 1
	#print(one_hot_label)

    #we dont want 0 and one in label
    one_hot_label[one_hot_label == 0] = 0.01
    one_hot_label[one_hot_label == 1] = .099
    return train,one_hot_label


if __name__ == '__main__':
    path = 'train-images-idx3-ubyte'
    path2 = "train-labels-idx1-ubyte"
    train,labels = get_train_label(path,path2)
    train,labels = convet(train,labels)
    x_train,x_test,y_train,y_test=train_test_split(train,labels,test_size=0.2)
    n = Neural(28*28,10)

    for t,l in zip(x_train,y_train):
        n.train(t,l)


   

    #(455, 384)


	
