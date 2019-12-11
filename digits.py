import cv2 
import numpy as np
from NeuralNetwork import NeuralNetwork as CNN


x,y = 200,120 #Starting coordinates of rectangle
w,h = 500,400 #Widht and Hegiht of recatngle

def reshape(arr):
    arr = arr.reshape(arr.shape[0],28*28) #reshaping size to (1,28*28)
    arr = arr.astype('float32')
    #converting to 1 and 0
    arr /= 255
    return arr

#one_hot_encoding
def one_hot_encoding(label):
    n_values = np.max(label) + 1
    return np.eye(n_values)[label] 

def convert(train,label):
    arr = reshape(train)
    label = one_hot_encoding(label)
    return arr,label

#imporiting dataset from ubyte file
def get_dataset():
    import idx2numpy
    files = ["train-images-idx3-ubyte","train-labels-idx1-ubyte"]
    trains = idx2numpy.convert_from_file(files[0])
    labels = idx2numpy.convert_from_file(files[1])
    train,labels = convert(trains,labels)
    return train,labels

def get_boundry(img):
    boundry = cv2.cvtColor(img[y:h,x:w],cv2.COLOR_BGR2GRAY)
    boundry = cv2.resize(boundry,(28,28)) # so that it would fit our 28*28 picture

    boundry = cv2.GaussianBlur(boundry, (5, 5), 0)#Reducing the noise

    #converting each pixel to either 0(black) or 1(white)
    boundry = cv2.adaptiveThreshold(boundry,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,11,5)
    return boundry


if __name__ == "__main__":
    
    #trainning our neural network with MNIST digit dataset
    train,labels = get_dataset()
    cnn = CNN(train.shape[1],len(labels[0]))
    cnn.fit(train,labels,epochs = 20)

    font = cv2.FONT_HERSHEY_COMPLEX
    cap = cv2.VideoCapture(0)
    _,frame = cap.read()

    while(True):
        _,img = cap.read()
        boundry = get_boundry(img) #getting only rectangle path
        cv2.imshow('critical',boundry)
        boundry = boundry.reshape(1,train.shape[1]) #making it proper for CNN input
        prob,res = cnn.evaluate(boundry)
        img = cv2.rectangle(img,(x,y),(w,h),255,2)
        text =   str(res[0]) +"   " +str(np.amax(prob) * 100) + "%"
        img = cv2.putText(img,text,(x,y),font,1,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow('original',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

