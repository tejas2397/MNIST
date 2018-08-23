#Importing the required libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#print(x_test.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train=x_train.reshape(60000,28,28,1)#to remove the error regarding the input shape due to the data lacking channel dimension

clf=Sequential()
#First convolution layer
clf.add(Convolution2D(32,3,3,input_shape=(28,28,1),activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))

#second convolution layer
clf.add(Convolution2D(32,3,3,activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
clf.add(Flatten())

#ANN
clf.add(Dense(output_dim=512,activation='relu'))
clf.add(Dense(output_dim=10,activation='softmax'))

#Compile
clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
x_test=x_test.reshape(10000,28,28,1)#to remove the error regarding the input shape due to the data lacking channel dimension

#to remove the error regarding target array expecting to have shape(10,) but gets (1,)
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

#fitting
clf.fit(x_train, y_train,batch_size=128,epochs=10,verbose=2,validation_data=(x_test, y_test))


