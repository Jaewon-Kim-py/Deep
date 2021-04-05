#import library
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import keras as keras

#load data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

#build model
class Convnet(Model):
  def __init__(self):
    super(Convnet, self).__init__()
    self.conv1 = Conv2D(filters = 8, kernel_size =3, padding = 'same', activation ='relu')
    self.pool1 = MaxPool2D(pool_size =(2,2), strides = 2)
    self.conv2 = Conv2D(filters = 16, kernel_size =3, padding ='same', activation = 'relu')
    self.pool2 = MaxPool2D(pool_size = (2,2), strides =2)
    self.flatten = Flatten()
    self.dense = Dense(10, activation = 'softmax')
  
  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.dense(x)

    return x

#define & compile
model = Convnet()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#test
temp_inputs = keras.Input(shape=(32,32,3))

model(temp_inputs)

#fit & evaluate
model.fit(x_train, y_train, epochs=5, batch_size = 32, validation_data = (x_test, y_test))

model.evaluate(x_test, y_test)
