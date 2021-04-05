#import library
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation
import keras as keras
from tensorflow.keras.regularizers import l2

#load data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

#CNN with Regularization, Dropout
class Convnet(Model):
  def __init__(self):
    super(Convnet, self).__init__()
    self.conv1 = Conv2D(filters = 8, kernel_size =3, padding = 'same', activation ='relu', kernel_regularizer= l2(0.001))
    self.pool1 = MaxPool2D(pool_size =(2,2), strides = 2)
    self.drop1 = Dropout(0.2)
    self.conv2 = Conv2D(filters = 16, kernel_size =3, padding ='same', activation = 'relu', kernel_regularizer= l2(0.001))
    self.pool2 = MaxPool2D(pool_size = (2,2), strides =2)
    self.drop2 = Dropout(0.2)
    self.flatten = Flatten()
    self.dense1 = Dense(64, activation = 'relu', kernel_regularizer= l2(0.001))
    self.dense2 = Dense(10, activation = 'softmax')
  
  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.drop1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.drop2(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)

    return x
#CNN with BatchNormalization(conv- batch_normalization - activation 으로 구성)
class Convnet(Model):
  def __init__(self):
    super(Convnet, self).__init__()
    self.conv1 = Conv2D(filters = 8, kernel_size =3, padding = 'same')
    self.batch1 = BatchNormalization()
    self.activation1 = Activation('relu')
    self.pool1 = MaxPool2D(pool_size =(2,2), strides = 2)
    self.conv2 = Conv2D(filters = 16, kernel_size =3, padding ='same')
    self.batch2 = BatchNormalization()
    self.activation2 = Activation('relu')
    self.pool2 = MaxPool2D(pool_size =(2,2), strides = 2)
    self.flatten = Flatten()
    self.dense1 = Dense(64)
    self.batch3 = BatchNormalization()
    self.dense2 = Dense(10, activation = 'softmax')
  
  def call(self, x):
    x = self.conv1(x)
    x = self.batch1(x)
    x = self.activation1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.batch2(x)
    x = self.activation2(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.batch3(x)
    x = self.dense2(x)

    return x
  
#train and evaluation
model = Convnet()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size = 32, validation_data = (x_test, y_test))
model.evaluate(x_test, y_test)
