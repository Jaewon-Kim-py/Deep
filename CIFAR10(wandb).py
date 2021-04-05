#import library
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Activation
import keras as keras
from tensorflow.keras.regularizers import l2

#install wandb
!pip install wandb

#wandb
import wandb
from wandb.keras import WandbCallback

#login
wandb.login

#init
wandb.init(project = 'cifar10')

#parameters
config = wandb.config
config.learning_rate = 0.001
config.batch_size = 32
config.filters1 = 8
config.dropout1 = 0.2
config.filter2 = 16
config.dropout2 = 0.2
config.dense = 64

#load data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

#subclassing
class Convnet(Model):
  def __init__(self):
    super(Convnet, self).__init__()
    self.conv1 = Conv2D(filters = config.filters1, kernel_size =3, padding = 'same', activation ='relu', kernel_regularizer= l2(0.001))
    self.pool1 = MaxPool2D(pool_size =(2,2), strides = 2)
    self.drop1 = Dropout(config.dropout1)
    self.conv2 = Conv2D(filters = config.filter2 , kernel_size =3, padding ='same', activation = 'relu', kernel_regularizer= l2(0.001))
    self.pool2 = MaxPool2D(pool_size = (2,2), strides =2)
    self.drop2 = Dropout(config.dropout2)
    self.flatten = Flatten()
    self.dense1 = Dense(config.dense, activation = 'relu', kernel_regularizer= l2(0.001))
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

opt = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)  
model = Convnet()
model.compile(optimizer= opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  

model.fit(x_train, y_train, epochs=10, batch_size = config.batch_size, validation_data = (x_test, y_test),callbacks=[WandbCallback()])
