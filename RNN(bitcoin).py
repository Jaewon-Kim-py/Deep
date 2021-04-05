#import library
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import SimpleRNN, LSTM, GRU

#setting
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 50
pd.options.display.max_columns = 40
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

# Data Loading
location = 'C:/Users/김재원/time/Cryptocurrency/Bitcoin.csv'
raw_all = pd.read_csv(location, index_col='Date')
raw_all.index = pd.to_datetime(raw_all.index)

# Parameters


sequence = 60
batch_size = 32
epoch = 10
verbose = 1
dropout_ratio = 0

# Feature Engineering
## Train & Test Split
criteria = '2020-01-01'
train = raw_all.loc[raw_all.index < criteria,:]
test = raw_all.loc[raw_all.index >= criteria,:]
print('Train_size:', train.shape, 'Test_size:', test.shape)

## Scaling
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

## X / Y Split
X_train, Y_train = [], []
for index in range(len(train_scaled) - sequence):
    X_train.append(train_scaled[index: index + sequence])
    Y_train.append(train_scaled[index + sequence])
X_test, Y_test = [], []
for index in range(len(test_scaled) - sequence):
    X_test.append(test_scaled[index: index + sequence])
    Y_test.append(test_scaled[index + sequence])

## Retype and Reshape
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)
print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)

# for MLP model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
print('Reshaping for MLP')
print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)

#MLP
import tensorflow as tf
from tensorflow import keras

class time(Model):
    def __init__(self):
        super(time, self).__init__()

        self.first = Dense(128, activation = 'relu')
        self.drop1 = Dropout(rate = 0.3)
        self.second = Dense(256, activation = 'relu')
        self.drop2 = Dropout(rate = 0.2)
        self.last = Dense(1)

    def call(self, inputs):
        x = self.first(inputs)
        x = self.drop1(x)
        x = self.second(x)
        x = self.drop2(x)
        x = self.last(x)

        return x

model = time()
model.compile(optimizer='adam', loss='mean_squared_error')
temp_inputs = keras.Input(shape=X_train.shape[1],)
model(temp_inputs)
model.summary()
model_subclassing = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 100, verbose =1)

#RNN
class rnn(Model):
    def __init__(self):
        super(rnn, self).__init__()
        
        self.first = SimpleRNN(128, return_sequences =True,activation = 'relu')
        self.drop1 = Dropout(rate = 0.3)
        self.second = SimpleRNN(256, return_sequences = True, activation = 'relu')
        self.drop2 = Dropout(rate = 0.3)
        self.third = SimpleRNN(128, return_sequences = True, activation = 'relu')
        self.drop3 = Dropout(rate = 0.3)
        self.fourth = SimpleRNN(64, return_sequences = True, activation = 'relu')
        self.drop4 = Dropout(rate = 0.3)
        self.flatten = Flatten()
        self.last = Dense(1)
        
    def call(self, x):
        x = self.first(x)
        x = self.drop1(x)
        x = self.second(x)
        x = self.drop2(x)
        x = self.third(x)
        x = self.drop3(x)
        x = self.fourth(x)
        x = self.drop4(x)
        x = self.flatten(x)
        x = self.last(x)
        
        return x
model = rnn()
model.compile(optimizer='adam', loss='mean_squared_error')
model_fit = model.fit(X_train, Y_train, 
                      batch_size=batch_size, epochs=epoch,
                      verbose=verbose)
