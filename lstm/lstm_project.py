import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset = pd.read_csv('dataset.csv')
#dataset = dataset[dataset['city_name'] == 'Valencia']
import time
import datetime
dataset['dates'] = pd.to_datetime(dataset['dates']).astype(int) // 10 **9                                                                                                    #dataset[time.mktime(datetime.datetime.strptime(dataset['dates'], "%Y-%m-%d"))]
print(dataset.head())
X = dataset.loc[:,['dates', 'amount']]
y = dataset.loc[:, ['amount']].values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print("train_data_size: "+str(len(train)), " test_data_size: "+str(len(test)))

# # convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
	    a = dataset[i:(i+time_step), 0]
	    dataX.append(a)
	    dataY.append(dataset[i + time_step, 0])
    print("dataX=", dataX)
    print("dataY=",dataY)
    return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
time_step = 30
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)
# plt.plot(X_train, linewidth=0.9, linestyle='-')
# plt.plot(X_test, linewidth=0.9, linestyle='--')
# plt.plot(y_train, linewidth=0.9, linestyle='-.')
# plt.plot(y_test, linewidth=0.9,linestyle=':')
# plt.show()
# reshape input to be [samples, time steps, features]
X_train = numpy.reshape(X_train, (X_train.shape[0],  X_train.shape[1], 1))
X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# create and fit the LSTM network
lstm = Sequential()
lstm.add(LSTM(units=64, input_shape = (X_train.shape[1], 1))) #input_dim=time_step
lstm.add(Dropout(0.2))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
lstm.fit(X_train, y_train, epochs=20, batch_size=20)

lstm.save("lstm.h5")
print("Model has been saved successfully!!")

