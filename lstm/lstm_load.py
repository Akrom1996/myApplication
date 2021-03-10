import pandas as pd
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset = pd.read_csv('dataset.csv')
#dataset = dataset[dataset['city_name'] == 'Valencia']
import time
import datetime
dataset['dates'] = pd.to_datetime(dataset['dates']).astype(int) // 10 **9                                                                                                    #dataset[time.mktime(datetime.datetime.strptime(dataset['dates'], "%Y-%m-%d"))]
print(dataset.head())
X = dataset.loc[:,['dates', 'amount']].values
y = dataset.loc[:, ['amount']].values

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(X)
# y_scaled = scaler.fit_transform(y)
# plt.plot(X[:,-1], c= 'blue', linewidth=0.9, linestyle = '-', label = 'Real value')
# plt.show()
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
plt.plot(train[:,-1], c= 'blue', linewidth=0.9, linestyle = '-')
plt.plot(test[:,-1], c= 'orange', linewidth=0.9, linestyle = '--')
print(train)
print(test)
print("train_data_size: "+str(len(train)), " test_data_size: "+str(len(test)))

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
	    a = dataset[i:(i+time_step), 0]
	    dataX.append(a)
	    dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
time_step = 30
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)

plt.plot(X_train[:,-1], c= 'red', linewidth=0.9, linestyle = '-')
plt.plot(y_train, linewidth=0.9, linestyle='-.')


plt.show()
# reshape input to be [samples, time steps, features]
X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


lstm = load_model('lstm.h5')

testPredict = lstm.predict(X_test)
trainPredict = lstm.predict(X_train)



inversed_test_pred = scaler.inverse_transform(testPredict)
inversed_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


testScore = math.sqrt(mean_squared_error(y_test[:], testPredict[:,-1]))
mae = mean_absolute_error(y_test[:], testPredict[:,-1])
print('Test Score: %.2f RMSE' % (testScore))
print('Test Score: %.2f MAE' % (mae))

mae = mean_absolute_error(y_train[:], trainPredict[:, -1])
trainScore = math.sqrt(mean_squared_error(y_train[:], trainPredict[:,-1]))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f MAE' % (mae))

# plt.plot(inversed_test_pred[:], c = 'orange', linewidth=0.9, linestyle = '-', label='Predicted value')
# plt.plot(inversed_y_test[:,-1], c= 'blue', linewidth=0.9, linestyle = '-', label = 'Real value')
# plt.legend()
# plt.show()