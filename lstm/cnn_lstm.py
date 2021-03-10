FILE_NAME = 'export_dataframe_nodeA.csv'

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#load_ext tensorboard
#print(tf.version.VERSION)


# split dataset 80% train, 15% validation, 5% dev
def split_dataset(dataset):
    size = dataset.shape[0]
    train_size = size * 80 // 100
    test_size = size * 15 // 100
    
    return dataset[0:train_size, :], dataset[train_size:(train_size + test_size), :], dataset[(train_size + test_size):size, :]

def plot_series(time, series, lab, form='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], form, label=lab)
    plt.xlabel("Minute")
    plt.ylabel("Price")
    plt.grid(True)

def tf_dataset(series_x, series_y, batch_size, shuffle_buffer, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((series_x, series_y))
    if shuffle:
        ds = ds.cache().shuffle(shuffle_buffer).batch(batch_size).repeat()
    else:
        ds = ds.cache().batch(batch_size).repeat()

    return ds

def create_window_dataset(ds, lb, window_size):
    windowed_dataset = []
    labels = []
    for i in range(window_size, ds.shape[0] + 1):
        windowed_dataset.append(ds[i - window_size:i])
        labels.append(lb[i - 1])
        
    return np.array(windowed_dataset), np.array(labels)

def get_metrics_result(metrics, true_labels, predicted_labels):    
    metrics_result = []
    for metric in metrics:
        metric.reset_states()
        metric.update_state(true_labels, predicted_labels)
        metrics_result.append(metric.result().numpy())
    
    return metrics_result

data = pd.read_csv(FILE_NAME, sep=',')
data = data[['dates', 'amount']]
print(data.head())

dataset = data['amount']
dataset.loc[dataset.shape[0]]= 0.0
dataset = dataset.iloc[1:]
data.iloc[:, -1] = dataset.values
dataset = data[:-1]

dataset_to_numpy = dataset.values
print(dataset.head())

#Split dataSet

train_dataset, cross_validation_dataset, dev_dataset = split_dataset(dataset_to_numpy)

print("Dataset shape: {:s}".format(str(dataset_to_numpy.shape)))
print("Train dataset shape: {:s}".format(str(train_dataset.shape)))
print("Cross validation dataset shape: {:s}".format(str(cross_validation_dataset.shape)))
print("Dev dataset shape: {:s}".format(str(dev_dataset.shape)))

#Plotting graph
'''
plt.figure(figsize=(10, 6))
plot_series(np.arange(train_dataset.shape[0]), train_dataset[:, -1], "train dataset")
plot_series(np.arange(train_dataset.shape[0], (cross_validation_dataset.shape[0] + train_dataset.shape[0])), cross_validation_dataset[:, -1], "cross validation dataset")
plot_series(np.arange((cross_validation_dataset.shape[0] + train_dataset.shape[0]), (cross_validation_dataset.shape[0]+ train_dataset.shape[0] + dev_dataset.shape[0])), dev_dataset[:, -1], "dev dataset")
plt.legend(loc='upper left')
plt.show()'''
#Data normalization


scaler = MinMaxScaler()
scaler = scaler.fit(train_dataset[:, 1:2])

train_dataset_normalized = scaler.transform(train_dataset[:, 1:2])
cross_validation_dataset_normalized = scaler.transform(cross_validation_dataset[:, 1:2])
dev_dataset_normalized = scaler.transform(dev_dataset[:, 1:2])

#Create window dataset
WINDOW_SIZE = 2
BATCH_SIZE = 10

windowed_dataset_train, labels_train = create_window_dataset(train_dataset[:, 1:2], train_dataset[:, -1], WINDOW_SIZE)

windowed_dataset_train, labels_train = create_window_dataset(train_dataset_normalized, train_dataset[:, -1], WINDOW_SIZE)
train_set = tf_dataset(windowed_dataset_train, labels_train, BATCH_SIZE, 1000)
unshuffled_train_set = tf_dataset(windowed_dataset_train, labels_train, BATCH_SIZE, 1000, False)

windowed_dataset_validation, labels_validation = create_window_dataset(cross_validation_dataset_normalized, cross_validation_dataset[:, -1], WINDOW_SIZE)
cross_validation_set = tf_dataset(windowed_dataset_validation, labels_validation, BATCH_SIZE, 1000, False)

windowed_dataset_dev, labels_dev = create_window_dataset(dev_dataset_normalized, dev_dataset[:, -1], WINDOW_SIZE)
dev_set = tf_dataset(windowed_dataset_dev, labels_dev, 1, 1000, False)
#print(windowed_dataset_dev)

TRAIN_STEP = math.ceil(windowed_dataset_train.shape[0] / BATCH_SIZE)
VALIDATION_STEP = math.ceil(windowed_dataset_validation.shape[0] / BATCH_SIZE)
DEV_STEP = windowed_dataset_dev.shape[0]
#print(windowed_dataset_train)
#print(windowed_dataset_train.shape[-3:])

# tf.keras.backend.clear_session()

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=64,
#                            kernel_size=5,
#                            strides=1,
#                            padding="causal",
#                            activation="relu",
#                            input_shape=(28,28,1)),#windowed_dataset_train.shape[-2:]
#     tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
#     tf.keras.layers.LSTM(128, return_sequences=True),
#     tf.keras.layers.LSTM(192, return_sequences=True),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(32, activation="relu"),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(1)
# ])

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(5e-4,
#                                                              decay_steps=1000000,
#                                                              decay_rate=0.98,
#                                                              staircase=False)

# model.compile(loss=tf.keras.losses.MeanSquaredError(),
#               optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.8),
#               metrics=['mae'])
# print(model.summary())

# class StopCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if logs.get('loss') < 0.08 and logs.get('val_loss') < 0.04:
#             print("\nReached the desired error so cancelling training!")
#             self.model.stop_training = True

# stop_callback = StopCallback()

# log_dir = "logs\\" + datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model = Sequential()

history = model.fit(train_set,
                    epochs=1000,
                    steps_per_epoch=TRAIN_STEP,
                    validation_data=cross_validation_set,
                    validation_steps=VALIDATION_STEP,
                    verbose=0,
                    callbacks=[stop_callback, tensorboard_callback])

forecast_train = model.predict(unshuffled_train_set, steps=TRAIN_STEP)
forecast_validation = model.predict(cross_validation_set, steps=VALIDATION_STEP)
forecast_dev = model.predict(dev_set, steps=DEV_STEP)

#Show predictions
total_forecast = np.concatenate((forecast_train[:,0], forecast_validation[:,0], forecast_dev[:,0]))
total_labels = np.concatenate((labels_train, labels_validation, labels_dev))

plt.figure(figsize=(18, 10))
plot_series(np.arange(total_labels.shape[0]), total_labels, "real value")
plot_series(np.arange(total_labels.shape[0]), total_forecast, "predicted value")
xpositions = [labels_train.shape[0], (labels_train.shape[0] + labels_validation.shape[0])]
for xp in xpositions:
    plt.axvline(x=xp, linestyle='--')
plt.legend(loc='upper left')

metrics = [
    tf.keras.metrics.MeanAbsoluteError(),
    tf.keras.metrics.MeanAbsolutePercentageError(),
    tf.keras.metrics.MeanSquaredError(),
    tf.keras.metrics.RootMeanSquaredError()
]
train_metrics = get_metrics_result(metrics, labels_train, forecast_train[:,0])
train_metrics.insert(0, 'Train')
val_metrics = get_metrics_result(metrics, labels_validation, forecast_validation[:,0])
val_metrics.insert(0, 'Validation')
dev_metrics = get_metrics_result(metrics,labels_dev, forecast_dev[:,0])
dev_metrics.insert(0, 'Dev')

COL_NAMES = ['', 'MAE', 'MAPE', 'MSE', 'RMSE']
metrics_table = pd.DataFrame([train_metrics, val_metrics, dev_metrics], columns=COL_NAMES)
metrics_table = metrics_table.set_index([''])
metrics_table.round(3)


