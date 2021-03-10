import pandas
import matplotlib.pyplot as plt
import sklearn
from keras.models import Sequential

dataset = pandas.read_csv('export_dataframe_nodeA.csv', usecols=[1], engine='python')
plt.plot(dataset)
plt.show()