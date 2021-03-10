


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#RNN 관련 라이브러리와 패키지 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# RNN 하이퍼 파라메터들 
TIME_STEP = 1
EPOCHS = 100
BATCH_SIZE = 10
DROP_OUT = 0.05


# 단계 1 : 학습 데이터 가져옴
dataset_train = pd.read_csv('export_dataframe_nodeA.csv') # 데이터 프레임 
dataset_train.dropna(inplace=True) #NaN을 모두 제거: 데이터 안에 NaN이 있는 경우
#필요한 데이터 선택 추출
#training_set_SO2 = dataset_train.iloc[:, 4:5].values
#training_set_PM10 = dataset_train.iloc[:, 5:6].values
training_set = dataset_train.iloc[:, 1:2].values
#training_set_O3 = dataset_train.iloc[:, 6:7].values
#training_set_NO2 = dataset_train.iloc[:, 7:8].values
#training_set_CO = dataset_train.iloc[:, 8:9].values
# 특징값을 0-1사이의 값으로 Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 하이퍼 파라미터인 timesteps을 가진 데이터 구조와 예측 값인 하나의 출력구조를 생성
X_train = []
y_train = []
for i in range(TIME_STEP, len(training_set)):
    X_train.append(training_set_scaled[i-TIME_STEP:i, 0])
    y_train.append(training_set_scaled[i, 0])

# X_train : 학습 데이터 갯수 - TIME_STEP
# y_train : TIME_STEP
X_train, y_train = np.array(X_train), np.array(y_train)

# RNN을 위한 3차원 데이터로 변환 (sample, timestep, feature) : Reshape
# X_train.shape[0] : 학습 데이터 갯수 - TIME_STEP, X_train.shape[1] : TIME_STEP
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# 단계 2 : RNN 구축
# RNN초기화
regressor = Sequential()

# 첫 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 40, return_sequences = False, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(DROP_OUT))

# 두 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(DROP_OUT))

# 세 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(DROP_OUT))
'''
# 네 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 65, return_sequences = True))
regressor.add(Dropout(DROP_OUT))
'''
# 다섯 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 60))
regressor.add(Dropout(DROP_OUT))

# 출력층 추가
regressor.add(Dense(units = 1))

# RNN 컴파일
regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics=['accuracy'])

# 학습 데이터를 RNN 적용
regressor.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)


#단계 3 : 학습된 RNN을 사용하여 예측
# 미세먼지 예측하고 결과를 그래프로 보여줌
# 예측할 데이터를 읽어옴 (실제 값들)
dataset_test = pd.read_csv('export_dataframe_nodeA.csv')




#필요한 데이터 선택 추출
#real_obj_value_SO2 = dataset_test.iloc[:, 4:5].values
#real_obj_value_PM10 = dataset_test.iloc[:, 5:6].values
real_obj_value = dataset_test.iloc[:, 1:2].values
#real_obj_value_O3 = dataset_test.iloc[:, 7:8].values
#real_obj_value_NO2 = dataset_test.iloc[:, 8:9].values
#real_obj_value_CO = dataset_test.iloc[:, 9:10].values

# 테스트 데이터 셋을 이용하여 예측된 PM 구하기 
dataset_combined = pd.concat((dataset_train['amount'], dataset_test['amount']), axis = 0)

# input_PM : 테스트 데이터 수 + TIME_STEP
input_obj = dataset_combined[len(dataset_combined) - len(dataset_test) - TIME_STEP:].values 
input_obj = input_obj.reshape(-1,1) #((테스트 데이터 수 + TIME_STEP),1)
input_obj = sc.transform(input_obj) #scale

#input_PM = X_test_scaled
X_test = []
for i in range(TIME_STEP, TIME_STEP+len(dataset_test)):
    X_test.append(input_obj[i-TIME_STEP:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_data = regressor.predict(X_test)
predicted_data = sc.inverse_transform(predicted_data)

'''
   모델 성능 평가하기 위한 방법 (RMSE)
   import math
   from sklearn.metrics import mean_squared_error
   rmse = math.sqrt(mean_squared_error(real_PM_value, predicted_PM25))
'''

# 예측된 결과와 실제 값을 그래프로 표시
plt.plot(real_obj_value, color = 'red', label = 'Actual Energy')
plt.plot(predicted_data, color = 'blue', label = 'Predicted Energy')
plt.title('Energy Prediction System')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.show()

#학습모델저장

#regressor.save('weight/Energy.h5')


