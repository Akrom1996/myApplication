#-*- coding: utf-8 -*-

# 데이터 전처리와 관련 라이브러리와 패키지 가져옴
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
EPOCHS = 50
BATCH_SIZE = 10
DROP_OUT = 0.1



# 단계 1 : 학습 데이터 가져옴

#string_dong = "Daemyeong"
#string_dong = 'Horim'
#string_dong = 'Ihyeon'
#string_dong = 'Jisan'
#string_dong = 'Manchon'
#string_dong = 'Nowon'
#string_dong = 'Seoho'
#string_dong = 'Shinam'
#string_dong = 'Suchang'
#string_dong = 'Taejeon'


dataset = pd.read_csv('export_dataframe_nodeA.csv') # 데이터 프레임
dataset.dropna(inplace=True) #NaN을 모두  제거: 데이터 안에 NaN이 있는 경우
DATA_LENGTH = int(len(dataset) * 0.8)

training_set = dataset.iloc[:DATA_LENGTH, 1:2].values

#training_set = dataset_train.iloc[:, 6:7].values # 여러 데이터 중 관심있는 PM2.5 데이터만 추출

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
regressor.add(LSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(DROP_OUT))
# 두 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 200, return_sequences = True))
regressor.add(Dropout(DROP_OUT))
# 세 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 200, return_sequences = True))
regressor.add(Dropout(DROP_OUT))

# 다섯 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 200))
regressor.add(Dropout(DROP_OUT))
# 출력층 추가
regressor.add(Dense(units = 1))

# RNN 컴파일
regressor.compile(optimizer = 'Adam', loss = 'mean_squared_error')

# 학습 데이터를 RNN 적용
regressor.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)


test_set_CO = dataset.iloc[DATA_LENGTH:, 1:2].values

real_PM_value = test_set_CO

dataset_test = real_PM_value.reshape(-1, 1)
dataset_test = sc.transform(dataset_test)



# 테스트 데이터 셋을 이용하여 예측된 PM 구하기 
dataset_combined = dataset['amount']

# input_PM : 테스트 데이터 수 + TIME_STEP
input_PM = dataset_combined[len(dataset_combined) - len(dataset_test) - TIME_STEP:].values 
input_PM = input_PM.reshape(-1,1) #((테스트 데이터 수 + TIME_STEP),1)
input_PM = sc.transform(input_PM) #scale

X_test = []
for i in range(TIME_STEP, TIME_STEP+len(dataset_test)):
    X_test.append(input_PM[i-TIME_STEP:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_PM25 = regressor.predict(X_test, batch_size=BATCH_SIZE)
predicted_PM25 = sc.inverse_transform(predicted_PM25)
predicted_PM25 = predicted_PM25[1:]

# 예측된 결과와 실제 값을 그래프로 표시
plt.rcParams["figure.figsize"] = (10,4)
plt.rcParams['lines.linewidth'] = 1
plt.plot(real_PM_value, color = 'gray', label = 'real energy amount')
plt.plot(predicted_PM25, color = 'orange', label = 'predict energy amount')
plt.title('Energy amount prediction')
plt.xlabel('Time')
plt.ylabel('amount')
plt.legend()
plt.show()

