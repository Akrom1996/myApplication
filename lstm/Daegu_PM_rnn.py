#-*- coding: utf-8 -*-

'''
   1) 미세먼지 예측을 위한 Recurrent Neural Network
   2) 사용 모델 : RNN
   3) 사용 데이터 : 대구 대명동, 호림동, 이현동, 지산동, 만촌동, 노원동, 서호동, 신암동, 수창동, 태전동
                    PM2.5, NO2, CO, O3, SO2 Data Data Set
                    (2008.01.01 01시 - 2017.12.31 24시)
                    매 한 시간마다 측정된 데이터 
   4) 데이터 수 : 약 87658개(Missing 데이터 포함) * 5가지 변수 * 지역 10곳
   대명동 : 87415
   호림동 : 87469
   이현동 : 87581
   지산동 : 87546
   만촌동 : 87406
   노원동 : 87460
   서호동 : 87192
   신암동 : 87432
   수창동 : 87427
   태전동 : 87502
'''

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
TIME_STEP = 30
EPOCHS = 5
BATCH_SIZE = 128
DROP_OUT = 0.1


# 단계 1 : 학습 데이터 가져옴

#string_dong = "Daemyeong"
#string_dong = 'Horim'
#string_dong = 'Ihyeon'
#string_dong = 'Jisan'
#string_dong = 'Manchon'
string_dong = 'Nowon'
#string_dong = 'Seoho'
#string_dong = 'Shinam'
#string_dong = 'Suchang'
#string_dong = 'Taejeon'


dataset = pd.read_csv('data/' + string_dong + '.csv') # 데이터 프레임
dataset.dropna(inplace=True) #NaN을 모두 제거: 데이터 안에 NaN이 있는 경우

#training_set_SO2 = dataset.iloc[:80000, 4:5].values
training_set_PM10 = dataset.iloc[:80000, 5:6].values
#training_set_O3 = dataset.iloc[:80000, 6:7].values
#training_set_NO2 = dataset.iloc[:80000, 7:8].values
#training_set_CO = dataset.iloc[:80000, 8:9].values

training_set = training_set_PM10

'''
# Taking care of missing data (평균 값 사용) : 데이터 안에 NaN이 있는 경우
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
'''
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
regressor.add(LSTM(units = 120, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(DROP_OUT))
# 두 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 180, return_sequences = True))
regressor.add(Dropout(DROP_OUT))
# 세 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 230, return_sequences = True))
regressor.add(Dropout(DROP_OUT))
'''
# 네 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 65, return_sequences = True))
regressor.add(Dropout(DROP_OUT))
'''
# 다섯 번째 LSTM층과 드롭 아웃을 사용하여 layer and some Dropout 규격화 시킴
regressor.add(LSTM(units = 120))
regressor.add(Dropout(DROP_OUT))
# 출력층 추가
regressor.add(Dense(units = 1))

# RNN 컴파일
regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')

# 학습 데이터를 RNN 적용
regressor.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)

#단계 3 : 학습된 RNN을 사용하여 예측
# 미세먼지 예측하고 결과를 그래프로 보여줌
# 예측할 데이터를 읽어옴 (실제 값들)

#test_set_SO2 = dataset.iloc[80000:, 4:5].values
test_set_PM10 = dataset.iloc[80000:, 5:6].values
#test_set_O3 = dataset.iloc[80000:, 6:7].values
#test_set_NO2 = dataset.iloc[80000:, 7:8].values
#test_set_CO = dataset.iloc[80000:, 8:9].values

real_PM_value = test_set_PM10

dataset_test = real_PM_value.reshape(-1, 1)
dataset_test = sc.transform(dataset_test)



# 테스트 데이터 셋을 이용하여 예측된 PM 구하기 
dataset_combined = dataset['PM10']

# input_PM : 테스트 데이터 수 + TIME_STEP
input_PM = dataset_combined[len(dataset_combined) - len(dataset_test) - TIME_STEP:].values 
input_PM = input_PM.reshape(-1,1) #((테스트 데이터 수 + TIME_STEP),1)
input_PM = sc.transform(input_PM) #scale


#input_PM = X_test_scaled
X_test = []
for i in range(TIME_STEP, TIME_STEP+len(dataset_test)):
    X_test.append(input_PM[i-TIME_STEP:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_PM25 = regressor.predict(X_test)
predicted_PM25 = sc.inverse_transform(predicted_PM25)
#predicted_PM25 = predicted_PM25[1:]


#모델 성능 평가하기 위한 방법 (RMSE)
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_PM_value, predicted_PM25))



# 예측된 결과와 실제 값을 그래프로 표시
plt.rcParams["figure.figsize"] = (10,4)
plt.rcParams['lines.linewidth'] = 1
plt.ylim([0,150])
plt.xlim([100, 150])
plt.plot(real_PM_value, color = 'gray', label = 'real PM10')
plt.plot(predicted_PM25, color = 'green', label = 'normal')
#plt.plot(adam, color = 'blue', label = 'Adam')
#plt.plot(ts15, color = 'blue', label = 'TS=15')
plt.title(string_dong +'dong PM10')
plt.xlabel('Time')
plt.ylabel('PM10')
plt.legend()
plt.show()

