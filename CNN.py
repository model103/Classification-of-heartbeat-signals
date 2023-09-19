import os
import gc
import math
import tensorflow as tf


import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings


warnings.filterwarnings('ignore')

train = pd.read_csv('./train.csv')  # 读取训练集
test = pd.read_csv('./testA.csv')  # 读取测试集


train.head()
test.head()


# 3.1数据预处理
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# 3.2 简单预处理
train_list = []

for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])

train = pd.DataFrame(np.array(train_list))
train.columns = ['id'] + ['s_' + str(i) for i in range(len(train_list[0]) - 2)] + ['label']
train = reduce_mem_usage(train)

test_list = []
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])

test = pd.DataFrame(np.array(test_list))
test.columns = ['id'] + ['s_' + str(i) for i in range(len(test_list[0]) - 1)]
test = reduce_mem_usage(test)

train.head()

# 4. 训练数据/测试数据准备
x_train = train.drop(['id', 'label'], axis=1)  # 去掉id和label标签
y_train = train['label']  # 只保留lable标签
x_test = test.drop(['id'], axis=1)
# print(type(x_train))
# print(x_train)
x_train_array = np.array(x_train)  # 转成array类型
y_train_array = np.array(y_train)  # label为0，1,2,3形式
x_test_array = np.array(x_test)
# print(y_train_array)




# print(y_train_array_class)

# print(y_train_array_class)


# print(np.shape(x_train_array))
# print(x_train_array)


# h划分交叉验证
folds = 5  # 4:1划分训练集和验证集
seed = 2021
train_X, test_X, train_Y, test_Y = train_test_split(x_train_array, y_train_array, test_size=1 / folds, random_state=seed)




print(np.shape(train_Y))
print(np.shape(test_Y))





#Model申明网络



class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=9, input_shape=(205, 1), activation='relu' , padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
        self.d2 = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


model = MnistModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_X, train_Y, batch_size=10000, epochs=1000, validation_split=0.2, validation_freq=1)

sub_model = tf.keras.models.Model(inputs = model.input, outputs = model.layers[-1].output)


#test_xx = x_train_array[0:20000]       #取前10000行做测试数据输出
#test_yy = y_train_array_class[0:20000]    #前10000行的真实标签
#print(testB)
result = sub_model.predict(test_X)

'''
loss = loss=sum(sum(abs(testA-result)))
print(result)
'''
result_1 = (result == result.max(axis=1, keepdims=1)).astype(float)  #将每行最大值置1，其他置0

# 将label转换为类似[0, 0, 1, 0]的形式
test_Y_class = np.zeros((len(test_Y), 4), dtype=np.float32)  # 将label转换为类似[0, 0, 1, 0]的形式
N = 0
for i in test_Y:
    test_Y_class[N][int(i)] = 1.
    N += 1
loss_1 = loss=sum(sum(abs(test_Y_class-result_1)))
print(result_1)
print(loss)
'''
dataA = pd.DataFrame(testA)
dataB = pd.DataFrame(result)
dataB_1 = pd.DataFrame(result_1)

writer = pd.ExcelWriter('结果.xlsx')		# 写入Excel文件
dataA.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
dataB.to_excel(writer, 'page_2', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
dataB_1.to_excel(writer, 'page_3', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
'''





model.summary()






'''
class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
        self.d2 = tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.d1(x)
        y = self.d2(x)
        return y


model = MnistModel()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

model.fit(x_train_array, y_train_array, batch_size=10, epochs=100, validation_split=0.2, validation_freq=1)
'''
