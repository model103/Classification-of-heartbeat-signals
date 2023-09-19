import os
import gc
import math
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

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

# 将label转换为类似[0, 0, 1, 0]的形式
y_train_array_class = np.zeros((len(y_train_array), 4), dtype=np.float32)  # 将label转换为类似[0, 0, 1, 0]的形式
N = 0
for i in y_train_array:
    y_train_array_class[N][int(i)] = 1.
    N += 1
# print(y_train_array_class)

# print(y_train_array_class)


# print(np.shape(x_train_array))
# print(x_train_array)


# h划分交叉验证
folds = 5  # 4:1划分训练集和验证集
seed = 2021
train_X, test_X, train_Y, test_Y = train_test_split(x_train_array, y_train_array_class, test_size=1 / folds,
                                                    random_state=seed)
'''
print(x_train_array)
print(y_train_array)
print(test_X)
print(train_Y)
'''
batch_size = 10000


def random_batch(x1, y1, batch_size):
    rnd_indices = np.random.randint(0, len(x1), batch_size)
    x_batch = x1[rnd_indices]
    y_batch = y1[rnd_indices]
    return x_batch, y_batch


# 创建神经网络
# 创建x，x是一个占位符，代表待识别的图片
x = tf.placeholder(tf.float32, [None, 205])
# w是softmax模型的参数，将一个784的输入转换为一个10位的输出
w = tf.Variable(tf.zeros([205, 4]))
# b是又一个softmax的参数，一般叫做“偏置项
b = tf.Variable(tf.zeros([4]))
# y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, w) + b)
# y_是实际的图像标签，同样以占位符表示
y_ = tf.placeholder(tf.float32, [None, 4])
# 根据y和y_构造交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 有了交叉熵，就可以使用梯度下降法针对模型的参数（w和b)进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个Session。只有在Session中才能运行优化步骤train_step
sess = tf.InteractiveSession()
# 运行之前必须要初始化所有的变量，分配内存
init = tf.global_variables_initializer()

print(type(train_X))

test = np.zeros((10000,205))


sess.run(init)
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是形状为（100,784)的图像数据，batch_ys是形如（100,10）的实际标签
    # batch_xs与batch_ys分别对应着x和y_两个占位符
    batch_xs, batch_ys = random_batch(train_X, train_Y, batch_size)
    # 在Session中运行train_step,运行时要传入占位符的值
    #print(batch_xs)
    sess.run(train_step, feed_dict={x: test, y_: batch_ys})
    #print(x.eval())

    # print(y.eval())
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy)

print(sess.run(accuracy, feed_dict={x: test_X, y_: test_Y}))