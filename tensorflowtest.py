# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

# 导入mnist教学的模块
from tensorflow.keras.datasets.mnist import input_data

# 读入mnist数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建x，x是一个占位符，代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])

# w是softmax模型的参数，将一个784的输入转换为一个10位的输出
w = tf.Variable(tf.zeros([784, 10]))

# b是又一个softmax的参数，一般叫做“偏置项
b = tf.Variable(tf.zeros([10]))

# y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, w) + b)

# y_是实际的图像标签，同样以占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

# 根据y和y_构造交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 有了交叉熵，就可以使用梯度下降法针对模型的参数（w和b)进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个Session。只有在Session中才能运行优化步骤train_step
sess = tf.InteractiveSession()
# 运行之前必须要初始化所有的变量，分配内存
tf.global_variables_initializer().run()


batch_size = 10



# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是形状为（100,784)的图像数据，batch_ys是形如（100,10）的实际标签
    # batch_xs与batch_ys分别对应着x和y_两个占位符
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中运行train_step,运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 在Session中运行Tensor可以得到Tensor的值
# 这里是获取最终模型的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))