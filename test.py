import numpy as np
from sklearn.model_selection import train_test_split
'''
x = [[1,2,3],
    [4,5,6],
    [7,8,9]]
print(np.shape(x))
x1 = np.array(x)
rnd_indices = np.random.randint(0, 3, 2)
print(rnd_indices)
print(x1[rnd_indices])
'''
'''
x_list=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y_list=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
x = np.array(x_list)
y = np.array(y_list)
print(type(x))
train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2,random_state=0)   #train_X, test_X, train_Y, test_Y会被转为list类型
print(train_X)
print(train_Y)

batch_size = 2

def random_batch(x1, y1, batch_size):
    rnd_indices = np.random.randint(0, len(x1), batch_size)
    x_batch = x1[rnd_indices]
    y_batch = y1[rnd_indices]
    return x_batch, y_batch

batch_xs, batch_ys = random_batch(train_X, train_Y, batch_size)

print(batch_xs)
print(batch_ys)
print(np.shape(batch_ys))
'''

a = np.array([[ 0.49596769,  1.15846407, -1.38944733],
       [-0.47042814, -0.07512128 , 1.90417981]])

b = (a == a.max(axis=1, keepdims=1)).astype(float)
print(a)
print(b)