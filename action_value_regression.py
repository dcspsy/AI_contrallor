# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import data_preprocessing

X, y = data_preprocessing.X, data_preprocessing.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.hidden = nn.Linear(6, 36)
        self.out = nn.Linear(36, 4)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

inputs_train, inputs_test = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(X_test))
target_train, target_test = Variable(torch.Tensor(y_train)), Variable(torch.Tensor(y_test))
num_epochs = 5000
for epoch in range(num_epochs):


    # forward
    out_train = model(inputs_train)  # 前向传播
    out_test = model(inputs_test)
    loss_train = criterion(out_train, target_train)  # 计算loss
    loss_test = criterion(out_test, target_test)
    # backward
    optimizer.zero_grad()  # 梯度归零
    loss_train.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if epoch % 100 == 0:
        print 'Epoch[{}/{}], loss_train: {:.6f}, loss_test: {:.6f}'.\
            format(epoch, num_epochs, loss_train.data[0], loss_test.data[0])


loss_val = criterion(model(inputs_test), target_test)
print loss_val

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(model(inputs_test).data.numpy()[500:600, 0])
plt.plot(target_test.data.numpy()[500:600, 0])
plt.subplot(2, 2, 2)
plt.plot(model(inputs_test).data.numpy()[500:600, 1])
plt.plot(target_test.data.numpy()[500:600, 1])
plt.subplot(2, 2, 3)
plt.plot(model(inputs_test).data.numpy()[500:600, 2])
plt.plot(target_test.data.numpy()[500:600, 2])
plt.subplot(2, 2, 4)
plt.plot(model(inputs_test).data.numpy()[500:600, 3])
plt.plot(target_test.data.numpy()[500:600, 3])
plt.show()


def eval_action_value(stats):
    """
    regression action values by stats,with DNN(36,36,1)
    """
    stats = Variable(torch.Tensor(stats))
    value = model(stats)
    value = value.data.numpy()
    if value.ndim == 1:
        value = data_preprocessing.y_scale.inverse_transform(value.reshape(1, 4))
    else:
        value = data_preprocessing.y_scale.inverse_transform(value)

    return value


def y_inverse_transform(data_y):
    if data_y.ndim == 1:
        value = data_preprocessing.y_scale.inverse_transform(data_y.reshape(1, 4))
    else:
        value = data_preprocessing.y_scale.inverse_transform(data_y)

    return value


def eval_action_value(stats):
    """
    regression action values by stats,with DNN(36,36,1)
    """
    stats = Variable(torch.Tensor(stats))
    value = model(stats)
    value = value.data.numpy()
    value = y_inverse_transform(value)

    return value

