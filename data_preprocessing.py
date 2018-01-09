import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


flow_data = np.load('data/flow_data')
stage_data = np.load('data/stage_data')
status_data = np.load('data/status_data')
reward_data = np.load('data/reward_data')
flow_next_data = np.load('data/flow_next_data')
stage_next_data = np.load('data/stage_next_data')
status_next_data = np.load('data/status_next_data')
reward_next_data = np.load('data/reward_next_data')
action_data = np.load('data/action_data')
# action_data = np.round(action_data/20)  # change action space's shape(1482, 4)->(259, 4)


def n_times_data(data, n=5):    # more data
    """
    used for 2-D data
    for 1-D data add <data = data.reashape(-1,1)> at begining
    """
    size = data.shape
    data_augmented = data
    for num in range(n-1):
        data_std = np.random.normal(loc=0, scale=np.std(data/10, axis=0), size=size)
        data_more = data + data_std
        data_augmented = np.concatenate((data_augmented, data_more), axis=0)
    return data_augmented


X = np.concatenate((status_data, stage_data), axis=1)
X = shuffle(n_times_data(X), random_state=0)  # shuffle
y = shuffle(n_times_data(action_data), random_state=0)

X_scale = MinMaxScaler()  # min_max scale
y_scale = MinMaxScaler()
X_scale.fit(X)
y_scale.fit(y)

X = X_scale.transform(X)
y = y_scale.transform(y)


clf = PCA(6)
X = clf.fit_transform(X)
# --------------------
actions = np.unique(action_data, axis=0)
actions = actions[np.abs(np.sum(actions, axis=1)) < 100]  # remove actions change too much


def delay(s):
    if s.ndim == 1:
        s = s.reshape(1, s.shape[0])

    delay_time = np.zeros((s.shape[0], 4))
    delay_time[:, 0] = s[:, 32] / sum((s[:, 17], s[:, 22], s[:, 23], s[:, 27], s[:, 28], s[:, 29]))
    delay_time[:, 1] = s[:, 33] / sum((s[:, 17], s[:, 20], s[:, 21], s[:, 22], s[:, 27], s[:, 28], s[:, 30], s[:, 31]))
    delay_time[:, 2] = s[:, 34] / sum((s[:, 17], s[:, 18], s[:, 22], s[:, 24], s[:, 27], s[:, 28]))
    delay_time[:, 3] = s[:, 35] / sum((s[:, 16], s[:, 17], s[:, 19], s[:, 22], s[:, 25], s[:, 26], s[:, 27], s[:, 28]))

    return delay_time


min_delay = np.percentile(delay(status_data), 30, axis=0)  # row 1 has negative number
min_stage = np.min(stage_data, axis=0)
max_stage = np.max(stage_data, axis=0)


def reward(delay_data):
    """
    input delay_data
    output reward by sigmod func
    """
    x = np.abs(delay_data - min_delay)
    x = np.sum(x, axis=1)
    return 1/(1+np.e**x)


def max_reward_action(status):
    best_action = np.zeros((status.shape[0], 4))
    best_reward = np.zeros((status.shape[0], 1))
    num = 0
    for stat in status:
        temp = np.tile(stat, actions.shape[0]).reshape((actions.shape[0], stat.shape[0]))
        temp[:, -4:] = temp[:, -4:] + actions
        temp = temp[(temp[:, -4:] >= min_stage).all(axis=1)]  # remove too little stage_value
        temp = temp[(temp[:, -4:] <= max_stage).all(axis=1)]  # remove too large stage_value
        temp = temp[np.sum(temp[:, -4:], axis=1) > 300]  # remove too little cycle_value
        temp = temp[np.sum(temp[:, -4:], axis=1) < 420]  # remove too little cycle_value
        R = reward(delay(temp))
        best_action[num] = actions[R.argmax()]
        best_reward[num] = R.max()
        num += 1
    return best_action, best_reward


r_target = reward(delay(status_data))
a, r_eval = max_reward_action(status_data)
plt.plot(r_target[:128])
plt.plot(r_eval[:128])
plt.show()
