# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
# read data from csv
date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
parse_dates = ['curr_time']
flow_data_read = pd.read_csv('origin_data/flow_origin_data.csv', encoding='gb2312', header=0,
                             date_parser=date_parser, parse_dates=parse_dates)
flow_data_read['date'] = flow_data_read['curr_time'].apply(lambda x: x.date())

date_time = flow_data_read[['date', 'curr_time']].drop_duplicates(['curr_time']).reset_index(drop=True)
date_index = flow_data_read['date'].unique()
time_index = flow_data_read['curr_time'].unique()

flow_data_read = flow_data_read.set_index(['date', 'cross_name'])

flows_index = [u'north_north_flow', u'west_north_flow', u'south_north_flow', u'east_north_flow',
               u'north_west_flow', u'west_west_flow', u'south_west_flow', u'east_west_flow',
               u'north_south_flow', u'west_south_flow', u'south_south_flow', u'east_south_flow',
               u'north_east_flow', u'west_east_flow', u'south_east_flow', u'east_east_flow']
cross_index = [u'观枫街-苏州大道东', u'华池街-旺墩路', u'星湖街-苏州大道东', u'华池街-苏州大道东', u'华池街-现代大道',
               u'观枫街-现代大道']
flow_map = {
        'north': {'input': flows_index[8:12], 'output': [flows_index[i] for i in [0, 4, 8, 12]]},
        'west': {'input': flows_index[12:16], 'output': [flows_index[i] for i in [1, 5, 9, 13]]},
        'south': {'input': flows_index[0:4], 'output': [flows_index[i] for i in [2, 6, 10, 14]]},
        'east': {'input': flows_index[4:8], 'output': [flows_index[i] for i in [3, 7, 11, 15]]}
            }
cross_map = {
    u'华池街-苏州大道东': {
        'north': u'华池街-现代大道', 'west': u'苏州大道东-月廊街', 'south': u'华池街-旺墩路', 'east': u'星湖街-苏州大道东'},
    u'观枫街-现代大道': {
        'north': None, 'west': None, 'south': u'观枫街-苏州大道东', 'east': u'华池街-现代大道'},
    u'华池街-现代大道': {
        'north': None, 'west': u'观枫街-现代大道', 'south': u'华池街-苏州大道东', 'east': None},
    u'观枫街-苏州大道东': {
        'north': u'观枫街-现代大道', 'west': None, 'south': None, 'east': u'华池街-苏州大道东'},
    u'华池街-旺墩路': {
        'north': u'华池街-苏州大道东', 'west': None, 'south': None, 'east': None},
    u'星湖街-苏州大道东': {
        'north': None, 'west': u'华池街-苏州大道东', 'south': None, 'east': None}
}


stage_map = {
    u'华池街-苏州大道东': {
        17: [4+31, 8+31],
        18: [1+31, 2+31, 3+31, 4+31, 7+31, 8+31],
        19: [3+31, 8+31],
        20: [4+31, 8+31],
        21: [2+31],
        22: [2+31],
        23: [1+31, 2+31, 3+31, 4+31, 7+31, 8+31],
        24: [1+31],
        25: [3+31, 7+31],
        26: [4+31, 7+31],
        27: [4+31],
        28: [1+31, 2+31, 3+31, 4+31, 7+31, 8+31],
        29: [1+31, 2+31, 3+31, 4+31, 7+31, 8+31],
        30: [1+31],
        31: [2+31],
        32: [2+31]
    },
    u'观枫街-现代大道': {
        17: None,
        18: None,
        19: None,
        20: None,
        21: [2+31],
        22: [2+31],
        23: [1+31, 2+31, 6+31],
        24: [1+31],
        25: None,
        26: [6+31],
        27: [6+31],
        28: [1+31, 2+31, 6+31],
        29: [1+31, 2+31, 6+31],
        30: [1+31],
        31: [2+31],
        32: [2+31]
    },
    u'华池街-现代大道': {
        17: [4+31, 8+31],
        18: [1+31, 2+31, 3+31, 4+31, 6+31, 7+31, 8+31],
        19: [3+31, 8+31],
        20: [4+31, 8+31],
        21: [2+31, 6+31],
        22: [2+31, 6+31],
        23: [1+31, 2+31, 3+31, 4+31, 6+31, 7+31, 8+31],
        24: [1+31, 6+31],
        25: [3+31, 7+31],
        26: [4+31, 7+31],
        27: [4+31, 7+31],
        28: [1+31, 2+31, 3+31, 4+31, 6+31, 7+31, 8+31],
        29: [1+31, 2+31, 3+31, 4+31, 6+31, 7+31, 8+31],
        30: [1+31],
        31: [2+31],
        32: [2+31]
    },
    u'观枫街-苏州大道东': {
        17: [3+31],
        18: [1+31, 2+31, 3+31, 5+31],
        19: [3+31],
        20: [3+31],
        21: [2+31],
        22: [2+31],
        23: [1+31, 2+31, 3+31, 5+31],
        24: [1+31],
        25: [3+31],
        26: [3+31],
        27: [3+31],
        28: [1+31, 2+31, 3+31, 5+31],
        29: [1+31, 2+31, 3+31, 5+31],
        30: [1+31, 5+31],
        31: [2+31, 5+31],
        32: [2+31, 5+31]
    },
    u'华池街-旺墩路': {
        17: [4+31, 8+31],
        18: [3+31, 4+31, 5+31, 8+31],
        19: [3+31, 8+31],
        20: [4+31, 8+31],
        21: [5+31],
        22: [5+31],
        23: [3+31, 4+31, 5+31, 8+31],
        24: [5+31],
        25: [3+31],
        26: [4+31],
        27: [4+31],
        28: [3+31, 4+31, 5+31, 8+31],
        29: [3+31, 4+31, 5+31, 8+31],
        30: [5+31],
        31: [5+31],
        32: [5+31]
    },
    u'星湖街-苏州大道东': {
        17: [4+31],
        18: [1+31, 3+31, 4+31, 5+31, 6+31],
        19: [3+31],
        20: [4+31],
        21: [1+31, 6+31],
        22: [1+31, 6+31],
        23: [1+31, 3+31, 4+31, 5+31, 6+31],
        24: [1+31, 6+31],
        25: [3+31],
        26: [4+31],
        27: [4+31],
        28: [1+31, 3+31, 4+31, 5+31, 6+31],
        29: [1+31, 3+31, 4+31, 5+31, 6+31],
        30: [1+31, 5+31],
        31: [1+31, 5+31],
        32: [1+31, 5+31]
    }
}
"""
0       苏州大道东-观枫街         2       东西左转
1       苏州大道东-观枫街         3       南北放行
2       苏州大道东-观枫街         1       东西直行
3       苏州大道东-华池街         1       东西直行
4       苏州大道东-华池街         2       东西左转
5       苏州大道东-华池街         3       南北直行
6       苏州大道东-华池街         4       南北左转
7       星湖街-苏州大道东         1       东西放行
8       星湖街-苏州大道东         3       南北直行
9       星湖街-苏州大道东         4       南北左转
10        旺墩路-华池街         4       南北左转
11        旺墩路-华池街         5       东西放行
12        旺墩路-华池街         3       南北直行
13       现代大道-华池街         2       东西左转
14       现代大道-华池街         3       南北直行
15       现代大道-华池街         4       南北左转
16       现代大道-华池街         1       东西直行
30       现代大道-观枫街         1       东西直行
31       现代大道-观枫街         2       东西左转
32       现代大道-观枫街         6        南放行
4462    星湖街-苏州大道东         5        东放行
4463    星湖街-苏州大道东         9        NaN
4464    星湖街-苏州大道东         6        西放行
5785     现代大道-华池街         8        北放行
12244    现代大道-华池街         7        南放行
147354  苏州大道东-观枫街         5        东放行
147475   现代大道-华池街         6        西放行
391739  苏州大道东-华池街         8        北放行
391792    旺墩路-华池街         8        北放行
391816  苏州大道东-华池街         7        南放行
640750    旺墩路-华池街       255        NaN"""

# set cross_name, date,time into index

flow_data = np.zeros((len(cross_index), len(time_index), 32))
flow_grouped = flow_data_read.groupby(['cross_name', 'curr_time']).mean()
for cross in range(len(cross_index)):
    if cross_map[cross_index[cross]]['north'] is not None:
        flow_data[cross][:,0:4] = flow_grouped.loc[cross_map[cross_index[cross]]['north']][flow_map['north']['input']]
    if cross_map[cross_index[cross]]['west'] is not None:
        flow_data[cross][:,4:8] = flow_grouped.loc[cross_map[cross_index[cross]]['west']][flow_map['west']['input']]
    if cross_map[cross_index[cross]]['south'] is not None:
        flow_data[cross][:,8:12] = flow_grouped.loc[cross_map[cross_index[cross]]['south']][flow_map['south']['input']]
    if cross_map[cross_index[cross]]['east'] is not None:
        flow_data[cross][:,12:16] = flow_grouped.loc[cross_map[cross_index[cross]]['east']][flow_map['east']['input']]
    flow_data[cross][:,16:20] = flow_grouped.loc[cross_index[cross]][flow_map['north']['output']]
    flow_data[cross][:,20:24] = flow_grouped.loc[cross_index[cross]][flow_map['west']['output']]
    flow_data[cross][:,24:28] = flow_grouped.loc[cross_index[cross]][flow_map['south']['output']]
    flow_data[cross][:,28:32] = flow_grouped.loc[cross_index[cross]][flow_map['east']['output']]

"""
----index
0-2:flow_data columns cross_index,time_index,date_index,
----input
0-4:north_south_flow,west_south_flow,south_south_flow,east_south_flow
5-8:north_east_flow,west_east_flow,south_east_flow,east_east_flow
9-12:north_north_flow,west_north_flow,south_north_flow,east_north_flow
13-16:north_west_flow, west_west_flow, south_west_flow, east_west_flow
----output
17-20:north_north_flow,north_west_flow,north_south_flow,north_east_flow
21-24:west_north_flow,west_west_flow,west_south_flow,west_east_flow
25-28:south_north_flow,south_west_flow,south_south_flow,south_east_flow
29-32:east_north_flow,east_west_flow,east_south_flow,east_east_flow

"""



# collect stage data by flowdata's time index
stage_data_read = pd.read_csv('origin_data/stage_origin_data.csv',
                              encoding='gb2312', date_parser=date_parser, parse_dates=['STAGE_START_TM'])
stage_data_read = stage_data_read[['ROAD_NAME', 'STAGE_START_TM', 'STAGE_SN', 'STAGE_SECONDS']]
stage_data_read.columns = ['cross_name', 'stage_time', 'stage_sn', 'val']

# encoding

stage_data_read['stage_time'] = pd.cut(stage_data_read['stage_time'], time_index, labels=range(1, len(time_index))).values
fix_cross_name = dict(zip([u'苏州大道东-观枫街', u'苏州大道东-华池街', u'星湖街-苏州大道东',u'旺墩路-华池街', u'现代大道-华池街', u'现代大道-观枫街'],
                          [0, 3, 2, 1, 4, 5]))

stage_data_read['cross_name'] = stage_data_read['cross_name'].replace(fix_cross_name)

# collect
stage_grouped = pd.pivot_table(stage_data_read, index=['cross_name', 'stage_time'], columns=['stage_sn'], values=['val'],
                            aggfunc=[np.sum], fill_value=0)


stage_data = np.zeros((len(cross_index), len(time_index), 10))
for cross in range(len(cross_index)):
    stage_data[cross][1:] = stage_grouped.loc[cross]

stage_data = stage_data[:, :, :-2]  # drop out stage_sn 9 & nan


action_data = np.zeros((len(cross_index), len(time_index), 8))
for cross in range(len(cross_index)):
    action_data[cross][1:] = stage_data[cross][1:] - stage_data[cross][:-1]

"""
----index
0-2:flow_data columns cross_index,time_index,date_index,
0-8:正常情况下东西左转,东西直行,南北左转,南北直行,东放行,西放行,南放行,北放行.
不同路口有差异
"""
# drop row 1 for each date
drop_rows = date_time.drop_duplicates('date').index
flow_data = np.delete(flow_data, drop_rows, 1)
stage_data = np.delete(stage_data, drop_rows, 1)
action_data = np.delete(action_data, drop_rows, 1)

status_data = np.concatenate((flow_data, stage_data), axis=2)


def occupancy(s, cross):
    """
    eg:s = status_data[:, :1500],cross=2
return delay,weight
    """
    if s.ndim == 1:
        s = s.reshape(1, s.shape[0])

    occupancy = np.zeros((s.shape[0], 16))
    for n in range(16):
        if stage_map[cross_index[cross]][17+n] is not None:
                occupancy[:, n] = s[:, 16 + n] / np.sum(s[:, stage_map[cross_index[cross]][17+n]], axis=1)

    occupancy[occupancy == np.inf] = 0
    occupancy[np.isnan(occupancy)] = 0
    occupancy = np.sum(occupancy, axis=1)
    return occupancy.T


training_data = status_data[:, :5151]
max_occ = [np.percentile(occupancy(training_data[i], i), 97) for i in range(len(cross_index))]


def reward(occ_data, cross):
    """
    input delay_data
    output reward by sigmod func
    """
    x = np.abs(occ_data - max_occ[cross]*0.9)
    return 1/(1+np.e**x)


reward_data=np.zeros((len(cross_index), 5229))
for i in range(reward_data.shape[0]):
    reward_data[i] = reward(occupancy(status_data[cross], cross), cross)


actions = [np.unique(action_data[i], axis=0) for i in range(len(cross_index))]
for ix in range(len(actions)):
    action = actions[ix]
    actions[ix] = action[np.abs(np.sum(action, axis=1)) < 100]  # remove actions change too much
    actions[ix] = np.row_stack((actions[ix], np.zeros(8)))


def max_reward_action(status):
    best_action = np.zeros((status.shape[0], status.shape[1], 8))
    best_reward = np.zeros((status.shape[0], status.shape[1]))
    for cross in range(status.shape[0]):
        num = 0
        cross_data = status[cross]
        for data in cross_data:
            temp = np.tile(data, actions[cross].shape[0]).reshape((actions[cross].shape[0], data.shape[0]))
            temp[:, -8:] = temp[:, -8:] + actions[cross]

            temp = temp[(temp[:, -8:] >= data[-8:]-20).all(axis=1)]  # remove too little stage_value
            temp = temp[(temp[:, -8:] <= data[-8:]+20).all(axis=1)]  # remove too large stage_value
            temp = temp[(temp[:, -8:] >= 0).all(axis=1)]
            temp = temp[np.sum(temp[:, -8:], axis=1) > data[-8:].sum() - 50]  # remove too little cycle_value
            temp = temp[np.sum(temp[:, -8:], axis=1) < data[-8:].sum() + 50]  # remove too large cycle_value

            R = reward(occupancy(temp, cross), cross)
            best_action[cross, num] = temp[R.argmax(), -8:] - data[-8:]
            best_reward[cross, num] = R.max()
            num += 1
    return best_action, best_reward


def area_reward(cross_reward):
    return np.sum(cross_reward, axis=0)


a, r = max_reward_action(status_data[:, :5151])


def plot_best_reward(cross,day):
    import matplotlib.pyplot as plt
    print "-- represent improved stage"
    print "f1 cross reward"
    plt.plot(reward(occupancy(status_data[cross, 128*day:128*(day+1)], cross), cross))
    plt.plot(r[cross, 128*day:128*(day+1)],'--')
    plt.show()
    print "f2 cycle "
    plt.plot(np.sum(status_data[cross, 128*day:128*(day+1), -8:] + a[cross, 128*day:128*(day+1)], axis=1), '--')
    plt.plot(np.sum(status_data[cross, 128*day:128*(day+1), -8:], axis=1))
    plt.show()
    print "f3 area score"
    plt.plot(area_reward(r[:,128*day:128*(day+1)]),'--')
    plt.plot(area_reward(reward_data[:,128*day:128*(day+1)]))
    plt.show()


flow_data.dump('./data/flow_data')
stage_data.dump('./data/stage_data')
status_data.dump('./data/status_data')
reward_data.dump('./data/reward_data')

flow_next_data.dump('./data/flow_next_data')
stage_next_data.dump('./data/stage_next_data')
status_next_data.dump('./data/status_next_data')
reward_next_data.dump('./data/reward_next_data')

action_data.dump('./data/action_data')



def max_reward_action(status):
    best_action = np.zeros((status.shape[0], status.shape[1], 8))
    best_reward = np.zeros((status.shape[0], status.shape[1]))
    for cross in range(status.shape[0]):
        num = 0
        cross_data = status[cross]
        for data in cross_data:
            temp = np.tile(data, actions[cross].shape[0]).reshape((actions[cross].shape[0], data.shape[0]))
            temp[:, -8:] = temp[:, -8:] + actions[cross]
            temp = temp[(temp[:, -8:] >= -200).all(axis=1)]  # remove too little stage_value
            temp = temp[(temp[:, -8:] <= 200).all(axis=1)]  # remove too large stage_value
            temp = temp[np.sum(temp[:, -8:], axis=1) > 300]  # remove too little cycle_value
            temp = temp[np.sum(temp[:, -8:], axis=1) < 420]  # remove too large cycle_value

            R = reward(occupancy(temp, cross), cross)
            best_action[cross,num] = temp[R.argmax(), -8:] - data[-8:]
            best_reward[cross,num] = R.max()
            num += 1
    return best_action, best_reward