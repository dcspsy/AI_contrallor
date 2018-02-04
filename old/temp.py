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

# index
date_time = flow_data_read[['date', 'curr_time']].drop_duplicates(['curr_time']).reset_index()
date_index = flow_data_read['date'].unique()
time_index = flow_data_read['curr_time'].unique()

flows_index = [u'north_north_flow', u'west_north_flow', u'south_north_flow', u'east_north_flow',
               u'north_west_flow', u'west_west_flow', u'south_west_flow', u'east_west_flow',
               u'north_south_flow', u'west_south_flow', u'south_south_flow', u'east_south_flow',
               u'north_east_flow', u'west_east_flow', u'south_east_flow', u'east_east_flow']
cross_index = [u'观枫街-苏州大道东', u'华池街-旺墩路', u'星湖街-苏州大道东', u'华池街-苏州大道东', u'华池街-现代大道',
               u'观枫街-现代大道']

# map
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
        '1': [3, 4, 9, 14, 7, 13], '2': [3, 4, 9, 14, 1, 11, 5, 15], '3': [3, 4, 9, 14, 2, 8],
        '4': [3, 4, 9, 14, 0, 12, 10, 6]}}

# transform flow data
flow_data = np.zeros((len(date_index), len(cross_index), len(time_index), 32))
aa = flow_data_read.groupby(['date', 'cross_name', 'curr_time']).mean()
for date in range(len(date_index)):
    for cross in range(len(cross_index)):
        for time in date_time[date_time['date'] == date_index[date]].index:
            if cross_map[cross_index[cross]]['north'] is not None:
                flow_data[date, cross, time][0:4] = \
                    aa.loc[date_index[date], cross_map[cross_index[cross]]['north'], time_index[time]][
                        flow_map['north']['input']]
            if cross_map[cross_index[cross]]['west'] is not None:
                flow_data[date, cross, time][4:8] = \
                    aa.loc[date_index[date], cross_map[cross_index[cross]]['west'], time_index[time]][
                        flow_map['west']['input']]
            if cross_map[cross_index[cross]]['south'] is not None:
                flow_data[date, cross, time][8:12] = \
                    aa.loc[date_index[date], cross_map[cross_index[cross]]['south'], time_index[time]][
                        flow_map['south']['input']]
            if cross_map[cross_index[cross]]['east'] is not None:
                flow_data[date, cross, time][12:16] = \
                    aa.loc[date_index[date], cross_map[cross_index[cross]]['east'], time_index[time]][
                        flow_map['east']['input']]
                flow_data[date, cross, time][16:20] = \
                    aa.loc[date_index[date], cross_index[cross],time_index[time]][flow_map['north']['output']]
                flow_data[date, cross, time][16:20] = \
                    aa.loc[date_index[date], cross_index[cross],time_index[time]][flow_map['west']['output']]
                flow_data[date, cross, time][16:20] = \
                    aa.loc[date_index[date], cross_index[cross],time_index[time]][flow_map['south']['output']]
                flow_data[date, cross, time][16:20] = \
                    aa.loc[date_index[date], cross_index[cross],time_index[time]][flow_map['east']['output']]

# transform stage data

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
stage_data_read = pd.pivot_table(stage_data_read, index=['cross_name', 'stage_time'], columns=['stage_sn'], values=['val'],
                            aggfunc=[np.sum], fill_value=0)

# merge
stage_data = np.zeros(13)
for cross in range(len(cross_index)):
    cross_stage_data = np.zeros((len(time_index), 13))
    cross_stage_data[:, 0] = cross
    cross_stage_data[:, 1] = np.arange(time_index.shape[0])
    cross_stage_data[:, 2] = date_time['date'].map(dict(zip(date_index, range(len(date_index)))))
    cross_stage_data[1:, 3:] = stage_data_read.loc[cross, :]
    stage_data = np.row_stack((stage_data, cross_stage_data))

stage_data = stage_data[1:]
print flow_data.shape, stage_data.shape

status_data = np.concatenate((flow_data, stage_data), axis=1)



def reward(s):
    r = sum((s[:, 1+16], s[:, 6+16], s[:, 7+16], s[:, 11+16], s[:, 12+16], s[:, 13+16]))/s[:, 32] + \
    sum((s[:, 1+16], s[:, 4+16], s[:, 5+16], s[:, 6+16], s[:, 11+16], s[:, 12+16], s[:, 14+16], s[:, 15+16]))/s[:, 33]+\
    sum((s[:, 1+16], s[:, 2+16], s[:, 6+16], s[:, 8+16], s[:, 11+16], s[:, 12+16])) / s[:,34] + \
    sum((s[:, 0+16], s[:, 1+16], s[:, 3+16], s[:, 6+16], s[:, 9+16], s[:, 10+16], s[:, 11+16], s[:, 12+16])) / s[:, 35]
    return r


stage_next_data = np.zeros((time_index.shape[0]-1, 4))
flow_next_data = np.zeros((time_index.shape[0]-1, 32))

stage_next_data[:-1] = stage_data[1:]
flow_next_data[:-1] = flow_data[1:]
status_next_data = np.concatenate((flow_next_data, stage_next_data), axis=1)

action_data = stage_next_data - stage_data


def drop_rows(data):  # drop error data
    save = date_index == np.append(date_index[1:], date_index[0])  # drop border data
    save[449:451] = False  # flow data error
    save[1062] = False  # stage data error
    return data[save]


flow_data = drop_rows(flow_data)
stage_data = drop_rows(stage_data)
status_data = drop_rows(status_data)
reward_data = reward(status_data)

flow_next_data = drop_rows(flow_next_data)
stage_next_data = drop_rows(stage_next_data)
status_next_data = drop_rows(status_next_data)
reward_next_data = reward(status_next_data)

action_data = drop_rows(action_data)


flow_data.dump('./data/flow_data')
stage_data.dump('./data/stage_data')
status_data.dump('./data/status_data')
reward_data.dump('./data/reward_data')

flow_next_data.dump('./data/flow_next_data')
stage_next_data.dump('./data/stage_next_data')
status_next_data.dump('./data/status_next_data')
reward_next_data.dump('./data/reward_next_data')

action_data.dump('./data/action_data')


def delay(s):
    """
    s[cross][time] eg:s = status_data[:2,:100]

    """
    if s.ndim == 1:
        s = s.reshape(1, s.shape[0])
    delay_time = np.zeros((s.shape[0], s.shape[1], 8))

    for c in range(s.shape[0]):
        for n in range(8):
            if stage_map[cross_index[c]][1+n] is not None:
                delay_time[c, :, n] = s[c, :, 32+n]/np.sum(s[c, :, stage_map[cross_index[c]][1+n]], axis=0)
    return delay_time


flow_data = np.zeros((len(cross_index), len(time_index), 32))
grouped = flow_data_read.groupby(['cross_name', 'curr_time']).mean()
for cross in range(len(cross_index)):
    if cross_map[cross_index[cross]]['north'] is not None:
        flow_data[cross][:, 0:4] = grouped.loc[cross_map[cross_index[cross]]['north']][flow_map['north']['input']]
    if cross_map[cross_index[cross]]['west'] is not None:
        flow_data[cross][:, 4:8] = grouped.loc[cross_map[cross_index[cross]]['west']][flow_map['west']['input']]
    if cross_map[cross_index[cross]]['south'] is not None:
        flow_data[cross][:, 8:12] = grouped.loc[cross_map[cross_index[cross]]['south']][flow_map['south']['input']]
    if cross_map[cross_index[cross]]['east'] is not None:
        flow_data[cross][:,12:16] = grouped.loc[cross_map[cross_index[cross]]['east']][flow_map['east']['input']]
    flow_data[cross][:,16:20] = grouped.loc[cross_index[cross]][flow_map['north']['output']]
    flow_data[cross][:,20:24] = grouped.loc[cross_index[cross]][flow_map['west']['output']]
    flow_data[cross][:,24:28] = grouped.loc[cross_index[cross]][flow_map['south']['output']]
    flow_data[cross][:,28:32] = grouped.loc[cross_index[cross]][flow_map['east']['output']]
            # flow data


stage_data = np.zeros((len(cross_index), len(time_index), 10))
for cross in range(len(cross_index)):
    stage_data[cross][1:] = stage_data_read.loc[cross]

'''
17-20:north_north_flow,north_west_flow,north_south_flow,north_east_flow
21-24:west_north_flow,west_west_flow,west_south_flow,west_east_flow
25-28:south_north_flow,south_west_flow,south_south_flow,south_east_flow
29-32:east_north_flow,east_west_flow,east_south_flow,east_east_flow
'''
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

s = status_data[:, :5151]
delay_time = np.zeros((s.shape[0], s.shape[1], 16))
for c in range(s.shape[0]):
    for n in range(16):
        if stage_map[cross_index[c]][17 + n] is not None:
            delay_time[c, :, n] = s[c, :, 16 + n] / np.sum(s[c, :, stage_map[cross_index[c]][17 + n]], axis=0)


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
            temp = temp[np.sum(temp[:, -8:], axis=1) > data[-8:].sum() - 50]  # remove too little cycle_value
            temp = temp[np.sum(temp[:, -8:], axis=1) < data[-8:].sum() + 50]  # remove too large cycle_value

            R = reward(occupancy(temp, cross), cross)
            best_action[cross, num] = temp[R.argmax(), -8:] - data[-8:]
            best_reward[cross, num] = R.max()
            num += 1
    return best_action, best_reward


a, r = max_reward_action(status_data[:, :5151])





