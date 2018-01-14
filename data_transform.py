# -*- coding:utf-8 -*-
import numpy as np

import pandas as pd


np.seterr(divide='ignore', invalid='ignore')
# read data from csv
date_parser = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M')
parse_dates = ['curr_time']
flow_data_read = pd.read_csv('/Users/pansmac/Desktop/ali/flow_origin.csv', encoding='gb2312', header=0,
                             date_parser=date_parser, parse_dates=parse_dates)
stage_data_read = pd.read_csv('/Users/pansmac/Desktop/ali/stage_origin.csv', date_parser=date_parser,
                              parse_dates=['阶段开始时刻']).values[:, [1, 3, 4]]

stage_data_read = pd.DataFrame(stage_data_read)
stage_data_read.columns = ['stage_no', 'value', 'curr_time']
'''
1	东西直行
2	东西左转
3	南北直行
4	南北左转
'''
# transform flows into inputs or outputs
flows_index = [u'north_north_flow', u'west_north_flow', u'south_north_flow', u'east_north_flow',
               u'north_west_flow', u'west_west_flow', u'south_west_flow', u'east_west_flow',
               u'north_south_flow', u'west_south_flow', u'south_south_flow', u'east_south_flow',
               u'north_east_flow', u'west_east_flow', u'south_east_flow', u'east_east_flow']
cross_index = list(flow_data_read.cross_name.unique())
flow_map = {
        'north': {'input': flows_index[8:12], 'output': [flows_index[i] for i in [0, 4, 8, 12]]},
        'west': {'input': flows_index[12:16], 'output': [flows_index[i] for i in [1, 5, 9, 13]]},
        'south': {'input': flows_index[0:4], 'output': [flows_index[i] for i in [2, 6, 10, 14]]},
        'east': {'input': flows_index[4:8], 'output': [flows_index[i] for i in [3, 7, 11, 15]]}
            }
cross_map = {
    u'华池街-苏州大道东': {
        'north': u'华池街-现代大道', 'west': u'苏州大道东-月廊街', 'south': u'华池街-旺墩路', 'east': u'星湖街-苏州大道东'}}

stage_map = {
    '1': [3, 4, 9, 14, 7, 13], '2': [3, 4, 9, 14, 1, 11, 5, 15], '3': [3, 4, 9, 14, 2, 8],
    '4': [3, 4, 9, 14, 0, 12, 10, 6]
}
#

flow_data_read['date'] = flow_data_read['curr_time'].apply(lambda x: x.date())
date_time = flow_data_read[['date', 'curr_time']].drop_duplicates(['curr_time'])
date_index = flow_data_read['date'].unique()
time_index = flow_data_read['curr_time'].unique()
flow_data = np.zeros(36)
stage_data = np.zeros((time_index.shape[0]-1, 4))

for cross in range(len(cross_index)):
    cross_flow_data = np.zeros((len(time_index), 36))
    cross_flow_data[:, 0] = cross
    cross_flow_data[:, 1] = np.arange(time_index.shape[0])
    cross_flow_data[:, 2] = date_time['date'].map(dict(zip(date_index, range(len(date_index)))))
    if cross_map[cross_index[cross]]['north'] is not None:
        cross_flow_data[:, 3:7] = flow_data_read[flow_data_read['cross_name'] == cross_map[cross_index[cross]]['north']][flow_map['north']['input']]
    if cross_map[cross_index[cross]]['west'] is not None:
        cross_flow_data[:, 7:11] = flow_data_read[flow_data_read['cross_name'] == cross_map[cross_index[cross]]['west']][flow_map['west']['input']]
    if cross_map[cross_index[cross]]['south'] is not None:
        cross_flow_data[:, 11:15] = flow_data_read[flow_data_read['cross_name'] == cross_map[cross_index[cross]]['south']][flow_map['south']['input']]
    if cross_map[cross_index[cross]]['east'] is not None:
        cross_flow_data[:, 15:19] = flow_data_read[flow_data_read['cross_name'] == cross_map[cross_index[cross]]['east']][flow_map['east']['input']]

    cross_flow_data[:, 19:23] = flow_data_read[flow_data_read['cross_name'] == cross_index[cross]][flow_map['north']['output']]
    cross_flow_data[:, 23:27] = flow_data_read[flow_data_read['cross_name'] == cross_index[cross]][flow_map['west']['output']]
    cross_flow_data[:, 27:31] = flow_data_read[flow_data_read['cross_name'] == cross_index[cross]][flow_map['south']['output']]
    cross_flow_data[:, 31:35] = flow_data_read[flow_data_read['cross_name'] == cross_index[cross]][flow_map['east']['output']]

    flow_data = np.row_stack((flow_data, cross_flow_data))

flow_data = flow_data[1:]

for row in range(time_index.shape[0] - 1):
    time_stage = (stage_data_read.curr_time >= time_index[row]) & (stage_data_read.curr_time < time_index[row+1])

    stage_data[row, 0] = stage_data_read[time_stage & (stage_data_read.stage_no == 1)].value.sum()
    stage_data[row, 1] = stage_data_read[time_stage & (stage_data_read.stage_no == 2)].value.sum()
    stage_data[row, 2] = stage_data_read[time_stage & (stage_data_read.stage_no == 3)].value.sum()
    stage_data[row, 3] = stage_data_read[time_stage & (stage_data_read.stage_no == 4)].value.sum()

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
