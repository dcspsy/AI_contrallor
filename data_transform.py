# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
# read data from csv
date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
parse_dates = ['curr_time']
flow_data_read = pd.read_csv('origin_data/flow_origin_data.csv', encoding='gb2312', header=0,
                             date_parser=date_parser, parse_dates=parse_dates)

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
        '1': [3, 4, 9, 14, 7, 13], '2': [3, 4, 9, 14, 1, 11, 5, 15], '3': [3, 4, 9, 14, 2, 8],
        '4': [3, 4, 9, 14, 0, 12, 10, 6]},
    u'观枫街-现代大道': {

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

flow_data_read['date'] = flow_data_read['curr_time'].apply(lambda x: x.date())


date_time = flow_data_read[['date', 'curr_time']].drop_duplicates(['curr_time'])
date_index = flow_data_read['date'].unique()
time_index = flow_data_read['curr_time'].unique()
flow_data = np.zeros(35)

for cross in range(len(cross_index)):
    cross_flow_data = np.zeros((len(time_index), 35))
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
"""
----index
0-2:flow_data columns cross_index,time_index,date_index,
----input
3-6:north_south_flow,west_south_flow,south_south_flow,east_south_flow
7-10:north_east_flow,west_east_flow,south_east_flow,east_east_flow
11-14:north_north_flow,west_north_flow,south_north_flow,east_north_flow
15-18:north_west_flow, west_west_flow, south_west_flow, east_west_flow
----output
19-22:north_north_flow,north_west_flow,north_south_flow,north_east_flow
23-26:west_north_flow,west_west_flow,west_south_flow,west_east_flow
27-30:south_north_flow,south_west_flow,south_south_flow,south_east_flow
31-34:east_north_flow,east_west_flow,east_south_flow,east_east_flow

"""
stage_map = {
    u'华池街-苏州大道东': {
        1: [20, 25, 30, 31, 26, 32], 2: [20, 25, 30, 31, 23, 24, 33, 34], 3: [20, 25, 30, 31, 21, 27],
        4: [20, 25, 30, 31, 22, 19, 28, 29], 7: [20, 25, 30, 31, 27, 28, 29], 8: [20, 25, 30, 31, 19, 21, 22]},
    u'观枫街-现代大道': {
        1: [20, 25, 30, 31, 26, 32], 2: [20, 25, 30, 31, 23, 24, 33, 34], 6: [20, 25, 30, 31, 27, 28, 29]},
    u'华池街-现代大道': {
        1: [20, 25, 30, 31, 26, 32], 2: [20, 25, 30, 31, 23, 24, 33, 34], 3: [20, 25, 30, 31, 21, 27],
        4: [20, 25, 30, 31, 22, 19, 28, 29], 7: [20, 25, 30, 31, 27, 28, 29], 8: [20, 25, 30, 31, 19, 21, 22]},
    u'观枫街-苏州大道东': {
        1: [20, 25, 30, 31, 26, 32], 2: [20, 25, 30, 31, 23, 24, 33, 34], 5: [20, 25, 30, 31, 32, 33, 34]},
    u'华池街-旺墩路': {
        3: [20, 25, 30, 31, 21, 27], 4: [20, 25, 30, 31, 22, 19, 28, 29],
        5: [20, 25, 30, 31, 32, 33, 34, 27, 28, 29], 8: [20, 25, 30, 31, 19, 21, 22]},
    u'星湖街-苏州大道东':{
        1: [20, 25, 30, 31, 32, 33, 34, 27, 28, 29], 3: [20, 25, 30, 31, 21, 27],
        4: [20, 25, 30, 31, 22, 19, 28, 29], 5: [20, 25, 30, 31, 32, 33, 34], 6: [20, 25, 30, 31, 27, 28, 29]}
}


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
stage_data = stage_data[:, :-2]



# def reward(s):
#     r = sum((s[:, 1+16], s[:, 6+16], s[:, 7+16], s[:, 11+16], s[:, 12+16], s[:, 13+16]))/s[:, 32] + \
#     sum((s[:, 1+16], s[:, 4+16], s[:, 5+16], s[:, 6+16], s[:, 11+16], s[:, 12+16], s[:, 14+16], s[:, 15+16]))/s[:, 33]+\
#     sum((s[:, 1+16], s[:, 2+16], s[:, 6+16], s[:, 8+16], s[:, 11+16], s[:, 12+16])) / s[:, 34] + \
#     sum((s[:, 0+16], s[:, 1+16], s[:, 3+16], s[:, 6+16], s[:, 9+16], s[:, 10+16], s[:, 11+16], s[:, 12+16])) / s[:, 35]
#     return r


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
