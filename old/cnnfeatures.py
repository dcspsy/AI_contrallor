# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
# read data from csv
date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
parse_dates = ['curr_time']
flow_data_read = pd.read_csv('origin_data/flow_origin_data.csv', encoding='gb2312', header=0,
                             date_parser=date_parser, parse_dates=parse_dates)

time_index = flow_data_read['curr_time'].unique()


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


"""
[0:4,0,1]->u'north_north_flow', u'west_north_flow', u'south_north_flow', u'east_north_flow'
[0:4,1,0]->u'north_west_flow', u'west_west_flow', u'south_west_flow', u'east_west_flow'
[0:4,2,1]->u'north_south_flow', u'west_south_flow', u'south_south_flow', u'east_south_flow'
[0:4,1,2]->u'north_east_flow', u'west_east_flow', u'south_east_flow', u'east_east_flow'
"""
def cube334(cross):
    data = np.zeros((len(time_index), 4, 3, 3))
    data[:, :, 0, 1] = flow_data_read[flow_data_read.cross_name == cross][
        [u'north_north_flow', u'west_north_flow', u'south_north_flow', u'east_north_flow']].values  # north
    data[:, :, 1, 0] = flow_data_read[flow_data_read.cross_name == cross][
        [u'north_west_flow', u'west_west_flow', u'south_west_flow', u'east_west_flow']].values  # west
    data[:, :, 2, 1] = flow_data_read[flow_data_read.cross_name == cross][
        [u'north_south_flow', u'west_south_flow', u'south_south_flow', u'east_south_flow']].values  # south
    data[:, :, 1, 2] = flow_data_read[flow_data_read.cross_name == cross][
        [u'north_east_flow', u'west_east_flow', u'south_east_flow', u'east_east_flow']].values  # east

    return data


flow_data = np.zeros((len(time_index), 4, 9, 9))
flow_data[:, :, 0:3, 0:3] = cube334(cross_index[5])
flow_data[:, :, 3:6, 0:3] = cube334(cross_index[0])
flow_data[:, :, 0:3, 3:6] = cube334(cross_index[4])
flow_data[:, :, 3:6, 3:6] = cube334(cross_index[3])
flow_data[:, :, 6:9, 3:6] = cube334(cross_index[1])
flow_data[:, :, 3:6, 6:9] = cube334(cross_index[2]) # flow data


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

stage_data = stage_data[3, :, :-2]  # stage_data

action_data = np.zeros((len(time_index), 8))
action_data[:-1] = stage_data[1:]-stage_data[:-1]  # action data


def occupancy(flow, stage):
    """
    stage_no:stage_name
    0:east-west straight
    1:east-west left trun
    2:south-north straight
    3:south-north left turn
    4:east go
    5:west go
    6:south go
    7:north go
    ----
    flow.index[0] ==stage.index[0]
    """

    if flow.ndim == 3:
        flow = flow.reshape(1, flow.shape[0], flow.shape[1], flow.shape[2])
    if stage.ndim == 1:
        stage = stage.reshape(1, stage.shape[0])


    occ = np.zeros((flow.shape[0], 16))

    stage_map = [[3, 7], [1, 5], [2, 6], [0, 1, 2, 3, 4, 5, 6, 7],
                 [0, 1, 2, 3, 4, 5, 6, 7], [1, 5], [3, 6], [0, 4],
                 [2, 7], [0, 1, 2, 3, 4, 5, 6, 7], [3, 6], [1, 4],
                 [1, 7], [0, 5], [0, 1, 2, 3, 4, 5, 6, 7], [1, 4]
                 ]
    occ[:, 0] = flow[:, 0, 0, 1] / np.sum(stage[:, stage_map[0]], axis=1)
    occ[:, 1] = flow[:, 1, 0, 1] / np.sum(stage[:, stage_map[1]], axis=1)
    occ[:, 2] = flow[:, 2, 0, 1] / np.sum(stage[:, stage_map[2]], axis=1)
    occ[:, 3] = flow[:, 3, 0, 1] / np.sum(stage[:, stage_map[3]], axis=1)
    occ[:, 4] = flow[:, 0, 1, 0] / np.sum(stage[:, stage_map[4]], axis=1)
    occ[:, 5] = flow[:, 1, 1, 0] / np.sum(stage[:, stage_map[5]], axis=1)
    occ[:, 6] = flow[:, 2, 1, 0] / np.sum(stage[:, stage_map[6]], axis=1)
    occ[:, 7] = flow[:, 3, 1, 0] / np.sum(stage[:, stage_map[7]], axis=1)
    occ[:, 8] = flow[:, 0, 2, 1] / np.sum(stage[:, stage_map[8]], axis=1)
    occ[:, 9] = flow[:, 1, 2, 1] / np.sum(stage[:, stage_map[9]], axis=1)
    occ[:, 10] = flow[:, 2, 2, 1] / np.sum(stage[:, stage_map[10]], axis=1)
    occ[:, 11] = flow[:, 3, 2, 1] / np.sum(stage[:, stage_map[11]], axis=1)
    occ[:, 12] = flow[:, 0, 1, 2] / np.sum(stage[:, stage_map[12]], axis=1)
    occ[:, 13] = flow[:, 1, 1, 2] / np.sum(stage[:, stage_map[13]], axis=1)
    occ[:, 14] = flow[:, 2, 1, 2] / np.sum(stage[:, stage_map[14]], axis=1)
    occ[:, 15] = flow[:, 3, 1, 2] / np.sum(stage[:, stage_map[15]], axis=1)

    occ[occ == np.inf] = 0
    occ[np.isnan(occ)] = 0

    return occ


def reward(flow, stage):
    occ = occupancy(flow, stage)
    occ_drop_turn_right_flow = occ[:, [0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15]]
    return np.max(occ_drop_turn_right_flow, axis=1)


reward_data = reward(flow_data, stage_data)  # reward data


def drop_error_data(data):
    drop_err = (stage_data <= 480).any(axis=1)
    return data[drop_err]


drop_error_data(flow_data).dump('./data/flow_data')
drop_error_data(stage_data).dump('./data/stage_data')
drop_error_data(reward_data).dump('./data/reward_data')
drop_error_data(action_data).dump('./data/action_data')