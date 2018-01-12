# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
# read data from csv
date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
parse_dates = ['curr_time']
flow_data_read = pd.read_csv('./origin_data/flow_origin_data.csv', encoding='gb2312', header=0,
                             date_parser=date_parser, parse_dates=parse_dates)

flows_index = [u'north_north_flow', u'west_north_flow', u'south_north_flow', u'east_north_flow',
               u'north_west_flow', u'west_west_flow', u'south_west_flow', u'east_west_flow',
               u'north_south_flow', u'west_south_flow', u'south_south_flow', u'east_south_flow',
               u'north_east_flow', u'west_east_flow', u'south_east_flow', u'east_east_flow']
cross_index = [u'华池街-苏州大道东']
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
    u'华池街-苏州大道东': {
        '1': [3, 4, 9, 14, 7, 13], '2': [3, 4, 9, 14, 1, 11, 5, 15], '3': [3, 4, 9, 14, 2, 8],
        '4': [3, 4, 9, 14, 0, 12, 10, 6]}}


def split_by_date(data, time_col='curr_time'):
    """
    :return: dict of each date data. key is time_col value's date
    """
    data_date = data[time_col].apply(lambda x: x.date())
    date_index = data_date.unique()

    return {index: data[data_date == index] for index in date_index}


# data_by_date = split_by_date(flow_data_read)

def transform_flow_data(data_all, time_col='curr_time', cross_index=cross_index):
    date_cross_flow = {}
    for date in data_all.keys():
        data = data_all[date]
        time_index = data[time_col].unique()
        flow = np.zeros((time_index.shape[0] - 1, 32))

        cross_flow = {}
        for cross in cross_index:
            for row in range(time_index.shape[0]-1):
                time = (data.curr_time >= time_index[row]) & (data.curr_time < time_index[row+1])
                is_cross = data.cross_name == cross_map[cross]['north']
                flow[row, :4] = data[time & is_cross][flow_map['north']['input']]
                is_cross = data.cross_name == cross_map[cross]['west']
                flow[row, 4:8] = data[time & is_cross][flow_map['west']['input']]
                is_cross = data.cross_name == cross_map[cross]['south']
                flow[row, 8:12] = data[time & is_cross][flow_map['south']['input']]
                is_cross = data.cross_name == cross_map[cross]['east']
                flow[row, 12:16] = data[time & is_cross][flow_map['east']['input']]
                is_cross = data.cross_name == cross
                flow[row, 16:20] = data[time & is_cross][flow_map['north']['output']]
                is_cross = data.cross_name == cross
                flow[row, 20:24] = data[time & is_cross][flow_map['west']['output']]
                is_cross = data.cross_name == cross
                flow[row, 24:28] = data[time & is_cross][flow_map['south']['output']]
                is_cross = data.cross_name == cross
                flow[row, 28:32] = data[time & is_cross][flow_map['east']['output']]

                cross_flow[cross] = flow

        date_cross_flow[date] = cross_flow
    return date_cross_flow


flow_data = transform_flow_data(split_by_date(flow_data_read))
# stage_data = np.zeros((time_index.shape[0]-1, 4))



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
