import numpy as np
from rl import Feature


flow_data = np.load('/Users/pansmac/PycharmProjects/AI_contrallor/data/flow_data')
stage_data = np.load('/Users/pansmac/PycharmProjects/AI_contrallor/data/stage_data')


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


class TrafficEnv(object):
    action_bound = [-200, 200]
    flow_space = np.zeros((4, 9, 9))
    action_space = np.zeros(8)
    feature_net = Feature()

    def __init__(self):
        self.traffic_flow = np.zeros((4, 9, 9))
        self.traffic_stage = np.zeros(8)
        self.action_space = np.zeros(8)
        self.r =np.zeros(1)
        self.mask = None
    def step(self, action):

        self.traffic_flow = flow_data[np.random.choice(range(flow_data.shape[0]))]
        done = False

        action = np.clip(action, *self.action_bound)

        self.traffic_flow = flow_data[np.random.choice(range(flow_data.shape[0]))]
        self.traffic_stage += action
        self.r = reward(self.traffic_flow, self.traffic_stage)

        flow = self.traffic_flow
        stage = self.traffic_stage
        r = self.r
        mask = self.mask

        return flow, stage, r, mask



    def render(self):
        pass


if __name__ == '__main__':
    env = TrafficEnv()
    for i in range(5):
        _, r, _ = env.step(env.sample_ation())
        print r



