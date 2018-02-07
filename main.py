
from env import TrafficEnv
from rl import DDPG
from replay_memory import ReplayMemory, Transition
import numpy as np
import torch


MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

env = TrafficEnv()
stats_dim = 9
action_dim = 8
a_bound = env.action_bound

agent = DDPG(GAMMA, TAU, hidden_size=100, num_inputs=9, num_outputs=8)
memory = ReplayMemory(MEMORY_CAPACITY)
# for i in range(MAX_EPISODES):
#     flow, stage = env.reset()
#     for j in range(MAX_EP_STEPS):
#         action = agent.select_action(flow, stage)
#         flow_, stage_, reward, done = env.step(action)
#
#         memory.push(flow, stage, action, flow_, stage_, reward)
#
#         if len(memory) > 5*BATCH_SIZE:
#             for _ in range(MAX_EP_STEPS):
#                 transitions = memory.sample(BATCH_SIZE)
#                 batch = Transition(*zip(*transitions))
#
#                 agent.update_parameters(batch)
#
#         flow, stage = flow_, stage_


def learn_history():
    flow_data = np.load('data/flow_data')
    stage_data = np.load('data/stage_data')
    action_data = np.load('data/action_data')
    action_data = action_data/200.
    reward_data = np.load('data/reward_data')
    for i in range(5000):
        memory.push(torch.Tensor(flow_data[i:i + 1]), torch.Tensor(stage_data[i:i + 1]),
                    torch.Tensor(action_data[i:i + 1]),
                    None,
                    torch.Tensor(flow_data[i + 1:i + 2]),
                    torch.Tensor(stage_data[i + 1:i + 2]),
                    torch.Tensor(reward_data[i:i + 1]))

    for _ in range(10000):
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        agent.update_parameters(batch)
        if _ % 100 == 0:
            print 'loss',agent.value_loss.data


learn_history()
transitions = memory.sample(5)
batch = Transition(*zip(*transitions))
print batch.action
print agent.select_action(batch.flow,batch.stage)


