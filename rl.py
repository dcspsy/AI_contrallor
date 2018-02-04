import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

MSELoss = nn.MSELoss()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, num_output):
        super(Actor, self).__init__()
        num_outputs = num_output

        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        mu = F.tanh(self.mu(x))
        return mu


class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, num_output):
        super(Critic, self).__init__()
        num_outputs = num_output
        self.bn0 = nn.BatchNorm1d(num_inputs)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.fill_(0)

        self.linear_action = nn.Linear(num_outputs, hidden_size)
        self.bn_a = nn.BatchNorm1d(hidden_size)
        self.bn_a.weight.data.fill_(1)
        self.bn_a.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        a = F.tanh(self.linear_action(actions))
        x = torch.cat((x, a), 1)
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        return V


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (4, 9, 9)
            nn.Conv2d(
                in_channels=4,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),  # output shape (16, 9, 9)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3),  # output shape (16, 3, 3)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 3, 3)
            nn.Conv2d(16, 32, 3),  # output shape (32, 1, 1)
            nn.ReLU(),  # activation
        )
        self.out = nn.Linear(32, 1)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # (batch_size, 32)
        output = self.out(x)
        return output


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.feature = Feature()

        self.actor = Actor(hidden_size, self.num_inputs, self.num_outputs)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.num_outputs)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.num_outputs)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.num_outputs)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def select_action(self, flow, stage, exploration=None):
        self.actor.eval()
        flow = Variable(torch.cat(flow))
        stage = torch.cat(stage)
        state = torch.cat((self.feature(flow).data, stage), dim=1)
        mu = self.actor((Variable(state, volatile=True)))
        self.actor.train()
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise())

        return mu.clamp(-1, 1)


    def update_parameters(self, batch):

        flow_batch = Variable(torch.cat(batch.flow))
        next_flow_batch = Variable(torch.cat(batch.next_flow))
        stage_batch = torch.cat(batch.stage)
        next_stage_batch = torch.cat(batch.next_stage)

        feature_bath = self.feature(flow_batch)
        next_feature_bath = self.feature(next_flow_batch)

        # state_batch = Variable(torch.cat(batch.state))
        state_batch = Variable(torch.cat((feature_bath.data, stage_batch), dim=1))
        # next_state_batch = Variable(torch.cat(batch.next_state), volatile=True)
        next_state_batch = Variable(torch.cat((next_feature_bath.data, next_stage_batch), dim=1), volatile=True)

        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        # mask_batch = Variable(torch.cat(batch.mask))

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = torch.unsqueeze(reward_batch, 1)
        expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = MSELoss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch), self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


    def store_transition(self, s, a, r, s_):
        pass





