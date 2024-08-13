# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import argparse
import collections
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space
from gym.wrappers import RecordVideo, TimeLimit
from torch.utils.tensorboard import SummaryWriter

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"


def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b


class ProcessObsInputEnv(gym.ObservationWrapper):
    """
    This wrapper handles inputs from `Discrete` and `Box` observation space.
    If the `env.observation_space` is of `Discrete` type,
    it returns the one-hot encoding of the state
    """

    def __init__(self, env):
        super().__init__(env)
        self.n = None
        if isinstance(self.env.observation_space, Discrete):
            self.n = self.env.observation_space.n
            self.observation_space = Box(0, 1, (self.n,))

    def observation(self, obs):
        if self.n:
            return one_hot(np.array(obs), self.n)
        return obs


# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst), np.array(done_mask_lst)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, env.action_space.n)

    def forward(self, x, device=DEVICE):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetworkMARL(nn.Module):
    def __init__(self, env):
        super(QNetworkMARL, self).__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space[0].shape).prod(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, env.action_space[0].n)

    def forward(self, x, device=DEVICE):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
