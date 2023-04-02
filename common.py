""" Common stuff for search strategies and RL algorithms """

import copy
import math
import random
import statistics
from collections import namedtuple, OrderedDict
from typing import List
import numpy as np
import sys
import pprint
from enum import Enum

#ml dependencies
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#compiler_gym dependency:
import compiler_gym
from compiler_gym.wrappers import TimeLimit, ConstrainedCommandline

#cross-cot deps
from rewards import const_factor_threshold
from action_spaces_presets import *
from logger import LogMode, Logger
from search_policies import *


class Runmode(Enum):
    GREEDY = 1
    RANDOM = 2
    RANDOM_POSITIVES = 3
    RANDOM_POSITIVES_USED = 4
    AC_BASIC = 5


flags = {}
flags.update({"episode_len": 15})  #"Number of transitions per episode."
flags.update({"hidden_size": 64})  #"Latent vector size."
flags.update({"log_interval": 100})  #"Episodes per log output."
flags.update({"iterations": 5})  #"Times to redo entire training."
flags.update({"exploration": 0.0})  #"Rate to explore random transitions."
flags.update({"mean_smoothing": 0.95})  #"Smoothing factor for mean normalization."
flags.update({"std_smoothing": 0.4})  #"Smoothing factor for std dev normalization."
flags.update({"learning_rate": 0.008})
flags.update({"episodes": 1000})
flags.update({"seed": 0})
flags.update({"log_mode": LogMode.SHORT})
flags.update({"logger": Logger(flags["log_mode"])})
flags.update({"actions_white_list": None}) # by default (if None), all actions from any action space are possible
flags.update({"patience": 5})  # patience for steps with no positive reward
flags.update({"search_iterations": 2})  # for non-RL search it is required dramatically more steps
FLAGS = flags

eps = np.finfo(np.float32).eps.item()

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


# === statistical values -- move to algorithms.statistical
class MovingExponentialAverage:
    """Simple class to calculate exponential moving averages."""

    def __init__(self, smoothing_factor):
        self.smoothing_factor = smoothing_factor
        self.value = None

    def next(self, entry):
        assert entry is not None
        if self.value is None:
            self.value = entry
        else:
            self.value = (
                entry * (1 - self.smoothing_factor) + self.value * self.smoothing_factor
            )
        return self.value


def remap_actions(awl):
    d = OrderedDict()
    for i, item in enumerate(awl):
        d[item] = i
    return d