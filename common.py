""" Common stuff for search strategies and RL algorithms """

import copy
import math
import random
import statistics
import json
import csv
import sys
import os
import glob
import pprint
from collections import namedtuple, OrderedDict
from typing import List
import numpy as np
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
from logger import LogMode, Logger, printRed, printGreen, printYellow, printLightPurple
from search_policies import *


class Runmode(Enum):
    GREEDY = 1
    RANDOM = 2
    RANDOM_POSITIVES = 3
    RANDOM_POSITIVES_USED = 4
    LEAST_FROM_POSITIVES = 5
    GREATEST_FROM_POSITIVE_SAMPLES = 6
    LEAST_FROM_POSITIVE_SAMPLES = 7
    AC_BASIC = 8


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
flags.update({"search_iterations": 100})  # for non-RL search it is required dramatically more steps
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


def read_action_log_from_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
        return data


def get_json_files_list(directory="results_oz"):
    return glob.glob(directory + r'/*.json')


class HistoryObservation(gym.ObservationWrapper):
    """
    This wrapper implements very simple characterization space:
    For the input representation (state), if there are N possible
    actions, then an action x is represented by a one-hot vector V(x)
    with N entries. A sequence of M actions (x, y, ...) is represented
    by an MxN matrix of 1-hot vectors (V(x), V(y), ...). Actions that
    have not been taken yet are represented as the zero vector. This
    way the input does not have a variable size since each episode has
    a fixed number of actions.

    Also, it supports multi-observation set by the parameter "hetero_observations_names"
    """

    def __init__(self, env, hetero_observations_=OrderedDict(), primary_metrics="TextSizeBytes"):
        super().__init__(env=env)
        self.x = len(env.action_spaces[0].names)
        self.y = len(env.action_spaces[0].names)
        self.prim_metrics = primary_metrics
        if len(FLAGS["actions_white_list"]) > 0:
            self.x = len(FLAGS["actions_white_list"])
            self.y = len(FLAGS["actions_white_list"])

        self.observation_space = gym.spaces.Box(
            low=np.full(self.x, 0, dtype=np.float32),
            high=np.full(self.y, float("inf"), dtype=np.float32),
            dtype=np.float32,
        )
        self.env = env
        self.hetero_os = hetero_observations_
        self.hetero_os_baselines = list()

    def reset(self, *args, **kwargs):
        self._steps_taken = 0
        self._state = np.zeros(
            (FLAGS['episode_len'] - 1, self.x), dtype=np.int32
        ) # drop history diagram
        super().reset(*args, **kwargs) # drop the environment state
        reset_state = [self._state] # add dropped history diagram as state_{0}
        # add extra observation spaces
        for k, v in self.hetero_os.items():
            obs = self.env.observation[k]
            if k == self.prim_metrics:
                obs = v # primary metrics should be taken from the original observation space (from previous)
            self.hetero_os_baselines.append(obs)
            reset_state.append(obs) # append dropped {observation space}_i as state_{i},i \in 1..len
            FLAGS["logger"].log("Reset: add baseline observation for " + str(k) + " : " + str(obs),
                                mode=LogMode.VERBOSE)

        return reset_state # should be specified by certain ObservationSpaces

    def step(self, action: int):
        assert self._steps_taken < FLAGS['episode_len']
        if self._steps_taken < FLAGS['episode_len'] - 1:
            # Don't need to record the last action since there are no
            # further decisions to be made at that point, so that
            # information need never be presented to the model.
            try:
                act = FLAGS["actions_filter_map"][action]
            except:
                act = action
            self._state[self._steps_taken][act] = 1 # act is only for observation states update, real action is "action"
        self._steps_taken += 1
        observable_a = self._state #, _, _, _ = super().step(action) # or simply return observation space
        # compose extra observations into the observation vector too
        observation_spaces_list = [self.env.observation.spaces[k] for (k, _) in self.hetero_os.items()]
        heterog_obs, b, c, d = self.env.step(action, observation_spaces=observation_spaces_list)
        observation = [observable_a, heterog_obs[0], heterog_obs[1]]

        return observation, b, c, d

    def observation(self, observation):
        return self._state