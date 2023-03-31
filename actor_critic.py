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
    AC_BASIC = 4


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

    def __init__(self, env, hetero_observations_names=OrderedDict()):
        super().__init__(env=env)
        self.x = len(env.action_spaces[0].names)
        self.y = len(env.action_spaces[0].names)
        if len(FLAGS["actions_white_list"]) > 0:
            self.x = len(FLAGS["actions_white_list"])
            self.y = len(FLAGS["actions_white_list"])

        self.observation_space = gym.spaces.Box(
            low=np.full(self.x, 0, dtype=np.float32),
            high=np.full(self.y, float("inf"), dtype=np.float32),
            dtype=np.float32,
        )
        self.env = env
        self.hetero_os = hetero_observations_names
        self.hetero_os_baselines = list()

    def reset(self, *args, **kwargs):
        self._steps_taken = 0
        self._state = np.zeros(
            (FLAGS['episode_len'] - 1, self.x), dtype=np.int32
        ) # drop history diagram
        super().reset(*args, **kwargs) # drop the environment state
        reset_state = [self._state] # add dropped history diagram as state_{0}
        # add extra observation spaces
        for k, _ in self.hetero_os.items():
            obs = self.env.observation[k]
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


# === AC Policies -- move to algorithms.rl.ac.policies
class BasicPolicy(nn.Module):
    """A very simple actor critic policy model."""

    def __init__(self, sz, PARAMS=FLAGS):
        super().__init__()
        self.affine1 = nn.Linear(
            (PARAMS['episode_len'] - 1) * sz, PARAMS['hidden_size']
        )
        self.affine2 = nn.Linear(PARAMS['hidden_size'], PARAMS['hidden_size'])
        self.affine3 = nn.Linear(PARAMS['hidden_size'], PARAMS['hidden_size'])
        self.affine4 = nn.Linear(PARAMS['hidden_size'], PARAMS['hidden_size'])

        # Actor's layer
        self.action_head = nn.Linear(PARAMS['hidden_size'], sz)

        # Critic's layer
        self.value_head = nn.Linear(PARAMS['hidden_size'], 1)

        # Action & reward buffer
        self.saved_actions: List[SavedAction] = []
        self.rewards: List[float] = []

        # Keep exponential moving average of mean and standard
        # deviation for use in normalization of the input.
        self.moving_mean = MovingExponentialAverage(PARAMS['mean_smoothing'])
        self.moving_std = MovingExponentialAverage(PARAMS['std_smoothing'])

    def forward(self, x):
        """Forward of both actor and critic"""
        # Initial layer maps the sequence of one-hot vectors into a
        # vector of the hidden size. Next layers stay with the same
        # size and use residual connections.
        x = F.relu(self.affine1(x))
        x = x.add(F.relu(self.affine2(x)))
        x = x.add(F.relu(self.affine3(x)))
        x = x.add(F.relu(self.affine4(x)))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


def select_action(model, state, exploration_rate=0.0, white_list=None, rev_w_=None):
    """Selects an action and registers it with the action buffer."""
    state = torch.from_numpy(state.flatten()).float()
    probs, state_value = model(state)

    # Create a probability distribution where the probability of
    # action i is probs[i].
    m = Categorical(probs)

    # Sample an action using the distribution, or pick an action
    # uniformly at random if in an exploration mode.
    while True: # do-while loop emulation
        if random.random() < exploration_rate:
            action = torch.tensor(random.randrange(0, len(probs)))
        else:
            action = m.sample()
        if white_list and action not in white_list:
            continue
        break

    # Save to action buffer. The drawing of a sample above simply
    # returns a constant integer that we cannot back-propagate
    # through, so it is important here that log_prob() is symbolic.
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # The action to take.
    if not rev_w_:
        return action.item()
    else:
        return rev_w_[action.item()]


def finish_episode(model, optimizer) -> float:
    """The training code. Calculates actor and critic loss and performs backprop."""
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # Calculate the true value using rewards returned from the
    # environment. We are iterating in reverse order while inserting
    # at each step to the front of the returns list, which implies
    # that returns[i] is the sum of rewards[j] for j >= i. We do not
    # use a discount factor as the episode length is fixed and not
    # very long, but if we had used one, it would appear here.
    for r in model.rewards[::-1]:
        R += r
        returns.insert(0, R)

    # Update the moving averages for mean and standard deviation and
    # use that to normalize the input.
    returns = torch.tensor(returns)
    model.moving_mean.next(returns.mean())
    model.moving_std.next(returns.std())
    returns = (returns - model.moving_mean.value) / (model.moving_std.value + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        # The advantage is how much better a situation turned out in
        # this case than the critic expected it to.
        advantage = R - value.item()

        # Calculate the actor (policy) loss. Because log_prob is
        # symbolic, back propagation will increase the probability of
        # taking the action that was taken if advantage is positive
        # and will decrease it if advantage is negative. In this way
        # we are learning a probability distribution without directly
        # being able to back propagate through the drawing of the
        # sample from that distribution.
        #
        # It may seem that once the critic becomes accurate, so that
        # the advantage is always 0, then the policy can no longer
        # learn because multiplication by 0 impedes back
        # propagation. However, the critic does not know which action
        # will be taken, so as long as there are worse-than-average or
        # better-than-average policies with a non-zero probability,
        # then the critic has to be wrong sometimes because it can
        # only make one prediction across all actions, so learning
        # will proceed.
        policy_losses.append(-log_prob * advantage)

        # Calculate critic (value) loss using L1 smooth loss.
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # Reset gradients.
    optimizer.zero_grad()

    # Sum up all the values of policy_losses and value_losses.
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss_value = loss.item()

    # Perform backprop.
    loss.backward()
    optimizer.step()

    # Reset rewards and action buffer.
    del model.rewards[:]
    del model.saved_actions[:]

    return loss_value


def single_pass_eval(env, reward_estimator=const_factor_threshold,
                     reward_if_list_func=lambda a: np.mean(a), need_reset=True):
    passes_list = FLAGS["reverse_actions_filter_map"]
    for k, v in passes_list.items():
        if need_reset:
            state = env.reset()
        prev_size = state[1]
        prev_runtime = reward_if_list_func(state[2])
        action = v
        state, r, d, _ = env.step(v)
        reward = reward_estimator(env.hetero_os_baselines[0], state[1], reward_if_list_func(env.hetero_os_baselines[1]), reward_if_list_func(state[2]), prev_size,prev_runtime)
        print("Action", env.action_spaces[0].names[action], "R", reward, "; size:", prev_size, "->", state[1], "; runtime:", prev_runtime, "->", reward_if_list_func(state[2]))


def one_pass_perform(env, prev_state, action, reward_estimator=const_factor_threshold, reward_if_list_func=lambda a: np.mean(a)):
    """
    one iteration of search: try all the passes, then return statistics
    """
    passes_list = FLAGS["reverse_actions_filter_map"]
    v = action
    ep_reward = 0
    prev_size = prev_state[1]
    prev_runtime = reward_if_list_func(prev_state[2])
    state, r, d, _ = env.step(v)
    reward = reward_estimator(env.hetero_os_baselines[0],
                              state[1],
                              reward_if_list_func(env.hetero_os_baselines[1]),
                              reward_if_list_func(state[2]),
                              prev_size,
                              prev_runtime)
    return {"action": env.action_spaces[0].names[v],
            "action_num": v,
            "reward": reward,
            "prev_size": prev_size, "size": state[1],
            "prev_runtime": prev_runtime, "runtime": reward_if_list_func(state[2]),
            "size gain %": (prev_size - state[1]) / prev_size * 100
            }


def examine_each_action(env, state, reward_estimator=const_factor_threshold, reward_if_list_func=lambda a: np.mean(a), step_lim=10):
    passes_list = FLAGS["reverse_actions_filter_map"]
    passes_results = []
    for k, v in passes_list.items():
        with copy.deepcopy(env) as copy_env:
            passes_results.append(one_pass_perform(copy_env, state, v, reward_estimator=reward_estimator, reward_if_list_func=reward_if_list_func))
    return passes_results


def max_subseq_from_start(seq: list, episode_reward=0.) -> list:
    """
    this is very slow (O(n)) implementation of max subseq search from start element
    """
    gain = episode_reward
    up_lim = len(seq) - 1
    for i in range(len(seq)):
        S = sum([s['size gain %'] for s in seq[:i]])
        if S > gain:
            gain = S
            up_lim = i
    return seq[:up_lim]


def search_strategy_eval(env, reward_estimator=const_factor_threshold,
                         reward_if_list_func=lambda a: np.mean(a),
                         step_lim=10, pick_pass=pick_random_from_positive, patience=FLAGS['patience']):
    state = env.reset()
    results = list()
    pat = 0
    action_log = []
    episode_reward = 0.0
    episode_size_gain = 0.0
    for i in range(step_lim):
        print("step", i)
        results = examine_each_action(env, state, reward_estimator=reward_estimator, reward_if_list_func=reward_if_list_func)
        best = pick_pass(results)[-1]
        state, reward, d, _ = env.step(best["action_num"])  # apply. state and reward updates
        action_log.append(best)
        try:
            episode_reward += reward
        except:
            pass
        episode_size_gain += best['size gain %']

        if best['reward'] <= .0:
            pat += 1
            if patience <= pat:
                print("=============PATIENCE LIMIT EXCEEDED===============")
                break

    print("====================================================")
    pprint.pprint(action_log)
    # find the subsequence from start, which gives max size gain
    action_log = max_subseq_from_start(action_log, episode_reward=episode_size_gain)
    return {"action_log": action_log, "episode_reward": episode_reward, "episode_size_gain": episode_size_gain}


def TrainActorCritic(env, PARAMS=FLAGS, reward_estimator=const_factor_threshold, reward_if_list_func=lambda a: np.mean(a)):
    bplen = len(env.action_spaces[0].names)
    try:
        bplen = len(FLAGS["actions_white_list"])
    except:
        pass
    model = BasicPolicy(bplen, PARAMS=PARAMS)
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['learning_rate']) # modify it
    # only for debug statistics
    max_ep_reward = -float("inf")
    avg_reward = MovingExponentialAverage(0.95)
    avg_loss = MovingExponentialAverage(0.95)
    for episode in range(1, PARAMS['episodes'] + 1):
        # Reset environment and episode reward.
        state = env.reset()
        ep_reward = 0
        prev_size = state[1]
        prev_runtime = reward_if_list_func(state[2])
        action_log = list()
        pat = 0
        while True:
            # Select action from policy.
            action = select_action(model, state[0], PARAMS['exploration'],
                                    white_list=FLAGS["actions_white_list"],
                                    rev_w_=FLAGS["reverse_actions_filter_map"])
            # Take the action

            action_log.append(env.action_spaces[0].names[action])
            state, reward, done, _ = env.step(action)
            # reward calculation

            reward = reward_estimator(env.hetero_os_baselines[0],
                                      state[1],
                                      reward_if_list_func(env.hetero_os_baselines[1]),
                                      reward_if_list_func(state[2]),
                                      prev_size,
                                      prev_runtime)
            if reward <= 0.:
                pat += 1
            if (FLAGS["patience"] <= pat):
                #print(episode, ": Patience limit exceed", action_log, "; Episode reward:", ep_reward + reward)
                done = True
            prev_size = state[1]
            prev_runtime = reward_if_list_func(state[2])
            size_rewards = [env.hetero_os_baselines[0], state[1]]
            runtime_rewards = [reward_if_list_func(env.hetero_os_baselines[1]), reward_if_list_func(state[2])]
            if size_rewards[1] < size_rewards[0]:
                print(episode, "Gained:", size_rewards, runtime_rewards, "size gain:", math.fabs(size_rewards[1] - size_rewards[0]) * 100/size_rewards[0], "%")
                print(action_log)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # Perform back propagation.
        loss = finish_episode(model, optimizer)
        # Update statistics.
        max_ep_reward = max(max_ep_reward, ep_reward)
        avg_reward.next(ep_reward)
        avg_loss.next(loss)

        # Log statistics.
        if episode == 1 or episode % FLAGS['log_interval'] == 0 or episode == FLAGS['episodes']:
            FLAGS["logger"].save_and_print(f"Episode {episode}\t"
                                f"Last reward: {ep_reward:.2f}\t"
                                f"Avg reward: {avg_reward.value:.2f}\t"
                                f"Best reward: {max_ep_reward:.2f}\t"
                                f"Last loss: {loss:.6f}\t"
                                f"Avg loss: {avg_loss.value:.6f}\t",
                                mode=LogMode.SHORT)
            FLAGS["logger"].save_and_print("Action Log: " + str(action_log), mode=LogMode.SHORT)
    final_perf_str = f"\nFinal performance (avg reward): {avg_reward.value:.2f}"
    final_avg_reward_str = f"Final avg reward versus own best: {avg_reward.value - max_ep_reward:.2f}"
    FLAGS["logger"].save_and_print(final_perf_str)
    FLAGS["logger"].save_and_print(final_avg_reward_str)

    # One could also return the best found solution here, though that
    # is more random and noisy, while the average reward indicates how
    # well the model is working on a consistent basis.
    return avg_reward.value


def make_env(extra_observation_spaces=None, benchmark=None, sz_baseline="TextSizeOz", actions_whitelist_names=None):
    if benchmark is None:
        benchmark = "cbench-v1/qsort"
    env = compiler_gym.make(  # creates a partially-empty env
                "llvm-v0",  # selects the compiler to use
                benchmark=benchmark,  # selects the program to compile
                observation_space=sz_baseline,  # initial observation
                reward_space=None,  # in future selects the optimization target
            )

    if actions_whitelist_names:
        FLAGS["actions_white_list"] = [env.action_spaces[0].names.index(action) for action in actions_whitelist_names]
        FLAGS["actions_filter_map"] = remap_actions(FLAGS["actions_white_list"])
        FLAGS["reverse_actions_filter_map"] = {v: k for k, v in FLAGS["actions_filter_map"].items()}
    env = TimeLimit(env, max_episode_steps=FLAGS["episode_len"])
    baseline_obs_init_val = env.reset()
    if not isinstance(extra_observation_spaces, OrderedDict):
        o = OrderedDict()
        o["TextSizeBytes"] = baseline_obs_init_val # this is only a baseline, and value is defaultly scalar -- chk type
        o["Runtime"] = np.zeros(1)
    else:
        o = extra_observation_spaces
    env = HistoryObservation(env, o)
    return env


def main(MODE="single_pass_validate"):
    call_evaluator = None
    if MODE == Runmode.AC_BASIC:
        pass
    elif MODE == Runmode.GREEDY:
        call_evaluator = pick_max_size_gain
    elif MODE == Runmode.RANDOM:
        call_evaluator = pick_random
    elif MODE == Runmode.RANDOM_POSITIVES:
        call_evaluator = pick_random_from_positive
    else:
        print("Incorrect run mode")
        sys.exit(1)

    with make_env(actions_whitelist_names=actions_oz_extra) as env:
        if FLAGS['iterations'] == 1:
            TrainActorCritic(env, reward_estimator=const_factor_threshold)
            return

        if MODE != Runmode.AC_BASIC:
            seq_list_lens = []
            for i in range(FLAGS["search_iterations"]):
                print("Iteration", i)
                seq_list_lens.append(search_strategy_eval(env, reward_estimator=const_factor_threshold, pick_pass=call_evaluator))
            positive_res = [s for s in seq_list_lens if s["episode_reward"] >= 0.]

            print("Iteration", i, "statistics:")
            __max_size_gain = max(positive_res, key=lambda a: a["episode_size_gain"])
            print("Max size gain:")
            pprint.pprint(__max_size_gain)
            print("On sequence:")
            seq = [(d['action'], d['size gain %']) for d in __max_size_gain['action_log']]
            for item in seq:
                print(item)
            print("---------------------------------------------------------------------------")
            sys.exit(0)
        else:
            torch.manual_seed(FLAGS['seed'])
            random.seed(FLAGS['seed'])
            performances = []
            for i in range(1, FLAGS['iterations'] + 1):
                FLAGS["logger"].save_and_print(f"\n*** Iteration {i} of {FLAGS['iterations']}")
                performances.append(TrainActorCritic(env, reward_estimator=const_factor_threshold))

            FLAGS["logger"].save_and_print("\n*** Summary")
            FLAGS["logger"].save_and_print(f"Final performances: {performances}\n")
            FLAGS["logger"].save_and_print(f"  Best performance: {max(performances):.2f}")
            FLAGS["logger"].save_and_print(f"Median performance: {statistics.median(performances):.2f}")
            FLAGS["logger"].save_and_print(f"   Avg performance: {statistics.mean(performances):.2f}")
            FLAGS["logger"].save_and_print(f" Worst performance: {min(performances):.2f}")


if __name__ == "__main__":
    main(Runmode.RANDOM)
