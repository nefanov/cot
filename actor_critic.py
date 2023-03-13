import math
import random
import statistics
from collections import namedtuple, OrderedDict
from typing import List
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#compiler_gym dependency:
import compiler_gym
from compiler_gym.wrappers import TimeLimit

flags = {}
flags.update({"episode_len": 5})  #"Number of transitions per episode."
flags.update({"hidden_size": 64})  #"Latent vector size."
flags.update({"log_interval": 100})  #"Episodes per log output."
flags.update({"iterations": 1})  #"Times to redo entire training."
flags.update({"exploration": 0.0})  #"Rate to explore random transitions."
flags.update({"mean_smoothing": 0.95})  #"Smoothing factor for mean normalization."
flags.update({"std_smoothing": 0.4})  #"Smoothing factor for std dev normalization."
flags.update({"learning_rate": 0.008})
flags.update({"episodes": 2000})
flags.update({"seed": 0})
flags.update({"is_debug": True})
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
        self.observation_space = gym.spaces.Box(
            low=np.full(len(env.action_spaces[0].names), 0, dtype=np.float32),
            high=np.full(len(env.action_spaces[0].names), float("inf"), dtype=np.float32),
            dtype=np.float32,
        )
        self.env = env
        self.hetero_os = hetero_observations_names

    def reset(self, *args, **kwargs):
        self._steps_taken = 0
        self._state = np.zeros(
            (FLAGS['episode_len'] - 1, self.action_space.n), dtype=np.int32
        ) # drop history diagram
        super().reset(*args, **kwargs) # drop the environment state
        reset_state = [self._state] # add dropped history diagram as state_{0}
        for k,_ in self.hetero_os.items():
            reset_state.append(self.env.observation[k]) # append dropped {observation space}_i as state_{i},i \in 1..len
            if FLAGS['is_debug']:
                print("Reset: add baseline observation for", k , ":", self.env.observation[k])

        return reset_state # should be specified by certain ObservationSpaces

    def step(self, action: int):
        assert self._steps_taken < FLAGS['episode_len']
        if self._steps_taken < FLAGS['episode_len'] - 1:
            # Don't need to record the last action since there are no
            # further decisions to be made at that point, so that
            # information need never be presented to the model.
            self._state[self._steps_taken][action] = 1
        self._steps_taken += 1
        observable_a = self._state #, _, _, _ = super().step(action) # or simply return observation space
        observation_spaces_list = [self.env.observation.spaces[k] for (k,_) in self.hetero_os.items()]
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


def select_action(model, state, exploration_rate=0.0):
    """Selects an action and registers it with the action buffer."""
    state = torch.from_numpy(state.flatten()).float()
    probs, state_value = model(state)

    # Create a probability distribution where the probability of
    # action i is probs[i].
    m = Categorical(probs)

    # Sample an action using the distribution, or pick an action
    # uniformly at random if in an exploration mode.
    if random.random() < exploration_rate:
        action = torch.tensor(random.randrange(0, len(probs)))
    else:
        action = m.sample()

    # Save to action buffer. The drawing of a sample above simply
    # returns a constant integer that we cannot back-propagate
    # through, so it is important here that log_prob() is symbolic.
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # The action to take.
    return action.item()


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


def TrainActorCritic(env, PARAMS=FLAGS, reward_estimator=lambda a: math.sqrt(sum([i*i for i in a[1:]]))):
    model = BasicPolicy(len(env.action_spaces[0].names), PARAMS=PARAMS)
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['learning_rate']) # modify it
    # only for debug statistics
    max_ep_reward = -float("inf")
    avg_reward = MovingExponentialAverage(0.95)
    avg_loss = MovingExponentialAverage(0.95)
    for episode in range(1, PARAMS['episodes'] + 1):
        # Reset environment and episode reward.
        state = env.reset()
        ep_reward = 0
        while True:
            # Select action from policy.
            action = select_action(model, state[0], PARAMS['exploration'])
            # Take the action
            state, reward, done, _ = env.step(action)
            print("OBS-RW:", state[1:], reward)
            # reward calculation
            reward = reward_estimator(state)
            # append reward to the model
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
        if (
                episode == 1
                or episode % FLAGS['log_interval'] == 0
                or episode == FLAGS['episodes']
        ):
            print(
                f"Episode {episode}\t"
                f"Last reward: {ep_reward:.2f}\t"
                f"Avg reward: {avg_reward.value:.2f}\t"
                f"Best reward: {max_ep_reward:.2f}\t"
                f"Last loss: {loss:.6f}\t"
                f"Avg loss: {avg_loss.value:.6f}\t",
                flush=True,
            )
    print(f"\nFinal performance (avg reward): {avg_reward.value:.2f}")
    print(f"Final avg reward versus own best: {avg_reward.value - max_ep_reward:.2f}")

    # One could also return the best found solution here, though that
    # is more random and noisy, while the average reward indicates how
    # well the model is working on a consistent basis.
    return avg_reward.value


def make_env(extra_observation_spaces=None, benchmark=None):
    if benchmark is None:
        benchmark = "cbench-v1/dijkstra"
    env  = compiler_gym.make(  # creates a partially-empty env
                "llvm-v0",  # selects the compiler to use
                benchmark=benchmark,  # selects the program to compile
                observation_space=None,  # default observation space
                reward_space=None,  # in future selects the optimization target
            )

    env = TimeLimit(env, max_episode_steps=5)
    if not isinstance(extra_observation_spaces, OrderedDict):
        o = OrderedDict()
        o["TextSizeBytes"] = 0
        o["Runtime"]= np.zeros(1)
    else:
        o = extra_observation_spaces
    env = HistoryObservation(env, o)
    return env


def main():
    """Main entry point."""


    torch.manual_seed(FLAGS['seed'])
    random.seed(FLAGS['seed'])

    with make_env() as env:
        print(f"Seed: {0}")
        print(f"Episode length: {5}")
        FLAGS['iterations'] = 10
        if FLAGS['iterations'] == 1:
            TrainActorCritic(env)
            return

        # Performance varies greatly with random initialization and
        # other random choices, so run the process multiple times to
        # determine the distribution of outcomes.
        performances = []
        for i in range(1, FLAGS['iterations'] + 1):
            print(f"\n*** Iteration {i} of {FLAGS['iterations']}")
            performances.append(TrainActorCritic(env))

        print("\n*** Summary")
        print(f"Final performances: {performances}\n")
        print(f"  Best performance: {max(performances):.2f}")
        print(f"Median performance: {statistics.median(performances):.2f}")
        print(f"   Avg performance: {statistics.mean(performances):.2f}")
        print(f" Worst performance: {min(performances):.2f}")


def test_run_actor_critic_smoke_test():
    main()


if __name__ == "__main__":
    test_run_actor_critic_smoke_test()