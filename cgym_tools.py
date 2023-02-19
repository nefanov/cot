import sys, os
import compiler_gym                      # imports the CompilerGym environments
from enum import Enum


class characterMode(Enum):
    PROGRAML = 0
    IR2VEC = "ir2vec"


def test_packages_integrity():
    env = compiler_gym.make(  # creates a new environment (same as gym.make)
        "llvm-v0",  # selects the compiler to use
        benchmark="cbench-v1/qsort",  # selects the program to compile
        observation_space="Ir2vecFlowAware",  # selects the observation space
        reward_space="IrInstructionCountOz",  # selects the optimization target
    )
    env.reset()  # starts a new compilation session
    #env.render()  # prints the IR of the program
    print("Agent step:")
    observation, reward, done, info = env.step(env.action_space.sample())  # applies a random optimization, updates state/reward/actions
    print(env.action_space.sample)
    print("observation:")
    print(observation)
    print("reward:")
    print(reward)
    print("info:")
    print(done)
    env.close()

def test_cycle():
    print("=====TEST #2=====")
    env = compiler_gym.make(  # creates a new environment (same as gym.make)
        "llvm-v0",  # selects the compiler to use
        benchmark="cbench-v1/qsort",  # selects the program to compile
        observation_space="Ir2vecFlowAware",  # selects the observation space
        reward_space="IrInstructionCountOz",  # selects the optimization target
    )
    env.reset()  # starts a new compilation session
    episode_reward = 0

    for i in range(1, 201):
        action = env.action_space.sample()
        #print("A", action)
        print("Action  #", action, ":", env.action_spaces[0].names[int(action.__repr__())])
        observation, reward, done, info = env.step(action)
        if done:
            break
        episode_reward += reward
        print(f"Step {i}, quality={episode_reward:.3%}")


if __name__ == '__main__':
    test_packages_integrity()
    print("Test 100 times Ir2Vec")
