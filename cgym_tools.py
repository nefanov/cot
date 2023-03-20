import sys, os
import compiler_gym                      # imports the CompilerGym environments
from enum import Enum
from rewards_env_wrappers import RuntimePointEstimateReward
import bench_util
from benchmarks import runnable_bench_onefile

import numpy as np

class CharacterMode(Enum):
    PROGRAML = 0
    IR2VECFA = "Ir2vecFlowAware"


class RewardMode(Enum):
    ObjectTextSizeB = "ObjectTextSizeBytes"
    RUNTIMEPOINTESTIMATE = "rper"

RUNCONFIG = dict()
RUNCONFIG['tmpdir'] = "/home/nefanov/compiler_experiments/cot_contrib"
RUNCONFIG['dfl_prog_name'] = "myapp"
RUNCONFIG['compiler_env'] = "llvm-v0"
RUNCONFIG['inital_reward_space'] = "External"
RUNCONFIG['runtime_observation_count'] = 10

'''
Experiment on clang/llvm environment class
'''
class Experiment:
    def __init__(self, bench, observation_space, reward_space, name="exp1"):
        if reward_space == RewardMode.RUNTIMEPOINTESTIMATE:
            # RS wrapped around CompilerGym RS
            self.env = compiler_gym.make(  # creates a new environment (same as gym.make)
                RUNCONFIG['compiler_env'],  # selects the compiler to use
                benchmark=bench,  # selects the program to compile
                observation_space=observation_space,  # selects the observation space
                reward_space="ObjectTextSizeBytes",  # selects the optimization target
            )

            self.env = RuntimePointEstimateReward(self.env)
            self.env.reset()

        elif reward_space == "External":
            # external RS
            self.env = compiler_gym.make(  # creates a new environment (same as gym.make)
                RUNCONFIG['compiler_env'],  # selects the compiler to use
                benchmark=bench,  # selects the program to compile
                observation_space=observation_space,  # selects the observation space
                #reward_space=reward_space,  # selects the optimization target
            )
        else:
            # CompilerGym RS
            self.env = compiler_gym.make(  # creates a new environment (same as gym.make)
                RUNCONFIG['compiler_env'],  # selects the compiler to use
                benchmark=bench,  # selects the program to compile
                observation_space=observation_space,  # selects the observation space
                reward_space=reward_space,  # selects the optimization target
            )

        self.name = name
        self.env.reset()  # starts a new compilation session

    def resetBenchmark(self, benchmark,
                       tmpdir=RUNCONFIG['tmpdir'],
                       fname=RUNCONFIG['dfl_prog_name'],
                       runtime_observation_count=RUNCONFIG['runtime_observation_count'],
                       run_args=list(),
                       rts=10):
        b = runnable_bench_onefile(self.env, runtime_observation_count=10,
                                   tmpdir=tmpdir, name=fname + ".c", run_args = run_args, run_timeout_seconds=rts)
        self.env.reset(benchmark=b)
        return self.env


    def experimentReset(self):
        self.env.reset()
        return

    def getActions(self):
        return self.env.action_spaces[0].names


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
        print("Action  #", action, ":", env.action_spaces[0].names[int(action.__repr__())])
        observation, reward, done, info = env.step(action)
        if done:
            break
        episode_reward += reward
        print(f"Step {i}, quality={episode_reward:.3%}")


def test_experiment(tmpdir=RUNCONFIG["tmpdir"], fname=RUNCONFIG["dfl_prog_name"]):
    ex = Experiment(
        bench=None,#bench_util.bench_uri_from_c_src(tmpdir+ "/" + fname + ".bc"),
        observation_space="ObjectTextSizeBytes",  # selects the observation space
        reward_space=RUNCONFIG['inital_reward_space'] #RewardMode.RUNTIMEPOINTESTIMATE,
    )
    #b = runnable_bench_onefile(ex.env, runtime_observation_count=10, tmpdir="/home/nefanov/compiler_experiments/cot_contrib", name = fname + ".c")
    #ex.env.reset(benchmark=b)
    ex.resetBenchmark(ex.env, runtime_observation_count=10, run_args=["10000","10000"], rts=10)
    print("=====DUMP ACTIONS=====")
    print(ex.getActions())
    print("=====REWARDS TESTING=====")

    episode_reward = 0.0
    print(ex.env.observation.spaces)
    prev_obs = None
    for i in range(1000):
        action = ex.env.action_space.sample()
        # print("A", action)
        print("Step", i, ", Action ", action, ":", ex.env.action_spaces[0].names[int(action.__repr__())])
        observation, reward, done, info = ex.env.step(action,
                 observation_spaces = [ ex.env.observation.spaces["Ir2vecFlowAware"],
                                        ex.env.observation.spaces["TextSizeBytes"],
                                        ex.env.observation.spaces["Runtime"],
                                        ex.env.observation.spaces["TextSizeOz"]])

        rwd = None if prev_obs == None else (prev_obs - np.mean(observation[2]))
        prev_obs = np.mean(observation[2])
        print("bytes:", observation[1], "Oz:", observation[3], "rt:", np.mean(observation[2]),
              "rt gain:", rwd)
        if done:
            print("DONE!!!")
            break

        #print(f"Step {i}, quality={episode_reward:.3%}")
    return


if __name__ == '__main__':
    test_packages_integrity()
    #print("Test 200 times Ir2Vec")
    #test_cycle()
    print("====Dump action space====")
    test_experiment()
    sys.exit(0)
