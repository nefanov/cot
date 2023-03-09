import sys, os
import compiler_gym                      # imports the CompilerGym environments
from enum import Enum
from rewards_env_wrappers import RuntimePointEstimateReward


class CharacterMode(Enum):
    PROGRAML = 0
    IR2VECFA = "Ir2vecFlowAware"


class RewardMode(Enum):
    ObjectTextSizeB = "ObjectTextSizeBytes"
    RUNTIMEPOINTESTIMATE = "rper"


class Experiment:
    def __init__(self, compiler, bench, observation_space, reward_space, name="exp1"):
        if reward_space == RewardMode.RUNTIMEPOINTESTIMATE:
            # RS wrapped around CompilerGym RS
            self.env = compiler_gym.make(  # creates a new environment (same as gym.make)
                "llvm-v0",  # selects the compiler to use
                benchmark=bench,  # selects the program to compile
                observation_space=observation_space,  # selects the observation space
                reward_space="ObjectTextSizeBytes",  # selects the optimization target
            )

            self.env = RuntimePointEstimateReward(self.env)
            self.env.reset()

        elif reward_space == "External":
            # external RS
            self.env = compiler_gym.make(  # creates a new environment (same as gym.make)
                "llvm-v0",  # selects the compiler to use
                benchmark=bench,  # selects the program to compile
                observation_space=observation_space,  # selects the observation space
                #reward_space=reward_space,  # selects the optimization target
            )
        else:
            # CompilerGym RS
            self.env = compiler_gym.make(  # creates a new environment (same as gym.make)
                "llvm-v0",  # selects the compiler to use
                benchmark=bench,  # selects the program to compile
                observation_space=observation_space,  # selects the observation space
                reward_space=reward_space,  # selects the optimization target
            )

        self.name = name
        self.env.reset()  # starts a new compilation session

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
        #print("A", action)
        print("Action  #", action, ":", env.action_spaces[0].names[int(action.__repr__())])
        observation, reward, done, info = env.step(action)
        if done:
            break
        episode_reward += reward
        print(f"Step {i}, quality={episode_reward:.3%}")


def test_experiment():
    ex = Experiment(
        "llvm-v0",  # selects the compiler to use
        bench="cbench-v1/crc32",  # selects the program to compile
        observation_space="ObjectTextSizeBytes",  # selects the observation space
        reward_space="External" #RewardMode.RUNTIMEPOINTESTIMATE,
    )
    print("=====DUMP ACTIONS=====")
    print(ex.getActions())
    print("=====REWARDS TESTING=====")

    episode_reward = 0.0
    for i in range(1000):
        action = ex.env.action_space.sample()
        # print("A", action)
        print("Action  #", action, ":", ex.env.action_spaces[0].names[int(action.__repr__())])
        observation, reward, done, info = ex.env.step(action,
                 observation_spaces = [ ex.env.observation.spaces["Ir2vecFlowAware"],
                                        ex.env.observation.spaces["TextSizeBytes"],
                                        ex.env.observation.spaces["Runtime"]])
        print("OBS-RW:", observation[1:], reward)
        if done:
            print("DONE!!!")
            break

        print(f"Step {i}, quality={episode_reward:.3%}")
    return

if __name__ == '__main__':
    test_packages_integrity()
    #print("Test 200 times Ir2Vec")
    #test_cycle()
    print("====Dump action space====")
    test_experiment()
    sys.exit(0)
