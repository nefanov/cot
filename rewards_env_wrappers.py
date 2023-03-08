from typing import Callable, Iterable

import numpy as np

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.spaces import RuntimeReward
from compiler_gym.spaces import Reward
from compiler_gym.wrappers import CompilerEnvWrapper

class SizeReward(Reward):
    """ Incremental size reward
    """

    def __init__(self):

        super().__init__(
            name="OTextSizeB",
            observation_spaces=["ObjectTextSizeBytes"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_size = 0

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.baseline_runtime = observation_view["ObjectTextSizeBytes"]

    def update(self, action, observations, observation_view):
        del action  # unused
        del observation_view  # unused
        return float(self.baseline_size - observations[0]) / self.baseline_size


class RuntimePointEstimateReward(CompilerEnvWrapper):
    """LLVM wrapper that uses a point estimate of program runtime as reward.
    RuntimePointEstimateReward --> RuntimeReward --> Reward --> "runtime" obs value
    This class wraps an LLVM environment and registers a new runtime reward
    space*. Runtime is estimated from one or more runtime measurements, after
    optionally running one or more warmup runs. At each step, reward is the
    change in runtime estimate from the runtime estimate at the previous step.
    
    *RuntimeReward(Reward): an example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    
    """
    def __init__(
        self,
        env: LlvmEnv,
        runtime_count: int = 30,
        warmup_count: int = 0,
        estimator: Callable[[Iterable[float]], float] = np.median,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param runtime_count: The number of times to execute the binary when
            estimating the runtime.

        :param warmup_count: The number of warmup runs of the binary to perform
            before measuring the runtime.

        :param estimator: A function that takes a list of runtime measurements
            and produces a point estimate.
        """
        super().__init__(env)

        self.env.unwrapped.reward.add_space(
            RuntimeReward(
                runtime_count=runtime_count,
                warmup_count=warmup_count,
                estimator=estimator,
            )
        )
        #self.env.unwrapped.reward.add_space(
        #    SizeReward(
        #
        #    )
        #)
        self.env.unwrapped.reward_space = "runtime"

        self.env.unwrapped.runtime_observation_count = runtime_count
        self.env.unwrapped.runtime_warmup_runs_count = warmup_count


    def fork(self) -> "RuntimePointEstimateReward":
        fkd = self.env.fork()
        # Remove the original "runtime" space so that we that new
        # RuntimePointEstimateReward wrapper instance does not attempt to
        # redefine, raising a warning.
        del fkd.unwrapped.reward.spaces["runtime"]
        return RuntimePointEstimateReward(
            env=fkd,
            runtime_count=self.reward.spaces["runtime"].runtime_count,
            warmup_count=self.reward.spaces["runtime"].warmup_count,
            estimator=self.reward.spaces["runtime"].estimator,
        )
