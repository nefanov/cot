from typing import Callable, Iterable

import numpy as np

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.spaces import RuntimeReward
from compiler_gym.wrappers import CompilerEnvWrapper
from typing import Callable, Iterable, List, Optional

from compiler_gym.errors import BenchmarkInitError, ServiceError
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType

class SizeReward(Reward):
    """ Incremental size reward
    """

    def __init__(self):

        super().__init__(
            name="OTextSizeB",
            observation_spaces=["ObjectTextSizeBytes", "Runtime"],
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


class RuntimeReward(Reward):
    def __init__(
        self,
        runtime_count: int,
        warmup_count: int,
        estimator: Callable[[Iterable[float]], float],
        default_value: int = 0,
    ):
        super().__init__(
            name="runtime",
            observation_spaces=["Runtime"],
            default_value=default_value,
            min=None,
            max=None,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.runtime_count = runtime_count
        self.warmup_count = warmup_count
        self.starting_runtime: Optional[float] = None
        self.previous_runtime: Optional[float] = None
        self.current_benchmark: Optional[str] = None
        self.estimator = estimator

    def reset(self, benchmark, observation_view) -> None:
        # If we are changing the benchmark then check that it is runnable.
        if benchmark != self.current_benchmark:
            if not observation_view["IsRunnable"]:
                raise BenchmarkInitError(f"Benchmark is not runnable: {benchmark}")
            self.current_benchmark = benchmark
            self.starting_runtime = None

        # Compute initial runtime if required, else use previously computed
        # value.
        if self.starting_runtime is None:
            self.starting_runtime = self.estimator(observation_view["Runtime"])

        self.previous_runtime = self.starting_runtime

    def update(
        self,
        actions: List[ActionType],
        observations: List[ObservationType],
        observation_view,
    ) -> float:
        del actions  # unused
        del observation_view  # unused
        runtimes = observations[0]
        if len(runtimes) != self.runtime_count:
            raise ServiceError(
                f"Expected {self.runtime_count} runtimes but received {len(runtimes)}"
            )
        runtime = self.estimator(runtimes)

        reward = self.previous_runtime - runtime
        self.previous_runtime = runtime
        return reward


class RuntimeAndSizeReward(Reward):
    def __init__(
        self,
        runtime_count: int,
        warmup_count: int,
        estimator: Callable[[Iterable[float]], float],
        default_value: int = 0,
        size_observation_type: str = "ObjectTextSizeBytes",
        is_debug: bool = True
    ):
        super().__init__(
            name="runtime",
            observation_spaces=["Runtime", size_observation_type],
            default_value=default_value,
            min=None,
            max=None,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.size_observation_type = size_observation_type
        self.runtime_count = runtime_count
        self.warmup_count = warmup_count
        self.starting_runtime: Optional[float] = None
        self.previous_runtime: Optional[float] = None
        self.starting_size: Optional[float] = 0.
        self.previous_size: Optional[float] = 0.
        self.current_benchmark: Optional[str] = None
        self.estimator = estimator
        self.is_debug = is_debug

    def reset(self, benchmark, observation_view) -> None:
        # If we are changing the benchmark then check that it is runnable.
        if benchmark != self.current_benchmark:
            if not observation_view["IsRunnable"]:
                raise BenchmarkInitError(f"Benchmark is not runnable: {benchmark}")
            self.current_benchmark = benchmark
            self.starting_runtime = None
            self.starting_size = 0.0

        # Compute initial runtime if required, else use previously computed
        # value.
        if self.starting_runtime is None:
            self.starting_runtime = self.estimator(observation_view["Runtime"])
        if self.starting_size == 0.0:
            self.starting_size = observation_view[self.size_observation_type]

        self.previous_runtime = self.starting_runtime
        self.previous_size = self.starting_size

    def update(
        self,
        actions: List[ActionType],
        observations: List[ObservationType],
        observation_view,
    ) -> float:
        """Calculate a reward for the given action.
                :param action: The action performed.
                :param observations: A list of observation values as requested by the
                    :code:`observation_spaces` constructor argument.
                :param observation_view: The
                    :class:`ObservationView <compiler_gym.views.ObservationView>`
                    instance.
        """
        del actions  # unused
        del observation_view  # unused

        runtimes = observations[0]

        if len(runtimes) != self.runtime_count:
            raise ServiceError(
                f"Expected {self.runtime_count} runtimes but received {len(runtimes)}"
            )
        runtime = self.estimator(runtimes)
        size = observations[1]
        delta_size =  (self.previous_size - size) / self.starting_size
        delta_runtime = (self.previous_runtime - runtime) / self.starting_runtime
        # put debug log there if need
        if self.is_debug == True:
            print("Debug Measuremens: {runtime delta:",
                  delta_runtime, "; size delta: (", delta_size, size,")")
        # default policy: const perf_degradation under min (size)

        reward = (1. + delta_runtime) * (delta_size)
        self.previous_runtime = runtime
        self.previous_size = size
        return reward



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
            RuntimeAndSizeReward(
                runtime_count=runtime_count,
                warmup_count=warmup_count,
                estimator=estimator,
            )
        )
        self.env.unwrapped.reward.add_space(
            SizeReward(

            )
        )
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
