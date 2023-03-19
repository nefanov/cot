import logging
import os
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
import numpy as np

from bench_util import *

#compiler_gym dependencies
from compiler_gym.datasets import Benchmark, BenchmarkInitError
from compiler_gym.service.proto import File
from compiler_gym.third_party.gccinvocation.gccinvocation import GccInvocation
from compiler_gym.util.commands import Popen
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.shell_format import join_cmd
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command
from compiler_gym.envs.llvm import llvm_benchmark

from compiler_gym.envs.llvm import LlvmEnv, llvm_benchmark
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType


def runnable_bench_onefile(env: LlvmEnv, tmpdir, runtime_observation_count: int,
                           name="program.c",
                           run_args=list(),
                           warmup_count=10,
                           run_timeout_seconds=10):
    env.reset()
    env.runtime_observation_count = runtime_observation_count
    env.runtime_warmup_runs_count = warmup_count

    benchmark = env.make_benchmark(Path(tmpdir) / name)
    benchmark.proto.dynamic_config.build_cmd.argument.extend(
        ["$CC", "$IN"] + llvm_benchmark.get_system_library_flags()
    )
    benchmark.proto.dynamic_config.build_cmd.outfile.extend(["a.out"])
    benchmark.proto.dynamic_config.build_cmd.timeout_seconds = 10
    benchmark.proto.dynamic_config.run_cmd.argument.extend(["./a.out"] + run_args)
    benchmark.proto.dynamic_config.run_cmd.timeout_seconds = run_timeout_seconds
    return benchmark
