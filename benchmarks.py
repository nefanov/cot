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
                           run_timeout_seconds=10,
                           extra_objects_list = None,
                           sys_settings = {}):
    env.reset()
    env.runtime_observation_count = runtime_observation_count
    env.runtime_warmup_runs_count = warmup_count
    compiler = "$CC"
    inputs = "$IN"
    output_bin = "a.out"
    arch = "native"
    sys_lib_flags = llvm_benchmark.get_system_library_flags()
    if 'compiler' in sys_settings.keys():
        compiler = sys_settings['compiler']
    if 'arch_triplet' in sys_settings.keys():
        if sys_settings['arch_triplet'].startswith("aarch64"):
            arch = "qemu-aarch64" # only this target is now supported
    if 'inputs' in sys_settings.keys():
        inputs = sys_settings['inputs']
    if 'sys_lib_flags' in sys_settings.keys():
        sys_lib_flags = sys_settings['sys_lib_flags']
    if 'output_bin' in sys_settings.keys():
        output_bin = sys_settings['output_bin']
    extra_obj = extra_objects_list if extra_objects_list else list()


    benchmark = env.make_benchmark(Path(tmpdir) / name)
    benchmark.proto.dynamic_config.build_cmd.argument.extend(
        [compiler, inputs] + extra_obj + sys_lib_flags
    )
    benchmark.proto.dynamic_config.build_cmd.outfile.extend([output_bin])
    benchmark.proto.dynamic_config.build_cmd.timeout_seconds = run_timeout_seconds
    if arch == "native":
        benchmark.proto.dynamic_config.run_cmd.argument.extend(["./" + output_bin] + run_args)
    elif arch == "qemu-aarch64":
        benchmark.proto.dynamic_config.run_cmd.argument.extend(["qemu-aarch64 -L " + sys_settings['target_libs_dir'] + " ./" + output_bin] + run_args)
    benchmark.proto.dynamic_config.run_cmd.timeout_seconds = run_timeout_seconds
    return benchmark
