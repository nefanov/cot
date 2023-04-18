import os
from gcc_reward import *


class gcc_benchmark:
    def __init__(self, from_dict={}):
        self.content = from_dict
        self.compile_cmds = list()
        self.pre_compile_cmds = list()
        self.run_cmd = list()
        self.filepath = None
        self.outfile = list()
        self.timeout_seconds = None

    def add_compile_cmd(self):
        pass

    def add_run_cmd(self):
        pass

    def make_benchmark(self, tmpdir, name="program.c", run_args=list(),
                    run_timeout_seconds=10,
                    extra_objects_list=None,
                    extra_include_dirs=None,
                    sys_settings=dict()
                       ):
        arch = "native"
        compiler = sys_settings.get('compiler', "$CC")
        inputs = sys_settings.get('inputs', "$IN")
        output_bin = sys_settings.get('output_bin', "a.out")
        sys_lib_flags = sys_settings.get('sys_lib_flags', self.get_system_library_flags())
        extra_obj = extra_objects_list if extra_objects_list else list()

        if 'arch_triplet' in sys_settings.keys():
            if sys_settings['arch_triplet'].startswith("aarch64"):
                arch = "qemu-aarch64"  # only this target is now supported

        if extra_include_dirs:
            for item in extra_include_dirs:
                sys_lib_flags.append('-isystem')
                sys_lib_flags.append(item)

        self.filepath = tmpdir + os.sep + name
        self.compile_cmds.extend(
            [compiler, inputs] + extra_obj + sys_lib_flags
        )
        self.outfile.extend([output_bin])
        self.timeout_seconds = run_timeout_seconds
        if arch == "native":
            self.run_cmd.extend(["./" + output_bin] + run_args)
        elif arch == "qemu-aarch64":
            self.run_cmd.argument.extend(
                ["qemu-aarch64 -L " + sys_settings['target_libs_dir'] + " ./" + output_bin] + run_args)


class gcc_env:
    def __init__(self):
        pass
    def make(  # creates a new environment (same as gym.make)
        config = dict(),  # selects the compiler to use
        benchmark=None,  # selects the program to compile
        observation_space=gcc_benchmark(),  # selects the observation space
        reward_space="ObjectTextSizeBytes",  # selects the optimization target
    ):
        pass
      
