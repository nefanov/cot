import os
import re
import subprocess
from gcc_reward import *
from rewards import const_factor_threshold

from common import FLAGS, printRed, printLightPurple, printGreen, printYellow

FLAGS['tmpdir'] = os.getcwd()


class gcc_benchmark:
    def __init__(self, from_dict={}, tmpdir=FLAGS['tmpdir']):
        self.content = from_dict
        self.compile_cmds = list()
        self.pre_compile_cmds = list()
        self.run_cmd = list()
        self.filepaths = list()
        self.outfile = list()
        self.timeout_seconds = None
        self.log_file_path = tmpdir + os.sep + "last_compile_log.txt"
        self.last_compile_success = False

    def add_compile_cmd(self):
        pass

    def add_run_cmd(self):
        pass

    def make_benchmark(self, tmpdir, names=["program.c"], run_args=list(),
                    run_timeout_seconds=10,
                    extra_objects_list=None,
                    extra_include_dirs=None,
                    sys_settings=dict(),
                       ):
        arch = "native"
        compiler = sys_settings.get('compiler', "gcc")
        output_bin = sys_settings.get('output_bin', "a.out")
        sys_lib_flags = sys_settings.get('sys_lib_flags', [])
        extra_obj = extra_objects_list if extra_objects_list else list()
        extra_compiler_flags = sys_settings.get('extra_compiler_flags', [])

        self.tmpdir = tmpdir

        if 'arch_triplet' in sys_settings.keys():
            if sys_settings['arch_triplet'].startswith("aarch64"):
                arch = "qemu-aarch64"  # only this target is now supported

        if extra_include_dirs:
            for item in extra_include_dirs:
                sys_lib_flags.append('-isystem')
                sys_lib_flags.append(item)

        for name in names: # name is relative path to TU from the tmp dir
            self.filepaths.append([tmpdir + os.sep + nm for nm in name.split(" ")])
            self.compile_cmds.append(
                [compiler] + [" ".join(self.filepaths[-1])] + extra_obj + sys_lib_flags + extra_compiler_flags
            )
        self.outfile.extend([output_bin]) # by default, there are only one running artifact
        self.timeout_seconds = run_timeout_seconds
        if arch == "native":
            self.run_cmd.extend(["./" + output_bin] + run_args)
        elif arch == "qemu-aarch64":
            self.run_cmd.argument.extend(
                ["qemu-aarch64 -L " + sys_settings['target_libs_dir'] + " ./" + output_bin] + run_args)


    def compile(self, opt=None):
        for _cmd in self.compile_cmds:
            cmd = _cmd[:]
            if opt:
                cmd.append(" ".join(opt))
            results = subprocess.run([" ".join(cmd)], text=True, shell=True, capture_output=True)
            with open(self.log_file_path, 'w+') as fp:
                output = "Building by cmd: " + str(cmd) + ":\n" + "STDOUT: " + str(results.stdout) + "\nSTDERR: " + str(results.stderr)
                fp.write(output)
                print(output)
            if (results.returncode != 0):
                printRed("Compile error on cmd: " + str(cmd))
                self.last_compile_success = False
                return 1, cmd
        printGreen("Compile success.")
        self.last_compile_success = True
        return 0, cmd

    def run(self, pre_cmd=["time"], need_compile=False):
        if need_compile is True:
            compile_ret = self.compile()
            if compile_ret[0] != 0:
                return 1
            else:
               pass

        if self.last_compile_success is False:
            printRed("run() error: inconsistent build")
            return 1, 0

        results = subprocess.run([" ".join(pre_cmd + self.run_cmd)], text=True, shell=True, capture_output=True)
        with open(self.log_file_path, 'w+') as fp:
            output = "Run by cmd: " + str(pre_cmd + self.run_cmd) + ":\n" + "STDOUT: " + str(
                results.stdout) + "\nSTDERR: " + str(
                results.stderr)
            fp.write(output)
            print(output)

        if results.returncode != 0:
            printRed("Run failed.")
            return 1, 0.
        else:
            printGreen("Run success.")
            out = results.stdout
            if "time" in pre_cmd:
                rt = re.findall(r'[0-9]+.[0-9]+elapsed', results.stderr)
                return 0, float((lambda text, suffix:
                       text[:-len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text)(rt[0], "elapsed"))
            else:
                return 0, 0.

    def get_text_size(self) -> int:
        """
            return: size of .TEXT section of compiled elf in bytes
        """
        if self.last_compile_success is False:
            printRed("run() error: inconsistent build")
            return 0

        results = subprocess.run([" ".join(['size', '-G', ]) + ' '+ self.outfile[0]], text=True, shell=True,
                                 capture_output=True)
        return int(list(filter(None,results.stdout.split('\n')[1].split('\t')[0].split(' ')))[0])

    def get_obj_size(self) -> int:
        """
        return: size of compiled elf in bytes
        """
        if self.last_compile_success is False:
            printRed("run() error: inconsistent build")
            return 0

        results = subprocess.run([" ".join(['du', '-sb', ]) + ' '+ self.outfile[0]], text=True, shell=True,
                                 capture_output=True)
        return int(results.stdout.split('\t')[0])

class gcc_env:
    def __init__(self,  # creates a new environment (same as gym.make)
        config = dict(),  # selects the compiler to use
        benchmark=gcc_benchmark(),  # selects the program to compile
        observation_space=None,  # selects the observation space
        reward_spaces=[],  # selects the optimization target
        action_space=list()):
        self.action_history = list()
        self.action_space = action_space
        self.benchmark = benchmark
        self.observation_space=observation_space
        self.config=config
        self.reward_spaces = reward_spaces

        pass

    def reset(self):
        self.benchmark.compile()
        self.action_history.clear()
        reward_metrics = list()
        for rw_meter in self.reward_spaces:
            reward_metrics.append(rw_meter.evaluate(env=self))
        self.state = [None] + reward_metrics
        return self.state


    def step(self, action):
        result = self.multistep([action])
        return result

    def reward_adapter(l: list, prev: list, hdrs: list, mode="const_runtime_min_text_sz_def_thr"):
        if len(prev) != len(l):
            print("reward metrics can't be matched")
            return -1
        if mode == "const_runtime_min_text_sz_def_thr":
            sz_name = "TextSizeBytes"
        elif mode == "const_runtime_min_obj_sz_def_thr":
            sz_name = "ObjSizeBytes"
        else:
            print("unexpected reward mode")
            return -1
        return const_factor_threshold(
                baseline_m=prev[hdrs.index(sz_name)],
                baseline_n=prev[hdrs.index("Runtime")],
                prev_m=prev[hdrs.index(sz_name)],
                prev_n=prev[hdrs.index("Runtime")],
                measured_m=l[hdrs.index(sz_name)],
                measured_n=l[hdrs.index("Runtime")]
            )

    def multistep(self, actions:list, reward_func = reward_adapter):
        prev_state = self.state
        done = False
        info = None

        seq = self.action_history + actions
        self.benchmark.compile(opt=seq)
        reward_metrics = list()
        for rw_meter in self.reward_spaces:
            reward_metrics.append(rw_meter.evaluate(env=self))

        reward = reward_func(reward_metrics, prev_state[1:], [r.kind for r in self.reward_spaces])
        self.action_history += actions
        state = [None] + [r for r in reward_metrics]
        return state, reward, done, info


if __name__ == '__main__':
    gbm = gcc_benchmark()
    gbm.make_benchmark(tmpdir=FLAGS['tmpdir'])
    env = gcc_env(benchmark=gbm, reward_spaces=[RuntimeRewardMetrics(), TextSizeBytesRewardMetrics()])
    env.reset()
    printLightPurple(str(env.step(action="-ftree-vectorize")))
