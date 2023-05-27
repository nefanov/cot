import copy
import os
import pprint
import re
import subprocess
import search_policies
import shutil
import itertools
from enum import Enum
import bench_configs
from standalone_reward import *
from routine import file_utils, sequences
from rewards import const_factor_threshold
from action_spaces_presets import load_as_from_file, actions_oz_extra, actions_oz_baseline


from common import FLAGS, printRed, printLightPurple, printGreen, printYellow

# dependencies from modules which are depended from CompilerGym: should be redesigned:
from experiment_runner import search_strategy_eval

FLAGS['tmpdir'] = os.getcwd()
FLAGS["reverse_actions_filter_map"] = {f:f for f in load_as_from_file("gcc_O2_size_red.txt")}


class Buildmode(Enum):
    GCC_DRIVER="driver"
    LLVM_PIPELINE="llvm"
    MAKE="make"
    CMAKE="cmake"


class standalone_benchmark:
    def __init__(self, from_dict={}, tmpdir=FLAGS['tmpdir'], build_mode=Buildmode.GCC_DRIVER):
        self.content = from_dict
        self.compile_cmds = list()
        self.pre_compile_cmds = list()
        self.run_cmd = list()
        self.filepaths = list()
        self.outfile = list()
        self.timeout_seconds = None
        self.log_file_path = tmpdir + os.sep + "last_compile_log.txt"
        self.last_compile_success = False
        self.build_mode = build_mode
        self.compiler = "standalone"
        self.opt_env_var_name = "OPT"
        self.pass_manager_ver = "old"

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def add_compile_cmd(self):
        pass

    def add_run_cmd(self):
        pass

    def make_benchmark(self, tmpdir, names=["program.c"], run_args=list(), run_timeout_seconds=10,
                        extra_objects_list=None, extra_include_dirs=None, sys_settings=dict(),
                       ):
        arch = "native"
        if self.build_mode == Buildmode.GCC_DRIVER:
            self.compiler = sys_settings.get('compiler', "standalone")
        if self.build_mode == Buildmode.LLVM_PIPELINE:
            self.compiler = sys_settings.get('compiler', 'clang')
            llvm_opt = sys_settings.get('opt', 'opt')
            opt_baseline = sys_settings.get('opt_baseline', '-O0')
        elif self.build_mode == Buildmode.MAKE:
            self.compiler = "make -C"
        output_run_artifact = sys_settings.get('output_bin', "a.out")
        sys_lib_flags = sys_settings.get('sys_lib_flags', [])
        self.pass_manager_ver = sys_settings.get('llvm_pass_manager',"old")
        extra_obj = extra_objects_list if extra_objects_list else list()
        extra_compiler_flags = sys_settings.get('extra_compiler_flags', [])
        self.opt_prepend = sys_settings.get('opt_prepend', [])
        self.opt_env_var_name = sys_settings.get('opt_var_name', 'OPT')
        self.tmpdir = tmpdir
        self.run_working_dir = sys_settings.get('run_working_dir', None)

        if 'arch_triplet' in sys_settings.keys():
            if sys_settings['arch_triplet'].startswith("aarch64"):
                arch = "qemu-aarch64"  # only this target is now supported

        if extra_include_dirs:
            for item in extra_include_dirs:
                sys_lib_flags.append('-isystem')
                sys_lib_flags.append(item)

        if self.build_mode == Buildmode.GCC_DRIVER:
            for name in names: # name is relative path to TU from the tmp dir
                self.filepaths.append([tmpdir + os.sep + nm for nm in name.split(" ")])
                self.compile_cmds.append(
                    [self.compiler] + [" ".join(self.filepaths[-1])] + extra_obj + sys_lib_flags + extra_compiler_flags
                )

        elif self.build_mode == Buildmode.LLVM_PIPELINE:
            for name in names: # name is relative path to TU from the tmp dir
                self.filepaths.append([tmpdir + os.sep + nm for nm in name.split(" ")])
                self.compile_cmds.append(
                    [self.compiler] +
                    ['-emit-llvm', '-c' ,'-O0', '-Xclang', '-disable-O0-optnone', '-Xclang', '-disable-llvm-passes'] +
                    [" ".join(self.filepaths[-1])] + ['-o', name+'_nonopt.bc'] +
                     sys_lib_flags + extra_compiler_flags
                )

                self.compile_cmds.append(
                    [llvm_opt] +
                    [opt_baseline, name + '_nonopt.bc', '-o', name + '_opt_baseline.bc']
                )

                self.compile_cmds.append(
                    [llvm_opt] +
                    [name + '_opt_baseline.bc', '-o', name + '_opt_optimized.bc']
                )
                self.compile_cmds.append(
                    [self.compiler] +
                    ['-c', ] +
                    [name + '_opt_optimized.bc'] + ['-o',  name + '.o'] +
                    sys_lib_flags + extra_compiler_flags
                )

            self.compile_cmds.append(
                [self.compiler] +
                [name + '.o' for name in names] +
                extra_obj + sys_lib_flags + extra_compiler_flags
            )

        elif self.build_mode == Buildmode.MAKE:
            try:
                self.filepaths.append(names[0]) # expect root target as a name
            except:
                self.filepaths.append(tmpdir + " all")
            env_vars = sys_settings.get('env_vars', {})
            self.compile_cmds.append(
                [self.compiler] + [tmpdir + " " + self.filepaths[-1] + " ".join([k+"="+v for k,v in env_vars.items()])]
            )
        if output_run_artifact != "a.out":
            self.outfile.append(self.tmpdir + os.sep + "a.out")
        self.outfile.extend([output_run_artifact]) # by default, there are only one running artifact
        self.timeout_seconds = run_timeout_seconds
        if arch == "native":
            self.run_cmd.extend(["./" + self.outfile[-1]] + run_args)
        elif arch == "qemu-aarch64":
            self.run_cmd.argument.extend(
                ["qemu-aarch64 -L " + sys_settings['target_libs_dir'] + " ./" + output_run_artifact] + run_args)

    def compile(self, opt=None):
        if not opt and len(self.opt_prepend) > 0:
            opt = list()
        if len(self.opt_prepend) > 0:
            opt += self.opt_prepend
        for _cmd in self.compile_cmds:
            cmd = _cmd[:]

            if opt:
                if self.build_mode == Buildmode.MAKE:
                    if not opt[0].startswith('\''):
                        opt[0] = '\'' + opt[0]
                        opt[-1] = opt[-1] + '\''
                    cmd.append(self.opt_env_var_name+"="+" ".join(opt))
                elif self.build_mode == Buildmode.GCC_DRIVER:
                    cmd.append(" ".join(opt))
                elif self.build_mode == Buildmode.LLVM_PIPELINE:
                    if not cmd[1].startswith("-O") and cmd[0] != self.compiler:
                        if self.pass_manager_ver == "old" or (" ".join(opt).startswith("-O")):
                            cmd.insert(1, " ".join(opt))
                        else:
                            cmd.insert(1,  '-passes="'+ ",".join(opt) +'"')
            results = subprocess.run([" ".join(cmd)], text=True, shell=True, capture_output=True)
            with open(self.log_file_path, 'w+') as fp:
                output = "Building by cmd: " + str(cmd) + ":\n" + "STDOUT: " + str(results.stdout) + "\nSTDERR: " + str(results.stderr)
                fp.write(output)
                #print(output)
            if (results.returncode != 0):
                printRed("Compile error on cmd: " + str(cmd))
                self.last_compile_success = False
                print(output)
                return 1, cmd
        #printGreen("Compile success.")
        #printYellow(output)
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
        if self.run_working_dir:
            shutil.copyfile(self.outfile[-1], self.run_working_dir + os.sep + self.outfile[-1])
        results = subprocess.run([" ".join(pre_cmd + self.run_cmd)], text=True, shell=True, capture_output=True,
                                 cwd=self.tmpdir if not self.run_working_dir else self.run_working_dir)
        with open(self.log_file_path, 'w+') as fp:
            output = "Run by cmd: " + str(pre_cmd + self.run_cmd) + ":\n" + "STDOUT: " + str(
                results.stdout) + "\nSTDERR: " + str(
                results.stderr)
            fp.write(output)
            #print(output)

        if results.returncode != 0:
            printRed("Run failed.")
            print(output)
            return 1, 0.
        else:
            #printGreen("Run success.")
            out = results.stdout
            #printGreen(out)
            #print("stderr", results.stderr)
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


class CompilerEnv:
    def __init__(self,  # creates a new environment (same as gym.make)
                 config=dict(),  # selects the compiler to use
                 benchmark=standalone_benchmark(),  # selects the program to compile
                 observation_space=None,  # selects the observation space
                 reward_spaces=[],  # selects the optimization target
                 action_space=list(FLAGS["reverse_actions_filter_map"].keys())
                 ):
        self.action_history = list()
        self.action_space = action_space
        self.benchmark = benchmark
        self.observation_space=observation_space
        self.config=config
        self.reward_spaces = reward_spaces
        printGreen("Creating new standalone environment")
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def reset(self):
        printGreen("Resetting standalone environment...")
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

    def multistep(self, actions: list, reward_func=reward_adapter):
        state, reward, done, info = self.probe(actions, reward_func)
        self.action_history += actions
        self.state = state
        return state, reward, done, info

    def probe(self, actions: list, reward_func=reward_adapter, need_pre_check={'need':False,
                                                                               'first_priority_metrics':[2]}):
        prev_state = self.state
        done = False
        info = None
        seq = self.action_history + actions
        self.benchmark.compile(opt=seq)
        if need_pre_check:
            '''
            If this action makes results worse than previous, propagate the flag "need_ignore" to use wherever next
            for decision making, and decide that this probe is failed
            '''
            for  i in need_pre_check['first_priority_metrics']:
                prior_metrics_val = self.reward_spaces[i-1].evaluate(env=self)
                if prev_state[i] <= prior_metrics_val:
                    return self.state, float('-inf'), done, {"need_ingore": True}
        reward_metrics = list()
        for rw_meter in self.reward_spaces:
            reward_metrics.append(rw_meter.evaluate(env=self))
        reward = reward_func(reward_metrics, prev_state[1:], [r.kind for r in self.reward_spaces])
        state = [None] + [r for r in reward_metrics]
        return state, reward, done, info


# ========================  stuff for simplified search algorithms =================================
def check_each_action(env: CompilerEnv, reward_if_list_func=np.mean):
    passes_results = []
    prev_state = env.state
    prev_size = prev_state[2]
    prev_runtime = reward_if_list_func(prev_state[1])
    for idx, action in enumerate(env.action_space):
        state, r, d, i = env.probe([action], need_pre_check={'need':True, 'first_priority_metrics': [2]})
        if i:
            if i.get("need_ignore", "False"):
                print("Ignoring action as non-effective:", action)
                continue
        passes_results.append( {"action": action,
                    "action_num": idx,
                    "reward": r,
                    "prev_size": prev_size,
                    "size": state[2],
                    "prev_runtime": prev_runtime,
                    "runtime": reward_if_list_func(state[1]),
                    "size gain %": (prev_size - state[2]) / prev_size * 100
                } )
    return passes_results


def search_episode(env: CompilerEnv, heuristics="least_from_positive_sampling", steps=FLAGS["episode_len"]):
    episode_reward = 0.0
    episode_size_gain = 0.0
    for i in range(steps):
        results = check_each_action(env)
        if heuristics == "least_from_positive_sampling":
            positive = search_policies.pick_least_from_positive_samples(results)
        state, reward, done, info = env.step(positive[0]['action'])

    return positive
    # ========================

def test_gnumake():
    gbm = standalone_benchmark(build_mode=Buildmode.MAKE)
    gbm.make_benchmark(tmpdir="third_party/cbench/cBench_V1.1/security_blowfish_d/src",
                    names=[""], run_args=["1"], sys_settings={'output_bin': "__run", 'opt_var_name': "CCC_OPTS", "opt_prepend":["-O0 "]})
    env = CompilerEnv(benchmark=gbm, reward_spaces=[RuntimeRewardMetrics(), ObjSizeBytesRewardMetrics()])
    state = env.reset()
    return env


def cbench_env(name, mode="default", settings=None):
    gbm = standalone_benchmark(build_mode=Buildmode.MAKE)
    placement = "third_party/cbench/cBench_V1.1/bzip2d/src"
    if name == "bzip2d":
        placement = "third_party/cbench/cBench_V1.1/bzip2d/src"
    if name in ["gsm", "telecom_gsm"]:
        placement = "third_party/cbench/cBench_V1.1/telecom_gsm/src"

    gbm.make_benchmark(tmpdir=placement,
                       names=[""], run_args=["1"], sys_settings={'output_bin': "__run", 'opt_var_name': "CCC_OPTS","opt_prepend":["-O2"]})
    env = CompilerEnv(benchmark=gbm, reward_spaces=[RuntimeRewardMetrics(), TextSizeBytesRewardMetrics()])
    state = env.reset()
    return env


def test_makebydriver_gcc():
    gbm = standalone_benchmark(build_mode=Buildmode.GCC_DRIVER)
    gbm.make_benchmark(tmpdir=FLAGS['tmpdir'], names=["program.c"])
    env = CompilerEnv(benchmark=gbm, reward_spaces=[RuntimeRewardMetrics(), TextSizeBytesRewardMetrics()])
    state = env.reset()
    return env


def test_makeby_clang_llvm():
    gbm = standalone_benchmark(build_mode=Buildmode.LLVM_PIPELINE)
    gbm.make_benchmark(tmpdir=FLAGS['tmpdir'], names=["program.c", "1.c"])
    env = CompilerEnv(benchmark=gbm, reward_spaces=[RuntimeRewardMetrics(), TextSizeBytesRewardMetrics()], action_space=actions_oz_extra)
    state = env.reset()
    return env
    # ===================================================================================================


def tune_by_clang_llvm_cbench(name="gsm", action_space=actions_oz_extra, sys_settings={}):
    print("Tuning for cbench test", name)
    cbench_test_path = bench_configs.cbench[name]["src"]
    gbm = standalone_benchmark(build_mode=Buildmode.LLVM_PIPELINE)
    sys_settings.update({'run_working_dir': cbench_test_path,
                         'extra_compiler_flags': bench_configs.cbench[name]["extra_c_flags"]})
    gbm.make_benchmark(tmpdir=FLAGS['tmpdir'],
                       names=file_utils.get_filenames_in_dir(directory=cbench_test_path, ext=".c"),
                       sys_settings=sys_settings,
                       run_args=bench_configs.cbench[name]["run_args"] +
                                  [os.path.join(FLAGS['tmpdir'],
                                  os.path.join(cbench_test_path, bench_configs.cbench[name]["test_data_file_path"]))] +
                                  bench_configs.cbench[name]["post_run_args"])
    env = CompilerEnv(benchmark=gbm, reward_spaces=[RuntimeRewardMetrics(), TextSizeBytesRewardMetrics()], action_space=action_space)
    state = env.reset()
    return env


def prepare_env_with_baseline(name="crc32", action_space=actions_oz_extra, base='-O2', sys_settings={}):
    env = tune_by_clang_llvm_cbench(name=name, action_space=action_space, sys_settings=sys_settings)
    baseline_state, _, _, _ = env.multistep(actions=[base])
    baseline_size = baseline_state[2]
    baseline_runtime = np.mean(baseline_state[1])
    env.reset()
    return env, baseline_size, baseline_runtime


def examine_sequence(name="crc32", action_space=actions_oz_extra, base='-O2'):
    print("Examine sequence for test", name, "with baseline", base)
    env, baseline_size, baseline_runtime = prepare_env_with_baseline(name=name, action_space=action_space, base=base)
    actions = action_space
    state, r, d, _ = env.multistep(actions=actions)
    size = state[2]
    runtime = np.mean(state[1])

    if runtime <= baseline_runtime and size < baseline_size:
        result = {'actions': actions, 'size_gain': (baseline_size - size) / 100.,
                         'rt_gain': (baseline_runtime - runtime) / 100.}
        printLightPurple("Sequence is better than baseline")
        print(runtime , " <= " , baseline_runtime, " and " , size , " < " ,
              baseline_size)
        pprint.pprint(result)
        return result
    else:
        return {}


def reordering_clang_llvm_cbench(name="crc32", action_space=actions_oz_extra, num_iterations=FLAGS["search_iterations"],
                                 base='-O2', sys_settings={}):
    env = tune_by_clang_llvm_cbench(name=name, action_space=action_space, sys_settings=sys_settings)
    seq_list = []

    baseline_state, _, _, _ = env.multistep(actions=[base])
    baseline_size = baseline_state[2]
    baseline_runtime = np.mean(baseline_state[1])
    env.reset()

    s_list = sequences.get_permutations(action_space, num=num_iterations)

    for i in range(num_iterations):
        printRed("Iteration " + str(i))
        actions = s_list[i]
        state, r, d, _ = env.multistep(actions=actions)
        size = state[2]
        runtime = np.mean(state[1])
        print(runtime)
        env.reset()
        print(runtime)
        print(runtime, "<=", baseline_runtime, "and", size, "<", baseline_size)
        if runtime <= baseline_runtime and size < baseline_size:
            seq_list.append({'actions': actions, 'size_gain': (baseline_size - size)/100.,'rt_gain': (baseline_runtime - runtime)/100.})
            printLightPurple("Sequence is better than baseline")
            print(runtime, "<=" , baseline_runtime, "and", size, "<", baseline_size)
            pprint.pprint(seq_list[-1])
    return seq_list


def test_heuristic_search(name="gsm", action_space=actions_oz_extra, sys_settings={}):
    env = tune_by_clang_llvm_cbench(name=name, action_space=action_space, sys_settings=sys_settings)
    seq_list = []
    for i in range(FLAGS["search_iterations"]):
        printRed("Iteration " + str(i))
        seq_list.append(search_strategy_eval(env,
             reward_estimator=const_factor_threshold,
             pick_pass=search_policies.pick_least_from_positive_samples,
             dump_to_json_file="results" + os.sep + "test_" + str(os.getpid()) + "_" + "_" + str(i) + ".json",
             mode='compiler_sa', examiner=check_each_action))
    positive_res = [s for s in seq_list if s["episode_reward"] >= 0.]
    return positive_res


def dobivator(name="gsm", base='-O2', action_space=actions_oz_extra, sys_settings={}):
    print("Experiment on adjusting", name, base)
    return test_heuristic_search(name=name, action_space=action_space, sys_settings=sys_settings.update({'opt_baseline': base}))


def test_reordering(name="gsm", action_space=actions_oz_baseline, sys_settings={}):
    res = reordering_clang_llvm_cbench(name=name, action_space=action_space, base="-O2", sys_settings=sys_settings)
    printLightPurple("Sequences better than -O2:")
    for i in res:
        pprint.pprint(i)


def test_seq():
    l = ['-dce', '-break-crit-edges', '-memcpyopt', '-licm', '-dse', '-newgvn', '-simplifycfg', '-mem2reg', '-instcombine', '-loop-reduce', '-adce', '-sroa', '-deadargelim', '-jump-threading', '-mergereturn', '-loop-deletion', '-aggressive-instcombine', '-gvn-hoist', '-prune-eh', '-lcssa', '-loop-simplify', '-early-cse-memssa', '-loop-versioning', '-gvn', '-early-cse-memssa', '-sccp', '-die', '-reassociate', '-loop-rotate', '-mergefunc']
    pprint.pprint(examine_sequence(name='gsm', action_space=l))


if __name__ == '__main__':
    l= []
    with open("passes/O2_simpl.txt","r") as pss:
        __l = pss.read()
        l = __l.split("\n")

    test_reordering(name="crc32", action_space=l, sys_settings={'llvm_pass_manager':"new", 'opt':'opt-15', 'compiler':'clang-15'})
    #test_seq()