from common import *
from actor_critic import HistoryObservation, TrainActorCritic
from search_algorithms import search_strategy_eval


def make_env(extra_observation_spaces=None, benchmark=None, sz_baseline="TextSizeOz", actions_whitelist_names=None):
    if benchmark is None:
        benchmark = "cbench-v1/qsort"
    env = compiler_gym.make(  # creates a partially-empty env
                "llvm-v0",  # selects the compiler to use
                benchmark=benchmark,  # selects the program to compile
                observation_space=sz_baseline,  # initial observation
                reward_space=None,  # in future selects the optimization target
            )

    if actions_whitelist_names:
        FLAGS["actions_white_list"] = [env.action_spaces[0].names.index(action) for action in actions_whitelist_names]
        FLAGS["actions_filter_map"] = remap_actions(FLAGS["actions_white_list"])
        FLAGS["reverse_actions_filter_map"] = {v: k for k, v in FLAGS["actions_filter_map"].items()}
    env = TimeLimit(env, max_episode_steps=FLAGS["episode_len"])
    baseline_obs_init_val = env.reset()
    if not isinstance(extra_observation_spaces, OrderedDict):
        o = OrderedDict()
        o["TextSizeBytes"] = baseline_obs_init_val # this is only a baseline, and value is defaultly scalar -- chk type
        o["Runtime"] = np.zeros(1)
    else:
        o = extra_observation_spaces
    env = HistoryObservation(env, o)
    return env


def main(MODE="single_pass_validate"):
    call_evaluator = None
    if MODE == Runmode.AC_BASIC:
        pass
    elif MODE == Runmode.GREEDY:
        call_evaluator = pick_max_size_gain
    elif MODE == Runmode.RANDOM:
        call_evaluator = pick_random
    elif MODE == Runmode.RANDOM_POSITIVES:
        call_evaluator = pick_random_from_positive
    elif MODE == Runmode.RANDOM_POSITIVES_USED:
        call_evaluator = pick_random_from_positive_used
    else:
        print("Incorrect run mode")
        sys.exit(1)

    with make_env(actions_whitelist_names=actions_oz_extra) as env:
        if FLAGS['iterations'] == 1:
            TrainActorCritic(env, reward_estimator=const_factor_threshold)
            return
        if MODE != Runmode.AC_BASIC:
            seq_list_lens = []
            for i in range(FLAGS["search_iterations"]):
                print("Iteration", i)
                seq_list_lens.append(search_strategy_eval(env, reward_estimator=const_factor_threshold, pick_pass=call_evaluator))
            positive_res = [s for s in seq_list_lens if s["episode_reward"] >= 0.]

            print("Iteration", i, "statistics:")
            __max_size_gain = max(positive_res, key=lambda a: a["episode_size_gain"])
            #print("Action log for max size gain:")
            #pprint.pprint(__max_size_gain)
            print("Max size gain on sequence:")
            seq = [(d['action'], d['size gain %']) for d in __max_size_gain['action_log']]
            for item in seq:
                print(item)
            print("Total size gain:",
                  (env.hetero_os_baselines[0] - __max_size_gain['action_log'][-1]['size']) / env.hetero_os_baselines[
                      0]*100,"%")
            print("Total perf diff:",
                  (env.hetero_os_baselines[1] - __max_size_gain['action_log'][-1]['runtime']) / env.hetero_os_baselines[
                      1] * 100, "%")
            print("---------------------------------------------------------------------------")
            sys.exit(0)
        else:
            torch.manual_seed(FLAGS['seed'])
            random.seed(FLAGS['seed'])
            performances = []
            for i in range(1, FLAGS['iterations'] + 1):
                FLAGS["logger"].save_and_print(f"\n*** Iteration {i} of {FLAGS['iterations']}")
                performances.append(TrainActorCritic(env, reward_estimator=const_factor_threshold))

            FLAGS["logger"].save_and_print("\n*** Summary")
            FLAGS["logger"].save_and_print(f"Final performances: {performances}\n")
            FLAGS["logger"].save_and_print(f"  Best performance: {max(performances):.2f}")
            FLAGS["logger"].save_and_print(f"Median performance: {statistics.median(performances):.2f}")
            FLAGS["logger"].save_and_print(f"   Avg performance: {statistics.mean(performances):.2f}")
            FLAGS["logger"].save_and_print(f" Worst performance: {min(performances):.2f}")


if __name__ == "__main__":
    main(Runmode.RANDOM_POSITIVES_USED)
