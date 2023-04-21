from common import *


def single_pass_eval(env, reward_estimator=const_factor_threshold,
                     reward_if_list_func=lambda a: np.mean(a), need_reset=True):
    passes_list = FLAGS["reverse_actions_filter_map"]
    for k, v in passes_list.items():
        if need_reset:
            state = env.reset()
        prev_size = state[1]
        prev_runtime = reward_if_list_func(state[2])
        action = v
        state, r, d, _ = env.step(v)
        reward = reward_estimator(env.hetero_os_baselines[0], state[1], reward_if_list_func(env.hetero_os_baselines[1]), reward_if_list_func(state[2]), prev_size,prev_runtime)
        print("Action", env.action_spaces[0].names[action], "R", reward, "; size:", prev_size, "->", state[1], "; runtime:", prev_runtime, "->", reward_if_list_func(state[2]))


def subsequence_eval(env, subseq, reward_estimator=const_factor_threshold, reward_if_list_func=lambda a: np.mean(a)):
    state = env.reset()
    results = []
    for __name in subseq:
        v = env.action_spaces[0].names.index(__name)
        prev_size = state[1]
        prev_runtime = reward_if_list_func(state[2])
        state, r, d, _ = env.step(v)
        reward = reward_estimator(env.hetero_os_baselines[0],
                              state[1],
                              reward_if_list_func(env.hetero_os_baselines[1]),
                              reward_if_list_func(state[2]),
                              prev_size,
                              prev_runtime)

        results.append({"action": env.action_spaces[0].names[v],
                        "action_num": v,
                        "reward": reward,
                        "prev_size": prev_size, "size": state[1],
                        "prev_runtime": prev_runtime, "runtime": reward_if_list_func(state[2]),
                        "size gain %": (prev_size - state[1]) / prev_size * 100
            })

    return results


def one_pass_perform(env, prev_state, action, reward_estimator=const_factor_threshold, reward_if_list_func=lambda a: np.mean(a)):
    """
    one iteration of search: try all the passes, then return statistics
    """
    passes_list = FLAGS["reverse_actions_filter_map"]
    v = action
    ep_reward = 0
    prev_size = prev_state[1]
    prev_runtime = reward_if_list_func(prev_state[2])
    state, r, d, _ = env.step(v)
    reward = reward_estimator(env.hetero_os_baselines[0],
                              state[1],
                              reward_if_list_func(env.hetero_os_baselines[1]),
                              reward_if_list_func(state[2]),
                              prev_size,
                              prev_runtime)
    return {"action": env.action_spaces[0].names[v],
            "action_num": v,
            "reward": reward,
            "prev_size": prev_size, "size": state[1],
            "prev_runtime": prev_runtime, "runtime": reward_if_list_func(state[2]),
            "size gain %": (prev_size - state[1]) / prev_size * 100
            }


def examine_each_action(env, state, reward_estimator=const_factor_threshold, reward_if_list_func=lambda a: np.mean(a), step_lim=10):
    passes_list = FLAGS["reverse_actions_filter_map"]
    passes_results = []
    for k, v in passes_list.items():
        with copy.deepcopy(env) as copy_env:
            passes_results.append(one_pass_perform(copy_env, state, v, reward_estimator=reward_estimator, reward_if_list_func=reward_if_list_func))
    return passes_results


def max_subseq_from_start(seq: list, episode_reward=0.) -> list:
    """
    this is very slow (O(n)) implementation of max subseq search from start element
    """
    gain = episode_reward
    up_lim = len(seq) - 1
    for i in range(len(seq)):
        S = sum([s['size gain %'] for s in seq[:i]])
        if S > gain:
            gain = S
            up_lim = i
    return seq[:up_lim]


def search_strategy_eval(env, reward_estimator=const_factor_threshold,
                         reward_if_list_func=lambda a: np.mean(a),
                         step_lim=10, pick_pass=pick_random_from_positive, patience=FLAGS['patience'],
                         pick_pass_args='action_log', dump_to_json_file=None, mode="llvm", examiner=examine_each_action):
    state = env.reset()
    results = list()
    pat = 0
    action_log = []
    episode_reward = 0.0
    episode_size_gain = 0.0
    for i in range(step_lim):
        printLightPurple("step " + str(i))
        if mode == 'llvm':
            results = examiner(env, state, reward_estimator=reward_estimator, reward_if_list_func=reward_if_list_func)
        elif mode == 'gcc':
            results = examiner(env, reward_if_list_func=reward_if_list_func)
        param = {}
        if pick_pass_args == 'action_log':
            param.update({pick_pass_args: [a['action_num'] for a in action_log]}) # some strategies uses previous actions log | etc
        best = pick_pass(results, **param)[-1]
        if mode == 'llvm':
            step = best["action_num"]
        elif mode == 'gcc':
            step = best['action']
        state, reward, d, _ = env.step(best["action"])  # apply. state and reward updates
        action_log.append(best)
        try:
            episode_reward += reward
        except:
            pass
        episode_size_gain += best['size gain %']

        if best['reward'] <= .0:
            pat += 1
            if patience <= pat:
                print("=============PATIENCE LIMIT EXCEEDED===============")
                break

    print("====================================================")
    pprint.pprint(action_log)
    if dump_to_json_file:
        with open(dump_to_json_file, "w+") as results:
            json.dump(action_log, results)

    # find the subsequence from start, which gives max size gain
    action_log = max_subseq_from_start(action_log, episode_reward=episode_size_gain)
    return {"action_log": action_log, "episode_reward": episode_reward, "episode_size_gain": episode_size_gain}

