from experiment_runner import *


def test_make_env():
    """
    test env for different baselines
    """
    print("------------------ test_make_env test --------------------")
    bm_prefix = "cbench-v1/"
    actions = actions_oz_baseline
    benchmarks = [bm_prefix+"blowfish", bm_prefix + "gsm", bm_prefix + "qsort", bm_prefix + "bzip2", bm_prefix + "tiff2bw", bm_prefix + "crc32", bm_prefix+"stringsearch"]
    baselines = ['TextSizeO3', 'TextSizeO0', 'TextSizeOz', 'TextSizeBytes']
    for b in benchmarks:
        print("Benchmark:", b)
        for baseline in baselines:
            with make_env(actions_whitelist_names=actions, benchmark=b, obj_baseline=baseline) as env:
                state = env.reset()
                print("baseline:", baseline, "size:", state[1], "runtime:", np.mean(state[2]))
                continue
        print('-'*30)


def test_presets():
    """
    test env for different baselines
    """
    print("------------------ test_make_env test --------------------")
    bm_prefix = "cbench-v1/"
    actions = actions_oz_baseline
    benchmarks = [bm_prefix+"blowfish", bm_prefix + "gsm", bm_prefix + "qsort", bm_prefix + "bzip2", bm_prefix + "tiff2bw", bm_prefix + "crc32", bm_prefix+"stringsearch"]
    baselines = ['TextSizeBytes']
    for b in benchmarks:
        print("Benchmark:", b)
        for baseline in baselines:
            with make_env(actions_whitelist_names=actions, benchmark=b, obj_baseline=baseline) as env:
                state = env.reset()
                print("baseline:", baseline, "size:", state[1], "runtime:", np.mean(state[2]))
                S,_,_,_ = env.step("-O2")
                continue
        print('-'*30)

#test_make_env()
test_presets()