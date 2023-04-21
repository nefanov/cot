import random

random.seed(0)


def pick_max_size_gain(results: list, **kwargs) -> list:
    results = sorted(results, key=lambda d: d['size gain %'])
    return results


def pick_all_positive_size_gain(results: list, **kwargs) -> list:
    results = sorted(results, key=lambda d: d['size gain %'])
    if results[-1]['size gain %'] <= 0:
        return [results[-1]]
    return [item for item in results if item['size gain %'] > 0]


def pick_all_positive_size_gain_rand_if_neg(results: list, **kwargs) -> list:
    results = sorted(results, key=lambda d: d['size gain %'])
    if results[-1]['size gain %'] <= 0.:
        return [random.choice(results)]
    return [item for item in results if item['size gain %'] > 0.]


def pick_least_from_positive(results: list, **kwargs) -> list:
    res = pick_all_positive_size_gain_rand_if_neg(results)
    return [res[0]]


def pick_least_from_positive_samples(results: list, **kwargs) -> list:
    res = pick_all_positive_size_gain_rand_if_neg(results)
    ns = 1
    try:
        ns = kwargs['num_samples']
    except:
        pass
    return [min([random.choice(res) for i in range(ns)], key=lambda d: d['size gain %'])]


def pick_greatest_from_positive_samples(results: list, **kwargs) -> list:
    res = pick_all_positive_size_gain_rand_if_neg(results)
    ns = 1
    try:
        ns = kwargs['num_samples']
    except Exception as e:
        pass
    return [max([random.choice(res) for i in range(ns)], key=lambda d: d['size gain %'])]


def pick_random_from_positive(results: list, **kwargs) -> list:
    return [random.choice(pick_all_positive_size_gain(results))]


def pick_random(results: list, **kwargs) -> list:
    return [random.choice(results)]


def pick_random_from_positive_used(results: list, **kwargs) -> list:
    res = pick_all_positive_size_gain(results)
    try:
        candidate = None
        history = kwargs['action_log']
        for i in range(len(res)):
            picked = random.choice(res)
            if picked['action_num'] not in history:
                candidate = picked
                return [candidate]

        if not candidate:
            print("Pick candidate randomly")
            return [random.choice(results)]
    except Exception as e:
        print(__name__, e, ": , just return random from whole results")
        return [pick_random(res)]
