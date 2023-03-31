import random

random.seed(0)

def pick_max_size_gain(results: list) -> list:
    results = sorted(results, key=lambda d: d['size gain %'])
    return results


def pick_all_positive_size_gain(results: list) -> list:
    results = sorted(results, key=lambda d: d['size gain %'])
    if results[-1]['size gain %'] <= 0:
        return [results[-1]]
    return [item for item in results if item['size gain %'] > 0]


def pick_random_from_positive(results: list) -> list:
    return [random.choice(pick_all_positive_size_gain(results))]


def pick_random(results: list) -> list:
    return [random.choice(results)]
