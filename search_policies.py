def pick_max_size_gain(results: list):
    results = sorted(results, key=lambda d: d['size gain %']).reverse()
    return results[0]


def pick_all_positive_size_gain(results: list):
    results = sorted(results, key=lambda d: d['size gain %']).reverse()
    return [item for item in results if item['size gain %'] > 0]