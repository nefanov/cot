def pick_max_size_gain(results: list):
    max_item = results[0]
    results = sorted(results, key=lambda d: d['size gain %']).reverse()
    for item in results:
        if item["size gain %"] > max_item["size gain %"]:
            max_item = item
    return max_item