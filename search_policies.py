def pick_max_size_gain(results: list):
    max_item = results[0]
    for item in results:
        if item["size gain %"] > max_item["size gain %"]:
            max_item = item
    return max_item