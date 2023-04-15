"""
1. Extract the best subsequences
2. Filter
2.1. Get intersection of the best subsequences
2.2. Construct more long sequences from the intersection, if the variety is big. Else, work with the best from 1.
3. Validation
Validate the sequences from 2 on each of the tests from benchmark.
"""

from common import *
from search_algorithms import subsequence_eval
from experiment_runner import make_env
import glob
import operator


def read_statistics(files_list):
    data = []
    for fentry in files_list:
        action_list = read_action_log_from_json(fentry)
        d = dict()
        d.update({'logfile': files_list})
        d.update({'action_list': action_list})
        data.append(d)
    return data


def get_non_neg_from_start(size_gains: list, rt_gains: list, start =0, patience_rt_max=3, patience_sz_max=1)-> tuple:
    rt = 0.
    size = 0.
    pat_rt = 0
    pat_sz = 0
    if start >= len(size_gains) - 1:
        return (start, start)
    for i in range(start, len(size_gains)):
        if rt_gains[i] < 0. or rt < 0.:
            pat_rt += 1
            if pat_rt > patience_rt_max:
                return (start,i-1)
        if size < 0.:
            pat_sz += 1
            if pat_sz > patience_sz_max:
                return (start,i-1)
        pat_sz = 0
        pat_rt = 0
        rt += rt_gains[i]
        size += size_gains[i]
    if size <= 0. or rt <= 0.:
        result = get_non_neg_from_start( size_gains[:-1], rt_gains[:-1], start)
        return result

    return (start, len(size_gains)-1)


def get_best_from_statistics(files_list: list, subseq_mining_algo=get_non_neg_from_start) -> list:
    stat = read_statistics(files_list)
    prefer_seq = []
    for estimation in stat:
        actions = [a['action'] for a in estimation['action_list']]
        action_nums = [a['action_num'] for a in estimation['action_list']]
        sizes = [a['size'] for a in estimation['action_list']]
        runtimes = [a['runtime'] for a in estimation['action_list']]
        size_gains = [a['size gain %'] for a in estimation['action_list']]
        rt_gains = [(a['prev_runtime'] - a['runtime']) / 100. for a in estimation['action_list']]
        subseq_ids = subseq_mining_algo(size_gains, rt_gains)
        #print(actions[subseq_ids[0]:subseq_ids[1]+1],
        #      estimation['action_list'][subseq_ids[0]]['size'], estimation['action_list'][subseq_ids[1]]['size'],
        #      estimation['action_list'][subseq_ids[0]]['runtime'], estimation['action_list'][subseq_ids[1]]['runtime'])

        if estimation['action_list'][subseq_ids[0]]['size'] > estimation['action_list'][subseq_ids[1]]['size'] and \
                estimation['action_list'][subseq_ids[0]]['runtime'] > estimation['action_list'][subseq_ids[1]]['runtime']:
            prefer_seq.append({
                'action_list': actions[subseq_ids[0]:subseq_ids[1] + 1],
                'action_nums': action_nums[subseq_ids[0]:subseq_ids[1] + 1],
                'sizes': sizes[subseq_ids[0]:subseq_ids[1] + 1],
                'runtimes': runtimes[subseq_ids[0]:subseq_ids[1] + 1],
                'size_gains': size_gains[subseq_ids[0]:subseq_ids[1] + 1],
                'rt_gains': rt_gains[subseq_ids[0]:subseq_ids[1] + 1],
                })

    print("===========PREFERRED SEQUENCES=============")
    pprint.pprint(prefer_seq)
    return prefer_seq


def test1():
    print(read_action_log_from_json("results_oz/28979_gsm_49.json"))


def sequential_slice(iterable, length):
    pool = tuple(iterable)
    assert 0 < length <= len(pool)
    tails = (pool[s:] for s in range(length))
    return zip(*tails)


def sequence_in_list(sequence, lst):
    pool = tuple(sequence)
    return any((pool == s for s in sequential_slice(lst, len(pool))))


def lcs(a, b):
    if len(a) > len(b):
        a, b = b, a
    for l in reversed(range(1, len(a)+1)):
        seq = [subseq for subseq in sequential_slice(a, l) if sequence_in_list(subseq, b)]
        if seq:
            break
    return seq


def static_validate_seq(item, l):
    size = len(l)
    for i in range(0, len(item['action_list']) - size + 1):
        if item['action_list'][i: i + size] == l and item['sizes'][i] > item['sizes'][i + size-1]:
            return True
    return False


def extract_subsequences(stat: list):
    freq_voc = {}
    for i, item_i in enumerate(stat):
        iter = i / len(stat) * 100
        if iter % 10 == 0 or i == len(stat) - 1:
            print("Processing subsequences: ", iter, "% completed")
        for item_j in stat:
            l = lcs(item_i['action_list'], item_j['action_list'])
            for subseq in l:
                list_subseq = list(subseq)
                if list_subseq == [] or len(list_subseq) == 1:
                    continue
                if static_validate_seq(item_j, list_subseq) and static_validate_seq(item_i, list_subseq):
                    val = freq_voc.get(str(list_subseq), {'actions': list_subseq, 'freq': 0})
                    val['freq'] += 1
                    freq_voc.update({str(list_subseq): val, })
    return freq_voc


def generate(subsequences: list):
    result = []
    return result


def filter_seq(sorted_freq_list, freq=2, length=3 ):
    res = []
    for item in sorted_freq_list:
        if len(item['actions']) >= length and item['freq'] >= freq:
            res.append(item)
    return res


def validate_subseq(input_list, validation_list, eval_func=subsequence_eval):
    valid_subsequences = dict()
    for test in validation_list:
        valid_subsequences[test] = []
        benchmark = "cbench-v1/" + test
        actions = actions_oz_extra
        with make_env(actions_whitelist_names=actions, benchmark=benchmark) as env:
            for subseq in input_list:
                printLightPurple("Test " + test + ": subsequence" + str(subseq))
                results = eval_func(env, subseq['actions'], reward_estimator=const_factor_threshold)
                if results[0]['prev_size'] < results[-1]['size'] or results[0]['prev_runtime'] < results[-1]['runtime']:
                    printRed("Test " + test + " : degrades on " + str(subseq['actions']))
                else:
                    print(results)
                    valid_subsequences[test].append(results)

    return valid_subsequences


def gen_pipeline(l: list):
    b_stat = get_best_from_statistics(l) # debug only
    freq = extract_subsequences(b_stat)

    sorted_freq_list = reversed(sorted([v for _, v in freq.items()], key=lambda d: d['freq']))
    res = filter_seq(sorted_freq_list, freq=5, length=4)
    print("========FILTERED RESULTS==========")
    for i, item in enumerate(res):
        print(i, ":", item)
    # validation
    validation_list = [
        "gsm",
        "bzip2",
        "stringsearch",
        "tiff2bw",
        "sha",
        "qsort",
        "patricia",
        "crc32",
    ]
    valid_subseq = validate_subseq(res, validation_list)
    print("==============VALIDATION=============")
    print("Validation on tests:", validation_list)
    print("--------------------------------------")
    print("Sequences which are good on all of the tests:")
    print("--------------------------------------")
    all_good_seq = {}

    for k, v in valid_subseq.items():
        all_good_seq[k] = []
        for estimation in v:
            sequence = [e['action'] for e in estimation]
            all_good_seq[k].append(sequence)
    pprint.pprint(all_good_seq)

    congl = {}
    for k,v in all_good_seq.items():
        for l in v:
            congl_value = congl.get(str(l), {'actions': [], 'test': [], 'freq': 0})
            congl_value['actions'] = l
            congl_value['test'].append(k)
            congl_value['freq'] += 1
            congl.update({str(l): congl_value})
    print("Conglomerate freq-dict for the best results_oz:")
    congl_list = []
    congl_list = [v for k,v in congl.items()]
    congl_list_chart = sorted(congl_list, key=lambda d: d['freq'])
    for v in congl_list_chart:
        print(v['actions'], ":", v['freq'], "- on tests:", v['test'])
    with open("validation_results.json", "w") as final:
        json.dump(congl_list, final)
    keys = congl_list_chart[0].keys()
    with open('validation_results.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(congl_list_chart)
    return


if __name__ == '__main__':
    test1()
    gen_pipeline(get_json_files_list("results_2"))
