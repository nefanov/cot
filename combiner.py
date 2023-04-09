"""
1. Extract the best subsequences
2. Filter
2.1. Get intersection of the best subsequences
2.2. Construct more long sequences from the intersection, if the variety is big. Else, work with the best from 1.
3. Validation
Validate the sequences from 2 on each of the tests from benchmark.
"""

from common import *
import glob


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


def get_best_from_statistics(files_list: list, subseq_mining_algo=get_non_neg_from_start)-> list:
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


def test1():
    print(read_action_log_from_json("results/28979_gsm_49.json"))


def test2(l: list):
    get_best_from_statistics(l)


if __name__ == '__main__':
    test1()
    test2(get_json_files_list("results"))
