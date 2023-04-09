"""
1. Extract the best subsequences
2. Filter
2.1. Get intersection of the best subsequences
2.2. Construct more long sequences from the intersection, if the variety is big. Else, work with the best from 1.
3. Validation
Validate the sequences from 2 on each of the tests from benchmark.
"""

from common import *


def read_statistics(files_list):
    data = []
    for fentry in files_list:
        action_list = read_action_log_from_json(fentry)
        data.append(action_list)
    return data

def get_best_from_statistics(data: list):
    pass

def test1():
    print(read_action_log_from_json("results/28979_gsm_49.json"))


if __name__ == '__main__':
    test1()