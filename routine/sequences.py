import numpy as np


def get_permutations(arr, num=200):
    return [list(np.random.permutation(arr)) for i in range(num)]


def test_get_permutations():
    print(get_permutations([i for i in range(100)], 200))