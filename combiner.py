"""
1. Extract the best subsequences
2. Filter
2.1. Get intersection of the best subsequences
2.2. Construct more long sequences from the intersection, if the variety is big. Else, work with the best from 1.
3. Validation
Validate the sequences from 2 on each of the tests from benchmark.
"""