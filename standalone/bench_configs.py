cbench = {
    'bzip2d': { # target
        "src": 'third_party/cbench/cBench_V1.1/bzip2d/src',
        "extra_c_flags": [],
        "run_args" : ["-d", "-k", "-f", "-c"],
        "test_data_file_path": '../../bzip2_data/1.bz2',
        "post_run_args": ["> /dev/null"]
    },
    'bzip2e': { # target
        "src": 'third_party/cbench/cBench_V1.1/bzip2e/src',
        "extra_c_flags": [],
        "run_args" : ["-z", "-k", "-f", "-c"],
        "test_data_file_path": '../../automotive_qsort_data/1.dat',
        "post_run_args": ["> /dev/null"]
    },
    'gsm': {
        "src": 'third_party/cbench/cBench_V1.1/telecom_gsm/src',
        "extra_c_flags": ["-DSASR", "-DSTUPID_COMPILER", "-DNeedFunctionPrototypes=1"],
        "run_args" : ["-fps", "-c"],
        "test_data_file_path": '../../telecom_gsm_data/1.au',
        "post_run_args": ["> /dev/null"]
    },
    'crc32': {
        "src": 'third_party/cbench/cBench_V1.1/telecom_CRC32/src',
        "extra_c_flags": ['-c'],
        "run_args" : [],
        "test_data_file_path": '../../telecom_data/1.pcm',
        "post_run_args": ["> /dev/null"]
    },
}