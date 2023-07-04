import re

actions_oz_baseline = [
        "-break-crit-edges",
        "-early-cse-memssa",
        "-gvn-hoist",
        "-gvn",
        "-instcombine",
        "-instsimplify",
        "-jump-threading",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-versioning",
        "-mem2reg",
        "-newgvn",
        "-reg2mem",
        "-simplifycfg",
        "-sroa",
    ]

actions_test = [

        "-simplifycfg",

    ]

actions_oz_extra = [
        "-simplifycfg",
        "-loop-simplify",
        "-jump-threading",
        "-lcssa",
        "-instcombine",
        "-break-crit-edges",
        "-early-cse-memssa",
        "-gvn-hoist",
        "-gvn",
        "-newgvn",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-versioning",
        "-aggressive-instcombine",
        "-deadargelim",
        "-dce",
        "-adce",
        "-die",
        "-dse",
        "-loop-deletion",
        "-licm",
        "-mem2reg",
        "-memcpyopt",
        "-mergefunc",
        "-mergereturn",
        "-prune-eh",
        "-reassociate",
        "-early-cse-memssa",
        "-sroa",
        "-sccp",
    ]


def load_as_from_file(fn: str)-> list:
        with open(fn,'r') as f:
                data = f.read()
                return re.split('; |, |\*|\n', data)
