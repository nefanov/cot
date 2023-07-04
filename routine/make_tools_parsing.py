import os, sys


def __parse_cbench_makefmt(s: str):
    res = dict()
    lines = s.split('\n')
    for i in range(len(lines)):
        if lines[i].startswith("ifeq"):
            print(lines[i+1])
            ll = lines[i+1].split("=")
            res.update({ll[0].lstrip().rstrip(): ll[1].lstrip().rstrip()})

        elif lines[i].startswith("all:"):
            cmdl = list()
            while not lines[i+1].startswith("clean:"):
                cmdl.append(lines[i+1].lstrip().rstrip())
                i += 1
            res.update({'cmd': cmdl})
    return res


def test_parse_utest():
    print(__parse_cbench_makefmt(
        "ifeq ($(CCC_OPTS),)\n\
 CCC_OPTS =\n\
endif\n\
ifeq ($(CCC_OPTS_ADD),)\n\
 CCC_OPTS_ADD =\n\
endif\n\
\n\
# Declarations\n\
ifeq ($(ZCC),)\n\
 ZCC = standalone\n\
endif\n\
ifeq ($(LDCC),)\n\
 LDCC = standalone\n\
endif\n\
ifeq ($(LD_OPTS),)\n\
 LD_OPTS = -o a.out\n\
endif\n\
\n\
# Actions\n\
all:\n\
        $(ZCC) $(CCC_OPTS) $(CCC_OPTS_ADD) -c *.c\n\
        $(LDCC) $(LD_OPTS) $(CCC_OPTS_ADD) -lm *.o\n\
\n\
clean:\n\
        rm -f a.out *.o *.a *.s *.i *.I"
    ))


def rewrite_makefile_for_llvm(fn, fnout, clang='clang', opt='opt'):
    with open(fn, 'r') as f_in:
        data = __parse_cbench_makefmt(f_in.read())
        lines = f_in.readlines()
        for i, l in enumerate(lines):
            pass
    with open(fnout, 'w+') as f_out:
        pass



test_parse_utest()
