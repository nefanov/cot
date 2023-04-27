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
 ZCC = gcc\n\
endif\n\
ifeq ($(LDCC),)\n\
 LDCC = gcc\n\
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

test_parse_utest()
