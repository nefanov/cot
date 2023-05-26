import random as rnd
import os
from subprocess import *
from time import perf_counter

def run_opt_and_compile(passes, legacy=False):
    files = [i[:-3] for i in os.listdir('../gsm/telecom_gsm/src/bc_unoptimized')]
    for name in files:
        if legacy:
            res = call(f'''opt-16 -enable-new-pm=0 {' '.join(['-' + i for i in passes])} -o ../gsm/telecom_gsm/src/bc_optimized/{name}.bc ../gsm/telecom_gsm/src/bc_unoptimized/{name}.bc''', shell=True)
        else:
            res = call(f'''opt-16 -passes="{','.join(passes)}" -o ../gsm/telecom_gsm/src/bc_optimized/{name}.bc ../gsm/telecom_gsm/src/bc_unoptimized/{name}.bc''', shell=True)
        if res != 0:
            os._exit(res)

    res = call('clang-16 -lm ../gsm/telecom_gsm/src/bc_optimized/* -o bench.elf', shell=True)
    if res != 0:
        os._exit(res)

def bench_time(target_time=0):
    agg_time = 0
    for i in range(0, 10):
        start_time = perf_counter()
        run('./bench.elf -fps -c ./1.au > /dev/null', shell=True)
        end_time = perf_counter()
        diff_time = (end_time - start_time)
        agg_time += diff_time
        if (target_time != 0) & (diff_time > (target_time + 0.3)):
            return diff_time
    return agg_time / 10

def import_passes(path, legacy=False, expand=False):
    f = open(path, 'r')
    if legacy:
        passes = [i[1:-1] for i in f.readlines()]
    elif expand:
        read_passes = [i[:-2] for i in f.readlines()]
        passes = []
        for string in read_passes:
            passes += expand_pass(string).split(',')
    else:
        passes = [i[:-2] for i in f.readlines()]
    f.close()
    return passes

def find_delim(string):
    open_index = string.find('(')
    counter = 1
    for i, elem in enumerate(list(string[open_index + 1:])):
        if elem == '(':
            counter += 1
        elif elem == ')':
            counter -= 1
        if counter == 0:
            close_index = i
            break
    return (open_index, close_index + open_index + 1)

def expand_pass(string):
    sign = [',']
    position = [-1]
    i = 0
    while i < len(string):
        elem = string[i]
        if elem in ('(', ','):
            sign.append(elem)
            position.append(i)
        elif elem == ')':
            j = -1
            while sign[j] != '(':
                j -= 1
            sign = sign[:j]
            pos = position[j]
            position = position[:j]
            name = string[position[-1]+1:pos]
            subelems = string[pos+1:i].split(',')
            n = len(subelems)
            namelen = len(name)
            for j, subelem in enumerate(subelems):
                subelems[j] = name + f'({subelem})'
            string = string[:position[-1]+1] + ','.join(subelems) + string[i+1:]
            i += (namelen + 2) * (n - 1)
        i += 1
    return string

passes = import_passes('../passes/O2_passes.flags', expand=False)
run_opt_and_compile(passes)
baseline_size = int(run('size bench.elf', shell=True, capture_output=True).stdout.split()[6])
print('baseline size: ', baseline_size)
baseline_time = bench_time();
print('baseline time: ', baseline_time)

f_baseline = open('baseline.log', 'w')
f_baseline.write(f'baseline size {baseline_size} bytes\n')
f_baseline.write(f'baseline time {baseline_time} seconds\n')
f_baseline.close()

f_log = open('./log.log', 'a')

hit_count = 0

while hit_count < 5:
    rnd.shuffle(passes)
    run_opt_and_compile(passes)
    size = int(run('size bench.elf', shell=True, capture_output=True).stdout.split()[6])
    if size > baseline_size:
        print('pass sequence failed size test')
        continue

    time = bench_time(baseline_time)
    if baseline_time - time > 0.1:
        time_decrease = baseline_time - time
        f_log.write(f'size {size} bytes\n')
        f_log.write(f'size decrease {baseline_size - size} bytes [{(baseline_size - size) / baseline_size}]\n')
        f_log.write(f'time {time} seconds\n')
        f_log.write(f'time decrease {time_decrease} seconds [{time_decrease / baseline_time}]\n')
        f_log.write(','.join(passes))
        f_log.write('\n')
        print(f'better pass sequence found and logged, time decrease {time_decrease}')
        hit_count += 1
        if time_decrease > 0.3:
            f_good_log = open('./fast.log', 'a')
            f_good_log.write(f'size {size} bytes\n')
            f_good_log.write(f'size decrease {baseline_size - size} bytes [{(baseline_size - size) / baseline_size}]\n')
            f_good_log.write(f'time {time} seconds\n')
            f_good_log.write(f'time decrease {time_decrease} seconds [{time_decrease / baseline_time}]\n')
            f_good_log.write(','.join(passes))
            f_good_log.write('\n')
            f_good_log.close()
    else:
        print(f'pass sequence failed time test with time {time}')

f_log.close()
