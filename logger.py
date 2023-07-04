import sys, os
from enum import Enum
from collections import OrderedDict
from datetime import datetime


def printRed(skk): print("\033[91m {}\033[00m".format(skk))


def printGreen(skk): print("\033[92m {}\033[00m".format(skk))


def printYellow(skk): print("\033[93m {}\033[00m".format(skk))


def printLightPurple(skk): print("\033[94m {}\033[00m".format(skk))


def printPurple(skk): print("\033[95m {}\033[00m".format(skk))


def printCyan(skk): print("\033[96m {}\033[00m".format(skk))


def printLightGray(skk): print("\033[97m {}\033[00m".format(skk))


def printBlack(skk): print("\033[98m {}\033[00m".format(skk))

class LogMode(Enum):
    OFF = 0
    SHORT = 1
    VERBOSE = 2


class Logger:
    def __init__(self, mode = LogMode.SHORT):
        self.storage = OrderedDict()
        self.mode = mode

    def log(self, entity, mode=LogMode.SHORT):
        if mode.value > self.mode.value:
            return
        else:
            dt = datetime.now()
            ts = datetime.timestamp(dt)
            self.storage[ts] = {dt:entity, 'logtype':mode}
        return

    def dump(self, mode=LogMode.SHORT):
        if mode.value > self.mode.value:
            return
        for k, entity in self.storage.items():
            if (entity['logtype'].value <= mode.value):
                print(entity)

    def print_last(self, mode=LogMode.SHORT):
        if mode.value <= self.mode.value:
            nrs = next(reversed(self.storage))
            if (nrs['logtype'].value <= mode.value):
                print(nrs)

    def save_and_print(self, entity, mode=LogMode.SHORT):
        self.log(entity, mode)
        print(entity)

    def clear(self):
        self.storage.clear()
