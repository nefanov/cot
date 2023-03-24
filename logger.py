import sys, os
from enum import Enum
from collections import OrderedDict
from datetime import datetime


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
