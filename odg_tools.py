'''
[2023] Nikolay Efanov
odg_tools module
Tools for manipulation with order-dependency graphs (ODG), which are frequently used in compiler phase-ordering task for subsequences of optimizing passes construction
'''
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum


class ODGMakingMode(Enum):
    FULLYCONNECTED = 0
    STOPLIST = 1
    ALLOWLIST = 2


class ODG:
    def __init__(self, fullseq, mode=ODGMakingMode.FULLYCONNECTED):
        print(sorted(nx.complete_graph(len(fullseq), nx.DiGraph())))
        if mode == ODGMakingMode.FULLYCONNECTED:
            mapping = {}
            for i, item in enumerate(fullseq):
                mapping.update({i: item})
            self.G = nx.relabel_nodes(nx.complete_graph(len(fullseq), nx.DiGraph()), mapping)

        elif mode == ODGMakingMode.STOPLIST:
            pass
        elif mode == ODGMakingMode.ALLOWLIST:
            pass
        else:
            print("Unknown graph making mode")

    def dump(self):
        nx.draw(self.G)
        plt.show()
