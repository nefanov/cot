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
    FROMPATHS = 3
    EMPTY = 4
    FROMGRAPH = 5


class ODG:
    def __init__(self, mode=ODGMakingMode.EMPTY):
        self.G = nx.DiGraph()

    def fromFullSeq(self, fullseq=[], mode=ODGMakingMode.FULLYCONNECTED):
        print(sorted(nx.complete_graph(len(fullseq), nx.DiGraph())))
        if mode == ODGMakingMode.FULLYCONNECTED:
            self.G = nx.complete_graph(len(fullseq), nx.DiGraph())
            for n in self.G.nodes:
                self.G.nodes[n].update({'passname': fullseq[n]})
            nx.set_edge_attributes(self.G, 1, 'weight')
        elif mode == ODGMakingMode.STOPLIST:
            pass
        elif mode == ODGMakingMode.ALLOWLIST:
            pass
        elif mode == ODGMakingMode.FROMPATHS:
            print("Please, provide the list of sequences")
        else:
            print("Unknown graph making mode")
        return self

    def fromSeqList(self, seq_list=[], mode=ODGMakingMode.FROMPATHS):
        if mode != ODGMakingMode.FROMPATHS:
            print("Incorrect mode")
            return
        self.G = nx.DiGraph()
        for seq in seq_list:
            for item in seq:
                pass
        return self


    def fromGraph(self, Graph=None, mode=ODGMakingMode.FROMGRAPH):
        self.G = Graph
        return self

    def plot(self):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos)
        node_labels = nx.get_node_attributes(self.G, 'passname')
        nx.draw_networkx_labels(self.G, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)
        plt.show()


odg = ODG().fromFullSeq(['A','B','C'])
odg.plot()
odg2 = ODG().fromGraph(Graph=odg)
print(odg2.G.edges(data=True))
