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
            mapping = {}
            for i, item in enumerate(fullseq):
                mapping.update({i : item})
            H = nx.relabel_nodes(self.G, mapping)
            for n in self.G.nodes:
                self.G.nodes[n].update({'passname': fullseq[n]})

        elif mode == ODGMakingMode.STOPLIST:
            pass
        elif mode == ODGMakingMode.ALLOWLIST:
            pass
        elif mode == ODGMakingMode.FROMPATHS:
            print("Please, provide the list of sequences")
        else:
            print("Unknown graph making mode")
        return self

    def add_attribute_to_edge(self, id_node_source, id_node_target, new_attr, value_attr):
        keydict = self.G[id_node_source][id_node_target]
        key = len(keydict)
        for k in keydict:
            if 'type' not in self.G.edge[id_node_source][id_node_target][k]:
                self.G.add_edge(id_node_source, id_node_target, key=k, new_attr=value_attr)

    def addPath(self, seq, incrementEdgeW=False):
        for i in range(1,(len(seq))):
            if not self.G.has_edge(seq[i - 1], seq[i]):
                self.G.add_edge(seq[i - 1], seq[i], weight=1)
            else:
                if incrementEdgeW:
                    if self.G.has_edge(seq[i-1], seq[i]):
                        self.G[seq[i-1]][seq[i]]['weight'] = self.G[seq[i-1]][seq[i]]['weight'] + 1
        return self


    def fromSeqList(self, seq_list=[], mode=ODGMakingMode.FROMPATHS):
        if mode != ODGMakingMode.FROMPATHS:
            print("Incorrect mode")
            return
        nx.set_edge_attributes(self.G, 1, 'weight')
        nodeNames = list(set([item for sublist in seq_list for item in sublist]))
        self.G.add_nodes_from(nodeNames)
        for item in nodeNames:
            self.G.nodes[item].update({'passname': item})

        for seq in seq_list: # for each sequence in the sequences list
            self = self.addPath(seq, True)
        return self

    def fromGraph(self, Graph=None, mode=ODGMakingMode.FROMGRAPH):
        self.G = Graph
        nx.set_edge_attributes(self.G, 1, 'weight')
        return self

    def plot(self):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos)
        node_labels = nx.get_node_attributes(self.G, 'passname')
        nx.draw_networkx_labels(self.G, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)
        plt.show()


if __name__ == '__main__':
    odg = ODG().fromFullSeq(['A', 'B', 'C'])
    odg2 = ODG().fromGraph(Graph=odg.G)
    odg2.addPath([0,1], True)


    odg3 = ODG().fromGraph(nx.DiGraph())
    odg3 = ODG().fromSeqList([['A','B'],['A','B'] , ['A','B'], ['A','B'], ['A','B'], ['A','B'], ['A','B'], ['A','B'] ,['A','C'], ['A','B','C']])
    print(odg3.G.edges(data=True))
    print(odg3.G.nodes)
    print(odg3.G.edges(data=True))
    odg3.plot()
