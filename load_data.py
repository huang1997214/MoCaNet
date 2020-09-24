import networkx as nx
import numpy as np
import scipy as sc
import os
import re

def load_data(dir, name):
    '''
    only for DD and ENZYMES
    :param dir:
    :param name:
    :return:
    '''
    data_dir = dir + "/" + name
    file_graph_indicator = data_dir + '/' + name + '_graph_indicator.txt'
    graph_indicator = {}
    with open(file_graph_indicator) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indicator[i] = int(line)
            i += 1

    file_node_attributes = data_dir + '/' + name + '_node_attributes.txt'
    node_attributes = []
    with open(file_node_attributes) as f:
        for line in f:
            line = line.strip("\s\n")
            attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
            node_attributes.append(np.array(attrs))

    file_graph_label = data_dir + '/' + name + '_graph_labels.txt'
    graph_labels=[]
    label_vals = []
    with open(file_graph_label) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    file_adj= data_dir + '/' + name + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(file_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indicator[e0]].append((e0,e1))
            index_graph[graph_indicator[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]
    graphs=[]
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]
        for u in G.nodes():
            G.node[u]['feat'] = node_attributes[u - 1]
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))

    return graphs











