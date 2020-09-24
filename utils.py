import os
import re
import copy
import torch
import numpy as np
import scipy as sc
import networkx as nx
import matplotlib.pyplot as plt
from torch.autograd import Variable
from param_parser import parameter_parser
from Motif_Gen import gen_triangle_motif, gen_square_motif
def cal_acc(prediction, label):
    acc = 0
    for i in range(len(prediction)):
        if prediction[i] == label[i]:
            acc += 1
    return acc / len(label)

def cal_prediction(prediction):
    prediction_mag = torch.sqrt((prediction ** 2).sum(dim=1))
    _, prediction_max_index = prediction_mag.max(dim=0)
    result = prediction_max_index.data.view(-1).item()
    return result

def margin_loss(scores, target, loss_lambda):
    scores = scores.squeeze()
    v_mag = torch.sqrt((scores**2).sum(dim=1, keepdim=True)).cuda()
    zero = Variable(torch.zeros(1)).cuda()
    m_plus = 0.9
    m_minus = 0.1
    max_l = torch.max(m_plus - v_mag, zero).view(1, -1)**2
    max_r = torch.max(v_mag - m_minus, zero).view(1, -1)**2
    T_c = Variable(torch.zeros(v_mag.shape))
    T_c = target
    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1)
    L_c = L_c.mean()
    return L_c

def draw_motif_graph(node_list, edge_list, new_edge_list, graph_name = 'graph.png', new_graph_name = 'new_graph.png'):
    remove_list = [e for e in edge_list if e not in new_edge_list]
    G = nx.Graph()
    for node in node_list:
        G.add_node(node)
    for edge in edge_list:
        G.add_edge(edge[0], edge[1])
    nx.draw(G)
    plt.savefig(graph_name, format='PNG')

    for edge in remove_list:
        G.remove_edges_from(edge)
    nx.draw(G)
    plt.savefig(new_graph_name, format='PNG')

def draw_graph_from_edge_list(edge_list, graph_dir):
    G = nx.from_edgelist(edge_list)
    nx.draw(G)
    plt.savefig(graph_dir, format='PNG')
    plt.close('all')

def draw_graph_from_adj_matrix(adj_matrix, graph_dir):
    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1:
                edge_list.append([i, j])
    draw_graph_from_edge_list(edge_list, graph_dir)

def load_data(args):
    dir = args.data_dir
    name = args.data_name
    data_dir = dir + "/" + name
    file_graph_indicator = data_dir + '/' + name + '_graph_indicator.txt'
    graph_indicator = {}
    with open(file_graph_indicator) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indicator[i] = int(line)
            i += 1

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

    if name == 'ENZYMES':
        file_node_attributes = data_dir + '/' + name + '_node_attributes.txt'
        node_attributes = []
        with open(file_node_attributes) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attributes.append(np.array(attrs))
        node_feature_dim = np.array(attrs).shape[0]
    elif name == 'PROTEINS':
        '''
        For the Proteins
        We first load it than transform it to one_hot 
        '''
        file_node_attributes = data_dir + '/' + name + '_node_attributes.txt'
        node_attributes = []
        int_map = []
        with open(file_node_attributes) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attributes.append(attrs[0])
                if attrs[0] not in int_map:
                    int_map.append(int(attrs[0]))
        for i in range(len(node_attributes)):
            node_attributes[i] = int(node_attributes[i])
        attr_map_to_int = {val: i for i, val in enumerate(int_map)}
        attr_after_map = [attr_map_to_int[l] for l in node_attributes]
        #print(attr_map_to_int)
        #print(attr_after_map)
        node_feature_dim = max(attr_after_map) + 1
        node_attributes = []
        for attr in attr_after_map:
            feat_temp = np.zeros(int(node_feature_dim))
            feat_temp[attr] = 1
            node_attributes.append(feat_temp)
    else:
        '''
        first calculate the node num
        '''
        print('')
        node_num_list = []
        node_indica_list = []
        with open(file_graph_indicator) as f:
            for line in f:
                line = line.strip("\n")
                node_indica = int(line)
                node_indica_list.append(node_indica)
        index = 1
        temp_node_num = 0
        for i in range(len(node_indica_list)):
            if i == len(node_indica_list)-1:
                temp_node_num += 1
                node_num_list.append(temp_node_num)
            if node_indica_list[i] == index:
                temp_node_num += 1
            else:
                node_num_list.append(temp_node_num)
                temp_node_num = 1
                index += 1
        '''
        Then generate the degree feature
        '''
        node_attributes = []
        pre_node_num = 0
        max_degree_list = []
        all_degree_list = []
        for i in range(len(adj_list)):
            node_num = node_num_list[i]
            temp_adj_matrix = np.zeros((node_num, node_num))
            for edge in adj_list[i + 1]:
                node1 = edge[0] - pre_node_num
                node2 = edge[1] - pre_node_num
                temp_adj_matrix[node1 - 1][node2 - 1] = 1
                temp_adj_matrix[node2 - 1][node1 - 1] = 1
            pre_node_num = pre_node_num + node_num_list[i]
            degree = np.sum(temp_adj_matrix, -1)
            max_degree = int(np.max(degree))
            temp_degree_list = degree.tolist()
            for degree_item in temp_degree_list:
                all_degree_list.append(degree_item)
            max_degree_list.append(max_degree)

        node_feature_dim = max(max_degree_list)
        for deg_num in all_degree_list:
            feat_temp = np.zeros(int(node_feature_dim))
            feat_temp[int(deg_num - 1)] = 1
            node_attributes.append(feat_temp)
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
    return graphs, node_feature_dim

class graph_structure():
    def __init__(self, netx_graph, adj_edges_list, node_feature_list, triangle_motif_edges_list, square_motif_edges_list):
        self.netx_graph = netx_graph
        self.adj_edges_list = adj_edges_list
        self.node_feature_list = node_feature_list
        self.triangle_motif_edges_list = triangle_motif_edges_list
        self.square_motif_edges_list = square_motif_edges_list

def generate_adj_matrix(feature_torch, edge_torch_list):
    node_num = feature_torch.shape[0]
    adj_matrix_torch_list = []
    for edge_torch in edge_torch_list:
        adj_matrix = np.zeros((node_num, node_num))
        for i in range(edge_torch.shape[1]):
            node_1 = edge_torch[0][i]
            node_2 = edge_torch[1][i]
            adj_matrix[node_1][node_2] = 1
        adj_matrix_torch_list.append(adj_matrix)
    return adj_matrix_torch_list

def generate_core_adj_matrix(args, edge_list, feature_torch):
    node_num = feature_torch.shape[0]
    core_node_num = args.Core_node_num
    if node_num<= core_node_num:
        core_adj_matrix = np.zeros((core_node_num, core_node_num))
        for i in range(len(edge_list)):
            node_1 = edge_list[i][0]
            node_2 = edge_list[i][1]
            core_adj_matrix[node_1][node_2] = 1
    else:
        edge_num_dict = {}
        core_adj_matrix = np.zeros((core_node_num, core_node_num))
        for edge in edge_list:
            if edge[0] not in edge_num_dict:
                edge_num_dict[edge[0]] = 1
            else:
                edge_num_dict[edge[0]] += 1
        sorted_edge_num_list = sorted(edge_num_dict.items(), key=lambda x: x[1], reverse=True)
        core_node_list = sorted_edge_num_list[0:core_node_num]
        new_index = 0
        node_dict = {}
        for node_id, edge_num in core_node_list:
            node_dict[node_id] = new_index
            new_index += 1
        for edge in edge_list:
            node_1 = edge[0]
            node_2 = edge[1]
            if node_1 not in node_dict or node_2 not in node_dict:
                continue
            map_node_1 = node_dict[node_1]
            map_node_2 = node_dict[node_2]
            core_adj_matrix[map_node_1][map_node_2] = 1
    return core_adj_matrix

def prepare_data(args, graph):
    edge_list = graph.adj_edges_list
    node_feature_list = graph.node_feature_list
    triangle_motif_edge_list = graph.triangle_motif_edges_list
    square_motif_edge_list = graph.square_motif_edges_list
    edges_torch = torch.t(torch.LongTensor(edge_list))
    triangle_motif_edge_torch = torch.t(torch.LongTensor(triangle_motif_edge_list))
    square_motif_edge_torch = torch.t(torch.LongTensor(square_motif_edge_list))
    feature_torch = torch.FloatTensor(node_feature_list)
    core_adj_matrix = generate_core_adj_matrix(args, edge_list, feature_torch)
    #edge_torch_list = [edges_torch, triangle_motif_edge_torch, square_motif_edge_torch]
    edge_torch_list = [edges_torch, triangle_motif_edge_torch]
    edge_torch_list = edge_torch_list[0:args.motif_num]
    adj_matrix_torch_list = generate_adj_matrix(feature_torch, edge_torch_list)
    return feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix

def process_graph(args, graph_list):
    processed_graph_list = []
    graph_size_list = []
    graph_sum_len = len(graph_list)
    index = 1
    if args.Use_Motif:
        print("Use the motif to train")
    else:
        print("Do not Use the motif to train")
    for g in graph_list:
        print('Processing Graph: ' + str(index) + '/' + str(graph_sum_len))
        index += 1
        edges = g.edges()
        edges_list = [[edge[0], edge[1]] for edge in edges]
        adj_edges_list = edges_list + [[edge[1], edge[0]] for edge in edges]
        feature_list = []
        for u in g.nodes():
            feature_list.append(g.node[u]['feat'])
        graph_size_list.append(len(feature_list))
        if args.Use_Motif:
            triangle_motif_edge_list = gen_triangle_motif(adj_edges_list)
            #square_motif_edge_list = gen_square_motif(adj_edges_list)
            square_motif_edge_list = adj_edges_list
        else:
            triangle_motif_edge_list = adj_edges_list
            square_motif_edge_list = adj_edges_list
        temp_graph = graph_structure(g, adj_edges_list, feature_list, triangle_motif_edge_list, square_motif_edge_list)
        processed_graph_list.append(temp_graph)
    return processed_graph_list, graph_size_list

def store_motif(args, graph_list):
    '''

    :param args:
    :param motif_type: can only be square/triangle
    :return:
    '''
    square_motif_store_dir = args.data_dir + '/' + args.data_name + '/' + 'square' + '_motif' + '.' + 'txt'
    triangle_motif_store_dir = args.data_dir + '/' + args.data_name + '/' + 'triangle' + '_motif' + '.' + 'txt'
    square_motif_file = open(square_motif_store_dir, "w+")
    triangle_motif_file = open(triangle_motif_store_dir, "w+")
    i = 0
    for g in graph_list:
        edges = g.edges()
        edges_list = [[edge[0], edge[1]] for edge in edges]
        adj_edges_list = edges_list + [[edge[1], edge[0]] for edge in edges]
        triangle_motif_edge_list = gen_triangle_motif(adj_edges_list)
        square_motif_edge_list = gen_square_motif(adj_edges_list)
        square_motif_file.write(str(square_motif_edge_list))
        triangle_motif_file.write(str(triangle_motif_edge_list))
        i += 1
        if i>5:
            break
    square_motif_file.close()
    triangle_motif_file.close()


def load_motif(args):
    square_motif = []
    triangle_motif = []
    square_motif_load_dir = args.data_dir + '/' + args.data_name + '/' + 'square' + '_motif' + '.' + 'txt'
    triangle_motif_load_dir = args.data_dir + '/' + args.data_name + '/' + 'triangle' + '_motif' + '.' + 'txt'
    square_motif_file = open(square_motif_load_dir, "r")
    triangle_motif_file = open(triangle_motif_load_dir, "r")
    for line in square_motif_file:
        print(line)
        print(type(line))
        print(list(line))

def process_graph_from_file(args, graph_list):
    processed_graph_list = []
    return processed_graph_list

def save_args(args, args_file_dir):
    args_fo = open(args_file_dir, "w+")
    args_fo.write('Epoch:' + str(args.Epoch) + '\n')
    args_fo.write('Dataset Name:' + str(args.data_name) + '\n')
    args_fo.write('GCN Setting:' + str(args.gcn_filters) + '\n')
    args_fo.write('Motif Num:' + str(args.motif_num) + '\n')
    args_fo.write('Pri_Cap_Dim:' + str(args.capsule_dimensions) + '\n')
    args_fo.write('Pri_Cap_Num:' + str(args.Pri_Cap_Num) + '\n')
    args_fo.write('Super_Node_Dim:' + str(args.Super_Node_Dim) + '\n')
    args_fo.write('Super_Node_Num:' + str(args.Super_Node_Num) + '\n')
    args_fo.write('Class_Cap_Num:' + str(args.Class_Capsule_Num) + '\n')
    args_fo.write('Lr:' + str(args.learning_rate) + '\n')
    args_fo.write('Lambda:' + str(args.lambd) + '\n')
    args_fo.write('Weight_Decay:' + str(args.weight_decay) + '\n')
    args_fo.write('Use_Motif:' + str(args.Use_Motif) + '\n')
    args_fo.write('Use Reconstruction loss:' + str(args.Use_Recon) + '\n')
    args_fo.write('Reconstruction loss type:' + str(args.Recon_type) + '\n')
    args_fo.write('Core_Node_Num:' + str(args.Core_node_num) + '\n')
    args_fo.write('"Show Reconstruction Result Or Not:' + str(args.Recon_vis) + '\n')
    args_fo.close()
'''
args = parameter_parser()
graphs, node_feature_dim = load_data(args)
graphs, graph_size_list = process_graph(args, graphs)
'''