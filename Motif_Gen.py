import torch
import numpy as np

def gen_triangle_motif(edges_list):
    '''
    . - .
    \./
    '''
    new_edge_list = []
    for edge_1 in edges_list:
        for edge_2 in edges_list:
            if edge_1[0] == edge_2[0] and edge_1[1] == edge_2[1]:
                continue
            if edge_1[0] == edge_2[1] and edge_1[1] == edge_2[0]:
                continue
            if edge_1[1] == edge_2[0]:
                node_1 = edge_2[1]
                node_2 = edge_1[0]
                test_edge = [node_1, node_2]
                if test_edge in edges_list:
                    new_node_1 = edge_1[0]
                    new_node_2 = edge_1[1]
                    new_node_3 = edge_2[1]
                    if [new_node_1, new_node_2] not in new_edge_list:
                        new_edge_list.append([new_node_1, new_node_2])
                    if [new_node_2, new_node_3] not in new_edge_list:
                        new_edge_list.append([new_node_2, new_node_3])
                    if [new_node_3, new_node_1] not in new_edge_list:
                        new_edge_list.append([new_node_3, new_node_1])
    if len(new_edge_list) <= 4:
        new_edge_list = edges_list
    return new_edge_list

def check_square(edge_1, edge_2, edge_3, edges_list):
    flag = True
    node1_1 = edge_1[0]
    node1_2 = edge_1[1]
    node2_1 = edge_2[0]
    node2_2 = edge_2[1]
    node3_1 = edge_3[0]
    node3_2 = edge_3[1]
    if node1_2 != node2_1:
        flag = False
    if node2_2 != node3_1:
        flag = False
    test_edge = [node3_2, node1_1]
    if test_edge not in edges_list:
        flag = False
    return flag

def gen_square_motif(edges_list):
    new_edge_list = []
    for edge_1 in edges_list:
        for edge_2 in edges_list:
            for edge_3 in edges_list:
                if edge_1 == edge_2 or edge_1 == edge_3 or edge_2 == edge_3:
                    continue
                flag = check_square(edge_1, edge_2, edge_3, edges_list)
                if flag == True:
                    new_node_1 = edge_1[0]
                    new_node_2 = edge_1[1]
                    new_node_3 = edge_2[1]
                    new_node_4 = edge_3[1]
                    if [new_node_1, new_node_2] not in new_edge_list:
                        new_edge_list.append([new_node_1, new_node_2])
                    if [new_node_2, new_node_3] not in new_edge_list:
                        new_edge_list.append([new_node_2, new_node_3])
                    if [new_node_3, new_node_4] not in new_edge_list:
                        new_edge_list.append([new_node_3, new_node_4])
                    if [new_node_4, new_node_1] not in new_edge_list:
                        new_edge_list.append([new_node_4, new_node_1])
    if len(new_edge_list) <= 4:
        new_edge_list = edges_list
    return new_edge_list