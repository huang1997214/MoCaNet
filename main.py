import torch
import os
import random
import random
import datetime
import numpy as np
import networkx as nx
from utils import load_data, process_graph, draw_motif_graph, prepare_data, generate_adj_matrix, cal_prediction, margin_loss, cal_acc, save_args, draw_graph_from_edge_list, draw_graph_from_adj_matrix
from param_parser import parameter_parser
from Model import Motif_Aware_CAPNN
args = parameter_parser()
now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
experiment_name = args.data_name
log_file_name = experiment_name + ' ' + now_time
log_file_path = args.log_dir + '/' + log_file_name
arg_dir = log_file_dir = log_file_path + '/' + 'Model_HP.txt'
os.makedirs(log_file_path)
save_args(args, arg_dir)
graphs, node_feature_dim = load_data(args)
print('Dataset:', args.data_name)
print('Graph num:', len(graphs))
graphs, graph_size_list = process_graph(args, graphs)
#weighted_avg_graph_size = int(0.35 * sum(graph_size_list)/len(graph_size_list))
weighted_avg_graph_size = args.Super_Node_Num
test_final_result_list = []
exp_time = 1

random.shuffle(graphs)
train_index = int(len(graphs) * 0.8)
val_index = int(len(graphs) * 0.9)
train_graphs = graphs[0:train_index]
val_graphs = graphs[train_index:val_index]
test_graphs = graphs[val_index:len(graphs)]
'''
while exp_time <= 1:
    if args.Recon_vis:
        recon_vis_path = log_file_path + '/' + 'Recon_Vis' + str(exp_time)
        os.makedirs(recon_vis_path)
    print('Experiemnt time:', exp_time)
    log_file_dir = log_file_path + '/' + str(exp_time) + '.txt'
    log_fo = open(log_file_dir, "w+")
    mod = Motif_Aware_CAPNN(args, node_feature_dim, weighted_avg_graph_size)
    mod = mod.cuda()
    optimizer = torch.optim.Adam(mod.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=args.weight_decay)
    test_final_result = -1
    best_val_acc = -1
    best_val_test_acc = []
    the_final_epoch = 0
    Recon_vis_index = 0
    for step in range(args.Epoch):

        mod.train()
        losses = 0
        sum_mar_loss = 0
        sum_rec_loss = 0
        result_list = []
        truth_list = []
        loss_list = []
        mar_loss_list = []
        rec_loss_list = []
        graph_num = len(graphs)
        batch_index = 1
        test_acc_list = []
        for g in graphs:
            label = g.netx_graph.graph['label']
            target = np.zeros(args.Class_Capsule_Num)
            target[label] = 1
            target = torch.FloatTensor(target).cuda()
            feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix = prepare_data(args, g)
            prediction, recon_loss, recon_core_adj = mod(args, feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix)
            if args.Recon_vis:
                if step<2 or step>15:
                    if feature_torch.shape[0] >= 40 and feature_torch.shape[0]<= 80:
                        if Recon_vis_index % 20 ==0:
                            for i in range(recon_core_adj.shape[0]):
                                for j in range(recon_core_adj.shape[1]):
                                    if recon_core_adj[i][j] >= 0.2:
                                        recon_core_adj[i][j] = 1
                            Ori_graph_show_dir = recon_vis_path + '/' + str(Recon_vis_index) + 'Original Graph.png'
                            Ori_core_graph_show_dir = recon_vis_path + '/' + str(Recon_vis_index) + 'Original Core Graph.png'
                            Recon_core_graph_show_dir = recon_vis_path + '/' + str(Recon_vis_index) + 'Reconstruction Graph.png'
                            draw_graph_from_adj_matrix(adj_matrix_torch_list[0], Ori_graph_show_dir)
                            draw_graph_from_adj_matrix(core_adj_matrix, Ori_core_graph_show_dir)
                            draw_graph_from_adj_matrix(recon_core_adj, Recon_core_graph_show_dir)
                            Recon_vis_index += 1
                        else:
                            Recon_vis_index += 1
            result = cal_prediction(prediction)
            result_list.append(result)
            truth_list.append(label)
            loss = margin_loss(prediction,
                               target,
                               args.lambd)
            mar_loss = loss
            rec_loss = args.theta * recon_loss

            loss = mar_loss + rec_loss

            #loss = mar_loss

            losses += loss
            sum_mar_loss += mar_loss
            sum_rec_loss += rec_loss
            if batch_index % 10 == 0:
                loss_list.append(losses)
                mar_loss_list.append(sum_mar_loss)
                rec_loss_list.append(sum_rec_loss)
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                losses = 0
                sum_mar_loss = 0
                sum_rec_loss = 0
            else:
                batch_index += 1
        temp_sum_mar_loss = 0
        for loss in mar_loss_list:
            temp_sum_mar_loss += loss
        avg_mar_loss = temp_sum_mar_loss/len(mar_loss_list)

        temp_sum_rec_loss = 0
        for loss in rec_loss_list:
            temp_sum_rec_loss += loss
        avg_rec_loss = temp_sum_rec_loss/len(rec_loss_list)
        train_acc = cal_acc(result_list, truth_list)
        print("Epoch", step, " train accuracy:", train_acc, 'Train margin loss:', avg_mar_loss, 'Train rec loss:', avg_rec_loss)
        log_fo.write("Epoch" + str(step) + " train accuracy:" + str(train_acc) + 'Train margin loss:' + str(avg_mar_loss) + 'Train rec loss:' + str(avg_rec_loss) + '\n')
        
        if step %3 ==0:

            mod.eval()
            val_result_list = []
            val_truth_list = []
            for g in val_graphs:
                label = g.netx_graph.graph['label']
                target = np.zeros(args.Class_Capsule_Num)
                target[label] = 1
                target = torch.FloatTensor(target)
                feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix = prepare_data(args, g)
                prediction, _, _ = mod(args, feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix)
                result = cal_prediction(prediction)
                val_result_list.append(result)
                val_truth_list.append(label)
            val_acc = cal_acc(val_result_list, val_truth_list)
            print("Epoch", step, " val accuracy:", val_acc)
            log_fo.write("Epoch" + str(step) + " val accuracy:" + str(val_acc) + '\n')
            if val_acc > best_val_acc:
                best_val_test_acc = []
                best_val_acc = val_acc
                test_result_list = []
                test_truth_list = []
                for g in test_graphs:
                    label = g.netx_graph.graph['label']
                    target = np.zeros(args.Class_Capsule_Num)
                    target[label] = 1
                    target = torch.FloatTensor(target)
                    feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix = prepare_data(args, g)
                    prediction, _, _ = mod(args, feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix)
                    result = cal_prediction(prediction)
                    test_result_list.append(result)
                    test_truth_list.append(label)
                test_acc = cal_acc(test_result_list, test_truth_list)
                best_val_test_acc.append(test_acc)
                print("\033[43m Epoch\033[0m", step, " test accuracy:", sum(best_val_test_acc) / len(best_val_test_acc))
                log_fo.write("Epoch" + str(step) + " test accuracy:" + str(sum(best_val_test_acc) / len(best_val_test_acc)) + '\n')
                test_final_result = sum(best_val_test_acc) / len(best_val_test_acc)
            elif val_acc == best_val_acc:
                test_result_list = []
                test_truth_list = []
                for g in test_graphs:
                    label = g.netx_graph.graph['label']
                    target = np.zeros(args.Class_Capsule_Num)
                    target[label] = 1
                    target = torch.FloatTensor(target)
                    feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix = prepare_data(args, g)
                    prediction, _, _ = mod(args, feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix)
                    result = cal_prediction(prediction)
                    test_result_list.append(result)
                    test_truth_list.append(label)
                test_acc = cal_acc(test_result_list, test_truth_list)
                best_val_test_acc.append(test_acc)
                print("\033[43m Epoch\033[0m", step, " test accuracy:", max(best_val_test_acc))
                log_fo.write("Epoch" + str(step) + " test accuracy:" + str(max(best_val_test_acc)) + '\n')
                test_final_result = max(best_val_test_acc)

    exp_time += 1
'''

mod = Motif_Aware_CAPNN(args, node_feature_dim, weighted_avg_graph_size)
mod = torch.load('model_v2.pkl')
mod = mod.cuda()
feature_np = np.zeros((len(graphs), args.Class_Capsule_Num * args.capsule_dimensions))
#feature_np = np.zeros((len(graphs), args.capsule_dimensions))
label_np = np.zeros(len(graphs))
vis_index = 0
for g in graphs:
    label = g.netx_graph.graph['label']
    label_np[vis_index] = label
    feature_torch, edge_torch_list, adj_matrix_torch_list, core_adj_matrix = prepare_data(args, g)
    prediction, _, _ = mod(args, feature_torch, edge_torch_list, adj_matrix_torch_list,
                                                 core_adj_matrix)
    feature = prediction.flatten()
    #feature = prediction[5]
    feature_np[vis_index] = feature.cpu().detach().numpy()
    vis_index += 1
    print(vis_index)
np.savetxt('feature_np.txt', feature_np, fmt='%0.6f')
np.savetxt('label_np.txt', label_np, fmt='%0.6f')
print(feature_np)
torch.save(mod, 'model_v2.pkl')

