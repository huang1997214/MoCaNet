import torch
from Layers import Motif_GCN_Layer, PrimaryCapsuleLayer, ClassCapsuleLayer, BatchedDiffPool, CapsuleLayer, reconstruction_layers, My_reconstruction_layers

class Motif_Aware_CAPNN(torch.nn.Module):
    def __init__(self, args, node_feature_num, weighted_avg_graph_size):
        super(Motif_Aware_CAPNN, self).__init__()
        self.node_feature_num = node_feature_num
        self.Motif_Layer = Motif_GCN_Layer(args, node_feature_num).cuda()
        self.Pooling_Layer = BatchedDiffPool(args.gcn_filters[-1], weighted_avg_graph_size, args.Super_Node_Dim).cuda()
        self.Pri_Cap_Layer = PrimaryCapsuleLayer(args, args.Pri_Cap_Num, args.capsule_dimensions).cuda()
        self.ClassCap_Layer = CapsuleLayer(args.Class_Capsule_Num, weighted_avg_graph_size * args.Pri_Cap_Num, args.capsule_dimensions, args.capsule_dimensions, num_iterations = 3).cuda()
        if args.Use_Recon:
            if args.Recon_type == 'ICLR':
                self.Recon_Layer = reconstruction_layers(self.node_feature_num, args.Class_Capsule_Num, args.capsule_dimensions).cuda()
            if args.Recon_type == 'MY':
                self.Recon_Layer = My_reconstruction_layers(args.Core_node_num, args.Class_Capsule_Num,args.capsule_dimensions).cuda()

    def forward(self, args, feature_torch ,edge_torch_list, adj_matrix_torch_list, core_adj_matrix):
        feature_torch = feature_torch.cuda()
        output = self.Motif_Layer(feature_torch, edge_torch_list)
        #output = feature_torch
        feature_mat_list = []
        for feature_mat in output:
            feature_mat = torch.unsqueeze(feature_mat, -1)
            feature_mat_list.append(feature_mat)
        output = feature_mat_list[0]
        for i in range(len(feature_mat_list) - 1):
            output = torch.cat((output, feature_mat_list[i + 1]), -1)
        output = output.permute(2, 0, 1).cuda()
        batched_adj_matrix_torch = torch.FloatTensor(adj_matrix_torch_list).cuda()
        new_feature, new_adj_mat = self.Pooling_Layer(output, batched_adj_matrix_torch)
        new_feature = new_feature.permute(1,0,2)
        output = self.Pri_Cap_Layer(new_feature)
        for i in range(len(output)):
            output[i] = output[i].view(output[i].shape[0] * output[i].shape[1], 1)
        pri_cap = output[0]
        for i in range(len(output) - 1):
            pri_cap = torch.cat((pri_cap, output[i+1]), -1)
        output = torch.squeeze(self.ClassCap_Layer(pri_cap))
        #cal_recon_loss
        recon_loss = 0
        recon = core_adj_matrix
        if args.Use_Recon:
            if args.Recon_type == 'ICLR':
                capsule_input = output
                v_mag = torch.sqrt((capsule_input ** 2).sum(dim=1))
                _, v_max_index = v_mag.max(dim=0)
                v_max_index = v_max_index.data
                capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()))
                capsule_masked[v_max_index, :] = capsule_input[v_max_index, :]
                capsule_masked = capsule_masked.view(1, -1).cuda()
                recon = self.Recon_Layer(capsule_masked)
                recon_loss = torch.sum((feature_torch-recon)**2)
            if args.Recon_type == 'MY':
                capsule_input = output
                v_mag = torch.sqrt((capsule_input ** 2).sum(dim=1))
                _, v_max_index = v_mag.max(dim=0)
                v_max_index = v_max_index.data
                capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()))
                capsule_masked[v_max_index, :] = capsule_input[v_max_index, :]
                capsule_masked = capsule_masked.view(1, -1).cuda()
                recon = self.Recon_Layer(capsule_masked)
                core_adj_matrix = torch.FloatTensor(core_adj_matrix).cuda()
                recon_loss = torch.sum((core_adj_matrix - recon) ** 2)
        return output, recon_loss, recon