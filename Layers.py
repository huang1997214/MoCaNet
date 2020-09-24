import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import scipy
import torch.nn as nn
from torch.autograd import Variable

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class Basic_GCN_Layers(torch.nn.Module):
    def __init__(self, args, node_feature_num):
        super(Basic_GCN_Layers, self).__init__()
        self.layer_list = [GCNConv(node_feature_num, args.gcn_filters[0]).cuda()]
        for i in range(len(args.gcn_filters)-1):
            self.layer_list.append(GCNConv(args.gcn_filters[i], args.gcn_filters[i+1]))
        self.layer_list = ListModule(*self.layer_list)

    def forward(self, features, edges):
        features = features.cuda()
        edges = edges.cuda()
        for layer in self.layer_list:
            features = torch.nn.functional.relu(layer(features, edges))
        return features

class Motif_GCN_Layer(torch.nn.Module):
    def __init__(self, args, node_feature_num):
        super(Motif_GCN_Layer, self).__init__()
        self.Motif_GCN_list = []
        for i in range(args.motif_num):
            self.Motif_GCN_list.append(Basic_GCN_Layers(args, node_feature_num))
        self.Motif_GCN_list = ListModule(*self.Motif_GCN_list)

    def forward(self, features, motif_edge_torch_list):
        output = []
        for i in range(len(motif_edge_torch_list)):
            edge_torch = motif_edge_torch_list[i]
            output.append(self.Motif_GCN_list[i](features, edge_torch))
        return output

class BatchedGraphSAGE(torch.nn.Module):
    def __init__(self, infeat, outfeat, device='cuda:0', use_bn = True, mean = False, add_self = False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.device = device
        self.mean = mean
        self.W = torch.nn.Linear(infeat, outfeat, bias = True)
        torch.nn.init.xavier_uniform_(self.W.weight, gain = torch.nn.init.calculate_gain('relu'))

    def forward(self, x, adj, mask=None):
        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).to(self.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(h_k.size(1)).to(self.device)
            h_k = self.bn(h_k)
        if mask is not None:
            h_k = h_k * mask.unsqueeze(2).expand_as(h_k)
        return h_k


class BatchedDiffPool(torch.nn.Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False, device='cuda:0', link_pred=False):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.device = device
        self.is_final = is_final
        self.embed = BatchedGraphSAGE(nfeat, nhid, device=self.device, use_bn=True)
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, device=self.device, use_bn=True)
        self.log = {}
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self, x, adj, mask=None, log=False):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        if self.link_pred:
            # TODO: Masking padded s_l
            self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
            self.entropy_loss = torch.distributions.Categorical(probs=s_l).entropy()
            if mask is not None:
                self.entropy_loss = self.entropy_loss * mask.expand_as(self.entropy_loss)
            self.entropy_loss = self.entropy_loss.sum(-1)
        return xnext, anext

class CapsuleLayer(torch.nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, num_iterations):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

    def squash(self, tensor, dim=-1):
        tensor = tensor
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):

        #print(x[None, None, :, None, :].shape)
        #print(self.route_weights[:, None, :, :, :].shape)
        priors = x[None, None, :, None, :] @ self.route_weights[:, None, :, :, :]
        logits = Variable(torch.zeros(*priors.size())).cuda()
        for i in range(self.num_iterations):
            probs = softmax(logits, dim=2).cuda()
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True)).cuda()
            #outputs = (probs * priors).sum(dim=2, keepdim=True)
            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True).cuda()
                logits = logits + delta_logits
        return outputs

class PrimaryCapsuleLayer(torch.nn.Module):
    def __init__(self, args, capsule_per_node, capsule_dim):
        super(PrimaryCapsuleLayer, self).__init__()
        self.motif_num = args.motif_num
        self.capsule_dim = capsule_dim
        self.gcn_filter_dim = args.Super_Node_Dim
        self.pri_capsule_num = capsule_per_node
        self.extract_layers = []
        for i in range(self.capsule_dim):
            unit = torch.nn.Conv1d(in_channels = self.motif_num,
                                   out_channels = self.pri_capsule_num,
                                   kernel_size = self.gcn_filter_dim,
                                   stride=1,
                                   bias=True)

            self.add_module("unit_" + str(i), unit)
            self.extract_layers.append(unit)

    def forward(self, feature):
        hidden = [self.extract_layers[i](feature) for i in range(len(self.extract_layers))]
        return hidden

class reconstruction_layers(torch.nn.Module):
    def __init__(self, node_feature_num, number_of_targets, capsule_dimensions):
        super(reconstruction_layers, self).__init__()
        self.number_of_features = node_feature_num
        self.reconstruction_layer_1 = torch.nn.Linear(number_of_targets * capsule_dimensions,
                                                      int((node_feature_num * 2) / 3))
        self.reconstruction_layer_2 = torch.nn.Linear(int((node_feature_num * 2) / 3),
                                                      int((node_feature_num * 3) / 2))
        self.reconstruction_layer_3 = torch.nn.Linear(int((node_feature_num * 3) / 2),
                                                      node_feature_num)
    def forward(self, input):
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_1(input))
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = self.reconstruction_layer_3(reconstruction_output)
        reconstruction_output = reconstruction_output.view(1, self.number_of_features)
        return reconstruction_output

class My_reconstruction_layers(torch.nn.Module):
    def __init__(self, core_node_num, number_of_targets, capsule_dimensions):
        super(My_reconstruction_layers, self).__init__()
        self.core_node_num = core_node_num
        self.number_of_targets = number_of_targets
        self.capsule_dimensions = capsule_dimensions
        self.middle_hidden = int((core_node_num * core_node_num)/2)
        self.reconstruction_layer_1 = torch.nn.Linear(number_of_targets * capsule_dimensions, self.middle_hidden)
        self.reconstruction_layer_2 = torch.nn.Linear(self.middle_hidden, core_node_num * core_node_num)
    def forward(self, input):
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_1(input))
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = reconstruction_output.view(self.core_node_num, self.core_node_num)
        return reconstruction_output
'''
The following code may not be useful
'''
class ClassCapsuleLayer(torch.nn.Module):
    def __init__(self, args, child_capsule_dim, child_capsule_num, parent_capsule_dim = 8, parent_capsule_num = 10):
        super(ClassCapsuleLayer, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(parent_capsule_num, child_capsule_num, child_capsule_dim, parent_capsule_dim))

    def forward(self, feature):
        prior_weights = x[None, :, :, None, :] @ self.W[:, None, :, :, :]
        return prior_weights

