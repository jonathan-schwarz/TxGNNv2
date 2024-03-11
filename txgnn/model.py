import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils import data
import pandas as pd
import copy
import os
import random

import warnings
warnings.filterwarnings("ignore")
from .utils import sim_matrix, exponential, obtain_disease_profile, obtain_protein_random_walk_profile, convert2str
from .graphmask.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from .graphmask.squeezer import Squeezer
from .graphmask.sigmoid_penalty import SoftConcrete
from .predictor import DistMultPredictor
#from .gp_predictor import DistMultPredictor

    
class AttHeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(AttHeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        
        self.attn_fc = nn.ModuleDict({
                name : nn.Linear(out_size * 2, 1, bias = False) for name in etypes
            })
    
    def edge_attention(self, edges):
        src_type = edges._etype[0]
        etype = edges._etype[1]
        dst_type = edges._etype[2]
        
        if src_type == dst_type:
            wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]], dim=1)
        else:
            if etype[:3] == 'rev':
                wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]], dim=1)
            else:
                wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]], dim=1)
        a = self.attn_fc[etype](wh2)
        return {'e_%s' % etype: F.leaky_relu(a)}

    def message_func(self, edges):
        etype = edges._etype[1]
        return {'m': edges.src['Wh_%s' % etype], 'e': edges.data['e_%s' % etype]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'h': h}
    
    def forward(self, G, feat_dict, return_att = False):
        with G.local_scope():        
            funcs = {}
            att = {}
            etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0]
            for srctype, etype, dsttype in etypes_all:
                Wh = self.weight[etype](feat_dict[srctype])
                G.nodes[srctype].data['Wh_%s' % etype] = Wh

            for srctype, etype, dsttype in etypes_all:
                G.apply_edges(self.edge_attention, etype=etype)
                if return_att:
                    att[(srctype, etype, dsttype)] = G.edges[etype].data['e_%s' % etype].detach().cpu().numpy()
                funcs[etype] = (self.message_func, self.reduce_func)
                
            G.multi_update_all(funcs, 'sum')
            
            return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}, att


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        self.in_size = in_size
        self.out_size = out_size
            
        self.gate_storage = {}
        self.gate_score_storage = {}
        self.gate_penalty_storage = {}
    
    
    def add_graphmask_parameter(self, gate, baseline, layer):
        self.gate = gate
        self.baseline = baseline
        self.layer = layer
    
    def forward(self, G, feat_dict):
        funcs = {}
        etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0]
        
        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
       
        return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}
 
    def gm_online(self, edges):
        etype = edges._etype[1]
        srctype = edges._etype[0]
        dsttype = edges._etype[2]
        
        if srctype == dsttype:
            gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]])
        else:
            if etype[:3] == 'rev':                
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]])
            else:
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]])
                
        #self.penalty += len(edges.src['Wh_%s' % etype])/self.num_of_edges * penalty
        #self.penalty += penalty
        self.penalty.append(penalty)
        
        self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0])
        
        message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype] + (1 - gate.unsqueeze(-1)) * self.baseline[etype][self.layer].unsqueeze(0)
        
        if self.return_gates:
            self.gate_storage[etype] = copy.deepcopy(gate.to('cpu').detach())
            self.gate_penalty_storage[etype] = copy.deepcopy(penalty_not_sum.to('cpu').detach())
            self.gate_score_storage[etype] = copy.deepcopy(gate_score.to('cpu').detach())
        return {'m': message}
    
    def message_func_no_replace(self, edges):
        etype = edges._etype[1]
        #self.msg_emb[etype] = edges.src['Wh_%s' % etype].to('cpu')
        return {'m': edges.src['Wh_%s' % etype]}
    
    def graphmask_forward(self, G, feat_dict, graphmask_mode, return_gates):
        self.return_gates = return_gates
        self.penalty = []
        self.num_masked = 0
        self.num_of_edges = G.number_of_edges()
        
        funcs = {}
        etypes_all = G.canonical_etypes
        
        for srctype, etype, dsttype in etypes_all:
            Wh = self.weight[etype](feat_dict[srctype])
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            
        for srctype, etype, dsttype in etypes_all:
            
            if graphmask_mode:
                ## replace the message!
                funcs[etype] = (self.gm_online, fn.mean('m', 'h'))
            else:
                ## normal propagation!
                funcs[etype] = (self.message_func_no_replace, fn.mean('m', 'h'))
                
        G.multi_update_all(funcs, 'sum')
        
        
        if graphmask_mode:
            self.penalty = torch.stack(self.penalty).reshape(-1,)
            #penalty_mean = torch.mean(self.penalty)
            #penalty_relation_reg = torch.sum(torch.log(self.penalty) * self.penalty)
            #penalty = penalty_mean + 0.1 * penalty_relation_reg
            penalty = torch.mean(self.penalty)
        else:
            penalty = 0 

        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}, penalty, self.num_masked


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, attention, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device,
                 id_to_dd_mapping):
        super(HeteroRGCN, self).__init__()

        if attention:
            self.layer1 = AttHeteroRGCNLayer(in_size, hidden_size, G.etypes)
            self.layer2 = AttHeteroRGCNLayer(hidden_size, out_size, G.etypes)
        else:
            self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
            self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)
        
        # Shape [hidden_size, n_rel_types]
        self.w_rels = nn.Parameter(torch.Tensor(len(G.canonical_etypes), out_size))
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain('relu'))
        rel2idx = dict(zip(G.canonical_etypes, list(range(len(G.canonical_etypes)))))
         
        self.pred = DistMultPredictor(n_hid = hidden_size, 
                                      w_rels = self.w_rels, 
                                      G = G, 
                                      rel2idx = rel2idx, 
                                      proto = proto, 
                                      proto_num = proto_num, 
                                      sim_measure = sim_measure, 
                                      bert_measure = bert_measure, 
                                      agg_measure = agg_measure, 
                                      num_walks = num_walks, 
                                      walk_mode = walk_mode, 
                                      path_length = path_length, 
                                      split = split, 
                                      data_folder = data_folder, 
                                      exp_lambda = exp_lambda, 
                                      device = device,
                                      id_to_dd_mapping = id_to_dd_mapping)

        self.attention = attention      
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.etypes = G.etypes
        self.id_to_dd_mapping = id_to_dd_mapping
        
    def forward_minibatch(self, pos_G, neg_G, blocks, G, mode = 'train', pretrain_mode = False):
        # TODO(schwarzjn): Why blocks[0]/[1]?
        input_dict = blocks[0].srcdata['inp']
        h_dict = self.layer1(blocks[0], input_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h = self.layer2(blocks[1], h_dict)
        # !h.keys(): dict_keys(['anatomy', 'biological_process', 'drug', 'gene/protein']) -> these are only the ones present in blocks[1]

        scores, out_pos = self.pred(pos_G, G, h, pretrain_mode, mode = mode + '_pos', block = blocks[1])
        scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg', block = blocks[1])
        return scores, scores_neg, out_pos, out_neg
        
    
    def forward(self, G, neg_G, eval_pos_G = None, return_h = False, return_att = False, mode = 'train', pretrain_mode = False): 
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
            if self.attention:
                h_dict, a_dict_l1 = self.layer1(G, input_dict, return_att)
                h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h, a_dict_l2 = self.layer2(G, h_dict, return_att)
            else:
                h_dict = self.layer1(G, input_dict)
                h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h = self.layer2(G, h_dict)

            if return_h:
                return h

            if return_att:
                return a_dict_l1, a_dict_l2

            # full batch
            if eval_pos_G is not None:
                # eval mode
                scores, out_pos = self.pred(eval_pos_G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
            else:
                # train mode
                scores, out_pos = self.pred(G, G, h, pretrain_mode, mode = mode + '_pos')
                scores_neg, out_neg = self.pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
    
    def graphmask_forward(self, G, pos_graph, neg_graph, graphmask_mode = False, return_gates = False, only_relation = None):
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
            h_dict_l1, penalty_l1, num_masked_l1 = self.layer1.graphmask_forward(G, input_dict, graphmask_mode, return_gates)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict_l1.items()}
            h, penalty_l2, num_masked_l2 = self.layer2.graphmask_forward(G, h_dict, graphmask_mode, return_gates)         
            
            scores_pos, out_pos = self.pred(pos_graph, G, h, False, mode = 'train_pos', only_relation = only_relation)
            scores_neg, out_neg = self.pred(neg_graph, G, h, False, mode = 'train_neg', only_relation = only_relation)
            return scores_pos, scores_neg, penalty_l1 + penalty_l2, [num_masked_l1, num_masked_l2]

    
    def enable_layer(self, layer):
        print("Enabling layer "+str(layer))
        
        for name in self.etypes:
            for parameter in self.gates_all[name][layer].parameters():
                parameter.requires_grad = True

            self.baselines_all[name][layer].requires_grad = True
    

    def count_layers(self):
        return 2
    
    def get_gates(self):
        return [self.layer1.gate_storage, self.layer2.gate_storage]
    
    def get_gates_scores(self):
        return [self.layer1.gate_score_storage, self.layer2.gate_score_storage]
    
    def get_gates_penalties(self):
        return [self.layer1.gate_penalty_storage, self.layer2.gate_penalty_storage]
    
    def add_graphmask_parameters(self, G):
        gates_all, baselines_all = {}, {}
        hidden_size = self.hidden_size
        out_size = self.out_size
        
        for name in G.etypes:
            ## for each relation type

            gates = []
            baselines = []

            vertex_embedding_dims = [hidden_size, out_size]
            message_dims = [hidden_size, out_size]
            h_dims = message_dims

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                gate_input_shape = [m_dim, m_dim]

                ### different layers have different gates
                gate = torch.nn.Sequential(
                    MultipleInputsLayernormLinear(gate_input_shape, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    Squeezer(),
                    SoftConcrete()
                )

                gates.append(gate)

                baseline = torch.FloatTensor(m_dim)
                stdv = 1. / math.sqrt(m_dim)
                baseline.uniform_(-stdv, stdv)
                baseline = torch.nn.Parameter(baseline, requires_grad=True)

                baselines.append(baseline)

            gates = torch.nn.ModuleList(gates)
            gates_all[name] = gates

            baselines = torch.nn.ParameterList(baselines)
            baselines_all[name] = baselines

        self.gates_all = nn.ModuleDict(gates_all)
        self.baselines_all = nn.ModuleDict(baselines_all)

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False
            
        self.layer1.add_graphmask_parameter(self.gates_all, self.baselines_all, 0)
        self.layer2.add_graphmask_parameter(self.gates_all, self.baselines_all, 1)
