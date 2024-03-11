import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
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

class DistMultPredictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device, id_to_dd_mapping):
        super().__init__()
        
        self.proto = proto
        self.sim_measure = sim_measure
        self.bert_measure = bert_measure
        self.agg_measure = agg_measure
        self.num_walks = num_walks
        self.walk_mode = walk_mode
        self.path_length = path_length
        self.exp_lambda = exp_lambda
        self.device = device
        self.W = w_rels
        self.rel2idx = rel2idx
        self.id_to_dd_mapping = id_to_dd_mapping
        
        #TODO(schwarzjn)
        self.etypes_dd = [('drug', 'contraindication', 'disease'), 
                          ('drug', 'indication', 'disease'),
                          ('drug', 'off-label use', 'disease'),
                          ('disease', 'rev_contraindication', 'drug'), 
                          ('disease', 'rev_indication', 'drug'),
                          ('disease', 'rev_off-label use', 'drug')
        ]
        
        self.node_types_dd = ['disease', 'drug']
        
        if proto:
            self.W_gate = {}
            for i in self.node_types_dd:
                temp_w = nn.Linear(n_hid * 2, 1)
                nn.init.xavier_uniform_(temp_w.weight)
                self.W_gate[i] = temp_w.to(self.device)
            self.k = proto_num
            self.m = nn.Sigmoid()
            if sim_measure in ['bert', 'profile+bert']:
                
                data_path = os.path.join(data_folder, 'kg.csv')
                        
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                                
                self.disease_dict = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].values)
                self.disease_dict.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].values))
                
                if bert_measure == 'disease_name':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/bert_basic.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/nodes.csv')
                    
                elif bert_measure == 'v1':
                    self.bert_embed = np.load('/n/scratch3/users/k/kh278/disease_embeds_single_def.npy')
                    df_nodes_bert = pd.read_csv('/n/scratch3/users/k/kh278/disease_nodes_for_BERT_embeds.csv')
                
                df_nodes_bert['node_id'] = df_nodes_bert.node_id.apply(lambda x: convert2str(x))
                self.id2bertindex = dict(zip(df_nodes_bert.node_id.values, df_nodes_bert.index.values))
                
            self.diseases_profile = {}
            self.sim_all_etypes = {}
            self.diseaseid2id_etypes = {}
            self.diseases_profile_etypes = {}
            
            disease_etypes = ['disease_disease', 'rev_disease_protein']
            disease_nodes = ['disease', 'gene/protein']
            
            for etype in self.etypes_dd:
                src, dst = etype[0], etype[2]
                if src == 'disease':
                    all_disease_ids = torch.where(G.out_degrees(etype=etype) != 0)[0]
                elif dst == 'disease':
                    all_disease_ids = torch.where(G.in_degrees(etype=etype) != 0)[0]
                if sim_measure == 'all_nodes_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes, disease_nodes) for i in all_disease_ids}
                elif sim_measure == 'protein_profile':
                    diseases_profile = {i.item(): obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein']) for i in all_disease_ids}
                elif sim_measure == 'protein_random_walk':
                    diseases_profile = {i.item(): obtain_protein_random_walk_profile(i, num_walks, path_length, G, disease_etypes, disease_nodes, walk_mode) for i in all_disease_ids}
                elif sim_measure == 'bert':
                    diseases_profile = {i.item(): torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]) for i in all_disease_ids}
                elif sim_measure == 'profile+bert':
                    diseases_profile = {i.item(): torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]]))) for i in all_disease_ids}
                    
                diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(len(all_disease_ids))))
                disease_profile_tensor = torch.stack([diseases_profile[i.item()] for i in all_disease_ids])
                sim_all = sim_matrix(disease_profile_tensor, disease_profile_tensor)
                self.sim_all_etypes[etype] = sim_all
                self.diseaseid2id_etypes[etype] = diseaseid2id
                self.diseases_profile_etypes[etype] = diseases_profile
                
    def apply_edges(self, edges):
        """Actual DistMult computation (Equation 4)."""
        h_u = edges.src['h']
        h_v = edges.dst['h']
        rel_idx = self.rel2idx[edges._etype]
        """
        print(edges._etype)
        file_name = 'data/pretrained_mine/complex_disease/fully_separate/train_{}'.format('_'.join(edges._etype)) + '_{}.npz'

        src_ids, target_ids, _ = map(lambda x: x.numpy(), edges.edges())
        if edges._etype[0] == 'drug':
            u_names = [self.id_to_dd_mapping['idx2name_drug'](x) for x in src_ids]
            v_names = [self.id_to_dd_mapping['idx2name_disease'](x) for x in target_ids]
        else:
            u_names = [self.id_to_dd_mapping['idx2name_disease'](x) for x in src_ids]
            v_names = [self.id_to_dd_mapping['idx2name_drug'](x) for x in target_ids]

        if os.path.isfile(file_name.format('pos')):
            # Positive already exits, write negative
            labels = np.zeros(shape=[h_u.shape[0], 1], dtype=np.float32)
            with open(file_name.format('neg'), 'wb') as f:
                np.savez(f, h_u=h_u.detach().cpu().numpy(), h_v=h_v.detach().cpu().numpy(), 
                         u_names=u_names, v_names=v_names, labels=labels, type=edges._etype, mode='neg')
        else:
            labels = np.ones(shape=[h_u.shape[0], 1], dtype=np.float32)
            with open(file_name.format('pos'), 'wb') as f:
                np.savez(f, h_u=h_u.detach().cpu().numpy(), h_v=h_v.detach().cpu().numpy(), 
                         u_names=u_names, v_names=v_names, labels=labels, type=edges._etype, mode='pos')
        """
        h_r = self.W[rel_idx]
        score = torch.sum(h_u * h_r * h_v, dim=1)
        # Logits
        return {'score': score}

    def forward(self, graph, G, h, pretrain_mode, mode, block = None, only_relation = None):
        with graph.local_scope():
            scores = {}
            s_l = []
            
            if len(graph.canonical_etypes) == 1:
                etypes_train = graph.canonical_etypes
            else:
                etypes_train = self.etypes_dd

            if only_relation is not None:
                if only_relation == 'indication':
                    etypes_train = [('drug', 'indication', 'disease'),
                                    ('disease', 'rev_indication', 'drug')]
                elif only_relation == 'contraindication':
                    etypes_train = [('drug', 'contraindication', 'disease'), 
                                   ('disease', 'rev_contraindication', 'drug')]
                elif only_relation == 'off-label':
                    etypes_train = [('drug', 'off-label use', 'disease'),
                                   ('disease', 'rev_off-label use', 'drug')]
                else:
                    return ValueError

            graph.ndata['h'] = h
            
            if pretrain_mode:
                # Pretraining

                # Fetch etypes for which we have data
                etypes_all = [i for i in graph.canonical_etypes if graph.edges(etype = i)[0].shape[0] != 0]
                for etype in etypes_all:
                    graph.apply_edges(self.apply_edges, etype=etype)    
                    out = torch.sigmoid(graph.edges[etype].data['score'])
                    s_l.append(out)
                    scores[etype] = out
            else:
                # Finetuning
                for etype in etypes_train:
                    if self.proto:
                        src, dst = etype[0], etype[2]
                        src_rel_idx = torch.where(graph.out_degrees(etype=etype) != 0)
                        dst_rel_idx = torch.where(graph.in_degrees(etype=etype) != 0)

                        src_h = h[src][src_rel_idx]
                        dst_h = h[dst][dst_rel_idx]

                        src_rel_ids_keys = torch.where(G.out_degrees(etype=etype) != 0)
                        dst_rel_ids_keys = torch.where(G.in_degrees(etype=etype) != 0)
                        src_h_keys = h[src][src_rel_ids_keys]
                        dst_h_keys = h[dst][dst_rel_ids_keys]

                        h_disease = {}

                        if src == 'disease':
                            h_disease['disease_query'] = src_h
                            h_disease['disease_key'] = src_h_keys
                            h_disease['disease_query_id'] = src_rel_idx
                            h_disease['disease_key_id'] = src_rel_ids_keys
                        elif dst == 'disease':
                            h_disease['disease_query'] = dst_h
                            h_disease['disease_key'] = dst_h_keys
                            h_disease['disease_query_id'] = dst_rel_idx
                            h_disease['disease_key_id'] = dst_rel_ids_keys

                        # TODO(schwarzjn): self.sim_measure = ['all_nodes_profile'] 
                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert', 'profile+bert']:

                            try:
                                sim = self.sim_all_etypes[etype][np.array([self.diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                            except:
                                
                                disease_etypes = ['disease_disease', 'rev_disease_protein']
                                disease_nodes = ['disease', 'gene/protein']
            
                                ## new disease not seen in the training set
                                for i in h_disease['disease_query_id'][0]:
                                    if i.item() not in self.diseases_profile_etypes[etype]:
                                        if self.sim_measure == 'all_nodes_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes, disease_nodes)
                                        elif self.sim_measure == 'protein_profile':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein'])
                                        elif self.sim_measure == 'protein_random_walk':
                                            self.diseases_profile_etypes[etype][i.item()] = obtain_protein_random_walk_profile(i, self.num_walks, self.path_length, G, disease_etypes, disease_nodes, self.walk_mode)
                                        elif self.sim_measure == 'bert':
                                            self.diseases_profile_etypes[etype][i.item()] = torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])
                                        elif self.sim_measure == 'profile+bert':
                                            self.diseases_profile_etypes[etype][i.item()] = torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes), torch.Tensor(self.bert_embed[self.id2bertindex[self.disease_dict[i.item()]]])))
                                            
                                profile_query = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]]
                                profile_query = torch.cat(profile_query).view(len(profile_query), -1)

                                profile_keys = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_key_id'][0]]
                                profile_keys = torch.cat(profile_keys).view(len(profile_keys), -1)

                                sim = sim_matrix(profile_query, profile_keys)

                            if src_h.shape[0] == src_h_keys.shape[0]:
                                ## during training...
                                coef = torch.topk(sim, self.k + 1).values[:, 1:]
                                coef = F.normalize(coef, p=1, dim=1)
                                embed = h_disease['disease_key'][torch.topk(sim, self.k + 1).indices[:, 1:]]
                            else:
                                ## during evaluation...
                                coef = torch.topk(sim, self.k).values[:, :]
                                coef = F.normalize(coef, p=1, dim=1)
                                embed = h_disease['disease_key'][torch.topk(sim, self.k).indices[:, :]]

                            # Embed shape [1010, 3, 25], Coef [1010, 3] -> [1010, 3, 1]
                            out = torch.mul(embed, coef.unsqueeze(dim = 2).to(self.device)).sum(dim = 1)
                        
                        # TODO(schwarzjn): Embedding gating (update the original disease embedding  with the disease-disease metric learning embedding)
                        if self.sim_measure in ['protein_profile', 'all_nodes_profile', 'protein_random_walk', 'bert', 'profile+bert']:
                            # for protein profile, we are only looking at diseases for now...
                            if self.agg_measure == 'learn':
                                coef_all = self.m(self.W_gate['disease'](torch.cat((h_disease['disease_query'], out), dim = 1)))
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == 'heuristics-0.8':
                                proto_emb = 0.8*h_disease['disease_query'] + 0.2*out
                            elif self.agg_measure == 'avg':
                                proto_emb = 0.5*h_disease['disease_query'] + 0.5*out
                            elif self.agg_measure == 'rarity':
                                if src == 'disease':
                                    # TODO(schwarzjn): Equation 16
                                    coef_all = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                elif dst == 'disease':
                                    # TODO(schwarzjn): Equation 16
                                    coef_all = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                # TODO(schwarzjn): Equation 17
                                proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                            elif self.agg_measure == '100proto':
                                proto_emb = out
                            # [1010, 25]
                            h['disease'][h_disease['disease_query_id']] = proto_emb
                        else:
                            if self.agg_measure == 'learn':
                                coef_src = self.m(self.W_gate[src](torch.cat((src_h, sim_emb_src), dim = 1)))
                                coef_dst = self.m(self.W_gate[dst](torch.cat((dst_h, sim_emb_dst), dim = 1)))
                            elif self.agg_measure == 'rarity':
                                # give high weights to proto embeddings for nodes that have low degrees
                                coef_src = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                                coef_dst = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                            elif self.agg_measure == 'heuristics-0.8':
                                coef_src = 0.2
                                coef_dst = 0.2
                            elif self.agg_measure == 'avg':
                                coef_src = 0.5
                                coef_dst = 0.5
                            elif self.agg_measure == '100proto':
                                coef_src = 1
                                coef_dst = 1

                            proto_emb_src = (1 - coef_src)*src_h + coef_src*sim_emb_src
                            proto_emb_dst = (1 - coef_dst)*dst_h + coef_dst*sim_emb_dst

                            h[src][src_rel_idx] = proto_emb_src
                            h[dst][dst_rel_idx] = proto_emb_dst
                        graph.ndata['h'] = h
                    import pdb; pdb.set_trace()
                    graph.apply_edges(self.apply_edges, etype=etype)   
                    # Sigmoid is taken in `TxGNN.py`
                    out = graph.edges[etype].data['score']
                    s_l.append(out)
                    scores[etype] = out

                    if self.proto:
                        # recover back to the original embeddings for other relations
                        h[src][src_rel_idx] = src_h
                        h[dst][dst_rel_idx] = dst_h

            if pretrain_mode:
                s_l = torch.cat(s_l)             
            else: 
                s_l = torch.cat(s_l).reshape(-1,).detach().cpu().numpy()

            return scores, s_l
