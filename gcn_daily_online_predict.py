#!/usr/bin/env python
# coding: utf-8

# In[29]:


import networkx as nx
import pandas as pd
import s3fs
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa
import os
import networkx as nx 
import random
import dgl
import torch as th
s3 = s3fs.S3FileSystem()
import torch
import pandas as pd
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dateutil.parser import parse as dt_parse
import time
import logging
device = "cpu"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('date_key')
args = parser.parse_args()
yesterday = dt_parse(args.date_key).strftime('%Y%m%d')
print(yesterday)
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, graph, feat_dict):
        funcs = {}

        for srctype, etype, dsttype in graph.canonical_etypes:
            Wh = self.weight[etype](feat_dict[srctype])
            graph.nodes[srctype].data['Wh_{}'.format(etype)] = Wh
            funcs[etype] = (fn.copy_u('Wh_{}'.format(etype), 'm'), fn.mean('m', 'h'))

        graph.multi_update_all(funcs, 'sum')

        return {ntype: graph.nodes[ntype].data['h'] for ntype in graph.ntypes}

class HeteroRGCN(nn.Module):
    def __init__(self, graph,target_node,node_feature,in_feats, h_dim, num_classes=2):
        super(HeteroRGCN, self).__init__()

#       embed_dict = {ntype: nn.Parameter(torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device)) for ntype in graph.ntypes}
        embed_dict = {ntype: torch.Tensor(graph.number_of_nodes(ntype), in_feats).to(device) for ntype in graph.ntypes}

        for key, embed in embed_dict.items():
            embed_dict[key] = nn.init.zeros_(embed)
            #xavier_uniform_(embed) 如何可自动设置

   #    embed_dict['user'] = nn.Parameter(graph.nodes['user'].data['f'].float())
        
            
        self.embed = embed_dict
        self.embed[target_node] = node_feature

        self.layer1 = HeteroRGCNLayer(in_feats, h_dim, graph.etypes)
        self.layer2 = HeteroRGCNLayer(h_dim, num_classes, graph.etypes)
        # self.layer1 = RelGraphConv(in_feats, h_dim, num_rels=2)
        # self.layer2 = RelGraphConv(h_dim, num_classes, num_rels=2)

    def forward(self, graph):
        h_dict = self.layer1(graph, self.embed)
        h_dict_2 = {k: F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(graph, h_dict_2)
#add softmax here 
        return h_dict,h_dict_2    #['user'] #改成通用的 

def build_graph(relations_list,relations_data_list):
    relations_data_dic = {}
    i = 0 
    for each in relations_list:
        relations_data_dic[each] = relations_data_list[i]
        i += 1
    graph = dgl.heterograph(
       relations_data_dic
    )
    
    print('Node types:', graph.ntypes)
    print('Edge types:', graph.etypes)
    print('Canonical edge types:', graph.canonical_etypes)
    for each in graph.canonical_etypes:
        print('graph number edges--'+str(each)+':',graph.number_of_edges(each))
    for each in graph.ntypes:
        print('graph number nodes--'+str(each)+':',graph.number_of_nodes(each))
    return graph


# In[11]:


relations_list = []
relations_data_list  = []

path_1 = 's3://xhs.alpha/reddm/dm_as_antispam_chuixue_gcn_user_use_relations_day_inc/dtm={}'.format(yesterday)
relations_1 = pq.ParquetDataset(path_1,filesystem=s3).read().to_pandas()
relation_1_foward_edge = list(zip(relations_1.node_1.values, relations_1.node_2.values))
relation_1_back_edge = list(zip(relations_1.node_2.values, relations_1.node_1.values))
relations_list.append(('user', 'use', 'user_use'))
relations_list.append(('user_use', 'use by', 'user'))
relations_data_list.append(relation_1_foward_edge)
relations_data_list.append(relation_1_back_edge)


path_1 = 's3://xhs.alpha/reddm/dm_as_antispam_chuixue_user_note_relations_day_inc/dtm={}'.format(yesterday)
relations_1 = pq.ParquetDataset(path_1,filesystem=s3).read().to_pandas()
relation_1_foward_edge = list(zip(relations_1.node_1.values, relations_1.node_2.values))
relation_1_back_edge = list(zip(relations_1.node_2.values, relations_1.node_1.values))
relations_list.append(('user', 'interact', 'note'))
relations_list.append(('note', 'interact by', 'user'))
relations_data_list.append(relation_1_foward_edge)
relations_data_list.append(relation_1_back_edge)


graph = build_graph(relations_list,relations_data_list)


# In[12]:


import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import RobustScaler

def node_feature_handle(df,categorical_variables_list,numerical_variables_list):
    df = df[categorical_variables_list + numerical_variables_list ]
    for each in categorical_variables_list:
        print(each)
        features_encoder = LabelBinarizer()
        features_encoder.fit(df[each])
        transformed = features_encoder.transform(df[each])
        ohe_df = pd.DataFrame(transformed)    
        df = pd.concat([df, ohe_df], axis=1)  
#   categorical_variables_list.append('no')
    df = df.drop(categorical_variables_list, axis=1)
    df.info()
    scaler = RobustScaler()
    df[numerical_variables_list] = scaler.fit_transform(df[numerical_variables_list])
    torch_tensor = torch.tensor(df.values)
    return torch_tensor


# In[13]:


def predict_scores(model,predict_node_type, graph):
    model.eval()
    with th.no_grad():
        logits,embed_vectors = model(graph)#[label_column]
        logits = logits[predict_node_type]
        logits = F.softmax(logits,dim = 1)
        return logits.cpu().detach().numpy()


# In[14]:


path_features =    's3://xhs.alpha/reddm/dm_as_antispam_chuixue_gcn_user_feature_day_inc/dtm={}'.format(yesterday)
user_nodes_features_pd = pq.ParquetDataset(path_features,filesystem=s3).read().to_pandas()


# In[15]:


cate_list = []
num_list  = ['days_active_in_l56d', 'days_active_in_l28d', 'days_active_in_l14d', 'days_active_in_l7d',
             'days_active_in_l3d','frequent_city_days_90d']
user_nodes_features_pd[cate_list] = user_nodes_features_pd[cate_list].fillna('null')
user_nodes_features_pd[num_list] =  user_nodes_features_pd[num_list].fillna(-100)


# In[16]:


user_node_feature = node_feature_handle(user_nodes_features_pd,cate_list,num_list)


# In[17]:


in_feats = user_node_feature.shape[1] #need change according to input feature 
h_dim = 16
num_classes = 2


# In[18]:


the_net = HeteroRGCN(graph,'user',user_node_feature.float(),in_feats, h_dim, num_classes).to(device=device)
the_net.load_state_dict(th.load("/apps/chuixue/gcn_model_heter_online_predict_4_28_to_4_30_temp1.pt"))


# In[19]:


scores = predict_scores(the_net,'user', graph)

print(scores)

# In[21]:


index_list = [i for i in range(scores.shape[0])]


# In[23]:


d = {'score': scores[:, 1].tolist(), 'node_no': index_list}
df = pd.DataFrame(data=d)



result_path ='/apps/chuixue/online/data/out_'+yesterday+'/'
os.mkdir(result_path)
target_path ='s3://xhs.alpha/reddm/dm_as_antispam_chuixue_gcn_daily_scores_day_inc/dtm={}'.format(yesterday)

logging.info('{}: uploading result'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
if not os.path.exists(result_path):
    os.makedirs(result_path)

def upload2s3(target_path,data):
    rm_command='aws s3 rm --recursive {}'.format(target_path)
    s3_command='aws s3 cp --recursive {} {}'.format(data,target_path)
    os.system(rm_command)
    os.system(s3_command)
    
def write2parquet(path,data):
    table=pa.Table.from_pandas(data)
    pq.write_table(table,path)

    
write2parquet(result_path+'000000_0',df)
upload2s3(target_path,result_path)
os.remove(result_path+'000000_0')
os.rmdir(result_path)
