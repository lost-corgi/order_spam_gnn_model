# import torch
# import dgl
# import numpy as np
# import scipy.sparse as ssp
# import tqdm
# import dask.dataframe as dd
import torch.nn.functional as F
import traceback
from _thread import start_new_thread
from functools import wraps

from torch.multiprocessing import Queue

def load_feature_subtensor(nfeats, input_nodes, is_pad, device):
    """
    Extracts features for a set of nodes.
    """
    if is_pad:
        batch_inputs = {}
        for k, v in nfeats.items():
            if k is 'user':
                batch_inputs[k] = F.pad(v[input_nodes[k]], (0, 4))
            else:
                batch_inputs[k] = F.pad(v[input_nodes[k]], (24, 0))
            batch_inputs[k] = batch_inputs[k].to(device)
    else:
        batch_inputs = {k: v[input_nodes[k]].to(device) for k, v in nfeats.items()}
    return batch_inputs

def load_subtensor(nfeats, labels, seeds, input_nodes, label_type, is_pad, device):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = load_feature_subtensor(nfeats, input_nodes, is_pad, device)
    batch_labels = labels[seeds[label_type]].to(device)
    return batch_inputs, batch_labels

#######################################################################
#
# Multithread wrapper
#
#######################################################################

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

# # This is the train-test split method most of the recommender system papers running on MovieLens
# # takes.  It essentially follows the intuition of "training on the past and predict the future".
# # One can also change the threshold to make validation and test set take larger proportions.
# def train_test_split_by_time(df, timestamp, user):
#     df['train_mask'] = np.ones((len(df),), dtype=np.bool)
#     df['val_mask'] = np.zeros((len(df),), dtype=np.bool)
#     df['test_mask'] = np.zeros((len(df),), dtype=np.bool)
#     df = dd.from_pandas(df, npartitions=10)
#     def train_test_split(df):
#         df = df.sort_values([timestamp])
#         if df.shape[0] > 1:
#             df.iloc[-1, -3] = False
#             df.iloc[-1, -1] = True
#         if df.shape[0] > 2:
#             df.iloc[-2, -3] = False
#             df.iloc[-2, -2] = True
#         return df
#     df = df.groupby(user, group_keys=False).apply(train_test_split).compute(scheduler='processes').sort_index()
#     print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
#     return df['train_mask'].to_numpy().nonzero()[0], \
#            df['val_mask'].to_numpy().nonzero()[0], \
#            df['test_mask'].to_numpy().nonzero()[0]
#
# def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
#     train_g = g.edge_subgraph(
#         {etype: train_indices, etype_rev: train_indices},
#         preserve_nodes=True)
#     # remove the induced node IDs - should be assigned by model instead
#     del train_g.nodes[utype].data[dgl.NID]
#     del train_g.nodes[itype].data[dgl.NID]
#
#     # copy features
#     for ntype in g.ntypes:
#         for col, data in g.nodes[ntype].data.items():
#             train_g.nodes[ntype].data[col] = data
#     for etype in g.etypes:
#         for col, data in g.edges[etype].data.items():
#             train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]
#
#     return train_g
#
# def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
#     n_users = g.number_of_nodes(utype)
#     n_items = g.number_of_nodes(itype)
#     val_src, val_dst = g.find_edges(val_indices, etype=etype)
#     test_src, test_dst = g.find_edges(test_indices, etype=etype)
#     val_src = val_src.numpy()
#     val_dst = val_dst.numpy()
#     test_src = test_src.numpy()
#     test_dst = test_dst.numpy()
#     val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
#     test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))
#
#     return val_matrix, test_matrix
#
# def linear_normalize(values):
#     return (values - values.min(0, keepdims=True)) / \
#         (values.max(0, keepdims=True) - values.min(0, keepdims=True))

# # construct_computation_graph(g, n_layers, label_df[label_entity_col_name].values, label_entity_type)
# def construct_computation_graph(graph, n_layers, seed_node_ids, seed_node_type):
#     sub_g = graph.in_subgraph({seed_node_type: seed_node_ids})

# def node_feature_handle(df,categorical_variables_list,numerical_variables_list):
#     df = df[categorical_variables_list + numerical_variables_list ]
#     for each in categorical_variables_list:
#         print(each)
#         features_encoder = LabelBinarizer()
#         features_encoder.fit(df[each])
#         transformed = features_encoder.transform(df[each])
#         ohe_df = pd.DataFrame(transformed)
#         df = pd.concat([df, ohe_df], axis=1)
# #   categorical_variables_list.append('no')
#     df = df.drop(categorical_variables_list, axis=1)
#     df.info()
#     scaler = RobustScaler()
#     df[numerical_variables_list] = scaler.fit_transform(df[numerical_variables_list])
#     torch_tensor = torch.tensor(df.values)
#     return torch_tensor
