import argparse
import numpy as np
import torch
import dgl
from hgraph_builder import *
import s3fs
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
import os
from dateutil.parser import parse as dt_parse
from train import train

# dsnodash = dt_parse(args.date_key).strftime('%Y%m%d')

def write2parquet(path, data):
    table = pa.Table.from_pandas(data)
    pq.write_table(table, path)

def upload2s3(target_path, data):
    rm_command = 'aws s3 rm --recursive {}'.format(target_path)
    s3_command = 'aws s3 cp --recursive {} {}'.format(data, target_path)
    os.system(rm_command)
    os.system(s3_command)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Train user-device graph")
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--sample-ratio', type=int, default=3)
    argparser.add_argument('--num-epochs', type=int, default=300)
    # argparser.add_argument('--input-dim', type=int, default=10)
    argparser.add_argument('--hidden-dim', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=5)
    argparser.add_argument('--fan-out', type=str, default='20,20,20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--is-pad', type=bool, default=True)
    argparser.add_argument('--user-table', type=str,
                           default='dm_as_gnn_user_place_order_user_normalized_feature_7d_inc')
    argparser.add_argument('--device-table', type=str, default='dm_as_gnn_user_place_order_device_feature_7d_inc')
    argparser.add_argument('--relation-table', type=str, default='dm_as_gnn_user_place_order_device_relation_7d_inc')
    argparser.add_argument('--label-table', type=str, default='dm_as_gnn_user_place_order_label_7d')
    argparser.add_argument('--out-table', type=str, default='dm_as_gnn_user_place_order_pred_7d_inc')
    argparser.add_argument('--label-entity', type=str, default='user')
    argparser.add_argument('--dsnodash', type=str, default='20210309')
    argparser.add_argument('--debug', action='store_true')
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    if not args.debug:
        s3 = s3fs.S3FileSystem()
        user_table_path = 's3://xhs.alpha/reddm/' + args.user_table + '/dtm=%s' % args.dsnodash
        user_features = pq.ParquetDataset(user_table_path, filesystem=s3).read().to_pandas()
        device_table_path = 's3://xhs.alpha/reddm/' + args.device_table + '/dtm=%s' % args.dsnodash
        device_features = pq.ParquetDataset(device_table_path, filesystem=s3).read().to_pandas()
        relation_table_path = 's3://xhs.alpha/reddm/' + args.relation_table + '/dtm=%s' % args.dsnodash
        relation_df = pq.ParquetDataset(relation_table_path, filesystem=s3).read().to_pandas()
        label_table_path = 's3://xhs.alpha/reddm/' + args.label_table + '/dtm=%s' % args.dsnodash
        labels = pq.ParquetDataset(label_table_path, filesystem=s3).read().to_pandas()
        # Build graph
        graph_builder = PandasGraphBuilder()
        graph_builder.add_entities(user_features, 'user_entity_id', 'user')
        graph_builder.add_entities(device_features, 'device_entity_id', 'device')
        graph_builder.add_binary_relations(relation_df, 'user_entity_id', 'device_entity_id', 'used')
        graph_builder.add_binary_relations(relation_df, 'device_entity_id', 'user_entity_id', 'used-by')
        g = graph_builder.build()
        dgl.save_graphs('./dataset/dgl_graph', [g])
        # Assign features.
        user_features = user_features.sort_values(by='user_entity_id').values[:, 1:]
        device_features = device_features.sort_values(by='device_entity_id').values[:, 1:]
        labels = labels.values
        np.random.shuffle(labels)
        pos_label_count = np.count_nonzero(labels[:, 1] > 0)
        neg_labels = labels[labels[:, 1] == 0]
        neg_labels = neg_labels[:pos_label_count*args.sample_ratio, :]
        labels = np.vstack((labels[labels[:, 1] > 0], neg_labels))
        np.savez_compressed('./dataset/feat_and_label', user_f=user_features, device_f=device_features, labels=labels)
    else:
        g = dgl.load_graphs('./dataset/dgl_graph')[0][0]
        np_ds = np.load('./dataset/feat_and_label.npz' % args.dsnodash)
        user_features, device_features, labels = np_ds['user_f'], np_ds['device_f'], np_ds['labels']

    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()
    val_num, test_num = labels.shape[0] // 10, labels.shape[0] // 10
    n_classes = labels[:, 1].max() + 1
    num_user_feature = user_features.shape[1]
    num_device_feature = device_features.shape[1]

    np.random.shuffle(labels)
    train_idx, val_idx, test_idx = torch.from_numpy(labels[val_num + test_num:, 0]),\
                                   torch.from_numpy(labels[:val_num, 0]), torch.from_numpy(labels[val_num:val_num + test_num, 0])
    np.savez_compressed('./dataset/train_val_test', train_idx=labels[val_num + test_num:, 0],
                        val_idx=labels[:val_num, 0], test_idx=labels[val_num:val_num + test_num, 0])
    expand_labels = np.empty(user_features.shape[0], dtype=np.float32)
    expand_labels[labels[:, 0]] = labels[:, 1]
    labels = torch.from_numpy(expand_labels).to(device)
    labels = torch.unsqueeze(labels, 1)

    user_features = torch.from_numpy(user_features).type(torch.float32).to(device)
    device_features = torch.from_numpy(device_features).type(torch.float32).to(device)
    entity_features = {'user': user_features, 'device': device_features}

    # g.edges['used'].data['weights'] = torch.ShortTensor(relation_df['relation_edge_weight'].values)
    # g.edges['used-by'].data['weights'] = torch.ShortTensor(relation_df['relation_edge_weight'].values)
    # del relation_df

    # prepare for training
    data = train_idx, val_idx, test_idx, num_user_feature + num_device_feature, num_user_feature, num_device_feature, \
           labels, n_classes, entity_features, g

    pred_np = train(args, device, data)
    out_df = pd.DataFrame(np.expand_dims(pred_np, 1))

    output_path = 'dataset/'
    target_path = 's3://xhs.alpha/reddm/%s/dtm=%s' % (args.dsnodash, args.out_table)
    write2parquet(output_path + '000000_0', out_df)
    upload2s3(target_path, output_path)
