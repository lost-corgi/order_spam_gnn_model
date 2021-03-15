"""Infering Relational Data with Graph Convolutional Networks
"""
import argparse
import dgl
import torch as th
import numpy as np
from train import train

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("use rdf to test hetero gnn model")
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    argparser.add_argument('--num-epochs', type=int, default=30)
    argparser.add_argument('--input-dim', type=int, default=10)
    argparser.add_argument('--hidden-dim', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=64)
    argparser.add_argument('--val-batch-size', type=int, default=128)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    # argparser.add_argument('--loss-func', type=str, default='CrossEntropyLoss')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--is-pad', type=bool, default=False)
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    test_mask = g.nodes[category].data.pop('test_mask')
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    train_mask = g.nodes[category].data.pop('train_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop('labels')

    # for etype in g.canonical_etypes:
    #     g.edges[etype].data['weights'] = th.ones(g.num_edges(etype))
    # remove below to test multi-class case
    labels[labels >= 1] = 1
    labels[labels < 1] = 0
    labels = labels.type(th.FloatTensor)
    labels = th.unsqueeze(labels, 1)
    labels = labels.to(device)
    # no val set in rdf datasets, use test set for testing functionality
    val_idx = test_idx
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    g.create_formats_()

    entity_features = {entity: th.randn(g.number_of_nodes(entity), args.input_dim, device=device) for entity in g.ntypes}
    data = train_idx, val_idx, test_idx, args.input_dim, 0, 0, labels, num_classes, entity_features, g
    args.label_entity = category

    test_accs = []
    for i in range(1):
        test_accs.append(train(args, device, data))
        print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
