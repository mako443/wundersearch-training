import argparse
from argparse import ArgumentParser

import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cross-modal search training arguments')
    # Training
    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_batches', default=None)
    parser.add_argument('--margin', default=0.2, type=float)
    parser.add_argument('--lr_gamma', default=1.0, type=float)
    parser.add_argument('--loss', default='PRL', type=str)

    parser.add_argument('--continue_path', default=None, type=str)
    
    # Model
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--bi_dir', action='store_true', help="Use a bi-directional LSTM in the text encoder.")
    parser.add_argument('--num_lstm', default=1, type=int, help="Number of LSTM layers")
    parser.add_argument('--use_glove', action='store_true', help="Use pre-trained GloVe embeddings instead of training an Embedding layer from scratch.")

    # Data
    parser.add_argument('--resize_image', type=int, default=250)

    # Eval
    parser.add_argument('--top_k', type=int, nargs='+', default=[5, 10, 25])

    args = parser.parse_args()

    if args.resize_image is not None:
        args.resize_image = int(args.resize_image)

    if args.continue_path is not None:
        assert osp.isfile(args.continue_path)

    assert args.loss in ('PRL', 'HRL')

    attribs=[
        f'em{args.embed_dim}',
        f'biDir' if args.bi_dir else None,
        f'lstm{args.num_lstm}' if args.num_lstm!=1 else None,
        f'GloVe' if args.use_glove else None,
        f'bs{args.batch_size}',
        f'ep{args.epochs}',
        f'maxb{args.max_batches}' if args.max_batches else None,
        f'm{args.margin:0.2f}',
        f'g{args.lr_gamma:0.2f}' if args.lr_gamma != 1.0 else None,
        f'{args.loss}' if args.loss != 'PRL' else None,
        f'rs{args.resize_image}' if args.resize_image else None
    ]
    plot_name = '_'.join([a for a in attribs if a is not None])

    return args, plot_name