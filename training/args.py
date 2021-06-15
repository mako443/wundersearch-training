import argparse
from argparse import ArgumentParser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cross-modal search training arguments')
    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--max_batches', default=None)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--margin', default=0.2, type=float)
    parser.add_argument('--top_k', type=int, nargs='+', default=[5, 10, 25])

    args = parser.parse_args()

    attribs=[
        f'em{args.embed_dim}',
        f'bs{args.batch_size}',
        f'ep{args.epochs}',
        f'maxb{args.max_batches}' if args.max_batches else None,
        f'm{args.margin:0.2f}'
    ]
    plot_name = '_'.join([a for a in attribs if a is not None])

    return args, plot_name