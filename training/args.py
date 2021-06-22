import argparse
from argparse import ArgumentParser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cross-modal search training arguments')
    # Training
    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_batches', default=None)
    parser.add_argument('--margin', default=0.2, type=float)
    parser.add_argument('--lr_gamma', default=1.0, type=float)
    
    # Model
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--bi_dir', action='store_true', help="Use a bi-directional LSTM in the text encoder.")

    # Data
    parser.add_argument('--resize_image', type=int, default=250)

    # Eval
    parser.add_argument('--top_k', type=int, nargs='+', default=[5, 10, 25])

    args = parser.parse_args()

    if args.resize_image is not None:
        args.resize_image = int(args.resize_image)

    attribs=[
        f'em{args.embed_dim}',
        f'bs{args.batch_size}',
        f'ep{args.epochs}',
        f'maxb{args.max_batches}' if args.max_batches else None,
        f'm{args.margin:0.2f}',
        f'g{args.lr_gamma:0.2f}' if args.lr_gamma != 1.0 else None,
        f'rs{args.resize_image}' if args.resize_image else None
    ]
    plot_name = '_'.join([a for a in attribs if a is not None])

    return args, plot_name