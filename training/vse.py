import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

import os
import os.path as osp
import numpy as np
from easydict import EasyDict
import time
import matplotlib.pyplot as plt

from models.vse import VisualSemanticEmbedding
from dataloading.flickr30k import Flickr30kDataset

from training.losses import PairwiseRankingLoss
from training.args import parse_arguments
from training.plots import plot_metrics

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device:', device, torch.cuda.get_device_name(0))
torch.autograd.set_detect_anomaly(True)   

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []
    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        images, texts = model(batch['images'].to(device), batch['captions'])

        loss = criterion(images, texts)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

        if args.max_batches and i_batch >= int(args.max_batches):
            break

    return np.mean(epoch_losses)

@torch.no_grad()
def eval_epoch(model, dataloader, args):
    """Run top-k retrieval for evaluation.
    Note that only one random caption is encoded for each image.

    Args:
        model ([type]): [description]
        dataloader ([type]): [description]
        top_k (tuple, optional): [description]. Defaults to (5, 10, 25).
    """
    model.eval()

    # Encode the images and captions
    image_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))

    offset = 0
    for batch in dataloader:
        batch_size = len(batch['images'])
        images, texts = model(batch['images'].to(device), batch['captions'])
        image_encodings[offset : offset + batch_size] = images.cpu().detach().numpy()
        text_encodings[offset : offset + batch_size] = texts.cpu().detach().numpy()
        offset += batch_size

    assert len(image_encodings) == len(text_encodings) == len(dataloader.dataset)

    # Run retrieval
    accuracies = {k: [] for k in args.top_k}
    for query_idx in range(len(text_encodings)):
        scores = image_encodings[:] @ text_encodings[query_idx]
        sorted_indices = np.argsort(-1.0 * scores)[0 : max(args.top_k)] # Scores high -> low
        for k in args.top_k:
            accuracies[k].append(query_idx in sorted_indices[0:k])

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
    return accuracies

if __name__ == "__main__":
    args, plot_name = parse_arguments()
    print('Plot:', plot_name)

    target_size = (500, 500) # For Flickr30k, seems to always have max-size 500
    transform = T.Compose([
        T.RandomCrop(target_size, pad_if_needed=True), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = Flickr30kDataset('./data/flickr30k', './splits/flickr30k/val.txt', transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)

    learning_rates = np.logspace(-2, -4, 3)

    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc = {k: {lr: [] for lr in learning_rates} for k in args.top_k}

    for lr in learning_rates:
        model = VisualSemanticEmbedding(dataset_train.get_known_words(), args.embed_dim)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = PairwiseRankingLoss(args.margin)

        for epoch in range(1, args.epochs+1):
            t0 = time.time()
            loss = train_epoch(model, dataloader_train, args)
            t1 = time.time()

            accs = eval_epoch(model, dataloader_train, args)
            for k in args.top_k:
                dict_acc[k][lr].append(accs[k])

            dict_loss[lr].append(loss)
            print(f'\t lr {lr:0.6f} epoch {epoch}: loss {loss:0.2f} ela {t1-t0:0.2f} ', end='')
            for k, v in accs.items():
                print(f'{k}-{v:0.2f} ', end="")
            print("\n", flush=True)                

    train_accs = {f'train-acc-{k}': dict_acc[k] for k in args.top_k}

    metrics = {
        'train-loss': dict_loss,
        **train_accs
    }
    plot_metrics(metrics, osp.join('plots', plot_name + '.png'))
    