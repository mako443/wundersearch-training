from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models

from models.textencoder import TextEncoder

class VisualSemanticEmbedding(torch.nn.Module):
    """Visual-semantic Embedding model from "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Model"
    """
    def __init__(self, known_words, embed_dim):
        super(VisualSemanticEmbedding, self).__init__()

        self.embed_dim = embed_dim
        
        # self.image_model = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.image_model = torchvision.models.mobilenet_v2(pretrained=True)
        self.image_model.classifier = nn.Identity() # Cut off before the last layer
        self.image_dim = list(self.image_model.parameters())[-1].shape[0]
        self.W_i = nn.Linear(self.image_dim, embed_dim)

        self.text_model = TextEncoder(known_words, embed_dim, bi_dir=False, num_layers=1)
        self.W_t = nn.Linear(embed_dim, embed_dim)

    def encode_images(self, images: torch.Tensor):
        x = images.to(self.device)
        x = self.image_model(x)
        x = self.W_i(x)
        return x

    def encode_texts(self, texts: List[str]):
        x = self.text_model(texts)
        x = self.W_t(x)
        return x

    def forward(self, images: torch.Tensor, texts: List[str]):
        images = self.encode_images(images)
        texts = self.encode_texts(texts)
        return images, texts

    @property
    def device(self):
        return next(self.W_i.parameters()).device    