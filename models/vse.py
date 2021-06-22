from typing import List

import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models

class ImageEncoder(torch.nn.Module):
    """Image encoder for cross-modal retrieval
    """
    def __init__(self, args):
        super(ImageEncoder, self).__init__()

        # self.image_model = torchvision.models.mobilenet_v3_small(pretrained=True) # Does not convert to CoreML via CoreMLTools
        self.image_model = torchvision.models.mobilenet_v2(pretrained=True)
        self.image_model.classifier = nn.Identity() # Cut off before the last layer
        self.image_dim = list(self.image_model.parameters())[-1].shape[0]
        self.linear = nn.Linear(self.image_dim, args.embed_dim) # W_i from the paper  

    def forward(self, images: torch.Tensor):
        x = images.to(self.device)
        x = self.image_model(x)
        x = self.linear(x)
        return x

    @property
    def device(self):
        return next(self.linear.parameters()).device 

class TextEncoder(torch.nn.Module):
    """Text encoder for cross-modal retrieval
    """
    def __init__(self, known_words, args, num_layers=1):
        super(TextEncoder, self).__init__()

        self.known_words = {c: (i+1) for i,c in enumerate(known_words)}
        self.known_words['<unk>'] = 0        
        self.word_embedding = nn.Embedding(len(self.known_words), args.embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=args.embed_dim, hidden_size=args.embed_dim, bidirectional=args.bi_dir, num_layers=num_layers)                       
        
        self.linear = nn.Linear(args.embed_dim, args.embed_dim) # W_t from the paper

    # TODO: translate this to Swift
    def prepare_sentences(self, sentences: List[str]):
        word_indices = [ [self.known_words.get(word, 0) for word in sentence.split()] for sentence in sentences]
        sentence_lengths = [len(w) for w in word_indices]
        batch_size, max_length = len(word_indices), max(sentence_lengths)
        padded_word_indices = np.zeros((batch_size,max_length), np.int)

        for i,caption_length in enumerate(sentence_lengths):
            padded_word_indices[i,:caption_length] = word_indices[i]
        
        return padded_word_indices, sentence_lengths

    def forward(self, word_indices: torch.Tensor, sentence_lengths, pad_sequences=True):
        """Encode a batch of sentences.
        Each word has to be encoded as an index in the known_words dict.
        Sentences have to be padded to equal lengths using prepare_sentences().
        NOTE: Deactivate pad_sequences to skip sequence padding for CoreMLTools conversion.

        Args:
            word_indices (torch.Tensor): Tensor of padded word indices.
            pad_sequences (bool): Use pack_padded_sequence for faster training of uneven sentence lengths in a batch.

        Returns:
            torch.Tensor: Encoded texts
        """
        batch_size = len(word_indices)

        word_embeddings = self.word_embedding(word_indices)
        if pad_sequences:
            word_embeddings = nn.utils.rnn.pack_padded_sequence(word_embeddings, torch.tensor(sentence_lengths), batch_first=True, enforce_sorted=False)   

        d = 2 * self.lstm.num_layers if self.lstm.bidirectional else 1 * self.lstm.num_layers
        h=torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)
        c=torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)

        _,(h,c) = self.lstm(word_embeddings, (h,c))
        text_encodings = torch.mean(h, dim=0) # [B, DIM], mean for possible bi-dir output

        text_encodings = self.linear(text_encodings)

        return text_encodings 

    @property
    def device(self):
        return next(self.linear.parameters()).device 

class VisualSemanticEmbedding(torch.nn.Module):
    """Visual-semantic Embedding model from "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Model"
    NOTE: Trace the encoder modules separately for CoreML conversion.
    """
    def __init__(self, known_words, args):
        super(VisualSemanticEmbedding, self).__init__()

        self.embed_dim = args.embed_dim

        self.image_encoder = ImageEncoder(args)
        self.text_encoder = TextEncoder(known_words, args)

    def forward(self, images: torch.Tensor, texts: List[str]):
        encoded_images = self.image_encoder(images)

        word_indices, sentence_lengths = self.text_encoder.prepare_sentences(texts)
        word_indices = torch.tensor(word_indices, dtype=torch.long, device=self.text_encoder.device)
        encoded_texts = self.text_encoder(word_indices, sentence_lengths)

        return encoded_images, encoded_texts

if __name__ == '__main__':
    model = VisualSemanticEmbedding(['a', 'b', 'c'], EasyDict(embed_dim=16, bi_dir=True))

# class VisualSemanticEmbedding(torch.nn.Module):
#     """Visual-semantic Embedding model from "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Model"
#     """
#     def __init__(self, known_words, embed_dim):
#         super(VisualSemanticEmbedding, self).__init__()

#         self.embed_dim = embed_dim
        
#         # self.image_model = torchvision.models.mobilenet_v3_small(pretrained=True) # Does not convert to CoreML via CoreMLTools
#         self.image_model = torchvision.models.mobilenet_v2(pretrained=True)
#         self.image_model.classifier = nn.Identity() # Cut off before the last layer
#         self.image_dim = list(self.image_model.parameters())[-1].shape[0]
#         self.W_i = nn.Linear(self.image_dim, embed_dim)

#         self.text_model = TextEncoder(known_words, embed_dim, bi_dir=False, num_layers=1)
#         self.W_t = nn.Linear(embed_dim, embed_dim)

#     def encode_images(self, images: torch.Tensor):
#         x = images.to(self.device)
#         x = self.image_model(x)
#         x = self.W_i(x)
#         return x

#     def encode_texts(self, texts: List[str]):
#         x = self.text_model(texts)
#         x = self.W_t(x)
#         return x

#     def forward(self, images: torch.Tensor, texts: List[str]):
#         images = self.encode_images(images)
#         texts = self.encode_texts(texts)
#         return images, texts

#     @property
#     def device(self):
#         return next(self.W_i.parameters()).device    