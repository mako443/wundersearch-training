from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextEncoder(torch.nn.Module):
    """LSTM wrapper to encode a batch of uneven sentences.
    """
    def __init__(self, known_words, embedding_dim, bi_dir=False, num_layers=1):
        super(TextEncoder, self).__init__()

        self.known_words = {c: (i+1) for i,c in enumerate(known_words)}
        self.known_words['<unk>'] = 0        
        self.word_embedding = nn.Embedding(len(self.known_words), embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, bidirectional=bi_dir, num_layers=num_layers)

    def forward(self, sentences: List[str]):
        """Encodes a batch of sentences [d1, d2, ..., d_B] with d_i a sententence, batch-size B. Sentences can have different sizes.

        Args:
            sentences (List[str]): The batch of sentences as a list.

        Returns:
            torch.Tensor: Sentence encodings
        """
        # sentences = self.simplify_sentences(sentences)

        word_indices = [ [self.known_words.get(word, 0) for word in sentence.split()] for sentence in sentences]
        sentence_lengths = [len(w) for w in word_indices]
        batch_size, max_length = len(word_indices), max(sentence_lengths)
        padded_indices = np.zeros((batch_size,max_length), np.int)

        for i,caption_length in enumerate(sentence_lengths):
            padded_indices[i,:caption_length] = word_indices[i]
        
        padded_indices = torch.from_numpy(padded_indices)
        padded_indices = padded_indices.to(self.device) #Possibly move to cuda

        embedded_words = self.word_embedding(padded_indices)
        sentence_inputs = nn.utils.rnn.pack_padded_sequence(embedded_words, torch.tensor(sentence_lengths), batch_first=True, enforce_sorted=False)   

        d = 2 * self.lstm.num_layers if self.lstm.bidirectional else 1 * self.lstm.num_layers
        h=torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)
        c=torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)

        _,(h,c) = self.lstm(sentence_inputs, (h,c))
        sentence_encodings = torch.mean(h, dim=0) # [B, DIM], mean for possible bi-dir output

        return sentence_encodings

    @property
    def device(self):
        return next(self.lstm.parameters()).device    