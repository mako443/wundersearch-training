from typing import List

import numpy as np
import os
import os.path as osp
import cv2
from PIL import Image
import json

import torch
from torch.utils.data import Dataset, DataLoader

from dataloading.utils import simplify_sentence, build_word_embedding, load_glove

class Flickr30kDataset(Dataset):
    """DataLoader for Flickr30k, loads the dataset image-wise and returns a random caption per image.
    """
    def __init__(self, base_path, split_filepath, transform=None):
        self.image_folder = osp.join(base_path, 'flickr30k_images')
        self.text_file = osp.join(base_path, 'results.csv')
        assert osp.isdir(self.image_folder) and osp.isfile(self.text_file) and osp.isfile(split_filepath)
        self.transform = transform

        # Load the split
        with open(split_filepath, 'r') as f:
            lines = f.readlines()
            self.image_names = [line.strip() for line in lines if '.jpg' in line]
        
        # Load the captions
        self.image_captions = {image_name: [] for image_name in self.image_names} # Captions as {image_name: [c0, c1, ...]}
        with open(self.text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                split = line.split('|')
                if len(split) != 3:
                    continue
                
                image_name, caption = split[0].strip(), split[2].strip()
                if image_name in self.image_captions:
                    self.image_captions[image_name].append(caption)

        self.num_captions = sum([len(v) for v in self.image_captions.values()])
        print(str(self))

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(osp.join(self.image_folder, image_name))
        
        caption = np.random.choice(self.image_captions[image_name])
        caption = simplify_sentence(caption)
        
        if self.transform:
            image = self.transform(image)

        return {
            'images': image,
            'captions': caption
        }

    def __repr__(self):
        return f'Flickr30kDataset: {self.num_captions} captions for {len(self)} images'

    def __len__(self):
        return len(self.image_names)

    def get_known_words(self):
        sentences = []
        for captions in self.image_captions.values():
            sentences.extend([simplify_sentence(sentence) for sentence in captions])
        words = []
        for sentence in sentences:
            words.extend(sentence.split())
        return list(np.unique(words))

    def get_word_counts(self):
        known_words = self.get_known_words()
        word_counts = {word: 0 for word in known_words}

        for captions in self.image_captions.values():
            for caption in captions:
                for word in simplify_sentence(caption).split():
                    word_counts[word] +=1
        return word_counts

    def get_most_frequent_words(self, coverage=0.95):
        """Filter the words by occurence to cover <coverage> of the mentioned words.

        Args:
            coverage (float, optional): Amount of occurences to cover. Defaults to 0.95.

        Returns:
            List[str]: Filtered words
        """
        word_counts = self.get_word_counts()
        words = np.array(list(word_counts.keys()))
        counts = np.array([word_counts[word] for word in words])

        indices = np.argsort(-1 * counts) # High -> Low
        words = words[indices]
        counts = counts[indices]
        cumsum = np.cumsum(counts)
        total_occurences = cumsum[-1]
        filter_index = np.where(cumsum >= total_occurences * coverage)[0][0] # The index at which <coverage> of the occurences are covered.

        print(f'Selected {filter_index+1} of {len(words)} words for {coverage:0.2f} coverage')

        return words[0:filter_index+1]

if __name__ == '__main__':
    dataset = Flickr30kDataset('./data/flickr30k', './splits/flickr30k/val.txt')            
    data = dataset[0]
    words = dataset.get_most_frequent_words()

    # Building the dictionary based on the training data
    glove_words, glove_embeds = load_glove('/home/ubuntu/Downloads/')
    word_indices, word_embeds = build_word_embedding(words, glove_words, glove_embeds)

    # Save the indices and embeddings
    with open('./data/flickr30k/word_indices.json', 'w') as f:
        json.dump(word_indices,f)
    np.save('./data/flickr30k/word_embeds.npy', word_embeds)