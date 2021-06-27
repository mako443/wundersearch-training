from typing import List
import re
import numpy as np
import os
import os.path as osp

def simplify_sentence(sentence):
    """Remove all characters expect A-Za-z0-9 and whitespaces and convert to lower case.

    Args:
        sentence (str): Input sentence

    Returns:
        str: Output sentence
    """
    # return sentence.replace('.','').replace(',','').replace(';','').lower()
    return re.sub(r'[^A-Za-z0-9 ]+', '', sentence).lower()

def prepare_glove(filepath_in, dirpath_out):
    assert osp.isfile(filepath_in) and osp.isdir(dirpath_out), "Input file or output directory not found."

    step = 50000
    for i in range(20):
        print('Section:', i, flush=True)
        start_idx = i*step
        arr = np.genfromtxt(filepath_in, dtype=str, invalid_raise=False, skip_header=start_idx, max_rows=step)
        
        words = arr[:, 0]
        embeds = np.float32(arr[:, 1:])
        np.save(osp.join(dirpath_out, f'{i:02.0f}_words.npy'), words)
        np.save(osp.join(dirpath_out, f'{i:02.0f}_embeds.npy'), embeds)

        if len(arr) < step:
            break

def load_glove(dirpath_in):
    assert osp.isdir(dirpath_in), "Input dir is not a directory."

    # Load words
    filenames = [f for f in os.listdir(dirpath_in) if 'words.npy' in f]
    words = []
    for filename in filenames:
        words.append(np.load(osp.join(dirpath_in, filename)).flatten())
    words = np.hstack(words)

    # Load vectors
    filenames = [f for f in os.listdir(dirpath_in) if 'embeds.npy' in f]
    embeds = []
    for filename in filenames:
        embeds.append(np.load(osp.join(dirpath_in, filename)))
    embeds = np.vstack(embeds)    

    assert len(words) == len(embeds)

    return words, embeds

def build_word_embedding(dataset_words, glove_words, glove_embeds):
    words = set(dataset_words).intersection(set(glove_words))
    words = list(words)
    print(f'Selected a union of {len(words)} words between {len(dataset_words)} dataset-words and {len(glove_words)} embed words.')


    glove_indices = np.array([np.where(glove_words == word)[0][0] for word in words]) # The index of each word in the glove-embeds
    word_embeds = glove_embeds[glove_indices]
    
    # Add a padding/unknown token with zero embedding
    words.insert(0, '<unk>')
    word_embeds = np.vstack((np.zeros_like(word_embeds[0]), word_embeds))

    # Convert to idx dictionary {word: embed_idx}
    word_indices = {word: i for (i, word) in enumerate(words)}

    return word_indices, word_embeds
    

if __name__ == '__main__':
    # GloVe pre-process
    prepare_glove('/home/ubuntu/Downloads/glove.6B.300d.txt', '/home/ubuntu/Downloads/')