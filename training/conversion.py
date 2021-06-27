import torch
import torch.nn as nn
import torchvision.transforms as T

import coremltools as ct
import numpy as np
import os
import os.path as osp
from easydict import EasyDict
from PIL import Image
import json

from models.vse import VisualSemanticEmbedding, ImageEncoder, TextEncoder
from dataloading.flickr30k import Flickr30kDataset

'''
PyTorch -> CoreML model conversion
NOTE: Has to be run with up-to-date PyTorch version, 1.8.1+cu102 and coremltools 4.1 
'''

# Options, care to have them correct!
checkpoint_path = './checkpoints/em1024_biDir_bs64_ep16_m0.25_g0.85_rs250.pth'
args = EasyDict(
    embed_dim = 1024,
    bi_dir = True,
    num_lstm = 2
)

# Load the model
dataset_train = Flickr30kDataset('./data/flickr30k', './splits/flickr30k/train.txt', transform=None)
model = VisualSemanticEmbedding(dataset_train.get_known_words(), args)
model.load_state_dict(torch.load(checkpoint_path))

model.cpu()
model.eval()

# Convert the image encoder
image_input = torch.rand(1, 3, 500, 500)
image_traced = torch.jit.trace(model.image_encoder, image_input)
image_cml = ct.convert(
    image_traced,
    inputs=[ct.ImageType(name="input_1", shape=image_input.shape, scale=1/255.0)] # Only scale to [0,1] here, normalization is part of model
)
image_cml.save("./converted/ImageEncoder.mlmodel")

# Convert the text encoder
input_text = "there is a cat"
word_indices, sentence_lengths = model.text_encoder.prepare_sentences([input_text, ])
word_indices = torch.tensor(word_indices, dtype=torch.long, device=model.text_encoder.device)
text_traced = torch.jit.trace(model.text_encoder, word_indices)
text_cml = ct.convert(
    text_traced,
    inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 32)), dtype=np.int64)] # Up to 32 words, only one sentence
)
text_cml.save('./converted/TextEncoder.mlmodel')

# Save the word dictionary
with open('./converted/known_words.json', "w") as f:
    json.dump(model.text_encoder.known_words, f)

# Encode a test image and text to verify equal outputs in PyTorch and CoreML
img = Image.open('test-image.jpg')
t = T.ToTensor()
img = t(img)

text = "there is a cat"

img_enc, text_enc = model.forward(torch.unsqueeze(img, dim=0), [text, ])
print('Image descriptor:')
print(img_enc[0, 0:10].cpu().detach().numpy())
print('Text descriptor:')
print(text_enc[0, 0:10].cpu().detach().numpy())

'''
Image descriptor:
[ 0.01818143  0.07606838 -0.64115065  0.4427544  -0.12222917 -0.22584526
  0.0897302   0.14317872 -0.02089451 -0.0122386 ]
Text descriptor:
[ 0.0127266  -0.0528392   0.04972628 -0.16436188  0.00232068 -0.0173766
  0.04756896  0.00259494 -0.06127802  0.10899224]
'''