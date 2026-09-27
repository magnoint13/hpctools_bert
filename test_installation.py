#!/bin/env python3

import torch

cuda_available = torch.cuda.is_available()
print("Torch version:     ", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:    ", cuda_available)
if cuda_available:
    print("GPU:               ", torch.cuda.get_device_name(0))

import datasets
import transformers

print()
print("Datasets version:    ", datasets.__version__)
print("Transformers version:", transformers.__version__)

from transformers import BertTokenizer, BertModel

print("\n==== Model test ====")
text = input("Input text > ")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)

print(output)
