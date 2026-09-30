#!/bin/env python3

import torch
import datasets
import transformers
import sys

print("Python:            ", sys.version)

cuda_available = torch.cuda.is_available()
print("Torch version:     ", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:    ", cuda_available)
if cuda_available:
    print("GPU:               ", torch.cuda.get_device_name(0))

print()
print("Datasets version:    ", datasets.__version__)
print("Transformers version:", transformers.__version__)
