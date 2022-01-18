import os
print(os.environ.get('CUDA_PATH'))
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)