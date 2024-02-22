import sys
import os


os.environ['CUDA_VISIBLE_DEVICES']='0'


import torch
print(sys.version)



import subprocess
#Check if cuda installed
# Function to execute a shell command and return the output
def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, text=True)
        return output
    except subprocess.CalledProcessError as e:
        return e.output

# Running nvcc --version
nvcc_version = run_command("nvcc --version")
print("NVCC Version:\n", nvcc_version)

# Running nvidia-smi
nvidia_smi = run_command("nvidia-smi")
print("NVIDIA SMI:\n", nvidia_smi)


#import subprocess

def check_cuda():
    try:
        # Run 'nvcc --version' command and capture its output
        output = subprocess.check_output(["nvcc", "--version"], text=True)
        if "release" in output:
            # Extract CUDA version from the output
            cuda_version = output.split("release")[-1].split(",")[0].strip()
            return f"CUDA is installed. Version: {cuda_version}"
        else:
            return "CUDA is installed, but the version could not be determined."
    except FileNotFoundError:
        return "CUDA is not installed or nvcc is not in the PATH."

# Run the check
result = check_cuda()
print(result)
print(torch.cuda.is_available())


exit()


####

#########################Colab script

# standard imports
from timeit import default_timer as timer
import time
import pickle
from tqdm.auto import tqdm # for loop progress bar
import itertools as it
import numpy as np
import random
from matplotlib import pyplot as plt

# machine learning imports
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader

#####################

# general parameters
dtype = torch.float32 # very important
#seed = 0 # fixed random seed for reproducibility
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # use GPU if available
example = False # dataset is MNIST example if True
ising = True # import ising dataset

# set functions for neural network
loss_fn = nn.CrossEntropyLoss   # 'MSELoss' or 'CrossEntropyLoss'
optimizer_fn = torch.optim.Adam     # 'Adam' or 'AdamW' or 'SGD'
activation = nn.ReLU    # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

# set parameters for neural network
weight_decay = 1.0
learning_rate = 1e-3 # lr
weight_multiplier = 10.0 # initialization-scale
input_dim = 16**2 # 28 x 28 array of pixel values in MNIST (16x16 in Ising data)
output_dim = 2 # 10 different digits it could recognize (2 diff phases for Ising)
hiddenlayers=[5]

# set torch data type and random seeds
# torch.set_default_dtype(dtype)
# random.seed(seed)
# for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
#     set_seed(seed)


with open("../Data/IsingML_L16_traintest.pickle", "rb") as handle:
    data = pickle.load(handle)
    print(f'Data length: {len(data[1])}')
    print(data[0])
    data = data[1]
# shuffle data list
random.shuffle(data)
# split data into input (array) and labels (phase and temp)
inputs, phase_labels, temp_labels = zip(*data)
# for now ignore temp labels
my_X = torch.Tensor(np.array(inputs)).to(dtype) # transform to torch tensor of FLOATS
my_y = torch.Tensor(np.array(phase_labels)).to(torch.long) # transform to torch tensor of INTEGERS
my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
print(my_X.dtype, my_y.dtype)
print("Created Ising Dataset")