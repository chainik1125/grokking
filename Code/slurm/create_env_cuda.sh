#!/bin/bash
#SBATCH --job-name=create_environment
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=secondary-eth
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL

# Load the Conda module, if necessary
module load anaconda/2022-May/3
module load cuda/11.6

# Define environment name
ENV_NAME="pytorch_env"

# Check if the environment already exists and remove it if it does
conda env list | grep $ENV_NAME &> /dev/null
if [ $? == 0 ]; then
    echo "Environment $ENV_NAME already exists. Removing it."
    conda env remove -n $ENV_NAME
fi

# Create a new Conda environment
conda create -n $ENV_NAME python=3.8 -y

# Activate the Conda environment
conda activate $ENV_NAME

# Install CUDA Toolkit and cuDNN (replace CUDA_VERSION with the desired version)
# CUDA_VERSION="11.4"  # Adjust to your desired version
# sudo apt update
# sudo apt install -y cuda-$CUDA_VERSION libcudnn$CUDA_VERSION-dev

# Create a new Conda environment
conda create -n pytorch_env python=3.8 -y

# Activate the Conda environment
conda activate pytorch_env

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install Plotly using pip
# pip install plotly
# pip install tqdm
# pip install matplotlib
#pip uninstall -U kaleido
# pip install -U kaleido

# Deactivate the environment
conda deactivate
