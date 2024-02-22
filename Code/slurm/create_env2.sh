#!/bin/bash
#SBATCH --job-name=create_environment
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=eng-research-gpu
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL




# Load the specific Conda module
module load anaconda/anaconda/2023-Mar/3
module load cuda/11.7

# conda init bash
# Create a new Conda environment
conda create -n pytorch_env python=3.10 -y

# Activate the Conda environment
source activate pytorch_env

# Install PyTorch
#conda install pytorch torchvision -c pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

conda list torch
#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script
python seed_test.py
# Deactivate the environment
conda deactivate
