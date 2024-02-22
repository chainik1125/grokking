#!/bin/bash


#SBATCH --job-name=cuda-check       # Job name
#SBATCH --output=cuda-check-%j.out  # Standard output and error log
#SBATCH --error=cuda-check-%j.err   # Standard error log
#SBATCH --partition=eng-research-gpu            # Partition (queue) to submit to
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gres=gpu:1                # Request GPU resource



# Load the specific Conda module
module load anaconda/anaconda/2023-Mar/3
module load cuda/11.7

# conda init bash
# Create a new Conda environment


# Activate the Conda environment
source activate pytorch_one

# Install PyTorch
#conda install pytorch torchvision -c pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torchvision

conda list torch
#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script
python seed_test.py
# Deactivate the environment
conda deactivate
