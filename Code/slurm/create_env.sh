#!/bin/bash
#SBATCH --job-name=create_environment
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=eng-research-gpu
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL

# Load the Conda module, if necessary
# module load anaconda/2022-May/3
# module load cuda/11.6

# # Create a new Conda environment
# conda create -n pytorch_env python=3.8 -y

# # Activate the Conda environment
# source activate pytorch_env

# # Install PyTorch
# conda install pytorch torchvision cudatoolkit=11.6 -c pytorch

# # Install Plotly using pip
# # pip install plotly




# # Deactivate the environment
# conda deactivate





# Load the specific Conda module
module load anaconda/2022-May/3
module load cuda

# conda init bash
# Create a new Conda environment
conda create -n pytorch_env python=3.8 -y

# Activate the Conda environment
source activate pytorch_env

# Install PyTorch
conda install pytorch torchvision -c pytorch 
conda install plotly -c plotly

# Deactivate the environment
conda deactivate
