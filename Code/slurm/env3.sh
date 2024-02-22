#!/bin/bash
#SBATCH --job-name=create_environment
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=eng-research-gpu
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL

# Load the Conda module, if necessary
module load anaconda/2022-May/3
module load cuda/11.6

# Define environment name
ENV_NAME="pytorch_env"
ENV_DIR="$HOME/.conda/envs/$ENV_NAME"

# Check if the environment exists
if conda info --envs | grep "$ENV_NAME"; then
    echo "Removing existing environment '$ENV_NAME'."
    conda env remove -n "$ENV_NAME"
fi

# Check if the environment directory exists and remove it
if [ -d "$ENV_DIR" ]; then
    echo "Removing corrupted environment directory '$ENV_NAME'."
    rm -rf "$ENV_DIR"
fi

# Check if removal was successful
if [ -d "$ENV_DIR" ]; then
    echo "Failed to remove the environment directory. Please check permissions."
    exit 1
fi

# Create the Conda environment
conda create -n "$ENV_NAME" python=3.8 -y

# Activate the Conda environment
source activate "$ENV_NAME"

# Install PyTorch
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch

# Optional: Install Plotly using pip
# pip install plotly

# Deactivate the environment
conda deactivate
