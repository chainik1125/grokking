#!/bin/bash

#SBATCH --job-name=cuda-check       # Job name
#SBATCH --output=cuda-check-%j.out  # Standard output and error log
#SBATCH --error=cuda-check-%j.err   # Standard error log
#SBATCH --partition=eng-research-gpu            # Partition (queue) to submit to
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gres=gpu:1                # Request GPU resource

# Load the CUDA module
echo "Loading CUDA module..."
module load cuda

# Check if CUDA module is loaded
echo "Checking loaded modules..."
module list

# Display CUDA module details
echo "Showing CUDA module details..."
module show cuda

# Check and display relevant environment variables
echo "Checking environment variables..."
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# End of script
