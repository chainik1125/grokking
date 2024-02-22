#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=secondary-eth
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
module load anaconda/2023-Mar/3
module load cuda/11.7
nvcc --version
nvidia-smi
#SBATCH --array=0-1

# Activate the Conda environment
source activate pytorch_one


#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script
python Ising_seed.py ${SLURM_ARRAY_TASK_ID}

# Deactivate the environment
# conda deactivate
