#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --partition=secondary
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-26
module load anaconda/2023-Mar/3
module load cuda/11.7
# nvcc --version
# nvidia-smi


# Activate the Conda environment
source activate pytorch_one

config=config.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
seed=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

# Extract the sex for the current $SLURM_ARRAY_TASK_ID
wd=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

grok=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

#Run that file hombre
srun python3  ../Ising_seed.py ${SLURM_ARRAY_TASK_ID} ${seed} ${wd} ${grok}

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
echo "This is array task ${SLURM_ARRAY_TASK_ID}, seed ${seed}, wd ${wd}, grok ${grok}" >> output.txt

#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script


# Deactivate the environment
# conda deactivate
