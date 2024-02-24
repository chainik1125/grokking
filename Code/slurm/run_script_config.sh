#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --partition=secondary-eth
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-2
module load anaconda/2023-Mar/3
module load cuda/11.7
# nvcc --version
# nvidia-smi


# Activate the Conda environment
source activate pytorch_one

config=config2.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
data_seed_start=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
data_seed_end=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
sgd_seed_start=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
sgd_seed_end=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
int_seed_start=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
int_seed_end=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

# Extract the sex for the current $SLURM_ARRAY_TASK_ID
wd=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)

grok=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)

train_size=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $10}' $config)
#Run that file hombre
srun python3  ../Ising_seed2.py ${SLURM_ARRAY_TASK_ID} ${data_seed_start} ${data_seed_end} ${sgd_seed_start} ${sgd_seed_end} ${init_seed_start} ${int_seed_end} ${wd} ${grok} ${train_size}

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
echo "This is array task ${SLURM_ARRAY_TASK_ID}, ${data_seed_start} ${data_seed_end} ${sgd_seed_start} ${sgd_seed_end} ${init_seed_start} ${int_seed_end} ${wd} ${grok} ${train_size}" >> output.txt

#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script


# Deactivate the environment
# conda deactivate
