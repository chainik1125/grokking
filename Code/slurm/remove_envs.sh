#!/bin/bash
#SBATCH --job-name=cuda-check       # Job name
#SBATCH --output=cuda-check-%j.out  # Standard output and error log
#SBATCH --error=cuda-check-%j.err   # Standard error log
#SBATCH --partition=eng-research-gpu            # Partition (queue) to submit to
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gres=gpu:1                # Request GPU resource
# List all conda environments except the base and strip the first 3 characters (which are spaces and an asterisk for the active environment)
envs=$(conda env list | awk '$1 != "base" {print $1}')

echo "The following Conda environments will be removed:"
echo "$envs"

# Loop through each environment and remove it
for env in $envs; do
    echo "Removing environment: $env"
    conda env remove -n $env
done

echo "All Conda environments have been removed, except the base environment."
