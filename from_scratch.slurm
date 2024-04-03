#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu-v100

source ~/software/miniconda3/bin/activate
cd ~/599-27-toki-pona
conda activate nlpa4
onmt_train -config config.yaml

##### /scratch/${SLURM_JOB_ID}
##### 5 seconds per epoch with batch size 64