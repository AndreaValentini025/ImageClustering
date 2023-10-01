#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH --output=log_%a.txt
#SBATCH --error=log_%s.txt
#SBATCH --partition=students-prod
#SBATCH --array=0-20

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID:" $SLURM_ARRAY_TASK_ID

python face_extraction.py $SLURM_ARRAY_TASK_ID
