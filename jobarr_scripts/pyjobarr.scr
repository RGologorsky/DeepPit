#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-15%100
#SBATCH --output=jobarray_output/savePreds_job_array_n4_%A_%a.out
module purge; module load singularity
singularity exec --bind /gpfs/home/gologr01/DeepPit/:/DeepPit/ cuda_1906.sif bash -c 'jupyter nbconvert --to script DeepPit/100b_test_dice_table.ipynb'
singularity exec --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ --bind /gpfs/home/gologr01/DeepPit/:/DeepPit/ new_monai_1906.sif python3 DeepPit/100b_test_dice_table.py
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
