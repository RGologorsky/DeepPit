#!/bin/bash
sbatch <<EOT
#!/bin/bash


#SBATCH --job-name=n4_parallel

#SBATCH --time=0-00:10:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB

#SBATCH --array=0-79%100
#SBATCH --output=jobarray_output/AVBIB_job_array_n4_%A_%a.out

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

module purge
module load singularity

# singularity exec --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ cuda_1906.sif python3 DeepPit/11c_Nyul_Udupa_all.py
singularity exec --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ cuda_1906.sif python3 DeepPit/11b_save_N4_bias_all.py

#singularity run --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ cuda_1906.sif
#python3 DeepPit/11b_save_N4_bias_all.py

hostname

exit 0
EOT
