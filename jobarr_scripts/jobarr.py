#!/usr/bin/env python

import os
import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time",    default=10, type=int)
parser.add_argument("-n", "--nchunks", default=10, type=int)
parser.add_argument("-o", "--output",  default="ABIDE", type=str)
parser.add_argument("-s", "--script",  default="DeepPit/11b_save_N4_bias_all", type=str)
args = parser.parse_args()

print("Before", args.time)
if args.time >= 10:
  time_str = str(args.time)
else:
  print(args.time)
  time_str = "0" + str(args.time) #f"{args.time:02}"  
  
print("Time str: ", time_str)

job_name = "n4_parallel"
job_file = "pyjobarr.scr"

# commands
nbconvert_cmd = f"jupyter nbconvert --to script {args.script}.ipynb"
run_cmd       = f"python3 {args.script}.py"

# binds
code_bind_dir = "/gpfs/home/gologr01/DeepPit/:/DeepPit/"
data_bind_dir = "/gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/"

cmds = f"{nbconvert_cmd};{run_cmd}" 

lines = "#!/bin/bash" + "\n"

lines += f"#SBATCH --nodes=1"               + "\n"
lines += f"#SBATCH --ntasks-per-node=1"     + "\n"
lines += f"#SBATCH --cpus-per-task=1"       + "\n"
lines += f"#SBATCH --time=0-00:{time_str}:00"       + "\n"
lines += f"#SBATCH --mem-per-cpu=16G"        + "\n"

# array
lines += f"#SBATCH --array=0-{args.nchunks-1}%100" + "\n"
lines += f"#SBATCH --output=jobarray_output/{args.output}_job_array_n4_%A_%a.out" + "\n"

# run
lines += f"module purge; module load singularity" + "\n"
lines += f"singularity exec --bind {code_bind_dir} cuda_1906.sif bash -c 'jupyter nbconvert --to script {args.script}.ipynb'" + "\n"
lines += f"singularity exec --bind {data_bind_dir} --bind {code_bind_dir} new_monai_1906.sif {run_cmd}" + "\n"


# write sbatch script
with open(job_file, "w") as fh:

  # info + commands
  fh.writelines(lines)


  # echo
  fh.writelines(f'echo "SLURM_JOBID: " $SLURM_JOBID\n')
  fh.writelines(f'echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID\n')
  fh.writelines(f'echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID\n')

  # module purge, run singularity, python script
  # fh.writelines(f"module purge\n")
  # fh.writelines(f"module load singularity\n")
  # fh.writelines(f'singularity exec --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ new_monai_1906.sif {run_cmd}\n')

os.system(f"sbatch {job_file}")
