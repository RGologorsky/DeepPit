#!/usr/bin/env python

import os
import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nchunks", default=1, type=int)
parser.add_argument("-t", "--time",    default=0, type=int)
parser.add_argument("-m", "--timemin", default=30, type=int)
parser.add_argument("-s", "--script",  default="DeepPit/100c_ensemble-do-save", type=str)
parser.add_argument("-f", "--sif",  default="new_monai_1906", type=str)
parser.add_argument("-p", "--partition", default="gpu4_dev", type=str)
parser.add_argument("-o", "--output",  default="train", type=str)

parser.add_argument("-a", "--taskid",  default=0, type=int)

args = parser.parse_args()

print("Before", args.time)
if args.time >= 10:
  time_str = str(args.time)
else:
  print(args.time)
  time_str = "0" + str(args.time) #f"{args.time:02}"  
 
if args.timemin >= 10:
  timemin_str = str(args.timemin)
else:
  print(args.timemin)
  timemin_str = "0" + str(args.timemin) #f"{args.time:02}" 
print("Time min str: ", timemin_str)


job_name = "ensemble"
job_file = "pyensemble.scr"

code_bind_dir = "/gpfs/home/gologr01/DeepPit/:/DeepPit/"
data_bind_dir = "/gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/"

nbconvert_cmd = f"jupyter nbconvert --to script {args.script}.ipynb"
run_cmd       = f"python3 {args.script}.py"

cmds = f"{nbconvert_cmd};{run_cmd}" 

lines = "#!/bin/bash" + "\n"

lines += f"#SBATCH --partition={args.partition}" + "\n"
lines += f"#SBATCH --nodes=1"               + "\n"
lines += f"#SBATCH --tasks-per-node=1"     + "\n"
lines += f"#SBATCH --cpus-per-task=5"       + "\n"
lines += f"#SBATCH --time=0-{time_str}:{timemin_str}:00"       + "\n"
lines += f"#SBATCH --mem-per-cpu=8G"        + "\n"
lines += f"#SBATCH --gres=gpu:1"            + "\n"

lines += f"#SBATCH --job-name=sing_job_ensemble"  + "\n"  # Job name
lines += f"#SBATCH --output=sing_ensemble_%j.log" + "\n"  # Standard output and error log

# array
lines += f"#SBATCH --array=0-{args.nchunks-1}" + "\n"
lines += f"#SBATCH --output=jobarray_output/{args.output}_job_array_n4_%A_%a.out" + "\n"

lines += f"module load python; module load singularity" + "\n"
lines += f"singularity exec --bind {code_bind_dir} {args.sif}.sif bash -c python3 'print(job_name)'" + "\n"
lines += f"singularity exec --bind {code_bind_dir} {args.sif}.sif bash -c python3 'import os; print(os.getcwd()); print(os.listdir())'" + "\n"
lines += f"singularity exec --bind {code_bind_dir} {args.sif}.sif bash -c 'jupyter nbconvert --to script {args.script}.ipynb'" + "\n"
#lines += f"singularity exec --bind {data_bind_dir} --bind {code_bind_dir} --nv {args.sif}.sif python3 DeepPit/mylaunch.py {args.script}.py taskid={args.taskid}" + "\n"

lines += f"singularity exec --bind {data_bind_dir} --bind {code_bind_dir} --nv {args.sif}.sif python3 {args.script}.py taskid={args.taskid}" + "\n"



# write sbatch script
with open(job_file, "w") as fh:
    fh.writelines(lines)
    
    # echo
    fh.writelines(f'echo "SLURM_JOBID: " $SLURM_JOBID\n')
    # fh.writelines(f'echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID\n')
    # fh.writelines(f'echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID\n')


os.system(f"sbatch {job_file}")
