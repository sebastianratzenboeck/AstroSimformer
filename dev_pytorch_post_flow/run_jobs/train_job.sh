#!/bin/bash
#SBATCH -J domain_adapter
#SBATCH -p gpu                 # GPU‐capable partition
#SBATCH -t 2-00:00             # walltime 2 days
#SBATCH --mem=128000           # total RAM in MB (≈128 GB)
#SBATCH -c 32                  # CPU cores
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.ratzenboeck@cfa.harvard.edu
#SBATCH --account=itc_lab
#SBATCH --test-only

module purge
module load python
mamba activate py_gpu_cuda12.4

cd $HOME/code/AstroSimformer/dev_pytorch_post_flow
python train_script.py