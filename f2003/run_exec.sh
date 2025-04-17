#!/bin/bash
#
#SBATCH --job-name=single-GPU
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#
#SBATCH --account=c01-eng
#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd
#
# Check assigned GPU
# srun --ntasks=1 rocm-smi
# srun --ntasks=1 rocminfo
#
srun --ntasks=1 --cpus-per-task=1 ./main
