#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=30G
#SBATCH -p GPU
#SBATCH --gres=gpu:gtxp:1
#SBATCH -t 3-0   # 3-0 (3 jours)
#SBATCH -o FSRCNN_train_mouse.out
#SBATCH -J FSR_train

source /home_expes/tools/python/Python-3.7.1-ubuntu_gpu/bin/activate

srun python /home/jhr11385/FSRCNN-pytorch/train.py
