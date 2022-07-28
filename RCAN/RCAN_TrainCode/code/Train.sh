#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=30G
#SBATCH -p GPU
#SBATCH --gres=gpu:gtxp:1
#SBATCH -t 3-0   # 3-0 (3 jours)
#SBATCH -o RCAN_mouse.out
#SBATCH -J RCAN

source /home_expes/tools/python/Python-3.6.7-ubuntu_gpu/bin/activate

srun python main.py --model RCAN --save RCAN_MOUSE --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --reset --chop --save_results --print_model --patch_size 32 --dir_data /home/jhr11385/ --data_train MOUSE --data_test MOUSE --n_train 17752 --n_val 4438 --offset_val 17752 --ext img --epochs 20

