#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=30G
#SBATCH -p GPU
#SBATCH --gres=gpu:titanxp:1
#SBATCH -t 3-0   # 3-0 (3 jours)
#SBATCH -o FSRCNN_test_mouse.out
#SBATCH -J FSR_test

source /home_expes/tools/python/Python-3.7.1-ubuntu_gpu/bin/activate

srun python /home/jhr11385/FSRCNN-pytorch/test.py --weights-file /home/jhr11385/FSRCNN-output/MOUSE_big_data_x2/best.pth --image-dir /home/jhr11385/MOUSE/LR/Test --label-dir /home/jhr11385/MOUSE/HR/Test --output-dir /home/jhr11385/FSRCNN-output/FSRCNN_bigdata_test
