#!/bin/bash/
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -p GPU
#SBATCH --gres=gpu:gtxp:1
#SBATCH -t 10:00:00
#SBATCH --mail-user=rehan.jhuboo@univ-st-etienne.fr
#SBATCH --mail-type=FAIL
#SBATCH -e RCAN_test.err
#SBATCH -o RCAN_test.out
#SBATCH -J RCAN

source /home_expes/tools/python/Python-3.6.7-ubuntu_gpu/bin/activate


# RCAN_BIX2
srun python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --test_only --save_results --chop --save 'RCAN' --testpath ~/LR/ --testset Set5
# RCAN_BIX3
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX3.pt --test_only --save_results --chop --save 'RCAN' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCAN_BIX4
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --save 'RCAN' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCAN_BIX8
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX8.pt --test_only --save_results --chop --save 'RCAN' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
##
# RCANplus_BIX2
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCANplus_BIX3
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCANplus_BIX4
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCANplus_BIX8
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 8 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5

