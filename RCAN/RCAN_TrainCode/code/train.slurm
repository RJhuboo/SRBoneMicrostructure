#!/bin/bash
#SBATCH --job-name=RCAN			# nom du job
#SBATCH --partition=gpu_p2		# partition
#SBATCH --qos=qos_gpu-t3		# QOS : t3=20h-512GPU, t4=100h-16GPU, dev=2h-35GPU
#SBATCH --output=RCAN_travail_%j.out	# fichier de sortie
#SBATCH --error=RCAN_travail_%j.err	# fichier d'erreur
#SBATCH --time=20:00:00			# temps maximal d'allocation "(HH:MM:SS)"
#SBATCH --nodes=4			# reserver n noeud (computer)			
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8			# reserver n GPU
#SBATCH --cpus-per-task=3		# reserver n cpu par tache (et memoire associee)
#SBATCH --hint=nomultithread		# desactiver l'hyperthreading
#SBATCH --account=tvs@gpu		# specifier projet gpu ou cpu

module purge				# nettoyer les modules herites par defaut
conda deactivate			# desactiver les environnements herites par defaut

module load pytorch-gpu/py3/1.5.0	# charger les modules

set -x				
srun python main.py --model RCAN --save RCAN_MOUSE --scale 2 --n_resgroups 10 --n_resblocks 30 --n_feats 64 --reset --chop --save_results --print_model --patch_size 16 --dir_data /gpfsstore/rech/tvs/uki75tv/ --data_train MOUSE --data_test MOUSE --n_train 17752 --n_val 4438 --offset_val 17752 --ext img --epochs 20 --n_colors 1 --n_GPUs 42  
