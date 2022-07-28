#!/bin/bash
#SBATCH --job-name=DAN			# nom du job
#SBATCH --partition=gpu_p2 
#SBATCH --qos=qos_gpu-t4		# QOS : t3=20h-512GPU, t4=100h-16GPU, dev=2h-35GPU
#SBATCH --output=DAN_test_%j.out	# fichier de sortie
#SBATCH --error=DAN_test_%j.err	# fichier d'erreur
#SBATCH --time=06:00:00			# temps maximal d'allocation "(HH:MM:SS)"
#SBATCH --nodes=1			# reserver n noeud (computer)
#SBATCH --gres=gpu:1			# reserver n GPU
#SBATCH --cpus-per-task=3		# reserver n cpu par tache (et memoire associee)
#SBATCH --hint=nomultithread		# desactiver l'hyperthreading
#SBATCH --account=tvs@gpu		# specifier projet gpu ou cpu

module purge				# nettoyer les modules herites par defaut
conda deactivate			# desactiver les environnements herites par defaut

module load pytorch-gpu/py3/1.5.0	# charger les modules

				
srun python test.py -opt=./options/setting1/test/test_setting1_x2_mouse.yml
