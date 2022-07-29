# TRAINING DATA

python ./prepare.py --image-dir ../datasets/MOUSE/LR/Train --label-dir ../datasets/MOUSE/HR/Train --output-path ../TRAININGBASE.h5 

# TESTING DATA

python ./prepare.py --image-dir ../datasets/MOUSE/LR/Test --label-dir ../datasets/MOUSE/HR/Test --output-path ../TESTINGBASE.h5 --eval True

