python ?/processing.py --dataroot ./datasets/MOUSE/Train --noise_sigma 0.1 --BPNN_mode False --batch_size 8 --input_nc 1 --output_nc 1 --netG unet_256 --name pix2pix_training --n_epochs 20 --n_epochs_decay 30 --model pix2pix --direction AtoB --preprocess None
