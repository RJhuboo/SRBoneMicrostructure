import argparse
import os
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import calc_patch_size, convert_rgb_to_y


@calc_patch_size
def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []


    for image_path in os.listdir(args.images_dir):
        hr = pil_image.open(os.path.join(args.label_dir,image_path))
        lr = pil_image.open(os.path.join(args.images_dir,image_path))
        hr_images = []
        print("loading data ", image_path, " ...")

        


        if args.with_aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                        tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                        tmp = tmp.rotate(r, expand=True)
                        hr_images.append(tmp)
        else:   
            hr_images.append(hr)


        for hr in hr_images:
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            #lr = lr.resize((lr.width // args.scale, lr.height // args.scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            
            
                 
          

            for i in range(0, lr.shape[0] , lr.shape[0]//2):
                for j in range(0, lr.shape[1] , lr.shape[1]//2):
                    lr_patches.append(lr[i:i+(lr.shape[0]//2), j:j+(lr.shape[1]//2)])
                    hr_patches.append(hr[i*args.scale:i*args.scale+(lr.shape[0]//2)*args.scale, j*args.scale:j*args.scale+(lr.shape[1]//2)*args.scale])
            #lr_patches.append(lr)
            #hr_patches.append(hr)

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    
    print("patches size : ", len(lr_patches))
    print("patches hr size :", len(hr_patches))
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()
    


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(os.listdir(args.images_dir)):
        print("avancement", i)
        hr = pil_image.open(os.path.join(args.label_dir,image_path))
        lr = pil_image.open(os.path.join(args.images_dir,image_path))

        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
    
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default = "/home/jhr11385/MOUSE/LR/Train")
    parser.add_argument('--label-dir', type=str,default = "/home/jhr11385/MOUSE/HR/Train")
    parser.add_argument('--output-path', type=str, default = "/home/jhr11385/TRAININGBASE.h5")
    parser.add_argument('--scale', type=int, default=2) 
    parser.add_argument('--with-aug', default = False, action='store_true')
    parser.add_argument('--eval', default = False, action='store_true')
    args = parser.parse_args()

    if not args.eval:
        print("debut train")
        train(args)
        print("C'est fini")
    else:
        print("debut eval")
        eval(args)
