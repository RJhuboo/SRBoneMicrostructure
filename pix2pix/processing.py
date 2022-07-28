"""General-purpose training, testing function.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import os
from options.process_options import ProcessOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import html
import torch
import numpy as np
import pickle


class Namespace:
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)

       
def train(model, train_loader, epoch, opt):
    ''' 
    - model : Model pix2pix or CycleGAN
    - train_loader : the dataset containing training data
    - optimizer : optimizer for the gradient descent
    - criterion : Loss function 
    '''
    model.train()
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    psnr_metric = []
    ssim_metric = []
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
            
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        psnr, ssim = model.metrics()
        psnr_metric.append(psnr)
        ssim_metric.append(ssim)
        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            print("psnr  %f, ssim %f" %(psnr,ssim))
        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            psnr_mean = np.mean(psnr_metric)
            ssim_mean = np.mean(ssim_metric)

            metric_dict_train["psnr"].append(psnr_mean)
            metric_dict_train["ssim"].append(ssim_mean)
            if opt.BPNN_mode == "True":
                loss_dis["BPNN"].append(losses["BPNN"])
            #else:
                #bpnn_mean = model.Loss_extraction()
                #loss_dis["BPNN"].append(bpnn_mean)
            loss_dis["G_L1"].append(losses["G_L1"])
            loss_dis["G_GAN"].append(losses["G_GAN"])
            loss_dis["D_fake"].append(losses["D_fake"])
            loss_dis["D_real"].append(losses["D_real"])
            if opt.display_wandb is True:
                directory_ml = os.path.join(opt.results_dir,opt.name)
                if os.path.exists(directory_ml) is False:
                    os.mkdir(directory_ml)
                with open(os.path.join(directory_ml,"loss.txt"),"wb") as f:
                    pickle.dump(loss_dis,f)
                with open(os.path.join(directory_ml,"metric.txt"),"wb") as f:
                    pickle.dump(metric_dict_train,f)

            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                print("suppose to display loss")
            psnr_metric = []                # initialize psnr 
            ssim_metric = []                 # initialize ssim
            bpnn_metric = []
        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
   


def test(model,test_loader, epoch, opt_test):
    web_dir = os.path.join(opt_test.results_dir, opt_test.name, '{}_{}'.format(opt_test.phase, epoch))  # define the website directory
    if opt_test.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt_test.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt_test.name, opt_test.phase, epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    model.eval()
    with torch.no_grad():
        ssim_list = []
        psnr_list = []
        #bpnn_list = []
        for i, data in enumerate(dataset_test):
            model.set_input(data)  # unpack data from data loader
            if i < opt_test.num_test:  # only apply our model to opt.num_test images.
                #visuals = model.get_current_visuals()  # get image results
                #img_path = model.get_image_paths()     # get image paths
                print("path where images are saves during validation : ", img_path)
                if i % 5 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (i, img_path))
                #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            psnr, ssim = model.metrics()
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            if opt_test.BPNN_mode == "False":
                bpnn = model.Loss_extraction()
                bpnn = bpnn.cpu().detach().numpy()
                bpnn_list.append(bpnn)
                bpnn = np.mean(bpnn_list)
        psnr, ssim = np.mean(psnr_list), np.mean(ssim_list)
        metric_dict_test["psnr test"].append(psnr)
        metric_dict_test["ssim test"].append(ssim)
        #if opt_test.BPNN_mode == "False":
          #  metric_dict_test["bpnn test metric"].append(bpnn)
        directory_ml = os.path.join(opt.results_dir,opt.name)
        with open(os.path.join(directory_ml,"metric_test.txt"),"wb") as f:
            pickle.dump(metric_dict_test,f)
        #webpage.save()  # save the HTML


''' main '''

opt = ProcessOptions().parse()   # get training options
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)    # get the number of images in the dataset.
print('The number of training images = %d' % dataset_size)

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
opt_test = Namespace(vars(opt))
# hard-code some parameters for test
opt_test.num_threads = 0   # test code only supports num_threads = 0
opt_test.batch_size = 1    # test code only supports batch_size = 1
print("in opt:",opt.batch_size,"in opt_test:", opt_test)
opt_test.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt_test.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt_test.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
opt_test.phase = 'test'
opt_test.eval = True

dataset_test = create_dataset(opt_test)  # create a dataset given opt.dataset_mode and other options

loss_dis = {"BPNN":[],"G_GAN":[],"G_L1":[],"D_fake":[],"D_real":[]}
metric_dict_train = {"psnr":[],"ssim":[]}
metric_dict_test = {"psnr test":[],"ssim test":[]}
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq> 
    train(model, dataset, epoch, opt ) # train the epoch
    test(model, dataset_test, epoch, opt_test) # test the epoch
