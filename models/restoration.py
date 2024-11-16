import torch
import torch.nn as nn
import utils
import torchvision
import os
import PIL
import torch.nn.functional as F
from models.color_restoration import  rgb_histogram, Loss_color, ColorRestoration
import numpy as np
import cv2

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)             

class DiffusiveRestoration:
    def __init__(self,  diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        #self.join = join

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')


    def restore(self, val_loader, validation='snow', r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
       # self.join.sample_validation_patches(val_loader, 'eval_1000', 1000)
        with torch.no_grad():
            for i, (x, reference, y) in enumerate(val_loader):
          
                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
              
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
             
                x_output = self.diffusive_restoration(x_cond, r=r).to('cuda')
        
                print(image_folder)
                
                #x_output = F.interpolate(x_output, size=(400, 600), mode='bilinear', align_corners=False)
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}.png").replace("'", '').replace("[", '').replace(']', ''))
                

    def diffusive_restoration(self, x_cond, r=None):

        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=64, r=16)
        corners = [(i, j) for i in h_list for j in w_list]
        noise_size = x_cond[:, :3, :, :]
        x = torch.randn(noise_size.size(), device=self.diffusion.device)    
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=64)

        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=192, r=16)
        corners = [(i, j) for i in h_list for j in w_list]
        #print(corners)
        noise_size = x_cond[:, :3, :, :]
        #print(noise_size.size())
        x = torch.randn(noise_size.size(), device=self.diffusion.device)
        x_output_2 = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=192)
        

        #x_output = (x_output_1  + x_output_2  ) / 2 
        #x_output = (x_output_1*1.05  + x_output_2*1.05 ) / 2

        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
       
        
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
