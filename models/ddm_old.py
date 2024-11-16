import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
from models.unet_nafnet import NAFNet
import math
import random
import pandas as pd
from datetime import datetime
#创建train_acc.csv和var_acc.csv文件，记录loss和accuracy

# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm



def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        betas = cosine_beta_schedule(num_diffusion_timesteps).numpy()
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, refer, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * torch.log(1.0 - a).sqrt()
    #f = open(r'/home/eileen/Diffusion/data/lol_residual/test/test.txt', 'w')
    #print('>>>>>>>>>>>>>>>>>>>>>>model')
    #print(model)
    #f.close()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float()).to('cuda')
  
    x0_pred = (x + output * (1.0 - a.sqrt())) / a.sqrt().to('cuda')
    noise_loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0).to('cuda')

    pixel_loss  = (x0_pred - x0[:, 3:, :, :]).square().sum(dim=(1, 2, 3)).mean(dim=0).to('cuda')
    
    print('>>noise_loss ', noise_loss.item(), '  <<pixel_loss ', pixel_loss.item())

    return noise_loss+ pixel_loss*0.001, noise_loss, pixel_loss

 

class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        #self.model = DiffusionUNet(config)
        self.model = NAFNet(config)
        self.model.to(self.device)
        # model_file = open('/home/eileen/Diffusion/ckpts/model.txt', 'w+')
        # print(self.model, file=model_file)
        # model_file.close()
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        #self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=8000, eta_min=1e-7) #1e-7太小了
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        self.getModelSize(self.model)

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)
            print('>>>>>load ckpts<<<<<')

        df = pd.DataFrame(columns=['time', 'epoch', 'step','noise loss','pixel loss'])
        #df.to_csv("/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/xloss.csv",index=False) 

        
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, reference, y) in enumerate(train_loader):
                #print(x.shape)

         
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
          
                reference = reference.to(self.device)
                reference = data_transform(reference)
                e = torch.randn_like(x[:, 3:, :, :])
                #LOL_Map e = torch.randn_like(x[:, 9:, :, :])
                #e = torch.randn_like(x[:, 4:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)#ba
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss, noise_loss, pixel_loss = noise_estimation_loss(self.model, x, reference, t, e, b)
                '''if (epoch >= 0 and epoch < 1500):
                    loss = noise_loss + 0.001 * pixel_loss
                elif (epoch >= 1500 and epoch <3000):
                    loss = noise_loss + 0.01 * pixel_loss
                elif (epoch >= 3000 and epoch <4500):
                    loss = noise_loss + 10 * pixel_loss
                elif (epoch >= 4500):
                    loss = noise_loss + 10 * pixel_loss'''
                ###pixel noise   
                loss = noise_loss + 0.001 * pixel_loss
                #loss = noise_loss

                if self.step % 10 == 0:
                    print(f"epoch: {epoch}, step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                time_now = "%s"%datetime.now()
                step = "Step[%d]"%self.step
                noise_loss = "%f"%noise_loss
                pixel_loss = "%f"%pixel_loss
                #list = [time_now, epoch, step, noise_loss, pixel_loss]
                #data = pd.DataFrame([list])
                #data.to_csv('/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/xloss.csv',mode='a',header=False,index=False)

                self.optimizer.zero_grad()
                
                loss.backward()
                self.optimizer.step()
                #print(self.optimizer.state_dict()['param_groups'][0]['lr'])
                #self.lr_scheduler.step()
                self.ema_helper.update(self.model)
                data_start = time.time()
                
                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step, epoch)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                #if self.step % 100 == 0 or self.step == 1:
                #if epoch % 5 == 0:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', 'no_attention', self.config.data.dataset + str(epoch)+ '_ddpm'))

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):

        #skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        skip = 100
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        if patch_locs is not None:
            print('seq:', seq)
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
            #xs = utils.dpm_solver(x, x_cond, seq, self.model, self.betas, eta=0.,
                                    #corners=patch_locs, p_size=patch_size)
        else:
            #print('****none')
            
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            #print('****last', len(xs))
            #xs = xs[0][15]
            xs = xs[0][-1]
            #xs = xs
            #print('****xs shape', xs.shape)
        return xs
    
    def sample_validation_patches(self, val_loader, step, epoch):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
             
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            #LOL_Map x_cond = x[:, :9, :, :].to(self.device)
            #x_cond = x[:, :4, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)
         
            now_loss = 0
            for i in range(n):
                x_cond = x_cond[:, :3, :, :]
                #print('x cond', x[i].shape)
                #print('refer', reference[i].shape)
                now_loss  += (x[i] - reference[i]).square().sum(dim=(0, 1, 2)).mean(dim=0)
                

