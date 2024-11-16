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
from models.color_restoration import  rgb_histogram, Loss_color, ColorRestoration
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


def color_restoration(color_model, x, x0):

    ### Color restoration
    
    x = x.to('cuda')
    x = color_model(x, x0[:, :3, :, :])   
    ###

    ### color loss
    his_gt = rgb_histogram(x0[:, 3:, :, :])
    his_denoise = rgb_histogram(x)
    L_color = Loss_color()
    hist1_norm = his_gt / his_gt.sum()
    hist2_norm = his_denoise / his_denoise.sum()
    hist_loss = torch.nn.functional.pairwise_distance(hist1_norm, hist2_norm)
    ##pixel_loss  = (x - x0[:, 3:, :, :]).square().sum(dim=(1, 2, 3)).mean(dim=0)
    pixel_loss  = torch.abs(x - x0[:, 3:, :, :]).sum(dim=(1, 2, 3)).mean(dim=0)
    color_loss = torch.mean(L_color(x)) + hist_loss
    ###
    return pixel_loss, color_loss


def noise_estimation_loss(model, x0, refer, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    #f = open(r'/home/eileen/Diffusion/data/lol_residual/test/test.txt', 'w')
    #print('>>>>>>>>>>>>>>>>>>>>>>model')
    #print(model)
    #f.close()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
    x0_pred = (x - output * (1.0 - a).sqrt()) / a.sqrt()

    noise_loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    pixel_loss  = (x0_pred - x0[:, 3:, :, :]).square().sum(dim=(1, 2, 3)).mean(dim=0)

    #print('>>noise_loss ', noise_loss.item(), '  <<pixel_loss ', pixel_loss.item(), '  <<color_loss ', color_loss.item())

    return  noise_loss, pixel_loss, x0_pred

    '''LOL_Map x = x0[:, 9:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :9, :, :], x], dim=1), t.float())'''
'''    x = x0[:, 4:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    
    output = model(torch.cat([x0[:, :4, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)'''

class JoinNet(nn.Module):
    def __init__(self, config):
        super(JoinNet, self).__init__()
        self.diffusion_net = NAFNet(config)
        self.color_net = ColorRestoration()
        self.stage_one = 7500 #3500
        self.stage_two = 9000 #5000
    
    def forward(self,  x, reference, t, e, b, epoch):
        l_noise, l_pixel1, l_pixel2, l_color = -1, -1, -1, -1
        if epoch <= self.stage_one:
            l_noise, l_pixel1, x0_pred = noise_estimation_loss(self.diffusion_net,  x, reference, t, e, b)
        elif epoch > self.stage_one and epoch <= self.stage_two:
            #l_pixel2, l_color = color_restoration(self.color_net, x0_pred, x) 先不使用颜色损失
            l_noise, l_pixel1, x0_pred = noise_estimation_loss(self.diffusion_net,  x, reference, t, e, b)
            l_pixel2, _ = color_restoration(self.color_net, x0_pred, x)
        else:
            l_noise, l_pixel1, x0_pred = noise_estimation_loss(self.diffusion_net,  x, reference, t, e, b)
            #l_pixel2, l_color = color_restoration(self.color_net, x0_pred, x)
            l_pixel2, _ = color_restoration(self.color_net, x0_pred, x)
        return l_noise, l_pixel1, l_pixel2, l_color


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        #self.model = DiffusionUNet(config)
        self.join_model = JoinNet(config)
        self.subNet1 = self.join_model.diffusion_net
        self.subNet2 = self.join_model.color_net
        self.join_model.to(self.device)
        #self.join_model = torch.nn.DataParallel(self.join_model)

        '''self.model = NAFNet(config)        
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)'''

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.join_model)

        #self.optimizer = utils.optimize.get_optimizer(self.config, self.join_model.parameters())
        self.optimizer1 = utils.optimize.get_optimizer(self.config, self.subNet1.parameters(), 2e-4)
        self.optimizer2 = utils.optimize.get_optimizer(self.config, self.subNet2.parameters(), 2e-4)
        parameters = list(self.subNet1.parameters()) + list(self.subNet2.parameters())
        self.optimizer = utils.optimize.get_optimizer(self.config, parameters, 1e-4)
       
        #self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=8000, eta_min=1e-7)
        #error self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60100, 75000, 110000, 16000, 185000], gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[120000, 300000, 480000], gamma=0.5)
 
        self.scheduler1 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, milestones=[600000], gamma=0.5)         # [180000, 240000, 420000] from step 152630 
   
        self.scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer2, milestones=[60000, 120000, 180000], gamma=0.5)
        #[120, 240, 480, 600], gamma=0.5)  #
        #self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=2e6, eta_min=1e-6)
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
       

    '''def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        #self.join_model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.join_model.diffusion_net.load_state_dict(checkpoint['state_dict_noise'], strict=True)
        #self.join_model.diffusion_net.load_state_dict(checkpoint['state_dict'], strict=True)
        self.join_model.color_net.load_state_dict(checkpoint['state_dict_color'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer1.load_state_dict(checkpoint['optimizer1'])
        self.optimizer2.load_state_dict(checkpoint['optimizer2'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])

        if ema:
            self.ema_helper.ema(self.join_model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))'''

    def load_ddm_ckpt(self, load_path, ema=False):
        
        checkpoint1 = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint1['epoch']
        self.step = checkpoint1['step']


        if self.start_epoch -1 < self.join_model.stage_two:
            self.join_model.diffusion_net.load_state_dict(checkpoint1['state_dict'], strict=True)
            self.optimizer1.load_state_dict(checkpoint1['optimizer'])

            if self.start_epoch -1 > self.join_model.stage_one:
                checkpoint2 = utils.logging.load_checkpoint(load_path.replace('noise', 'color'), None)
                self.join_model.color_net.load_state_dict(checkpoint2['state_dict'], strict=True)
                self.optimizer2.load_state_dict(checkpoint2['optimizer'])
        #self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.join_model.load_state_dict(checkpoint1['state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint1['optimizer'])
            #self.ema_helper.load_state_dict(checkpoint1['ema_helper'])

    def multi_train(self, DATASET):

        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)
            print('>>>>>load ckpts<<<<<')

        df = pd.DataFrame(columns=['time', 'epoch', 'step','noise loss','pixel loss1','pixel loss2'])
        df.to_csv("/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/loss.csv",mode='a', index=False) 

        #last_params = [param.clone() for param in self.join_model.color_net.parameters()]
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            ### Stage One
            #python3 train_diffusion.py --config /home/eileen/Diffusion/WeatherDiffusion/configs/lol_origin.yml --resume /home/eileen/Diffusion/ckpts/NAFNet_color/LOL_Origin2250noise_ddpm.pth.tar 
            if epoch <= self.join_model.stage_one:
                
                for i, (x, reference, y) in enumerate(train_loader):

                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                    reference = reference.flatten(start_dim=0, end_dim=1) if reference.ndim == 5 else reference
                    n = x.size(0)    
                    self.join_model.diffusion_net.train()
                    self.join_model.color_net.eval()                
                    self.step += 1

                    x = x.to(self.device)
                    x = data_transform(x)
                    reference = reference.to(self.device)
                    reference = data_transform(reference)
                    e = torch.randn_like(x[:, 3:, :, :])
                    b = self.betas
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)#ba
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                    noise_loss, pixel_loss1, _, _ = self.join_model(x, reference, t, e, b, epoch)
                    loss = noise_loss + pixel_loss1*0.01  ##when epoch =1250, pixel_loss plus 0.01 #epoch 6750 0.1
                    self.optimizer1.zero_grad()
        
                    loss.backward()
                    self.optimizer1.step()
                    self.scheduler1.step()

                    if self.step % 10 == 0:
                        print(f"step: {self.step}, Lnoise: {noise_loss.item()}, Lpixel1: {pixel_loss1.item()}")
                    time_now = "%s"%datetime.now()
                    step = "Step[%d]"%self.step
                    noise_loss = "%f"%noise_loss
                    pixel_loss1 = "%f"%pixel_loss1
                    pixel_loss2 = -1
                    list = [time_now, epoch, step, noise_loss, pixel_loss1, pixel_loss2]
                    data = pd.DataFrame([list])
                    data.to_csv('/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/loss.csv',mode='a',header=False,index=False)
                    
                    if self.step % self.config.training.validation_freq == 0:
                            self.join_model.diffusion_net.eval()
                            self.sample_validation_patches(val_loader, self.step, epoch)

            elif epoch > self.join_model.stage_one and epoch <= self.join_model.stage_two:
                

                for i, (x, reference, y) in enumerate(train_loader):

                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                    reference = reference.flatten(start_dim=0, end_dim=1) if reference.ndim == 5 else reference
                    n = x.size(0)  
                    self.join_model.diffusion_net.eval()
                    self.join_model.color_net.train()                  
                    self.step += 1

                    x = x.to(self.device)
                    x = data_transform(x)
                    reference = reference.to(self.device)
                    reference = data_transform(reference)
                    e = torch.randn_like(x[:, 3:, :, :])
                    b = self.betas
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)#ba
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                    _, _, pixel_loss2, _ = self.join_model(x, reference, t, e, b, epoch)
                    loss = pixel_loss2
                    self.optimizer2.zero_grad()
        
                    loss.backward()
                    self.optimizer2.step()
                    self.scheduler2.step()

                    if self.step % 10 == 0:
                        print(f"step: {self.step}, Lpixel2: {pixel_loss2.item()}")
                    
                    time_now = "%s"%datetime.now()
                    step = "Step[%d]"%self.step
                    noise_loss = -1
                    pixel_loss1 = -1
                    pixel_loss2 = "%f"%pixel_loss2
                    list = [time_now, epoch, step, noise_loss, pixel_loss1, pixel_loss2]
                    data = pd.DataFrame([list])
                    data.to_csv('/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/loss.csv',mode='a',header=False,index=False)
                    
                    if self.step % self.config.training.validation_freq == 0:
                            self.join_model.eval()
                            self.sample_validation_patches(val_loader, self.step, epoch)

            else:
                

                for i, (x, reference, y) in enumerate(train_loader):

                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                    reference = reference.flatten(start_dim=0, end_dim=1) if reference.ndim == 5 else reference
                    n = x.size(0)     
                    self.join_model.diffusion_net.train()
                    self.join_model.color_net.train()               
                    self.step += 1

                    x = x.to(self.device)
                    x = data_transform(x)
                    reference = reference.to(self.device)
                    reference = data_transform(reference)
                    e = torch.randn_like(x[:, 3:, :, :])
                    b = self.betas
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)#ba
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                    noise_loss, pixel_loss1, pixel_loss2, _ = self.join_model(x, reference, t, e, b, epoch)
                    loss = noise_loss + pixel_loss1 + 0.5*pixel_loss2
                    self.optimizer.zero_grad()
        
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    if self.step % 10 == 0:
                        print(f"step: {self.step}, Lnoise: {noise_loss.item()}, Lpixel1: {pixel_loss1.item()}, Lpixel2: {pixel_loss2.item()}")
                    time_now = "%s"%datetime.now()
                    step = "Step[%d]"%self.step
                    noise_loss = "%f"%noise_loss
                    pixel_loss1 = "%f"%pixel_loss1
                    pixel_loss2 = "%f"%pixel_loss2
                    list = [time_now, epoch, step, noise_loss, pixel_loss1, pixel_loss2]
                    data = pd.DataFrame([list])
                    data.to_csv('/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/loss.csv',mode='a',header=False,index=False)
                    
                    if self.step % self.config.training.validation_freq == 0:
                            self.join_model.eval()
                            self.sample_validation_patches(val_loader, self.step, epoch)
        
            
            self.ema_helper.update(self.join_model)

            '''current_params = [param.clone() for param in self.join_model.color_net.parameters()]
            is_updated = any((param != last_param).any() for param, last_param in zip(current_params, last_params))

            if is_updated:
                print("网络进行了一次参数更新")
            last_params = current_params'''
            
            '''if epoch % 250 == 0:
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    #'state_dict': self.join_model.state_dict(),
                    'state_dict_noise': self.join_model.diffusion_net.state_dict(),
                    'state_dict_color': self.join_model.color_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'optimizer1': self.optimizer1.state_dict(),
                    'optimizer2': self.optimizer2.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join(self.config.data.data_dir, 'ckpts', 'NAFNet_color', self.config.data.dataset + str(epoch)+ '_ddpm'))'''
            if epoch  %25 == 0:
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    #'state_dict': self.join_model.state_dict(),
                    'state_dict': self.join_model.diffusion_net.state_dict(),
                    'optimizer': self.optimizer1.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join(self.config.data.data_dir, 'ckpts', 'NAFNet_color', self.config.data.dataset + str(epoch)+ 'noise_ddpm'))
                if epoch > self.join_model.stage_one and epoch<= self.join_model.stage_two:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        #'state_dict': self.join_model.state_dict(),
                        'state_dict': self.join_model.color_net.state_dict(),
                        'optimizer': self.optimizer2.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', 'NAFNet_color', self.config.data.dataset + str(epoch)+ 'color_ddpm'))
                if epoch > self.join_model.stage_two:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        #'state_dict': self.join_model.state_dict(),
                        'state_dict': self.join_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', 'NAFNet_color', self.config.data.dataset + str(epoch)+ 'join_ddpm'))
                print('>>>>>>Saving models')
            print('$$$$optimizer Learning rate:', epoch, ',', self.optimizer.param_groups[0]['lr'])
            print('$$$$optimizer1 Learning rate:', epoch, ',', self.optimizer1.param_groups[0]['lr'])
            print('$$$$optimizer2 Learning rate:', epoch, ',', self.optimizer2.param_groups[0]['lr'])


    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)
            print('>>>>>load ckpts<<<<<')

        df = pd.DataFrame(columns=['time', 'epoch', 'step','noise loss','pixel loss1','pixel loss2','color_loss'])
        df.to_csv("/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/loss.csv",mode='a', index=False) 

        
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, reference, y) in enumerate(train_loader):
                #print(x.shape)

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                reference = reference.flatten(start_dim=0, end_dim=1) if reference.ndim == 5 else reference
                n = x.size(0)
                data_time += time.time() - data_start
                #self.model.train()
                self.join_model.train()
                
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                reference = reference.to(self.device)
                reference = data_transform(reference)
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)#ba
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                '''noise_loss, pixel_loss1, x0_pred = noise_estimation_loss(self.model,  x, reference, t, e, b)
                
                loss1 = noise_loss + pixel_loss1 

                self.optimizer1.zero_grad()
        
                loss1.backward()
                self.optimizer1.step()
                self.scheduler1.step()'''

                noise_loss, pixel_loss1, pixel_loss2, _ = self.join_model(x, reference, t, e, b)


                ##make a conflic here when add pixel_loss1 and pixel_loss2 loss1 = noise_loss + pixel_loss1 
                loss1 = noise_loss + pixel_loss1
                loss2 = pixel_loss2 
                if epoch <= self.join_model.stage_one:
                    loss = loss1
                elif epoch > self.join_model.stage_one and epoch <= self.join_model.stage_two:
                    loss = loss2
                else:
                    loss = loss1 + 0.5*loss2

                self.optimizer.zero_grad()
        
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                '''if (epoch >= 0 and epoch < 1500):
                    loss = noise_loss + 0.1 * pixel_loss
                elif (epoch >= 1500 and epoch <3000):
                    loss = noise_loss + 1 * pixel_loss
                elif (epoch >= 3000 and epoch <4500):
                    loss = noise_loss + 10 * pixel_loss
                elif (epoch >= 4500):
                    loss = noise_loss + 10 * pixel_loss'''

                if self.step % 10 == 0:
                    #print('>>noise_loss ', noise_loss.item(), '  <<pixel_loss1 ', pixel_loss1.item(), '  <<pixel_loss2 ', pixel_loss2.item(),  '<<color_loss ', color_loss.item())
                    print(f"step: {self.step}, Lpixel1: {pixel_loss1.item()}, Lnoise: {noise_loss.item()}, Lpixel2: {pixel_loss2.item()}, Lcolor: {color_loss.item()}")

                time_now = "%s"%datetime.now()
                step = "Step[%d]"%self.step
                noise_loss = "%f"%noise_loss
                pixel_loss1 = "%f"%pixel_loss1
                pixel_loss2 = "%f"%pixel_loss2
                color_loss = "%f"%color_loss
                list = [time_now, epoch, step, noise_loss, pixel_loss1, pixel_loss1, color_loss]
                data = pd.DataFrame([list])
                data.to_csv('/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/loss.csv',mode='a',header=False,index=False)

                
                #if self.optimizer.param_groups[0]['lr'] != pred_lr:
                
                self.ema_helper.update(self.join_model)
                data_start = time.time()
                
                if self.step % self.config.training.validation_freq == 0:
                    self.join_model.eval()
                    self.sample_validation_patches(val_loader, self.step, epoch)

                #if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                if self.step % 10000== 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.join_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', 'NAFNet_color', self.config.data.dataset + str(epoch)+ '_ddpm'))
            print('$$$$Learning rate:', self.optimizer.param_groups[0]['lr'])

    def color_restoration(self,  x, x0):
        output = self.join_model.color_net(x, x0)
        return output

    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):

        #skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        skip = 50
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        if patch_locs is not None:
            print('seq:', seq)
            print('>>>>overlapping')
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.join_model.diffusion_net,  self.betas, eta=0.,
                                                             corners=patch_locs, p_size=patch_size)
            #xs = utils.dpm_solver(x, x_cond, seq, self.model, self.betas, eta=0.,
                                    #corners=patch_locs, p_size=patch_size)
            #xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        else:
            #print('****none')
            
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.join_model.diffusion_net,  self.betas, eta=0.)
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
            for i, (x, reference, y) in enumerate(val_loader):

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                reference = reference.flatten(start_dim=0, end_dim=1) if reference.ndim == 5 else reference
                break
            
            x0 = x.to('cuda')
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            reference = reference.to('cuda')
            #LOL_Map x_cond = x[:, :9, :, :].to(self.device)
            #x_cond = x[:, :4, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)

            before_color = x.to('cuda')
            ### color restoration
            x = x.to('cuda')
            
            x = self.color_restoration(x, x0)
            ###
            before_color = inverse_data_transform(before_color)
            x = inverse_data_transform(x)
            #print(x)
            x_cond = inverse_data_transform(x_cond)
            
            now_loss1, now_loss2 = 0, 0
            #dff = pd.DataFrame(columns=['time', 'epoch', 'step','loss1', 'loss2'])
            #dff.to_csv("/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/eval_loss.csv",mode='a', index=False) 
            for i in range(n):
                x_cond = x_cond[:, :3, :, :]
                #print('x cond', x[i].shape)
                #print('refer', reference[i].shape)
                now_loss1  += (before_color[i] - reference[i]).square().sum(dim=(0, 1, 2)).mean(dim=0)
                now_loss2  += (x[i] - reference[i]).square().sum(dim=(0, 1, 2)).mean(dim=0)
    
                #utils.logging.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_input.png"))
                utils.logging.save_image(before_color[i], os.path.join(image_folder, str(step), f"{i}before_color.png"))
                utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{i}output.png"))
                utils.logging.save_image(reference[i], os.path.join(image_folder, str(step), f"{i}refer.png"))
            now_loss1 /= n
            now_loss2 /= n
            time_now = "%s"%datetime.now()
            step = "Step[%d]"%self.step
            loss1 = "%f"%now_loss1
            loss2 = "%f"%now_loss2
            list = [time_now, epoch, step, loss1, loss2]
            data = pd.DataFrame([list])
            data.to_csv('/home/eileen/Diffusion/WeatherDiffusion/results/images/LOL_Origin/eval_loss.csv',mode='a',header=False,index=False)
