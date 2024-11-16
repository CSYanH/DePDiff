# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        #print(emb.shape)
        return emb


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, temb_dim = None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(temb_dim // 2, c * 4)
        ) if temb_dim else None

        dw_channel = c * DW_Expand
        #print('dw_channel ', dw_channel)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        ##ablation SSA 
        #donot use attention 
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2*2   , out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #self.conv3 = nn.Conv2d(in_channels=dw_channel // 2   , out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        ##no SSA self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.ssa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=0,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
    

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x_in):
        #print('********block', np.array(x_in).shape)
    
        x, t = x_in
      
        ###time embedding as cvprw's work
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(t, self.mlp)
        ###

        init = x
        x = self.norm1(x)
        ###cvprw add two parameters
        x = x * (scale_att + 1) + shift_att
        ###
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        #print('>>input x', x.shape)
        #dont use attention 
        x_channel = x * self.sca(x)
        #print('>>x_channel', x_channel.shape)
        
        ##ablation SSA 
        #don t use attention 
        maxpool_channel, _ = torch.max(x, dim=1, keepdim=True)
        #print('>>maxpool_channel', maxpool_channel.shape)

        ##ablation SSA 
        #do not use attention 
        x_spatial = x * self.ssa(maxpool_channel)
        #print('>>x_spatial', x_spatial.shape)
      

        ##ablation SSA 
        #don not use attention 
        x_merge = torch.cat((x_channel, x_spatial), dim=1)
        #print('>>x_merge', x_merge.shape) 

        ##ablation SSA  
        #do not use attention x_merge-> x 
        x = self.conv3(x_merge)
       # x = self.conv3(x)

        ##no SSA x = self.conv3(x_channel)

        x = self.dropout1(x)

        y = init + x * self.beta

        #x = self.conv4(self.norm2(y))
        ###here also use the two parameters 
        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
       
        ###
        x = self.sg(x)
        #x_channel2 = x * self.sca(x)

        x = y + x * self.gamma
        #print('x_out', x.shape)

        return x, t


class NAFNet(nn.Module):

    def __init__(self, config):
    
        super().__init__()
        self.config = config
        img_channel, out_channel = config.model.in_channels, config.model.out_ch
        width = config.model.width
        middle_blk_num = config.model.middle_blk_num
        enc_blk_nums, dec_blk_nums= config.model.enc_blk_nums, config.model.dec_blk_nums
        

        temb_dim = width * 4
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, temb_dim*2),
            SimpleGate(),
            nn.Linear(temb_dim, temb_dim)
        )

        # self.intro = nn.Conv2d(in_channels=img_channel*2 , out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)

        self.intro = nn.Conv2d(in_channels=img_channel*2 , out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        i=0
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, temb_dim) for _ in range(num)]
                )
            )
            
            if i % 2==0 :
                #print('i :', i, chan)
                self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
                chan = chan * 2
            else:
                chan = chan
                self.downs.append(nn.Conv2d(chan, chan, 1, 1))
            
            i = i + 1
            

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, temb_dim) for _ in range(middle_blk_num)]
            )

        j= 1
        for num in dec_blk_nums:
            if j % 2==0:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(chan, chan * 2, 1, bias=False),
                        nn.PixelShuffle(2)
                    )
                )
                chan = chan // 2
            else:
                self.ups.append(nn.Conv2d(chan, chan, 1, 1))
                chan = chan
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, temb_dim) for _ in range(num)]
                )
            )
            j = j +1

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, x, t):

        #print('######################forward') 
        #print('x:', x.shape)
        #print('t:', t.shape)
        t = self.time_mlp(t)
        #print('t_mlp:', t.shape)

        B, C, H, W = x.shape
        x = self.check_image_size(x)
        #print('check img size x:', x.shape)
        in_res = x

        x = self.intro(x)
        #print('intro x:', x.shape)

        encs = []

        #print('ZZZZZZZip', list(zip(self.encoders, self.downs)))
        i= 0
        for encoder, down in zip(self.encoders, self.downs):
            #print('input encoder x:', x.shape)
            x, _ = encoder([x, t])
            #print('output encoder x:', x.shape)
            encs.append(x)
            #print('enco', i, ' ', x.shape)
            if i%2 ==0:
                x = down(x)
            i = i+1
            #print('output x:', x.shape)

        #print('before mid, x nd t:', x.shape, ' ', t.shape)
        x, _ = self.middle_blks([x, t])
        #print('middle  x:', x.shape)

        j =1 
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            #print('dec', j, ' ', x.shape)
            if j%2 ==0:
                x = up(x)
            #print('up x:', x.shape)
            x = x + enc_skip
            #print('skip x:', x.shape)
            x, _ = decoder([x, t])
            j = j+1
            #print('output decoder x:', x.shape)

        x = self.ending(x)
        #print('ending x:', x.shape)
        ###cvprw's work delte this x = x + inp
        x = x[..., :H, :W]
        #print('return x:', x.shape)
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
