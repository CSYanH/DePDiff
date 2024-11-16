import math
import torch
import torch.nn as nn
import cv2
import numpy

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm









def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    #print(timesteps)
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    #print('half', half_dim)
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    #print(emb.shape)
    return emb


def Gaussian(x, u, d):
    """
    参数:
    x -- 变量
    u -- 均值
    d -- 标准差

    返回:
    p -- 高斯分布值
    """
    ### 代码开始 ### (≈ 3~5 行代码)
    
    
    d_2 = d * d * 2
    
    zhishu = -(numpy.square(x - u) / d_2)
    
    exp = numpy.exp(zhishu)
    
    pi = numpy.pi
    
    xishu = 1 / (numpy.sqrt(2 * pi) * d)
   
    p = xishu * exp
  
    return p
    ### 代码结束 ###


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
    '''
    res = Gaussian(x.detach().cpu(), 0, 1)
    res = res.cuda()
    return x*res'''



def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def gradient(x):
    def sub_gradient(x):
            
        left_shift_x, right_shift_x, grad = torch.zeros_like(
            x), torch.zeros_like(x), torch.zeros_like(x)
        left_shift_x[:, :, 0:-1] = x[:, :, 1:]
        right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            
        grad = 0.5 * (left_shift_x - right_shift_x)
        return grad
    res1 = sub_gradient(x)
    res2 = sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)
        

    return res1, res2 


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                #x = self.maxpool(x)
                x = self.conv_shortcut(x)
            else:
                #x = self.maxpool(x)
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        #LOL_Map in_channels = config.model.in_channels * 4 if config.data.conditional else config.model.in_channels
        #in_channels = config.model.in_channels * 2+1 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                #print('>>>block', block_in)
                if curr_res in attn_resolutions:
                    #print('**********', curr_res, attn_resolutions)
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                #print("i_level", i_level)
                #print('i_block', i_block)
                #print('skip_in', skip_in)
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                #print('i_level', str(i_level))
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        #self.adjust = nn.Parameter(torch.ones(1))
        #self.adjust = nn.Parameter(torch.Tensor([1.02]), requires_grad=False)

    def forward(self, x, t):
        '''
        assert x.shape[2] == x.shape[3] == self.resolution'''

        #calculate canny
        #print(x[:, :3, :, :].shape)
        '''origin = x[:, :3, :, :]
        
        input_img = origin
        stru = origin
        noise = origin/
        for i in range(x.shape[0]):
            
            
            r = input_img[i, 0, :, :]
            g = input_img[i, 1, :, :]
            b = input_img[i, 2, :, :]
            gray = 0.068 * b + 0.691 * g + 0.241 * r
            gray = gray.squeeze(0)
            gray = gray.detach().cpu().numpy()
            gray = gray.astype(numpy.uint8)
            
            stru_numpy = cv2.Canny(gray, 10, 25)
            
            stru[i] = torch.tensor(stru_numpy).unsqueeze(0)
            #print('canny', stru[i].shape)


        #calculate noise_map
        for i in range(x.shape[0]):
            tmp = origin[i]
            
            tmp = torch.log(torch.clamp(tmp + 1e-3, min=1e-3))
            tmp = tmp.unsqueeze(0)
            tmp = tmp[:, 0:3].exp() 
            
            color_map = tmp / (tmp.sum(dim=1, keepdim=True) + 1e-4)
            dx, dy = gradient(color_map)
            noise[i] = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]

        
        x = torch.cat([x[:, :3, :, :], stru, noise, x[:, 3:, :, :]], dim=1)'''

        # timestep embedding
        #print('x:', x.shape)
        #print('t:', t.shape)
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        #print('t_emb:', temb.shape)

        # downsampling
        #print(x.shape)
        hs = [self.conv_in(x)]
        #print(self.conv_in(x).shape)
        #hs.shape torch.Size([1, 64, 128, 96, 96])
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                #print('down: ', h.shape)
                if len(self.down[i_level].attn) > 0:
                    #print('>>>>>h', h.shape)
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        tmp = hs.copy()

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                hh = h
                #print('i_level: ', i_level, ' i_block: ', i_block, ' shape', (torch.cat([hh, tmp.pop()], dim=1)).shape)
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        #h = h * self.adjust
        #print('unet return')
        return h
