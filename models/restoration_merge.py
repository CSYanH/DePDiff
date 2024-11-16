import torch
import torch.nn as nn
import utils
import torchvision
import os


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestorationMerge:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestorationMerge, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='lowlight', r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                             
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                print(image_folder)
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}.png").replace("'", '').replace("[", '').replace(']', ''))

    def diffusive_restoration(self, x_cond, r=None):
   
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=64, r=16)
        
        corners = [(i, j) for i in h_list for j in w_list]
        noise_size = x_cond[:, :3, :, :]
        #print(noise_size.size())
        x = torch.randn(noise_size.size(), device=self.diffusion.device)
        x_output_2 = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=64)

        return x_output_2

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        if r is None:
            print("r is none")
        else:
            print("r is not none", r)
        r = 32 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
