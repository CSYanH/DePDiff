import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop
from models.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

#def generalized_steps(x, x_cond, seq, model, b, eta=0.):
def generalized_steps(x, x_cond, seq, model,  b,  eta=0.):
    
# betas = ....
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            #et = model(torch.cat([x_cond, xt], dim=1), t, merge, step) merge
            #et = model(torch.cat([x_cond, xt], dim=1), t,  x_cond[:, :3, :, :]) canny
            et = model(torch.cat([x_cond, xt], dim=1), t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def dpm_solver(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=b)

    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        model_kwargs=None,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")

    x_sample = dpm_solver.sample(
        x,
        x_cond,
        p_size,
        steps=10,
        order=2,
        skip_type="time_uniform",
        method="singlestep_fixed",
    )   
    return x_sample

def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        noise_size = x_cond[:, :3, :, :]
        #x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        x_grid_mask = torch.zeros_like(noise_size, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            #et_output = torch.zeros_like(x_cond, device=x.device)
            et_output = torch.zeros_like(noise_size, device=x.device)
            
            if manual_batching:
                #print("manual_batching")
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                #print(xt_patch.shape)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                #print('^^^^len', str(len(corners)))
                for i in range(0, len(corners), manual_batching_size):
                    #outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                              # xt_patch[i:i+manual_batching_size]], dim=1), t, x_cond_patch[i:i+manual_batching_size])
                    outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                               xt_patch[i:i+manual_batching_size]], dim=1), t)
                    for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):

                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch], dim=1), t)

            #et = et_output
            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

            xs.append(xt_next.to('cpu'))
    return xs, x0_preds
