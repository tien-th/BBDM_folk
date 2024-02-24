import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
import torch.utils.data


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class ConditionalDDPM(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        self.beta = torch.linspace(0.0001, 0.02, T)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

        # if self.mt_type == "linear":
        #     m_min, m_max = 0.001, 0.999
        #     m_t = np.linspace(m_min, m_max, T)
        # elif self.mt_type == "sin":
        #     m_t = 1.0075 ** np.linspace(0, T, T)
        #     m_t = m_t / m_t[-1]
        #     m_t[-1] = 0.999
        # else:
        #     raise NotImplementedError
        # m_tminus = np.append(0, m_t[:-1])

        # variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        # variance_tminus = np.append(0., variance_t[:-1])
        # variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        # posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        # to_torch = partial(torch.tensor, dtype=torch.float32)
        # self.register_buffer('m_t', to_torch(m_t))
        # self.register_buffer('m_tminus', to_torch(m_tminus))
        # self.register_buffer('variance_t', to_torch(variance_t))
        # self.register_buffer('variance_tminus', to_torch(variance_tminus))
        # self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        # self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

# TODO: skip sample DDIM ???
        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
                
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)
        
        self.steps = np.flip(self.steps.numpy(), axis=0)
        self.ddim_alpha = self.alpha_bar[self.steps.copy()].clone().to(torch.float32)
        
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_alpha_prev = torch.cat([self.alpha_bar[0:1], self.alpha_bar[self.steps[:-1].copy()]])
        self.ddim_sigma = (self.eta * 
                           ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                            (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
        self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5
        
        
    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()
    
# TODO
    def forward(self, x, y, add_cond, context=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # print(f'input image size: {h}x{w}') 
        # print(f'true image size: {img_size}x{img_size}')
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, add_cond, context, t)
    
# TODO
    def p_losses(self, x0, y, add_cond, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        # x_t, objective = self.q_sample(x0, y, t, noise)
        x_t, objective = self.q_sample(x0, t, noise)
        # print(f'x_t shape: {x0.shape}') 
        x_t_hat = torch.cat([x_t, y, add_cond], dim=1) 
        objective_recon = self.denoise_fn(x_t_hat, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        # x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
       
        log_dict = {
            "loss": recloss,
            # "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x0, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        # m_t = extract(self.m_t, t, x0.shape)
        # var_t = extract(self.variance_t, t, x0.shape)
        # sigma_t = torch.sqrt(var_t)
        alpha_bar = self.alpha_bar.clone().to(x0.device)
        mean = gather(alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(alpha_bar, t) 
        
        # TODO: doinggggggg 
        

        if self.objective == 'noise':
            objective = noise
        # elif self.objective == 'grad':
        #     # objective = m_t * (y - x0) + sigma_t * noise
        # elif self.objective == 'ysubx':
        #     objective = y - x0
        else:
            raise NotImplementedError()

        return (
            mean + (var ** 0.5) * noise,
            objective
        )
# TODO
    # def predict_x0_from_objective(self, x_t, y, t, objective_recon):
    #     if self.objective == 'noise': 
            
        
    #     # if self.objective == 'grad':
    #     #     x0_recon = x_t - objective_recon
    #     # elif self.objective == 'noise':
    #     #     m_t = extract(self.m_t, t, x_t.shape)
    #     #     var_t = extract(self.variance_t, t, x_t.shape)
    #     #     sigma_t = torch.sqrt(var_t)
    #     #     x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
    #     # elif self.objective == 'ysubx':
    #     #     x0_recon = y - objective_recon
    #     else:
    #         raise NotImplementedError
    #     return x0_recon
    
    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                                temperature: float,
                                repeat_noise: bool, t=None):
        alpha = self.ddim_alpha[index] 
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index] 
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t
        if sigma == 0.:
            noise = 0.
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else: 
            noise = torch.randn(x.shape, device=x.device)
        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0
    
# TODO
    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

# TODO
    @torch.no_grad()
    def p_sample(self, x_t, y, add_cond, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            x_t_hat = torch.cat([x_t, y, add_cond], dim=1) 
            objective_recon = self.denoise_fn(x_t_hat, timesteps=t, context=context)
            # x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            _, x0_recon = self.get_x_prev_and_pred_x0(objective_recon, i, x_t, temperature=1., repeat_noise=False)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            x_t_hat = torch.cat([x_t, y, add_cond], dim=1) 
            objective_recon = self.denoise_fn(x_t_hat, timesteps=t, context=context)
            x_prev, x0_recon = self.get_x_prev_and_pred_x0(objective_recon, i, x_t, temperature=1., repeat_noise=False)
            
            return x_prev, x0_recon 
    #         n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

    #         objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
    #         x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
    #         if clip_denoised:
    #             x0_recon.clamp_(-1., 1.)

    #         m_t = extract(self.m_t, t, x_t.shape)
    #         m_nt = extract(self.m_t, n_t, x_t.shape)
    #         var_t = extract(self.variance_t, t, x_t.shape)
    #         var_nt = extract(self.variance_t, n_t, x_t.shape)
    #         sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
    #         sigma_t = torch.sqrt(sigma2_t) * self.eta

    #         noise = torch.randn_like(x_t)
    #         x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
    #                         (x_t - (1. - m_t) * x0_recon - m_t * y)

    #         return x_tminus_mean + sigma_t * noise, x0_recon
    
    
# TODO
    @torch.no_grad()
    def p_sample_loop(self, y, add_cond, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
            
        shape = y.shape
        bs = shape[0]
        
        x = torch.randn(shape, device=y.device)
        
        time_steps = np.flip(self.steps)
        for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)): 
            index = len(time_steps) - i - 1
            x, x0_recon = self.p_sample(x, y, add_cond, context, index, clip_denoised=clip_denoised)

        return x
        # if sample_mid_step:
        #     imgs, one_step_imgs = [y], []
        #     for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
        #         img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
        #         imgs.append(img)
        #         one_step_imgs.append(x0_recon)
        #     return imgs, one_step_imgs
        # else:
        #     img = y
        #     for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
        #         img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
        #     return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)