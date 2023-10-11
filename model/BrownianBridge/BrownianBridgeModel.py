import pdb
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
import random

from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel, ConfidenceNetwork
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler


class BrownianBridgeModel(nn.Module):
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
        # self.conf_net = ConfidenceNetwork(**vars(model_params.ConfNetParams)) 

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

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

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        # self.conf_net.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()
        # return itertools.chain(self.denoise_fn.parameters(), self.conf_net.parameters())

    def forward(self, x, y, t, uncer_map=None, context=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.full((b,), t, device=device, dtype=torch.long)
        return self.p_losses(x, y, context, t, uncer_map)

    def p_losses(self, x0, y, context, t, uncer_map=None, noise=None):
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

        x_t, objective = self.q_sample(x0, y, t, noise)    
        
        if uncer_map == None:
            uncer_map = torch.zeros(x_t.shape, device=x_t.device) 
        
        x_t_hat = torch.cat([x_t, uncer_map], 1)
        objective_recon, conf = self.denoise_fn(x_t_hat, timesteps=t, context=context)

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        # conf = self.conf_net(torch.cat([x_t_hat, x0_recon], 1))

        x0_eff = conf * x0_recon + (1 - conf) * x0

        n_uncer_map = conf * x0_recon

        # reconstruction loss
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
            conf_loss1 = (x0 - x0_eff).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
            conf_loss1 = F.mse_loss(x0, x0_eff)
        else:
            raise NotImplementedError()
        
        # confidence loss
        lambda1 = 0.5
        lambda2 = 0.8
        sng = 1e-9
        
        conf_loss2 = -(1.0 / (h * w)) * torch.sum(torch.log(conf + sng))
        
        with torch.no_grad():
            if conf_loss2 < 0.25:
                lambda2 = 0.09 * lambda2 * (np.exp(5.4 * conf_loss2.cpu().item()) - 0.98)

        tot_loss = recloss + lambda1 * conf_loss1 + lambda2 * conf_loss2

        log_dict = {
            "rec_loss": recloss,
            "conf_loss1": conf_loss1,
            "conf_loss2": conf_loss2,
            "loss": tot_loss,
            "x0_recon": x0_recon
        }
        return tot_loss, n_uncer_map, log_dict

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, uncer_map, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            x_t_hat = torch.cat([x_t, uncer_map], 1) 
            objective_recon, conf = self.denoise_fn(x_t_hat, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            # conf = self.conf_net(torch.cat([x_t_hat, x0_recon], 1))
            n_uncer_map = conf * x0_recon

            return x0_recon, x0_recon, n_uncer_map
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            x_t_hat = torch.cat([x_t, uncer_map], 1) 
            objective_recon, conf = self.denoise_fn(x_t_hat, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            # conf = self.conf_net(torch.cat([x_t_hat, x0_recon], 1))
            n_uncer_map = conf * x0_recon

            return x_tminus_mean + sigma_t * noise, x0_recon, n_uncer_map

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        b, c, h, w = y.shape
        uncer_map = torch.zeros(y.shape, device=y.device)
        
        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon, n_uncer_map = self.p_sample(x_t=imgs[-1], y=y, uncer_map=uncer_map, context=context, i=i, clip_denoised=clip_denoised)
                uncer_map = n_uncer_map
                imgs.append(img)
                one_step_imgs.append(x0_recon)

            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon, n_uncer_map = self.p_sample(x_t=img, y=y, uncer_map=uncer_map, context=context, i=i, clip_denoised=clip_denoised)
                uncer_map = n_uncer_map
                
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)