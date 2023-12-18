import itertools
import pdb
import random
import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
import os
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.additional_condition_train_path = model_config.additional_condition_train_path
        self.additional_condition_val_path = model_config.additional_condition_val_path
        
        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_name, x_cond, stage, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
            # add_cond = self.get_additional_condition(x, x_cond_latent)
            add_cond = self.get_additional_condition(x_name, x_cond_latent, stage)
        context = self.get_cond_stage_context(x_cond)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), add_cond, context)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    def get_additional_condition(self, x_name, x_cond_latent, stage):    
        additional_condition_path = self.additional_condition_val_path
        
        if stage == 'train':
            additional_condition_path = self.additional_condition_train_path
        
        conditions = []
        
        for i in range(x_cond_latent.shape[0]):
            np_cond = np.load(os.path.join(additional_condition_path, f'{x_name[i]}.npy'), allow_pickle=True)
            tensor = torch.from_numpy(np_cond)
    
            conditions.append(tensor.unsqueeze(0).unsqueeze(0))
        
        return torch.cat(conditions, dim=0).to(x_cond_latent.device) 
            
    # def get_additional_condition(self, x, x_cond_latent):
    #     return self.get_segmented_condition(x, x_cond_latent)
    #     # return self.get_detected_condition(x, x_cond_latent)
    
    # def get_detected_condition(self, x, x_cond_latent):
    #     STATIC_THRESH_HOLD = 100
    #     MIN_AREA = 10
        
    #     def are_boxes_overlapping(box1, box2):
    #         return not (box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1])
        
    #     conditions = []
        
    #     for i in range(x.shape[0]):
    #         x_clone = x[i].clone()
    #         x_norm = x_clone.mul_(0.5).add_(0.5).clamp_(0, 1.).mul(255).permute(1, 2, 0).to('cpu').numpy()
            
    #         _, thresh = cv.threshold(x_norm, STATIC_THRESH_HOLD, 255, 0)
    #         thresh = thresh.astype(np.uint8)
            
    #         contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
    #         all_bounding_boxes = []

    #         for i in range(len(contours)):
    #             x1, y1, w1, h1 = cv.boundingRect(contours[i])
    #             all_bounding_boxes.append((x1, y1, x1 + w1, y1 + h1))

    #         sorted_boxes = sorted(all_bounding_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    #         filtered_boxes = [box for box in sorted_boxes if (box[2] - box[0]) * (box[3] - box[1]) >= MIN_AREA]
            
    #         tensor = torch.zeros(x_cond_latent.shape[2], x_cond_latent.shape[3])
            
    #         if len(filtered_boxes) > 0:
    #             non_overlapping_boxes = [filtered_boxes[0]]

    #             for current_box in filtered_boxes[1:]:
    #                 if all(not are_boxes_overlapping(existing_box, current_box) for existing_box in non_overlapping_boxes):
    #                     non_overlapping_boxes.append(current_box)
                
    #             for box in non_overlapping_boxes:
    #                 x1, y1, x2, y2 = box
    #                 x1 = int(1. * x1 / x.shape[3] * x_cond_latent.shape[3])
    #                 x2 = int(1. * x2 / x.shape[3] * x_cond_latent.shape[3])
    #                 y1 = int(1. * y1 / x.shape[2] * x_cond_latent.shape[2])
    #                 y2 = int(1. * y2 / x.shape[2] * x_cond_latent.shape[2])
    #                 tensor[y1:y2+1, x1:x2+1] = 1
    
    #         conditions.append(tensor.unsqueeze(0).unsqueeze(0))
            
    #     return torch.cat(conditions, dim=0).to(x_cond_latent.device) 

    # def get_segmented_condition(self, x, x_cond_latent):
    #     STATIC_THRESH_HOLD = 100
    #     conditions = []
        
    #     for i in range(x.shape[0]):
    #         x_clone = x[i].clone()
    #         x_norm = x_clone.mul_(0.5).add_(0.5).clamp_(0, 1.).mul(255).permute(1, 2, 0).to('cpu').numpy()
    #         _, thresh = cv.threshold(x_norm, STATIC_THRESH_HOLD, 255, 0)
            
    #         thresh = cv.resize(thresh, (x_cond_latent.shape[2], x_cond_latent.shape[3]))
    #         thresh[thresh > 0] = 1
  
    #         conditions.append(torch.from_numpy(thresh).unsqueeze(0).unsqueeze(0))
        
    #     return torch.cat(conditions, dim=0).to(x_cond_latent.device)
    
    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, x_name, stage, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, cond=True)
        # add_cond = self.get_additional_condition(x, x_cond_latent)
        add_cond = self.get_additional_condition(x_name, x_cond_latent, stage)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     add_cond=add_cond,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples, add_cond
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      add_cond=add_cond,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out, add_cond

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out