from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
import yaml
import argparse
import omegaconf 
import torch
import torchvision.transforms as transforms
import os
from tqdm import tqdm

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ct_path = "/home/PET-CT/splited_data_15k/train/A"
pet_path = "/home/PET-CT/splited_data_15k/train/B"

image_size = 256

label_fol = "/home/PET-CT/thaind/segmentation_model/data/train/labels"
img_fol = "/home/PET-CT/thaind/segmentation_model/data/train/images"

make_dir(label_fol)
make_dir(img_fol)

f = open('/home/PET-CT/thaind/BBDM_folk/configs/conditional_LBBDM.yaml', 'r')
dict_config = yaml.load(f, Loader=yaml.FullLoader)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

nconfig = dict2namespace(dict_config)
ltbbdm = LatentBrownianBridgeModel(nconfig.model)

import numpy as np 
import os
import cv2 as cv
import matplotlib.pyplot as plt 
from PIL import Image 

def extract_segmented_map(np_img, latent_np):
    STATIC_THRESH_HOLD = 100
    
    pet_img = np_img.copy() 
    pet_img = pet_img / 32767. * 255.
    pet_img = pet_img.astype(np.uint8)
    ret, thresh = cv.threshold(pet_img, STATIC_THRESH_HOLD, 255, 0)

    thresh = cv.resize(thresh, (latent_np.shape[0], latent_np.shape[1]))
    thresh[thresh > 0] = 1
    thresh = np.expand_dims(thresh, axis=-1)
    
    return thresh

for file in tqdm(os.listdir(pet_path)):
    if file.endswith(".npy"):
        ct_img = np.load(os.path.join(ct_path, file), allow_pickle=True)
        x = ct_img / 2047.
        x = Image.fromarray(x)

        image = transform(x) 
        image = image.unsqueeze(0)

        latent = ltbbdm.encode(image)
        latent = (latent / 4. + 0.5).clamp(0., 1.)
        latent_np = latent.squeeze().permute(1, 2, 0).cpu().numpy()
        latent_np = latent_np * 255.
        latent_np = latent_np.astype(np.uint8)

        img = Image.fromarray(latent_np)
        # Save numpy image
        img.save(os.path.join(img_fol, file[:-4] + ".png"))   
        
        pet_img = np.load(os.path.join(pet_path, file), allow_pickle=True)
        segmented_map = extract_segmented_map(pet_img, latent_np) 
        
        np.save(os.path.join(label_fol, file[:-4] + ".npy"), segmented_map)

