from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv
import os
import torch.nn.functional as F

class SegmentedPETCTDataset(Dataset):
    def __init__(self, ct_paths, pet_paths, image_size=(256, 256), ct_max_pixel = 255.0, pet_max_pixel = 255.0, flip=False):
        self.image_size = image_size
        self.ct_paths = ct_paths
        self.pet_paths = pet_paths
        self._length = len(ct_paths)
        self.ct_max_pixel = float(ct_max_pixel)
        self.pet_max_pixel = float(pet_max_pixel)
        self.flip = flip
        
    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        ct_path = self.ct_paths[index]
        pet_path = self.pet_paths[index]
        
        ct_image, segmented_map = None, None
        
        np_ct_image = np.load(ct_path, allow_pickle=True)
        np_ct_image = np_ct_image / float(self.ct_max_pixel)

        ct_image = Image.fromarray(np_ct_image) 
        ct_image = transform(ct_image) 

        np_pet_image = np.load(pet_path, allow_pickle=True)
        np_pet_image = np_pet_image / float(self.pet_max_pixel)
            
        np_segmented_map = self.extract_segmented_map(np_pet_image)
        
        if p:
            np_segmented_map = np.fliplr(np_segmented_map)
            
        segmented_map = torch.from_numpy(np_segmented_map.copy()).unsqueeze(0)
        # segmented_map = torch.from_numpy(np_segmented_map.copy())
        
        image_name = Path(ct_path).stem
        
        return ct_image, segmented_map, image_name
    
    def extract_segmented_map(self, np_img):
        STATIC_THRESH_HOLD = 100
        
        clone_img = np_img.copy() 
        clone_img = (clone_img * 255.).astype(np.uint8)
        
        _, thresh = cv.threshold(clone_img, STATIC_THRESH_HOLD, 255, 0)
        thresh[thresh > 0] = 1
        
        return thresh
    
    # def extract_segmented_map(self, np_img, np_latent):
    #     STATIC_THRESH_HOLD_1 = 50
    #     STATIC_THRESH_HOLD_2 = 100
        
    #     clone_img_1 = np_img.copy() 
    #     clone_img_1 = (clone_img_1 * 255.).astype(np.uint8)
        
    #     _, thresh_1 = cv.threshold(clone_img_1, STATIC_THRESH_HOLD_1, 255, 0)
    #     thresh_1 = cv.resize(thresh_1, (np_latent.shape[0], np_latent.shape[1]))
    #     thresh_1[thresh_1 > 0] = 1
        
    #     clone_img_2 = np_img.copy() 
    #     clone_img_2 = (clone_img_2 * 255.).astype(np.uint8)
        
    #     _, thresh_2 = cv.threshold(clone_img_2, STATIC_THRESH_HOLD_2, 255, 0)
    #     thresh_2 = cv.resize(thresh_2, (np_latent.shape[0], np_latent.shape[1]))
    #     thresh_2[thresh_2 > 0] = 1
        
    #     return thresh_1 + thresh_2
    
class ConditionalSegmentedPETCTDataset(Dataset):
    def __init__(self, ct_paths, pet_paths, labels_path, num_labels, image_size=(256, 256), ct_max_pixel = 255.0, pet_max_pixel = 255.0, flip=False):
        self.image_size = image_size
        self.ct_paths = ct_paths
        self.pet_paths = pet_paths
        self._length = len(ct_paths)
        self.labels_path = labels_path
        self.num_labels = num_labels
        self.ct_max_pixel = float(ct_max_pixel)
        self.pet_max_pixel = float(pet_max_pixel)
        self.flip = flip
        
    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        ct_path = self.ct_paths[index]
        pet_path = self.pet_paths[index]
        
        ct_image, segmented_map, label_emb, label = None, None, None, None
        
        image_name = Path(ct_path).stem
        
        np_ct_image = np.load(ct_path, allow_pickle=True)
        np_ct_image = np_ct_image / float(self.ct_max_pixel)

        ct_image = Image.fromarray(np_ct_image) 
        ct_image = transform(ct_image) 
        
        label_emb, label = self.get_label(ct_image, image_name)

        np_pet_image = np.load(pet_path, allow_pickle=True)
        np_pet_image = np_pet_image / float(self.pet_max_pixel)
            
        np_segmented_map = self.extract_segmented_map(np_pet_image)
        
        if p:
            np_segmented_map = np.fliplr(np_segmented_map)
            
        segmented_map = torch.from_numpy(np_segmented_map.copy()).unsqueeze(0)
        # segmented_map = torch.from_numpy(np_segmented_map.copy())
        
        return ct_image, segmented_map, label_emb, label, image_name
    
    def get_label(self, img, img_name):
        with open(os.path.join(self.labels_path, f'{img_name}.txt'), 'r') as f:
            label = int(f.readline().strip())

        label_emb = np.eye(self.num_labels)[label]
        label_emb = np.expand_dims(label_emb, axis=(0, 1))
        label_emb = np.tile(label_emb, (img.shape[1], img.shape[2], 1)) 
        label_emb = label_emb.transpose((2, 0, 1))
        label_emb = torch.from_numpy(label_emb).float() 
   
        return label_emb, label
    
    def extract_segmented_map(self, np_img):
        STATIC_THRESH_HOLD = 100
        
        clone_img = np_img.copy() 
        clone_img = (clone_img * 255.).astype(np.uint8)
        
        _, thresh = cv.threshold(clone_img, STATIC_THRESH_HOLD, 255, 0)
        thresh[thresh > 0] = 1
        
        return thresh  