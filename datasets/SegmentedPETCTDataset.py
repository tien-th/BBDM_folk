from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv

class SegmentedPETCTDataset(Dataset):
    def __init__(self, ct_paths, pet_paths, enc_dec, image_size=(256, 256), ct_max_pixel = 255.0, pet_max_pixel = 255.0, flip=False):
        self.image_size = image_size
        self.ct_paths = ct_paths
        self.pet_paths = pet_paths
        self._length = len(ct_paths)
        self.ct_max_pixel = float(ct_max_pixel)
        self.pet_max_pixel = float(pet_max_pixel)
        self.flip = flip
        self.enc_dec = enc_dec
        
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
        
        ct_latent, segmented_map = None, None
        
        try:
            np_ct_image = np.load(ct_path, allow_pickle=True)
            np_ct_image = np_ct_image / float(self.ct_max_pixel)

            ct_image = Image.fromarray(np_ct_image) 
            ct_image = transform(ct_image) 
            ct_image = ct_image.unsqueeze(0)

            ct_latent = self.enc_dec.encode(ct_image)
            ct_latent = (ct_latent / 4. + 0.5).clamp(0., 1.)
            
            np_ct_latent = ct_latent.squeeze().permute(1, 2, 0).cpu().numpy()
            np_ct_latent = (np_ct_latent * 255.).astype(np.uint8)
            
            ct_latent = torch.from_numpy(np_ct_latent).permute(2, 0, 1)

            np_pet_image = np.load(pet_path, allow_pickle=True)
            np_pet_image = np_pet_image / float(self.pet_max_pixel)
                
            np_segmented_map = self.extract_segmented_map(np_pet_image, np_ct_latent)
            
            if p:
                np_segmented_map = np.fliplr(np_segmented_map)
                
            segmented_map = torch.from_numpy(np_segmented_map.copy()).unsqueeze(0)
            
        except BaseException as e:
            print(ct_path)
            print(pet_path)
        
        image_name = Path(ct_path).stem
        
        return ct_latent, segmented_map, image_name
    
    def extract_segmented_map(self, np_img, np_latent):
        STATIC_THRESH_HOLD = 100
        
        clone_img = np_img.copy() 
        clone_img = (clone_img * 255.).astype(np.uint8)
        
        _, thresh = cv.threshold(clone_img, STATIC_THRESH_HOLD, 255, 0)
        thresh = cv.resize(thresh, (np_latent.shape[0], np_latent.shape[1]))
        thresh[thresh > 0] = 1
        
        return thresh
      