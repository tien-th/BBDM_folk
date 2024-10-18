from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import torch

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, type, image_size=(256, 256), max_pixel=255.0, neighbors=4, flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.type = type 
        self.neighbors = neighbors
        self.max_pixel = float(max_pixel) if max_pixel else None
        self.flip = flip
        self.to_normal = to_normal     

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
        ])

        img_path = self.image_paths[index]
        image = None
        
        try:
            np_image = np.load(img_path, allow_pickle=True).copy()
            
            if self.type == 'ct':
                np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
            else: 
                np_image = np_image / float(self.max_pixel)
                
            np_image = (np_image - 0.5) * 2.  
            
            assert len(np_image.shape) == 3, "Expected 3D image (C, H, W)"

            if self.neighbors:
                np_image = np.pad(np_image, ((self.neighbors, self.neighbors), (0, 0), (0, 0)), mode='constant', constant_values=-1)

            image_slices = []
            for i in range(np_image.shape[0]):
                slice_img = np_image[i].copy() 
                slice_img = Image.fromarray(slice_img) 
                slice_img = transform(slice_img)  
                slice_img = torch.from_numpy(np.array(slice_img))
                image_slices.append(slice_img.squeeze(0))

            image = torch.stack(image_slices, dim=0)
            
        except BaseException as e:
            print(img_path, e)

        patient_name = Path(img_path).parent.name
        
        return image, patient_name

      