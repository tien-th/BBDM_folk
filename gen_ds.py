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

# ct_path = "/home/PET-CT/tiennh/test_code/ct" 
ct_path = "/home/PET-CT/splited_data_15k/train/A"

# pet_path = "/home/PET-CT/tiennh/test_code/ctB"
pet_path = "/home/PET-CT/splited_data_15k/train/B"

image_size = 256

label_fol = "/home/PET-CT/tiennh/test_code/train/labels"
img_fol = "/home/PET-CT/tiennh/test_code/train/images"

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
STATIC_THRESH_HOLD = 100

# Function to convert bounding boxes in xmin, ymin, xmax, ymax format to YOLO format.
def bbox2yolo(bbox):
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3],   
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]


def are_boxes_overlapping(box1, box2):
    return not (box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1])

def extract_bb(np_img):
    pet_img = np_img.copy() 
    pet_img = pet_img / 32767. * 255.
    pet_img = pet_img.astype(np.uint8)
    ret, thresh = cv.threshold(pet_img, STATIC_THRESH_HOLD, 255, 0)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    all_bounding_boxes = []

    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
        # cv.rectangle(all_bounding_boxes_img, (x, y), (x + w, y + h), (0, 0, 0), 1)
        all_bounding_boxes.append((x, y, x + w, y + h))
    
    

    sorted_boxes = sorted(all_bounding_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)

    min_area = 10  # Adjust this value as needed

    # Filter out boxes that are too small
    filtered_boxes = [box for box in sorted_boxes if (box[2] - box[0]) * (box[3] - box[1]) >= min_area]

    # Initialize the list of non-overlapping boxes with the largest one (assuming the largest box is not too small)
    non_overlapping_boxes = [bbox2yolo(filtered_boxes[0])]

    # Go through the rest of the boxes and add them if they do not overlap with the existing ones
    for current_box in filtered_boxes[1:]:
        if all(not are_boxes_overlapping(existing_box, current_box) for existing_box in non_overlapping_boxes):
            non_overlapping_boxes.append(bbox2yolo(current_box))
    
    return non_overlapping_boxes

for file in tqdm(os.listdir(pet_path)):
    if file.endswith(".npy"):
        try :
            pet = np.load(os.path.join(pet_path, file), allow_pickle=True)
            yolo_non_overlapping_boxes = extract_bb(pet) 
            # Normalize boxes by image size
            yolo_normalized_boxes = [(x / float(image_size), y / float(image_size), x2 / float(image_size), y2 / float(image_size)) for x, y, x2, y2 in yolo_non_overlapping_boxes]

            # Save normalized boxes to txt file
            for box in yolo_normalized_boxes:
                with open(os.path.join(label_fol, file[:-4] + ".txt"), "a") as f:
                    f.write("0 " + " ".join([str(a) for a in box]) + "\n")
            
            ct_img = np.load(os.path.join(ct_path, file), allow_pickle=True)
            x = ct_img / 2047.
            x = Image.fromarray(x)

            image = transform(x) 
            # image = (image - 0.5) * 2.
            image = image.unsqueeze(0)

            latent = ltbbdm.encode(image)
            latent = (latent / 4. + 0.5).clamp(0., 1.)
            latent_np = latent.squeeze().permute(1, 2, 0).cpu().numpy()
            latent_np = latent_np * 255.
            latent_np = latent_np.astype(np.uint8)

            img = Image.fromarray(latent_np)
            # Save numpy image
            img.save(os.path.join(img_fol, file[:-4] + ".png"))
    
        except: 
            continue

