import numpy as np
import argparse
import omegaconf 
import os
from PIL import Image
import yaml
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import joblib

CKPT_PATH = '/home/PET-CT/thaind/kmeans/ckpts_kmeans_latent'
DATA_PATH = '/home/PET-CT/splited_data_15k/train/A'
CT_MAX = 2047
IMAGE_SIZE = 256

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir

def main():
    ct_paths = get_image_paths_from_dir(os.path.join(DATA_PATH))
    
    transform = transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    f = open('/home/PET-CT/thaind/BBDM_folk/configs/Template_CPDM.yaml', 'r')
    dict_config = yaml.load(f, Loader=yaml.FullLoader)

    nconfig = dict2namespace(dict_config)
    ltbbdm = LatentBrownianBridgeModel(nconfig.model).to('cuda:0')
    ltbbdm.eval()
    
    ct_images = []

    for ct_path in ct_paths:
        ct_image = np.load(ct_path, allow_pickle=True)
        ct_image = ct_image / float(CT_MAX)
        
        ct_image = Image.fromarray(ct_image) 
        ct_image = transform(ct_image) 
        ct_image = ct_image.unsqueeze(0).to('cuda:0')

        ct_image = ltbbdm.encode(ct_image)
        
        ct_image = ct_image.cpu().numpy()
        
        ct_images.append(ct_image.flatten())
    
    X = np.array(ct_images)
    
    for k in range(2, 11):  
        kmeans_model = KMeans(n_clusters=k, random_state=0, verbose=1)
        kmeans_model.fit(X)
        
        ckpt_path = make_dir(CKPT_PATH)
        joblib.dump(kmeans_model, os.path.join(ckpt_path, f'kmeans_model_k={k}.joblib'))
    
if __name__ == '__main__':
    main()