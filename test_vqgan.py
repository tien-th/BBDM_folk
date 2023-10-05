from model.VQGAN.vqgan import VQModel 
import yaml
import argparse 
import omegaconf.dictconfig
import os
import torch 
import numpy as np  

if __name__ == '__main__':

    f = open('/mnt/disk1/mbbank/tien/BBDM_folk/configs/test_vqgan.yaml', 'r')

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    i = 0
    file_names = []
    for npy_file in os.listdir('/mnt/disk3/tiennh/data4vq/test'):
        i += 1
        if i == 1000: 
            break
        file_names.append(npy_file)

    from tqdm import tqdm
    def cal_mae(weight_path, test_dir_path, file_names): 
        f = open('/mnt/disk1/mbbank/tien/BBDM_folk/configs/test_vqgan.yaml', 'r')
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
        nconfig = dict2namespace(dict_config) 
        nconfig.model.VQGAN.params.ckpt_path = weight_path 
        vq = VQModel(**vars(nconfig.model.VQGAN.params))
        
        # load npy file in test_dir_path 
        maes = []

        for npy_file in tqdm(file_names):
        
            x_np = np.load(os.path.join(test_dir_path, npy_file), allow_pickle=True)
            x = torch.from_numpy(x_np)
            x = x.unsqueeze(0)
            input = x 
            quant, diff, _ = vq.encode(input)
            dec = vq.decode(quant)
            # caculate MAE between input and dec
            mae = torch.abs(dec - x ).mean()
            maes.append(mae.item())
            # print(f'File: {npy_file}, MAE: {torch.mean(diff).item()}')
        print(f'MAE: {np.mean(maes)}')
        print("len :", len(maes))
        
    cal_mae('/mnt/disk3/tiennh/taming-transformers/last_82.ckpt', '/mnt/disk3/tiennh/data4vq/test', file_names)
    print("=====================================================")
    cal_mae('/mnt/disk3/tiennh/taming-transformers/last_90.ckpt', '/mnt/disk3/tiennh/data4vq/test', file_names)
    print("======================================================")
    cal_mae('/mnt/disk3/tiennh/taming-transformers/last_55.ckpt', '/mnt/disk3/tiennh/data4vq/test', file_names)