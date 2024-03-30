import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch

# Define the function to compute MAE
def compute_mae(image1, image2):
    return np.abs(image1 - image2).mean()

def compute_mape(image1, image2):
    return np.abs((image1 - image2) / (image1 + 1e-6)).mean() * 100

loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda:0')

# Paths to the folders
gt_folder = "/home/PET-CT/splited_data_15k/test/B"
# pre_folder = "results/108_CT2PET_UncerBBDM3c/LBBDM-f4/sample_to_eval/200"
pre_folder = "/home/PET-CT/tiennh/thesis/BBDM_folk_reg1/results/BBDM+REG_x0_1_0.1_0.005_xavier/LBBDM-f4/sample_to_eval/200"
# Lists to store the computed metrics for each pair
ssim_scores = []
psnr_scores = []
mae_scores = []
mape_scores = []
lpips_scores = []

# Iterate through the files in the ground truth folder
for filename in os.listdir(gt_folder):
    # Make sure the file is a numpy array
    if filename.endswith(".npy"):
        # Construct the paths for the corresponding ground truth and predicted files
        try:
            gt_path = os.path.join(gt_folder, filename)
            pre_path = os.path.join(pre_folder, filename)
        
            # Load the images as numpy arrays
            gt_img = np.load(gt_path, allow_pickle=True)
            pre_img = np.load(pre_path, allow_pickle=True)
        except:
            continue   
        
        # Preprocess the predicted image
        pre_img = pre_img.mean(axis=-1) / 32767.0
        
        # Normalize the ground truth image
        gt_img = gt_img / 32767.0
        
        pre_img = pre_img.astype(np.float32)
        gt_img = gt_img.astype(np.float32)
        
        # Calculate the SSIM, PSNR, and MAE for this pair
        ssim_score = ssim(pre_img, gt_img, data_range=1)
        psnr_score = psnr(pre_img, gt_img, data_range=1)
        mae = compute_mae(pre_img, gt_img)
        mape = compute_mape(1-gt_img, 1-pre_img)

        pre_img = torch.from_numpy(pre_img)
        pre_img = (pre_img - 0.5) * 2
        pre_img = pre_img.unsqueeze(0)
        pre_img = pre_img.expand(3, -1, -1)
        pre_img = pre_img.unsqueeze(0).to('cuda:0')
        
        gt_img = torch.from_numpy(gt_img)
        gt_img = (gt_img - 0.5) * 2
        gt_img = gt_img.unsqueeze(0)
        gt_img = gt_img.expand(3, -1, -1)
        gt_img = gt_img.unsqueeze(0).to('cuda:0')

        lpips_score = loss_fn_vgg(pre_img, gt_img) 
        lpips_score = lpips_score.detach().cpu().numpy()[0, 0, 0, 0]
        
        # Append the scores to the corresponding lists
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        mae_scores.append(mae * 32767)
        mape_scores.append(mape)
        lpips_scores.append(lpips_score)

# Calculate the mean scores over all pairs
mean_ssim = np.mean(ssim_scores)
mean_psnr = np.mean(psnr_scores)
mean_mae = np.mean(mae_scores)
mean_mape = np.mean(mape_scores)
mean_lpips = np.mean(lpips_scores)

# Print the mean metrics
print("Mean SSIM: {}".format(mean_ssim))
print("Mean PSNR: {}".format(mean_psnr))
print("Mean MAE: {}".format(mean_mae))
print("Mean MAPE: {}".format(mean_mape))
print("Mean LPIPS: {}".format(mean_lpips))