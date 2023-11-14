# train
# python3 main.py --config configs/UncerBBDM.yaml --train --sample_at_start --save_top --gpu_ids 2
# python3 main.py --config configs/UncerBBDM3c.yaml --train --sample_at_start --save_top --gpu_ids 3
python3 main.py --config configs/UncerBBDM.yaml --sample_to_eval --gpu_ids 2 --resume_model results/UncerBBDM_1Unet_confloss_15k_v1/LBBDM-f4/checkpoint/top_model_epoch_88.pth  --resume_optim results/UncerBBDM_1Unet_confloss_15k_v1/LBBDM-f4/checkpoint/top_optim_sche_epoch_88.pth
# python3 main.py --config configs/UncerBBDM3c.yaml --sample_to_eval --gpu_ids 3 --resume_model results/108_CT2PET_UncerBBDM3c_/LBBDM-f4/checkpoint/top_model_epoch_126.pth  --resume_optim results/108_CT2PET_UncerBBDM3c_/LBBDM-f4/checkpoint/top_optim_sche_epoch_126.pth

#test
# python3 main.py --config configs/LBBDM_7_21.yaml --sample_to_eval --gpu_ids 2 --resume_model results/108_CT2PET_7_17/LBBDM-f4/checkpoint/last_model.pth --resume_optim results/108_CT2PET_7_17/LBBDM-f4/checkpoint/last_optim_sche.pth

#preprocess and evaluation
## rename
#python3 preprocess_and_evaluation.py -f rename_samples -r root/dir -s source/dir -t target/dir

## copy
#python3 preprocess_and_evaluation.py -f copy_samples -r root/dir -s source/dir -t target/dir

## LPIPS
#python3 preprocess_and_evaluation.py -f LPIPS -s source/dir -t target/dir -n 1

## max_min_LPIPS
#python3 preprocess_and_evaluation.py -f max_min_LPIPS -s source/dir -t target/dir -n 1

## diversity
#python3 preprocess_and_evaluation.py -f diversity -s source/dir -n 1

## fidelity
#fidelity --gpu 0 --fid --input1 path1 --input2 path2