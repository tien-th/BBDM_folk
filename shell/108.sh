# train
# python3 main.py --config configs/LBBDMxSAGxVq13.yaml --train --sample_at_start --save_top --gpu_ids 3 --resume_model results/LBBDMxSAGxVq13/LBBDM-f4/checkpoint/last_model.pth --resume_optim results/LBBDMxSAGxVq13/LBBDM-f4/checkpoint/last_optim_sche.pth
#test
# python3 main.py --config configs/LDM.yaml --train --sample_at_start --save_top --gpu_ids 1
# python3 main.py --config configs/LDM.yaml --sample_to_eval --gpu_ids 0 --resume_model /home/PET-CT/thaind/BBDM_folk/results/224_conditional_LDM_attenmap/LDM-f4/checkpoint/top_model_epoch_84.pth
python3 main.py --seed 345 --config configs/conditional_LBBDM.yaml --train --sample_at_start --save_top --gpu_ids 2
# python3 main.py --config configs/conditional_LBBDM.yaml --sample_to_eval --gpu_ids 2 --resume_model /home/PET-CT/thaind/BBDM_folk/results/229_conditional_LBBDM_atten/LBBDM-f4/checkpoint/top_model_epoch_50.pth
#python3 main.py --config configs/LBBDMxVq13.yaml --train --sample_at_start --save_top --gpu_ids 1 --resume_model results/LBBDMxVq13/LBBDM-f4/checkpoint/latest_model_28.pth --resume_optim results/LBBDMxVq13/LBBDM-f4/checkpoint/latest_optim_sche_28.pth
# OMP_NUM_THREADS=12 
# -m torch.distributed.run --nnodes=1 --nproc_per_node=8 
# --sample_to_eval
# --resume_model results/LBBDMxSAGxVq13/LBBDM-f4/checkpoint/top_model_epoch_192.pth --resume_optim results/LBBDMxSAGxVq13/LBBDM-f4/checkpoint/top__optim_sche_epoch_192.pth 

# python3 main.py --config configs/LBBDMxVq13.yaml --sample_to_eval --gpu_ids 0 --resume_model results/LBBDMxVq13_10k/LBBDM-f4/checkpoint/top_model_epoch_36.pth --resume_optim results/LBBDMxVq13_10k/LBBDM-f4/checkpoint/top_optim_sche_epoch_36.pth

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