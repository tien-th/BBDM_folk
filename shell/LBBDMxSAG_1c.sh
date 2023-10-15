# train
# python3 main.py --config configs/LBBDMxSAG_Vq13.yaml --train --sample_at_start --save_top --gpu_ids 2

#test
python3 main.py --config configs/LBBDMxSAG_Vq13.yaml --sample_to_eval --gpu_ids 3 --resume_model results/LBBDMxSAGxVq13_0.0/LBBDM-f4/checkpoint/top_model_epoch_200.pth --resume_optim results/LBBDMxSAGxVq13_0.0/LBBDM-f4/checkpoint/top__optim_sche_epoch_200.pth 

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