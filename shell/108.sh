#train
python3 main.py --config configs/LBBDM_7_17.yaml --train --sample_at_start --save_top --gpu_ids 3

#test
# python3 main.py --config configs/ct2pet_7_1.yaml --sample_to_eval --gpu_ids 3

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