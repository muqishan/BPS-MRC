# Train 
CUDA_VISIBLE_DEVICES=1  python3 mrc_train.py --model_type electra --model_name_or_path BioM-ELECTRA-Large-SQuAD2-BioASQ8B \
--train_file SQuAD/20241031/checkData/Train.json \
--predict_file SQuAD/20241031/checkData/Dev.json \
--do_lower_case \
--do_train \
--do_eval \
--threads 20 \
--version_2_with_negative \
--num_train_epochs 10 \
--learning_rate 5e-6 \
--weight_decay 5e-5 \
--adam_epsilon 1e-8 \
--max_seq_length 512 \
--doc_stride 128 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8   \
--logging_steps 50 \
--save_steps 500 \
--max_grad_norm 0.5 \
--overwrite_output_dir \
--output_dir mrc_runs/mrc20250523_large_4 \
--overwrite_cache


# eval
CUDA_VISIBLE_DEVICES=0  python3 mrc_eval.py --model_type electra --model_name_or_path mrc_runs/mrc20241031_10_ \
--predict_file SQuAD/20241031/checkData/Test.json \
--do_lower_case \
--threads 20 \
--version_2_with_negative \
--max_seq_length 512 \
--doc_stride 128 \
--per_gpu_eval_batch_size 8 \
--output_dir mrc_runs/mrc20241031_10_/evla 

# 18base 10_