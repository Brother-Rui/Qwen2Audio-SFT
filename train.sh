process_port=29503
experiment_name=Qwen2-Audio-7B_train
model_dir=/path/to/Qwen2-Audio-7B
train_data_path=/path/to/data.json
#dev_data_file=/path/to/Apollo2-7B_Prodata/dev.json
output_dir=/path/to/ckpts
log_folder=/path/to/logs/${experiment_name}
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log

accelerate launch \
   --config_file /path/to/src/sft/training_config/zero.yaml \
   --num_processes 8 \
   --num_machines 1 \
   --main_process_port ${process_port} \
   --num_cpu_threads_per_process 2 \
   --deepspeed_multinode_launcher standard /path/to/src/sft/train_qwen2audio_resume_val.py \
   --model_path ${model_dir} \
   --experiment_name ${experiment_name} \
   --gradient_accumulation_steps 2 \
   --train_data_path ${train_data_path} \
   --output_dir ${output_dir} \
   --log_dir ./wandb_logs \
   --n_epochs 1 \
   --train_bsz_per_gpu 1 \
   --learning_rate 1e-5 \
   --eval_step -1 \
   --save_step -1 \
   --warmup_rates 0.03 \
   --max_ckpts 3 \
   --gradient_checkpointing  > ${log_folder}/$log_name 2>&1 &
   # --checkpoint_path path 恢复训练
