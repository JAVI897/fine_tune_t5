#!/bin/sh
export WANDB_PROJECT=instruct_dial-t0
export WANDB_DISABLED=true
deepspeed ./fine_tune_t5_small.py \
    --model_name_or_path ./models/t5-small \
    --do_train \
    --do_eval \
    --train_file examples_spanish.json \
    --validation_file examples_spanish_test.json \
    --text_column prompt \
    --target_column output \
    --output_dir ./tmpsmall \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps 36\
    --gradient_checkpointing \
    --learning_rate 5e-05 \
    --deepspeed tsmall-config.json \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 10\
    --evaluation_strategy steps\
    --num_train_epochs 2\
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 200\
    --eval_steps 200\
    --logging_steps 25\ 