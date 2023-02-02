#!/bin/sh
export WANDB_PROJECT=instruct_dial-t0
export WANDB_DISABLED=true
deepspeed ./fine_tune.py \
    --model_name_or_path ./models/t0-8-bit-all_tasksv2-m1-t1 \
    --do_train \
    --do_eval \
    --deepspeed ds-config.json \
    --train_file examples_spanish.json \
    --validation_file examples_spanish_test.json \
    --text_column prompt \
    --target_column output \
    --output_dir ./tmp \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps 10\
    --gradient_checkpointing false\
    --learning_rate 5e-05 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 10\
    --evaluation_strategy steps\
    --num_train_epochs 2\
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 200\
    --eval_steps 200\
    --logging_steps 1\ 