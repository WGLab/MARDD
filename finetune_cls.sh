#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"

#  loss_type should be one of "cross_entropy", "focal_loss", "class_balanced_cross_entropy", or "class_balanced_focal_loss".
cd ${MAIN_DIR}
export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_cls.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path ${MAIN_DIR}/src/dataset/train_4field_list_image_only_new.json \
    --image_folder /scr1/users/wangz12/train_imgs \
    --eval_path ${MAIN_DIR}/src/dataset/test_4field_list_image_only_new.json \
    --eval_image_folder /scr1/users/wangz12/test_imgs \
    --freeze_llm True \
    --freeze_vision_tower False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --loss_type "class_balanced_focal_loss" \
    --focal_alpha "0.050, 0.082, 0.103, 0.107, 0.125, 0.126, 0.154, 0.163, 0.178, 0.236" \
    --focal_gamma 3.0 \
    --class_balanced_beta 0.9999 \
    --num_labels 10 \
    --disable_flash_attn2 False \
    --output_dir /scr1/users/wangz12/output/qwen2_cls_image_only_new \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --head_lr 4e-5 \
    --vision_lr 6e-6 \
    --merger_lr 2e-5 \
    --weight_decay 0.02 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --eval_strategy "epoch" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_f1" \
    --greater_is_better True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --dataloader_num_workers 8 \
    # --use_liger True \
    # --lora_enable True \
    # --use_dora False \
    # --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    # --lora_rank 64 \
    # --lora_alpha 64 \
    # --lora_dropout 0.05 \
    # --num_lora_modules -1 \