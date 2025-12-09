cd ${MAIN_DIR}
export PYTHONPATH=src:$PYTHONPATH

python inference.py \
    --model_ckpt /scr1/users/wangz12/output/qwen_cls18\
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --data_path /scr1/users/wangz12/synthetic_images/metadata.json\
    --image_folder /scr1/users/wangz12/synthetic_images \
    --save_model_path /scr1/users/wangz12/output/qwen_cls18\
    --batch_size 8 \
    --num_workers 8 \
    --save_pred \
    --safe-serialization
