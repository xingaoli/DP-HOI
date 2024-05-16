cd /path/to/gen-vlkt
EXP_DIR=exps/vcoco_gen_vlkt_s_r50_dec_3layers
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env \
        main.py \
        --pretrained /path/to/dphoi-res50-gen-vcoco.pth \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --hoi_path /path/to/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --ft_clip_with_small_lr \
        --with_clip_label \
        --with_obj_clip_label \
        --batch_size 4  \
        --fix_backbone