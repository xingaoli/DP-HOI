cd /path/to/cdn_dn
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env \
        main.py \
        --output_dir logs \
        --dataset_file hico \
        --start_epoch 0 \
        --pretrained  /path/to/dphoi-res50-hico-cdn.pth  \
        --hoi_path /path/to/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 60 \
        --lr_drop 30 \
        --use_nms_filter \
        --batch_size 4 \
        --use_dn \
        --use_ccs