cd /path/to/HOICLIP-main
EXP_DIR=exps/hico/hoiclip    
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port 29000 \
        --use_env \
        main.py \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path /path/to/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 60 \
        --lr_drop  30 \ 
        --use_nms_filter \
        --fix_clip \
        --batch_size 8 \
        --pretrained /path/to/dphoi_res50_hico_hoiclip.pth \
        --with_clip_label \
        --with_obj_clip_label \
        --gradient_accumulation_steps 1 \
        --num_workers 8 \
        --opt_sched "multiStep" \
        --dataset_root GEN \
        --model_name HOICLIP \
        --del_unseen \
        --zero_shot_type unseen_verb \
        --verb_pth ./tmp/verb.pth \
        --training_free_enhancement_path ./training_free_ehnahcement/