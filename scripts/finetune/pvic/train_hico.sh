cd /path/to/pvic-main
DETR=base python main.py --pretrained /path/to/dphoi_res50_3layers_hicodet.pth \
                         --output-dir outputs/pvic-detr-r50-hicodet \
                         --world-size 4 \
                         --batch-size 4 \
                         --dec-layers 3 \