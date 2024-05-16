cd /path/to/upt-main
python main.py  --pretrained  /path/to/dphoi_res50_3layers_hicodet.pth \
                --output-dir checkpoints/upt-r50-hicodet \
                --world-size 4 \
                --batch-size 4 \
                --dec-layers 3 \
