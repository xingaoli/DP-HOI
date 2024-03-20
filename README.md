# Disentangled Pre-training for Human-Object Interaction Detection
Zhuolong Li,
Xingao Li,
Changxing Ding,
Xiangmin Xu

The paper is accepted to CVPR2024.

<div align="center">
  <img src="img/overview_dphoi.png" width="900px" />
</div>

## Preparation

### Environment
Our implementation uses environment the same as [HOICLIP](https://github.com/Artanic30/HOICLIP),
please follow HOICLIP to set up pytorch environment.

### Dataset

The dataset structure is:
```
qpic
 |─ data
 |   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |─ images
 |       |   |─ test2015
 |       |   └─ train2015
 
 |   └─ v-coco
 |       |─ annotations
 |       |   |─ trainval_vcoco.json
 |       |   |─ test_vcoco.json
 |       |   |─ corre_vcoco.npy
 |       |   |─ vcoco_test.json
 |       |   |─ instances_vcoco_all_2014.json
 |       |─ images
 |       |   |─ train2014
 |       |   └─ val2014
 
 |   └─ hoia
 |       |─ annotations
 |       |   |─ test_2019.json
 |       |   |─ train_2019.json
 |       |   |─ obj_clipvec.npy
 |       |   |─ sim_index_hoia.pickle
 |       |   └─ corre_hoia.npy
 |       |─ images
 |       |   |─ test
 |       |   └─ trainval
```

The annotations file,
pre-trained weights and 
trained parameters can be downloaded [here]()

Please download the images at the official website for the datasets above.


### Trained parameters

## Training
After the preparation, you can start the training with the following command.

For the HICO-DET training.
```
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained params/detr-r50-pre.pth \
--hoi \
--dataset_file hico \
--hoi_path data/hico_20160224_det \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
```

For the V-COCO training.
```
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained params/detr-r50-pre-vcoco.pth \
--hoi \
--dataset_file vcoco \
--hoi_path data/v-coco \
--num_obj_classes 81 \
--num_verb_classes 29 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
```

## Evaluation
The evaluation is conducted at the end of each epoch during the training. The results are written in `outputs/log.txt` like below:

You can also conduct the evaluation with trained parameters as follows.

HICO-DET
```
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained params/checkpoint.pth \
--hoi \
--dataset_file hico \
--hoi_path data/hico_20160224_det \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--eval \
```

For the official evaluation of V-COCO, a pickle file of detection results have to be generated. You can generate the file as follows.
```
python generate_vcoco_official.py \
--param_path outputs/vcoco/ts_model/checkpoint.pth \
--save_path vcoco.pickle \
--hoi_path data/v-coco
```

## Results
HICO-DET.
|| Full (D) | Rare (D) | Non-rare (D) | Full(KO) | Rare (KO) | Non-rare (KO) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|ours (CDN)| 34.27 | 30.02 | 35.54 | 37.05 | 33.09 | 38.23 |
|ours (GEN_s)| 34.40 | 31.17 | 35.36 | 38.25 | 35.64 | 39.03 |
|ours (HOICLIP)| 36.56 | 34.36 | 37.22 | - | - | - |

D: Default, KO: Known object

V-COCO.
|| Scenario 1 | 
| :--- | :---: |
|ours (GEN_s)| 66.2|

## Citation
Please consider citing our paper if it helps your research.
```
@inproceedings{disentangled_cvpr2024,
author = {Zhuolong Li,Xingao Li,Changxing Ding,Xiangmin Xu},
title = {Disentangled Pre-training for Human-Object Interaction Detection},
booktitle={CVPR},
year = {2024},
}
```

## Acknowledgement
[HOICLIP](https://github.com/Artanic30/HOICLIP)
