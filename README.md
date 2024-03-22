# Disentangled Pre-training for Human-Object Interaction Detection
Zhuolong Li<sup>\*</sup>,
Xingao Li<sup>\*</sup>,
Changxing Ding,
Xiangmin Xu

The paper is accepted to CVPR2024.

<div align="center">
  <img src="img/overview_dphoi.png" width="900px" />
</div>

## Preparation

### Environment
1. Install the dependencies.
```
pip install -r requirements.txt
```
2. Clone and build CLIP.
```
git clone https://github.com/openai/CLIP.git && cd CLIP && python setup.py develop && cd ..
```

### Dataset
1. haa500 dataset

&emsp; Download the haa500 dataset from the following URL and unzip it to the `DP-HOI/pre_datasets` folder.
```
http://xxxx
```
&emsp; run `pre_haa500.py`
```
python ./pre_datasets/pre_haa500.py
```
&emsp; Move the processed haa500 dataset to the `DP-HOI/data` folder.

2. kinetics700 dataset

&emsp; Download the kinetics700 dataset from the following URL and unzip it to the `DP-HOI/pre_datasets` folder.
```
http://xxxx
```
&emsp; run `pre_kinetics700.py`
```
python ./pre_datasets/pre_kinetics700.py
```
&emsp; Move the processed kinetics700 dataset to the `DP-HOI/data` folder.

3. flickr30k dataset

&emsp; Download the flickr30k dataset from the following URL and directly unzip it to the `DP-HOI/data` folder.
```
http://xxxx
```
&emsp; Move the processed json file in the `DP-HOI/pre_datasets/train_flickr30k.json` to the `DP-HOI/data/flickr30k/annotations` folder

4. vg dataset

&emsp; Download the vg dataset from the following URL and directly unzip it to the `DP-HOI/data` folder.
```
http://xxxx
```
&emsp; Move the processed json file in the `DP-HOI/pre_datasets/train_vg.json` to the `DP-HOI/data/vg/annotations` folder

5. objects365 dataset

&emsp; Download the objects365 dataset from the following URL and directly unzip it to the `DP-HOI/data` folder.
```
http://xxxx
```
&emsp; Move the processed json file in the `DP-HOI/pre_datasets/train_objects365_10k.json` to the `DP-HOI/data/objects365/annotations` folder

When you have completed the above steps, the pre-training dataset structure is:
```
DP-HOI
 |─ data
 |   └─ coco
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |─ images
 |       |   |─ test2015
 |       |   └─ train2015
 
 |   └─ object365
 |       |─ annotations
 |       |   |─ train_objects365_10k.json
 |       |─ images
 |       |   |─ train2014
 
 |   └─ haa500
 |       |─ annotations
 |       |   |─ train_haa500_50k.json
 |       |─ images
 |       |   └─ train

 |   └─ kinetics700
 |       |─ annotations
 |       |   |─ train_kinetics700_10k.json
 |       |─ images
 |       |   └─ train

 |   └─ flickr30k
 |       |─ annotations
 |       |   |─ train_flickr30k.json
 |       |─ images
 |       |   └─ train

 |   └─ vg
 |       |─ annotations
 |       |   |─ train_vg.json
 |       |─ images
 |       |   └─ train
```

### Initial parameters
To speed up the pre-training process, consider using DETR's pre-trained weights for initialization. 
Download the pretrained model of DETR detector for ResNet50 , and put it to the `params` directory.


## Pre-training
After the preparation, you can start training with the following commands.
```
sh ./config/train.sh
```

## Fine-tuning
After pre-training, you can start fine-tuning with the following commands.

Firstly, you can convert pre-training parameters to downstream model as follows.
```
python ./util/convert_parameters_hico.sh --fine-tuning_model_name --input_dir --output_dir
```
Then, fine-tuning according to the official code of the corresponding model.
An example of fine-tuning on HOICLIP is provided below.

```
python ./util/convert_parameters_hico.sh HOICLIP ./logs/pre-trained/checkpoint0039.pth ./dphoi_res50.pth
mv dphoi_res50.pth ../HOICLIP/params
cd ../HOICLIP
sh ./scripts/train_hico.sh
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
