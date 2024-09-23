from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from PIL import Image
import json
import datasets.transforms as T

class K700Action(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_anno = self.annotations[idx]
        target = {}
        imgs = []
        for imgname in video_anno['file_name']:
            img = Image.open(self.img_folder / imgname).convert('RGB')
            w, h = img.size
            img, target = self._transforms(img, target)
            imgs.append(img)
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['dataset_name'] = video_anno['dataset_name']
        target['video_name'] = video_anno['video_name']
        target['action_annotations'] =  video_anno['action_annotations']
        action_num = 700
        action_one_hot = torch.zeros((action_num,), dtype=torch.float32)
        for action_anno in target['action_annotations']:
            action_one_hot[action_anno['action_id']] = 1
        target['action_labels'] = torch.stack([action_one_hot])
        return imgs, target

def make_k700_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
                T.RandomRescale(scale_range=(256, 320)),
                T.RandomCrop(size=[256,256]),
                T.RandomHorizontalFlip(p=0.5),
                normalize
            ])

def build_k700(image_set, args):
    root = Path(args.action_k700_path)
    assert root.exists(), f'provided action k700 path {root} does not exist'

    image_set = 'train'
    anno_name = 'train_kinetics700.json'
    PATHS = {'train': (root / 'images', root / 'annotations' / anno_name)}

    img_folder, anno_file = PATHS[image_set]
    dataset = K700Action(image_set, img_folder, anno_file, transforms=make_k700_transforms(image_set))
    return dataset