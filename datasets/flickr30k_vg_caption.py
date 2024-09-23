from collections import defaultdict
import numpy as np
import datasets.transforms as T
import util.misc as utils
import torch
from pathlib import Path
import json
from PIL import Image

class Flickr30k_VG_Caption(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_anno = self.annotations[idx]
        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        target = {}
        w, h = img.size
        img, target = self._transforms(img, target)
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['dataset_name'] = img_anno['dataset_name'] 
        target['file_name'] = img_anno['file_name']
        target['triplet_captions'] = img_anno['triplet_captions']
        target['cluster_category'] = img_anno['cluster_category']
        return img, target

def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales2 = [int(v * 800 / 800) for v in scales]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomColorJitter(),
            T.RandomGrayscale(),
            T.RandomGaussianBlur(),
            T.RandomResize(scales2, max_size=int(1333)),
            normalize,
        ])

def build_caption(image_set, args):
    root = Path(args.caption_path)
    assert root.exists(), f'provided action haa500 path {root} does not exist'

    image_set = 'train'
    anno_name = 'Flickr30k_VG_cluster_dphoi.json'
    PATHS = {'train': (root, root / 'annotations' / anno_name)}
    img_folder, anno_file = PATHS[image_set]
    dataset = Flickr30k_VG_Caption(image_set, img_folder, anno_file, 
                            transforms=make_transforms(image_set))
    return dataset