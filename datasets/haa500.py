from collections import defaultdict
import numpy as np
import datasets.transforms as T
import util.misc as utils
import torch
from pathlib import Path
import json
from PIL import Image

class Haa500Action(torch.utils.data.Dataset):

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
        target['action_annotations'] = img_anno['action_annotations']
        action_num = 500
        action_one_hot = torch.zeros((action_num,), dtype=torch.float32)
        for action_anno in target['action_annotations']:
            action_one_hot[action_anno['action_id']] = 1
        target['action_labels'] = torch.stack([action_one_hot])
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

def build_haa500(image_set, args):
    root = Path(args.action_haa500_path)
    assert root.exists(), f'provided action haa500 path {root} does not exist'

    image_set = 'train'
    anno_name = 'train_haa500.json'
    PATHS = {'train': (root / 'images', root / 'annotations' / anno_name)}
    img_folder, anno_file = PATHS[image_set]
    dataset = Haa500Action(image_set, img_folder, anno_file, 
                            transforms=make_transforms(image_set))
    return dataset