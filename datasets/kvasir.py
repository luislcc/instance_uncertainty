import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Kvasir(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    KvasirClass = namedtuple('KvasirClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        KvasirClass('background', 0, 0, 'background', 0, False, False, (0, 0, 0)),
        KvasirClass('polyp', 255, 1, 'polyp', 1, True, False, (255, 255, 255)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'images')

        self.targets_dir = os.path.join(self.root, 'masks')
        self.transform = transform

        print(self.images_dir)

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        self.images = [os.path.join(self.images_dir,split,file_name) for file_name in os.listdir(os.path.join(self.images_dir,split))]
        self.targets = [os.path.join(self.targets_dir,file_name) for file_name in os.listdir(self.targets_dir)]


    @classmethod
    def encode_target(cls, target):
        #print(np.array(target))
        target_array = np.array(target)
        target = (target_array > 127).astype(int)
        return target
        #return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index]).convert('L')
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)


        uncertainty_mask_name = self.images[index].split('/')[-1][:-4]
        uncertainty_mask_path = f'uncertainty_masks/{uncertainty_mask_name}_uncertainty_mask.npy'

        if os.path.exists(os.path.join(self.root,uncertainty_mask_path)):
            uncertainty_mask = np.load(os.path.join(self.root,uncertainty_mask_path))
        else:
            uncertainty_mask = []
        
        target = np.expand_dims(target, axis=0)
        
        return (image,self.images[index]), (target.astype(np.float32),uncertainty_mask)

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data