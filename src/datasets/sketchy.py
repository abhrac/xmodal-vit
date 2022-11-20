import glob
import os
import random
from typing import List, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


class SketchyTree:
    def __init__(self):
        self.tree: Dict[int, Dict[int, (str, List[str])]] = {}

    def add_instance(self, class_label: int, instance_label: int, photo_path: str, sketches: List[str]):
        if class_label not in self.tree.keys():
            self.tree[class_label]: dict = {}

        self.tree[class_label][instance_label] = (photo_path, sketches)

    def get_class_labels(self) -> List[int]:
        return list(self.tree.keys())

    def get_class_instances(self, class_label: int) -> List[int]:
        return list(self.tree[class_label].keys())

    def get_photo_path(self, class_label: int, instance_label: int) -> str:
        return self.tree[class_label][instance_label][0]

    def get_sketch_correspondences(self, class_label: int, instance_label: int) -> List[str]:
        return self.tree[class_label][instance_label][1]


class Sketchy(Dataset):
    def __init__(self, photo_folder_path, transform=ToTensor(), subset_size=None, class_index=None):
        self.photo_folder = ImageFolder(photo_folder_path)
        self.sketch_folder = ImageFolder(photo_folder_path.replace("photo", "sketch"))
        self.num_classes = len(self.photo_folder.class_to_idx)
        self.transform = transform
        if subset_size is None:
            self.num_photos = len(self.photo_folder.imgs)
        else:
            self.num_photos = subset_size

        sketchy_tree: SketchyTree = SketchyTree()

        photo_objects = self.photo_folder.imgs[:self.num_photos]
        if class_index is not None:
            photo_objects = self.photo_folder.imgs[(class_index * 100): (class_index * 100) + self.num_photos]

        random.shuffle(photo_objects)

        for photo_path, class_label in photo_objects:
            to_sketches = photo_path.replace("photo", "sketch").split('/')
            sketch_class_folder = '/'.join(to_sketches[:-1])
            instance_name: str = to_sketches[-1].split('.')[0]
            instance_label: int = instance_name_hash(instance_name)
            sketch_correspondences = glob.glob(os.path.join(sketch_class_folder, instance_name + '-?.png'))

            # Update Sketchy-Tree with the current instance
            sketchy_tree.add_instance(class_label, instance_label, photo_path, sketch_correspondences)

        self.sketchy_tree: SketchyTree = sketchy_tree

    def __getitem__(self, idx):
        sketchy_tree = self.sketchy_tree
        class_label: int = sketchy_tree.get_class_labels()[idx]
        instance: int = random.choice(sketchy_tree.get_class_instances(class_label))
        photo_path: str = sketchy_tree.get_photo_path(class_label, instance)
        sketch_anchor_path, sketch_positive_path = random.sample(
            sketchy_tree.get_sketch_correspondences(class_label, instance), 2)

        photo = self.transform((Image.fromarray(np.array(Image.open(photo_path).convert('RGB')))))
        sketch_anchor = self.transform((Image.fromarray(np.array(Image.open(sketch_anchor_path).convert('RGB')))))
        sketch_positive = self.transform((Image.fromarray(np.array(Image.open(sketch_positive_path).convert('RGB')))))

        return photo, sketch_anchor, sketch_positive, class_label

    def __len__(self):
        return len(self.sketchy_tree.get_class_labels())


def instance_name_hash(instance_name: str) -> int:
    name_tokens: List[str] = instance_name.split('_')
    return int(str(ord(name_tokens[0][0])) + name_tokens[0][1:] + name_tokens[1])
