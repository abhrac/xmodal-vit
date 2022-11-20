import glob
import random
from typing import List, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class InstanceMappings:
    def __init__(self):
        self.map: Dict[int, List[str]] = {}

    def add_instance(self, instance_label: int, sketches: List[str]):
        self.map[instance_label] = sketches

    def get_instances(self) -> List[int]:
        return list(self.map.keys())

    def get_sketch_correspondences(self, instance_label: int) -> List[str]:
        return self.map[instance_label]


class QMULShoeV2(Dataset):
    def __init__(self, photo_folder_path: str, sketch_folder_path: str, transform=ToTensor()) -> None:
        photos_paths = glob.glob(photo_folder_path + '/*.*')
        photos = []
        sketches = []
        instance_labels = []

        photo_sketch_maps = InstanceMappings()

        for photo_path in photos_paths:
            photo_instance_name = int(photo_path.split('/')[-1].split('.')[0])
            sketch_correspondences = glob.glob(sketch_folder_path + '/' + str(photo_instance_name) + '_?.png')
            photo_sketch_maps.add_instance(photo_instance_name, sketch_correspondences)

            if len(sketch_correspondences) == 1:
                sketch_correspondences.append(sketch_correspondences[0])
            if len(sketch_correspondences) == 0:
                continue

            sketch_correspondences = list(sketch_correspondences)
            photos.append(photo_path)
            sketches.append(sketch_correspondences)
            instance_labels.append((photo_path, photo_instance_name))

        self.transform = transform
        self.photo_sketch_maps = photo_sketch_maps
        self.photos = photos
        self.sketches = sketches
        self.instance_labels = instance_labels

    def __getitem__(self, idx):
        photo = self.transform((Image.fromarray(np.array(Image.open(self.photos[idx]).convert('RGB')))))

        sketch_anchor, sketch_positive = random.sample(self.sketches[idx], k=2)
        anchor_path, positive_path = sketch_anchor, sketch_positive
        sketch_anchor = self.transform((Image.fromarray(np.array(Image.open(sketch_anchor).convert('RGB')))))
        sketch_positive = self.transform((Image.fromarray(np.array(Image.open(sketch_positive).convert('RGB')))))
        instance_label = self.instance_labels[idx]

        return photo, sketch_anchor, sketch_positive, [instance_label[0], instance_label[1], anchor_path, positive_path]

    def __len__(self):
        return len(self.photos)
