import argparse
import random

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from datasets.chair import QMULChairV2
from datasets.shoe import QMULShoeV2
from datasets.sketchy import Sketchy
from networks.photo_encoder_student import PhotoEncoder
from networks.sketch_encoder_student import SketchEncoder


def main():
    parser = argparse.ArgumentParser(description="Training script for the Sketch Encoder (Student)")
    parser.add_argument("--dataset", default="ShoeV2", help="ShoeV2, ChairV2, Sketchy")
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[INFO] Setting SEED: " + str(args.seed))

    image_size = 224

    if args.dataset == 'ShoeV2':
        from params import shoe as p
        test_set = QMULShoeV2(photo_folder_path=p.photo_folder_path_test,
                              sketch_folder_path=p.sketch_folder_path_test,
                              transform=Compose([Resize(image_size), ToTensor()]))
    elif args.dataset == 'ChairV2':
        from params import chair as p
        test_set = QMULChairV2(photo_folder_path=p.photo_folder_path_test,
                               sketch_folder_path=p.sketch_folder_path_test,
                               transform=Compose([Resize(image_size), ToTensor()]))
    elif args.dataset == 'Sketchy':
        from params import sketchy as p
        test_set = Sketchy(photo_folder_path=p.photo_folder_path_test,
                           transform=Compose([Resize(image_size), ToTensor()]))

    batch_size = 1024
    device = p.photo_student_device

    test_set_size = len(test_set)
    print('Test set size: {}'.format(test_set_size))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    photo2xat_student = PhotoEncoder(feature_dim=p.student_embed_dim, output_dim=p.teacher_out_dim).to(device)
    photo2xat_student.encoder = timm.create_model(p.student_encoder, pretrained=False).to(device)
    photo2xat_student.load_state_dict(torch.load(p.photo_student_tgt_dir + '/photo_student_20000.pth'))

    sketch2xat_student = SketchEncoder(feature_dim=p.student_embed_dim, output_dim=p.teacher_out_dim).to(device)
    sketch2xat_student.encoder = timm.create_model(p.student_encoder, pretrained=False).to(device)
    sketch2xat_student.load_state_dict(torch.load(p.sketch_student_tgt_dir + '/sketch_student_20000.pth'))

    acc_at_1 = 0
    acc_at_10 = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            photo, sketch, _, instance_id = data
            photo = photo.to(device)

            photo = photo.to(device)
            sketch = sketch.to(device)

            gallery_repr = photo2xat_student.forward_features(photo)
            query_repr = sketch2xat_student.forward_features(sketch)

            similarity_matrix = torch.argsort(torch.matmul(query_repr, gallery_repr.T), dim=1, descending=True)
            acc_at_1 += (similarity_matrix[:, 0] == torch.tensor(range(len(photo))).to(device)).sum() / len(photo)

            acc_at_10 += (similarity_matrix[:, :10] == torch.unsqueeze(torch.tensor(range(len(photo))), dim=1).to(
                device)).sum() / len(photo)

        print(f'Acc@1: {acc_at_1 / len(test_set) * 100}, Acc@10: {acc_at_10 / len(test_set) * 100}')


if __name__ == '__main__':
    main()
