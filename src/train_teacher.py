import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from datasets.chair import QMULChairV2
from datasets.shoe import QMULShoeV2
from datasets.sketchy import Sketchy
from networks.teacher import ModalityFusionNetwork
from utils.losses import XAQC


def main():
    parser = argparse.ArgumentParser(description="Training script for the Modality Fusion Network (Teacher)")
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
    batch_size = 16
    num_epochs = 20000

    if args.dataset == 'ShoeV2':
        from params import shoe as p
        train_set = QMULShoeV2(photo_folder_path=p.photo_folder_path_train,
                               sketch_folder_path=p.sketch_folder_path_train,
                               transform=Compose([Resize(image_size), ToTensor()]))
    elif args.dataset == 'ChairV2':
        from params import chair as p
        train_set = QMULChairV2(photo_folder_path=p.photo_folder_path_train,
                                sketch_folder_path=p.sketch_folder_path_train,
                                transform=Compose([Resize(image_size), ToTensor()]))
    elif args.dataset == 'Sketchy':
        from params import sketchy as p
        train_set = Sketchy(photo_folder_path=p.photo_folder_path_train,
                            transform=Compose([Resize(image_size), ToTensor()]))

    device = p.teacher_device
    train_set_size = int(len(train_set))
    print('Dataset size: {}, Train set size: {}'
          .format(train_set_size, train_set_size))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    teacher = ModalityFusionNetwork(image_size, 3, feature_dim=p.teacher_out_dim, heads=3,
                                    encoder_backbone=p.teacher_encoder).to(device)
    ce_criterion = nn.CrossEntropyLoss()

    xaqc_teacher = XAQC(K=p.xa_queue_size_teacher, dim=p.teacher_out_dim).to(device)

    lr = 3e-6
    weight_decay = 1e-4
    fp16_precision = True
    optimizer = torch.optim.Adam(teacher.parameters(), lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=fp16_precision)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs, eta_min=0,
                                                           last_epoch=-1)

    os.makedirs(p.teacher_tgt_dir, exist_ok=True)

    for epoch in range(num_epochs):

        epoch_train_contrastive_loss = 0

        for batch_idx, data in enumerate(tqdm(train_loader)):
            teacher.train()
            photo, sketch_anchor, sketch_positive, instance_label = data
            photo = photo.to(device)
            sketch_anchor, sketch_positive = sketch_anchor.to(device), sketch_positive.to(device)

            optimizer.zero_grad()

            photo2sketch = teacher.forward_features(photo, sketch_anchor)[0]
            sketch2photo = teacher.forward_features(photo, sketch_positive)[1]

            logits, labels = xaqc_teacher(photo2sketch, sketch2photo)
            teacher_loss = ce_criterion(logits, labels)

            loss = teacher_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_contrastive_loss = epoch_train_contrastive_loss + teacher_loss.item()

            teacher.eval()
            if batch_idx % 10 == 0:
                scheduler.step()

        print('Epoch Train: [', epoch, '] Contrast Loss: ', epoch_train_contrastive_loss)

        if epoch % 100 == 0:
            print('Updating Modality Fusion Network (Teacher) checkpoint...')
            torch.save(
                teacher.state_dict(),
                os.path.join(p.teacher_tgt_dir, 'teacher_' + str(epoch) + '.pth'))


if __name__ == '__main__':
    main()
