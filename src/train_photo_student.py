import argparse
import os
import random

import numpy as np
import timm
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from datasets.chair import QMULChairV2
from datasets.shoe import QMULShoeV2
from datasets.sketchy import Sketchy
from networks.photo_encoder_student import PhotoEncoder
from networks.teacher import ModalityFusionNetwork
from utils.losses import XAQC
from utils.losses import XMRDDistance, XMRDAngle


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

    device = p.photo_student_device
    teacher_dir = os.path.join(p.teacher_tgt_dir, 'teacher_20000.pth')
    os.makedirs(p.photo_student_tgt_dir, exist_ok=True)

    train_set_size = int(len(train_set))
    print('Train set size: {}'.format(train_set_size))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    teacher = ModalityFusionNetwork(image_size, 3, feature_dim=p.teacher_out_dim, heads=3,
                                    encoder_backbone=p.teacher_encoder).to(device)

    teacher.load_state_dict(torch.load(teacher_dir))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    photo2xat_student = PhotoEncoder(feature_dim=p.student_embed_dim, output_dim=p.teacher_out_dim).to(device)

    photo2xat_student.encoder = timm.create_model(p.student_encoder, pretrained=True).to(device)

    xaqc_student = XAQC(K=p.xa_queue_size_student, dim=p.teacher_out_dim).to(device)
    xmrd_distance = XMRDDistance()
    xmrd_angle = XMRDAngle()
    ce_loss = nn.CrossEntropyLoss()

    lr = 1e-5
    weight_decay = 1e-5
    fp16_precision = True
    optimizer = torch.optim.Adam(photo2xat_student.parameters(), lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=fp16_precision)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)

    for epoch in range(num_epochs):
        total_train_loss = 0

        for batch_idx, data in enumerate(tqdm(train_loader)):
            photo2xat_student.train()
            photo, sketch_q, sketch_k, _ = data
            photo = photo.to(device)
            sketch_q = sketch_q.to(device)
            sketch_k = sketch_k.to(device)

            optimizer.zero_grad()

            query = photo2xat_student.forward_features(sketch_q)
            with torch.no_grad():
                key = teacher(photo, sketch_q)
            logits, labels = xaqc_student(query, key)
            loss = ce_loss(logits.to(device), labels.to(device))

            loss = loss + xmrd_distance(query, key)
            loss = loss + xmrd_angle(photo, sketch_q, sketch_k, teacher, photo2xat_student, modality='sketch')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss = total_train_loss + loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch}: Train Loss: ', avg_train_loss)

        if epoch % 5 == 0:
            print('Updating Photo-to-XAT student checkpoint...')
            torch.save(photo2xat_student.state_dict(),
                       p.photo_student_tgt_dir + '/photo_student_' + str(epoch) + '.pth')

        scheduler.step()


if __name__ == '__main__':
    main()
