import os
import random
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from config.script_config import parse_script_args
from core.engine import Trainer
from core.utils_train import smart_load_weights, resolve_device
from core.warmup_scheduler import GradualWarmupScheduler
from data.dataset_utils import TestSetLoader, TrainSetLoader
from data.split import load_dataset
from losses.loss import build_loss
from models.LCMamNet import LCMamNet

warnings.filterwarnings("ignore")


DATASET_CFG = {
    "IRSTD-1k": {
        "norm_mode": "zscore-irstd-1k",
        "mean": [0.3430047],
        "std": [0.1557629],
    },
    "NUAA-SIRST": {
        "norm_mode": "zscore-nuaa-sirst",
        "mean": [0.3963288],
        "std": [0.1357631],
    },
    "NUDT-SIRST": {
        "norm_mode": "zscore-nudt-sirst",
        "mean": [0.4227806],
        "std": [0.1295010],
    },
}


DEFAULTS = {
    "project": "runs/train",
    "dataset": "IRSTD-1k",
    "root": "datasets",
    "mode": "TXT",
    "split_method": "split",
    "norm_mode": "zscore-irstd-1k",
    "name": "lcmamnet",
    "seed": 123,
    "base_size": 256,
    "crop_size": 256,
    "epochs": 800,
    "patience": 240,
    "warmup_epochs": 5,
    "train_batch_size": 12,
    "test_batch_size": 12,
    "workers": 8,
    "optimizer": "AdamW",
    "lr": 1.0e-3,
    "min_lr": 1.0e-5,
    "weight_decay": 5.0e-2,
    "grad_clip": 1.0,
    "use_amp": False,
    "amp_dtype": "fp16",
    "loss_mode": "bce30_softiou70",
    "bce_pos_weight": None,
    "model": "LCMamNet",
    "resume": None,
    "load": None,
    "suffix": ".png",
    "device": "cuda",
}


def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random Seed set to {seed}")


def main(args):
    seed_pytorch(args.seed)

    device = resolve_device(args.device)
    torch.cuda.set_device(device)

    if args.dataset not in DATASET_CFG:
        raise ValueError(f"Unknown dataset={args.dataset}. Choose from {list(DATASET_CFG)}.")
    cfg = DATASET_CFG[args.dataset]
    args.norm_mode = cfg["norm_mode"]
    mean, std = cfg["mean"], cfg["std"]

    dataset_dir = args.root + "/" + args.dataset
    train_img_ids, val_img_ids, _ = load_dataset(args.root, args.dataset, args.split_method)

    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    trainset = TrainSetLoader(
        dataset_dir,
        img_id=train_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=input_transform,
        suffix=args.suffix,
    )
    testset = TestSetLoader(
        dataset_dir,
        img_id=val_img_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=input_transform,
        suffix=args.suffix,
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )

    model = LCMamNet(n_channels=1, n_classes=1).cuda()
    criterion = build_loss(args)
    if args.load:
        print(f"Loading weights from: {args.load}")
        model = smart_load_weights(model, args.load)
    else:
        print("Training from scratch (no --load provided).")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr,
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=args.warmup_epochs,
        after_scheduler=scheduler_cosine,
    )

    trainer = Trainer(args, model, train_loader, test_loader, optimizer, scheduler)
    trainer.criterion = criterion

    print(f"Start training {args.name} on {args.dataset}")
    best_iou = 0
    early_stop_counter = 0
    for epoch in range(0, args.epochs):
        trainer.training(epoch)
        current_iou = trainer.testing(epoch)
        if scheduler:
            scheduler.step()
        if current_iou > best_iou:
            best_iou = current_iou
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered. Best mIoU: {best_iou:.4f}")
                break


if __name__ == "__main__":
    main(parse_script_args(DEFAULTS))
