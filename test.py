import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from config.script_config import parse_script_args
from core.utils_train import smart_load_weights, resolve_device
from data.dataset_utils import TestSetLoader
from data.split import load_dataset
from metrics.irstd_metrics import PD_FA, mIoU, SamplewiseSigmoidMetric
from models.LCMamNet import LCMamNet


DEFAULTS = {
    "model": "LCMamNet",
    "weights": "weights/LCMamNet_irstd1k.pt",
    "root": "datasets",
    "dataset": "IRSTD-1k",
    "split_method": "split",
    "suffix": ".png",
    "base_size": 256,
    "crop_size": 256,
    "batch_size": 1,
    "workers": 4,
    "norm_mode": "auto",
    "warmup_batches": 10,
    "max_batches": 0,
    "device": "cuda",
}

DATASET_NORM = {
    "IRSTD-1k": "zscore-irstd-1k",
    "NUAA-SIRST": "zscore-nuaa-sirst",
    "NUDT-SIRST": "zscore-nudt-sirst",
}

DATASET_ALIASES = {
    "NUAA": "NUAA-SIRST",
    "NUAA-SIRST": "NUAA-SIRST",
    "NUDT": "NUDT-SIRST",
    "NUDT-SIRST": "NUDT-SIRST",
    "IRSTD-1K": "IRSTD-1k",
    "IRSTD1K": "IRSTD-1k",
}

ZSCORE_STATS = {
    "zscore-nuaa-sirst": (0.3963288, 0.1357631),
    "zscore-nudt-sirst": (0.4227806, 0.1295010),
    "zscore-irstd-1k": (0.3430047, 0.1557629),
}


def resolve_dataset_and_ids(root, dataset, split_method):
    root_path = Path(root)
    effective_dataset = DATASET_ALIASES.get(dataset.upper(), dataset)
    if not (root_path / effective_dataset).exists():
        raise FileNotFoundError(f"Dataset folder not found: {root_path / effective_dataset}")

    _, val_img_ids, split_file = load_dataset(root, effective_dataset, split_method)
    if not val_img_ids:
        raise ValueError(f"No validation ids found for dataset={dataset}")
    return effective_dataset, val_img_ids, split_file


def build_transform(norm_mode):
    if norm_mode == "minmax":
        mean, std = [0.5], [0.5]
    elif norm_mode in ZSCORE_STATS:
        mean, std = [ZSCORE_STATS[norm_mode][0]], [ZSCORE_STATS[norm_mode][1]]
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def select_eval_output(preds):
    if isinstance(preds, (list, tuple)):
        return preds[-1]
    return preds


def main(args):
    device = resolve_device(args.device)

    effective_dataset, val_img_ids, split_file = resolve_dataset_and_ids(
        args.root, args.dataset, args.split_method
    )
    norm_mode = DATASET_NORM.get(effective_dataset, args.norm_mode) if args.norm_mode == "auto" else args.norm_mode
    args.norm_mode = norm_mode
    dataset_dir = str(Path(args.root) / effective_dataset)
    transform = build_transform(norm_mode)
    testset = TestSetLoader(
        dataset_dir,
        img_id=val_img_ids,
        transform=transform,
        base_size=args.base_size,
        crop_size=args.crop_size,
        suffix=args.suffix,
    )
    loader = DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = LCMamNet(n_channels=1, n_classes=1).to(device).eval()
    if args.weights:
        model = smart_load_weights(model, args.weights)

    miou_metric = mIoU()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    pdfa_metric = PD_FA()

    total_images_timed = 0
    total_time = 0.0
    total_batches = len(loader)
    warmup_batches = min(args.warmup_batches, total_batches)

    pbar = tqdm(enumerate(loader), total=total_batches, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    with torch.no_grad():
        for batch_idx, (data, labels) in pbar:
            if args.max_batches and batch_idx >= args.max_batches:
                break

            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            preds = select_eval_output(model(data)).float()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            miou_metric.update(preds, labels)
            niou_metric.update(preds, labels)
            pdfa_metric.update(preds, labels)

            _, mean_iou = miou_metric.get()
            _, n_iou = niou_metric.get()
            pd, fa = pdfa_metric.get()

            if batch_idx >= warmup_batches:
                total_time += elapsed
                total_images_timed += data.size(0)

            avg_ms = (total_time / max(total_images_timed, 1)) * 1000.0
            fps = total_images_timed / max(total_time, 1e-12)
            pbar.set_description(
                f"{batch_idx + 1:>4}/{total_batches:<4} mIoU {mean_iou:>7.4f} "
                f"nIoU {n_iou:>7.4f} PD {pd:>7.4f} FA {fa:>11.8f} {avg_ms:>8.3f}ms {fps:>8.2f}fps"
            )

    pixacc, mean_iou = miou_metric.get()
    _, n_iou = niou_metric.get()
    pd, fa = pdfa_metric.get()
    avg_ms = (total_time / max(total_images_timed, 1)) * 1000.0
    fps = total_images_timed / max(total_time, 1e-12)

    print("\nBenchmark Summary")
    print(f"model           : {args.model}")
    print(f"weights         : {args.weights}")
    print(f"dataset         : {effective_dataset}")
    print(f"split_file      : {split_file}")
    print(f"num_images      : {len(val_img_ids)}")
    print(f"pixAcc          : {pixacc:.8f}")
    print(f"mIoU            : {mean_iou:.8f}")
    print(f"nIoU            : {n_iou:.8f}")
    print(f"PD              : {pd:.8f}")
    print(f"FA              : {fa:.8f}")
    print(f"latency_ms_img  : {avg_ms:.3f}")
    print(f"fps             : {fps:.2f}")


if __name__ == "__main__":
    main(parse_script_args(DEFAULTS))
