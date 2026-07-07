import csv
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler
from tqdm import tqdm

from data.dataset_utils import *
from metrics.irstd_metrics import *
from losses.loss import *
from core.utils_train import increment_path, get_gpu_mem


class Trainer(object):
    def __init__(self, args, model, train_loader, test_loader, optimizer, scheduler=None):
        self.args = args
        self.model = model
        self.train_data = train_loader
        self.test_data = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.start_epoch = 0
        self.best_iou = 0

        project = getattr(args, 'project', 'runs/train')
        name = getattr(args, 'name', 'lcmamnet')
        self.save_dir = increment_path(Path(project) / name, mkdir=True)
        self.weights_dir = self.save_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / 'args.yaml', 'w') as f:
            yaml.dump(vars(args), f, sort_keys=False)

        self.csv_path = self.save_dir / "results.csv"

        if not args.resume:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'test_loss', 'mIoU', 'lr'])

        print(f"\nLogging results to {self.save_dir}")

        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_amp = self._parse_bool(getattr(args, 'use_amp', True))
        self.amp_dtype_name = str(getattr(args, 'amp_dtype', 'bf16')).lower()
        self.amp_dtype = self._resolve_amp_dtype(self.amp_dtype_name)
        self.scaler_enabled = self.use_amp and self.device_type == 'cuda' and self.amp_dtype == torch.float16
        self.scaler = GradScaler('cuda', enabled=self.scaler_enabled)

        if self.use_amp:
            print(f"AMP enabled (dtype={self.amp_dtype_name}, scaler={self.scaler_enabled})")
        else:
            print("AMP disabled")

        self.mIoU_metric = mIoU()
        self.grad_clip = float(getattr(args, 'grad_clip', 2.0))

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "y", "on")
        return bool(value)

    def _resolve_amp_dtype(self, amp_dtype_name):
        if not self.use_amp:
            return torch.float16

        if amp_dtype_name in ("bf16", "bfloat16"):
            bf16_supported = (
                self.device_type == 'cuda'
                and hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
            if bf16_supported:
                return torch.bfloat16
            print("BF16 not supported on this runtime, fallback to FP16 autocast.")
            self.amp_dtype_name = "fp16"
            return torch.float16

        if amp_dtype_name in ("fp16", "float16", "half"):
            self.amp_dtype_name = "fp16"
            return torch.float16

        print(f"Unknown amp_dtype={amp_dtype_name}, fallback to bf16/fp16 auto-detect.")
        self.amp_dtype_name = "bf16"
        return self._resolve_amp_dtype("bf16")

    def _pick_pred(self, preds):
        if isinstance(preds, (list, tuple)):
            return preds[-1]
        return preds

    def save_ckpt(self, epoch, mIoU, is_best=False):
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.weights_dir / 'last.pt')
        if is_best:
            torch.save(state_dict, self.weights_dir / 'best.pt')

    def training(self, epoch):
        criterion = getattr(self, 'criterion', None)
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        self.model.train()
        losses = AverageMeter()

        print(f"\n{'Epoch':>12} {'GPU_mem':>10} {'Train_Loss':>12} {'Instances':>10} {'Size':>8}")

        pbar = tqdm(enumerate(self.train_data), total=len(self.train_data),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        self.optimizer.zero_grad()
        for i, (data, labels) in pbar:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with torch.autocast(
                device_type=self.device_type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                preds = self._pick_pred(self.model(data))
                loss = self.criterion(preds, labels)
                raw_loss = loss

            # 跳过 NaN/Inf loss，避免污染参数。
            if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                print(f"NaN/Inf loss at Epoch {epoch} Batch {i}. Skipping step.")
                self.optimizer.zero_grad()
                continue

            if self.scaler_enabled:
                self.scaler.scale(raw_loss).backward()
            else:
                raw_loss.backward()

            if self.scaler_enabled:
                self.scaler.unscale_(self.optimizer)

            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            else:
                grad_norm = torch.tensor(0.0, device=data.device)

            # 跳过 NaN/Inf 梯度。
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"NaN/Inf gradient (norm={grad_norm}) at Epoch {epoch} Batch {i}. Skipping step.")
                self.optimizer.zero_grad()
                if self.scaler_enabled:
                    self.scaler.update()
                continue

            if self.scaler_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            losses.update(raw_loss.item(), data.size(0))

            mem = get_gpu_mem()
            epoch_str = f"{epoch + 1}/{self.args.epochs}"
            desc = f"{epoch_str:>12} {mem:>10} {losses.avg:>12.4g} {data.size(0):>10} {self.args.crop_size:>8}"
            pbar.set_description(desc)

        self.train_loss = losses.avg

    def testing(self, epoch):
        criterion = getattr(self, 'criterion', None)
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        self.model.eval()
        self.mIoU_metric.reset()
        losses = AverageMeter()

        print(f"{'Epoch':>12} {'GPU_mem':>10} {'Test_Loss':>12} {'mIoU':>10}")

        pbar = tqdm(enumerate(self.test_data), total=len(self.test_data),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        with torch.no_grad():
            for i, (data, labels) in pbar:
                data = data.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                with torch.autocast(
                    device_type=self.device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    preds = self._pick_pred(self.model(data))

                preds = torch.nan_to_num(preds.float(), nan=0.0)
                loss_fn = getattr(self, 'criterion', SoftIoULoss())
                loss = loss_fn(preds, labels)
                if torch.isnan(loss):
                    loss = torch.tensor(0.0, device=preds.device)
                losses.update(loss.item(), preds.size(0))

                self.mIoU_metric.update(preds, labels)
                _, mean_iou = self.mIoU_metric.get()

                mem = get_gpu_mem()
                epoch_str = f"{epoch + 1}/{self.args.epochs}"
                desc = f"{epoch_str:>12} {mem:>10} {losses.avg:>12.4g} {mean_iou:>10.4g}"
                pbar.set_description(desc)

            test_loss = losses.avg
            current_miou = mean_iou
            desc = f"{epoch_str:>12} {mem:>10} {losses.avg:>12.4g} {current_miou:>10.4g}"
            pbar.set_description(desc)

        current_lr = self.optimizer.param_groups[0]['lr']

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                self.train_loss,
                test_loss,
                current_miou,
                current_lr,
            ])

        is_best = current_miou > self.best_iou
        if is_best:
            self.best_iou = current_miou

        self.save_ckpt(epoch + 1, current_miou, is_best)

        return current_miou



