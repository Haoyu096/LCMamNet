from PIL import Image, ImageOps
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch


class TrainSetLoader(Dataset):
    """Iceberg Segmentation datasets."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, base_size=512, crop_size=480, transform=None, suffix='.png'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir + '/' + 'masks'
        self.images = dataset_dir + '/' + 'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.aug_mode = 'arg04'

    def _sync_transform_arg01(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.float32)
        return img_np, mask_np

    def _sync_transform_arg02(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.TRANSPOSE)
            mask = mask.transpose(Image.TRANSPOSE)

        crop_size = self.crop_size
        w, h = img.size

        padh = crop_size - h if h < crop_size else 0
        padw = crop_size - w if w < crop_size else 0
        if padh > 0 or padw > 0:
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            w, h = img.size

        mask_np_for_loc = np.array(mask)
        has_target = mask_np_for_loc.sum() > 0

        if has_target and random.random() < 0.5:
            ys, xs = np.where(mask_np_for_loc > 0)
            idx = random.randint(0, len(ys) - 1)
            target_y, target_x = ys[idx], xs[idx]

            x1_min = max(0, target_x - crop_size + 1)
            x1_max = min(w - crop_size, target_x)
            y1_min = max(0, target_y - crop_size + 1)
            y1_max = min(h - crop_size, target_y)

            x1 = random.randint(x1_min, x1_max)
            y1 = random.randint(y1_min, y1_max)
        else:
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)

        img_crop = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask_crop = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        img_np = np.array(img_crop)
        mask_np = np.array(mask_crop, dtype=np.float32)
        return img_np, mask_np

    def _sync_transform_arg03(self, img, mask):
        target_size = self.base_size
        img = img.resize((target_size, target_size), Image.BILINEAR)
        mask = mask.resize((target_size, target_size), Image.NEAREST)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.TRANSPOSE)
            mask = mask.transpose(Image.TRANSPOSE)

        if random.random() < 0.5:
            img_np = np.array(img, dtype=np.float32)
            gain = random.uniform(0.9, 1.1)
            bias = random.uniform(-8.0, 8.0)
            img_np = np.clip(img_np * gain + bias, 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(img_np)

        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.float32)
        return img_np, mask_np

    def _sync_transform_arg04(self, img, mask):
        target_size = self.base_size
        img = img.resize((target_size, target_size), Image.BILINEAR)
        mask = mask.resize((target_size, target_size), Image.NEAREST)
        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.float32)
        return img_np, mask_np

    def _sync_transform(self, img, mask, img_id=None):
        if self.aug_mode == 'arg04':
            return self._sync_transform_arg04(img, mask)
        if self.aug_mode == 'arg03':
            return self._sync_transform_arg03(img, mask)
        if self.aug_mode == 'arg01':
            return self._sync_transform_arg01(img, mask)
        if self.aug_mode not in ('arg01', 'arg02', 'arg03', 'arg04'):
            raise ValueError(f'Unsupported aug_mode={self.aug_mode}. Use arg01/arg02/arg03.')
        return self._sync_transform_arg02(img, mask)

    def __getitem__(self, idx):

        img_id = self._items[idx]
        img_path = self.images + '/' + img_id + self.suffix
        label_path = self.masks + '/' + img_id + self.suffix

        img = Image.open(img_path).convert('L')  # 统一转为单通道灰度
        mask = Image.open(label_path)

        # synchronized transform
        img, mask = self._sync_transform(img, mask, img_id=img_id)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation datasets."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir + '/' + 'masks'
        self.images = dataset_dir + '/' + 'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask,
                                            dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]
        img_path = self.images + '/' + img_id + self.suffix
        label_path = self.masks + '/' + img_id + self.suffix
        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
