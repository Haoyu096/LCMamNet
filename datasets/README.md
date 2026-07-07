# Datasets

This folder shows the expected layout for each dataset. Only **two sample image/mask
pairs** are shipped per dataset as a format reference; the full `train.txt` / `test.txt`
split lists are included in full. To reproduce the paper results, download the complete
datasets and place every image/mask under the corresponding folders.

## Layout

```
datasets/
└── <DATASET>/
    ├── images/        # <id>.png  — input infrared images
    ├── masks/         # <id>.png  — binary target masks (same filename as the image)
    └── split/
        ├── train.txt  # one image id per line (no file extension)
        └── test.txt
```

`<DATASET>` is one of `IRSTD-1k`, `NUAA-SIRST`, `NUDT-SIRST`. Image ids in the split
files must match the `<id>.png` filenames in `images/` and `masks/`.

## Sample ids included

| Dataset     | Sample ids          |
| ----------- | ------------------- |
| IRSTD-1k    | `XDU0`, `XDU1`      |
| NUAA-SIRST  | `Misc_1`, `Misc_10` |
| NUDT-SIRST  | `000001`, `000002`  |

## Sources

- IRSTD-1k: https://github.com/RuiZhang97/ISNet
- NUAA-SIRST: https://github.com/YimianDai/sirst
- NUDT-SIRST: https://github.com/YeRen123455/Infrared-Small-Target-Detection
