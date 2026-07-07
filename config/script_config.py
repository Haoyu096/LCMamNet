import argparse
from types import SimpleNamespace


def _infer_arg_type(value):
    if isinstance(value, bool):
        return str
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


def _serialize_default(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return value


def _deserialize_value(raw, default):
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")
    if default is None:
        return raw
    try:
        return type(default)(raw)
    except Exception:
        return raw


def parse_script_args(defaults, descriptions=None, finalize=None):
    descriptions = descriptions or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_config", action="store_true")

    for key, value in defaults.items():
        parser.add_argument(
            f"--{key}",
            type=_infer_arg_type(value),
            default=None,
            help=descriptions.get(key, f"Override for {key}"),
        )

    parsed = parser.parse_args()
    merged = {}
    for key, value in defaults.items():
        override = getattr(parsed, key)
        merged[key] = value if override is None else _deserialize_value(override, value)

    # 让派生字段(如按 dataset 决定 norm_mode)在打印前定稿, 保证 --show_config 显示真实生效值。
    if finalize is not None:
        merged = finalize(merged)

    if parsed.show_config:
        print("Current script config:")
        for key, value in merged.items():
            print(f"  {key} = {value}")

    return SimpleNamespace(**merged)
