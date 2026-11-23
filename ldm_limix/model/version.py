from packaging.version import InvalidVersion, Version

FROZEN_V1_VERSION = Version("1.0")
FROZEN_V2_VERSION = Version("2.0")

# Backward-compatible alias: v1_0-only frozen check.
FROZEN_VERSION = FROZEN_V1_VERSION


def parse_arch_version(value) -> Version:
    if isinstance(value, Version):
        return value
    text = str(value).strip().lstrip("vV")
    try:
        return Version(text)
    except InvalidVersion as exc:
        raise ValueError(f"Invalid arch_version: {value!r}") from exc


def is_frozen_version(ver) -> bool:
    """True for the V1.0 frozen line (arch_version < 2.0)."""
    return parse_arch_version(ver) < FROZEN_V2_VERSION


def resolve_arch_line(ver) -> str:
    v = parse_arch_version(ver)
    if v < FROZEN_V2_VERSION:
        return "v1_0"
    if v <= FROZEN_V2_VERSION:
        return "v2_0"
    raise ValueError(
        f"Unsupported arch_version={v}; this tree only supports V1.0 and V2.0"
    )


def resolve_arch_version(ckpt: dict | None = None, model_config: dict | None = None) -> Version:
    """Resolve architecture version from a checkpoint or model_config.

    Missing version on a checkpoint is treated as 1.0 (legacy weights).
    Missing version on a model_config is treated as 2.0.
    """
    if ckpt is not None:
        for src in (
            ckpt.get("arch_version"),
            (ckpt.get("config") or {}).get("arch_version"),
            (ckpt.get("config_info") or {}).get("arch_version"),
            ((ckpt.get("config_info") or {}).get("model_config") or {}).get("arch_version"),
        ):
            if src is not None and str(src).strip() != "":
                return parse_arch_version(src)
        return FROZEN_V1_VERSION

    if model_config is not None and model_config.get("arch_version") not in (None, ""):
        return parse_arch_version(model_config["arch_version"])
    return FROZEN_V2_VERSION
