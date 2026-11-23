"""Architecture router. Model implementations live in model.v1_0 and model.v2_0."""
from ldm_limix.model.v1_0.transformer import FeaturesTransformer as FeaturesTransformer_V1_0
from ldm_limix.model.v2_0.transformer import FeaturesTransformer as FeaturesTransformer_V2_0
from ldm_limix.model.version import resolve_arch_line

FeaturesTransformer = FeaturesTransformer_V2_0

__all__ = [
    "FeaturesTransformer",
    "FeaturesTransformer_V1_0",
    "FeaturesTransformer_V2_0",
    "get_features_transformer_cls",
]


def get_features_transformer_cls(arch_version=None):
    if arch_version is None:
        return FeaturesTransformer_V2_0
    line = resolve_arch_line(arch_version)
    if line == "v1_0":
        return FeaturesTransformer_V1_0
    return FeaturesTransformer_V2_0
