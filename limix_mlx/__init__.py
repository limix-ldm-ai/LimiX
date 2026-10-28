"""MLX backend for LimiX-2 (Apple silicon).

The forward pass runs on MLX while all preprocessing, ensembling and decoding
is inherited from the reference ``LimiXPredictor``. MLX is imported lazily, so
``import limix_mlx`` succeeds on platforms where MLX is unavailable.

    from limix_mlx import LimiXMLXPredictor
    clf = LimiXMLXPredictor(model_path="<LimiX-2.ckpt>")
"""

__all__ = ["LimiXMLXPredictor"]


def __getattr__(name):  # PEP 562: defer the MLX import to first use.
  if name in __all__:
    from limix_mlx.api import LimiXMLXPredictor
    return LimiXMLXPredictor
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
