# LimiX-2 on MLX (Apple silicon)

Native MLX port of [stable-ai/LimiX-2](https://huggingface.co/stable-ai/LimiX-2)
(400M-param tabular foundation model: classification, regression,
missing-value imputation), plus a drop-in sklearn-style predictor.

Same pattern as the TabFM MLX port: identical module/parameter names, so
weight conversion is mechanical and parity is verifiable layer by layer.

## Layout

| File | Purpose |
|---|---|
| `lxm_model.py` | MLX core blocks: RMSNorm, SoftmaxScalingMLP, sample attention (separate X/Y), decoupled structural-task attention (DStI), gated SiLU MLP, pre-norm `smf` layers |
| `lxm_top.py` | MLX encoders (mask-emb/numeric/fusion, cls/reg Y), parameter-free preprocessing, decoders, adapters, `FeaturesTransformer`, config-driven `build_model` |
| `convert.py` | torch `.ckpt` → `.npz` (no key remapping; `np.savez` — `mx.savez` caps kwargs at 1024) |
| `loader.py` | `build_model(config)` + `load()` with strict 1496-key check; `__main__` key-coverage smoke test |
| `api.py` | `LimiXMLXPredictor`: subclasses the reference `LimiXPredictor`, reuses ALL preprocessing/ensemble/decode logic, swaps only the forward backend to MLX |
| `shim.py` | `nvtx`/`triton` stubs so the reference torch code imports on Mac CPU (Triton norms already fall back to torch ops off-CUDA) |
| `parity.py` | raw-forward parity vs torch CPU (controls the per-forward pos-emb noise) |
| `e2e.py` | sklearn-dataset smoke test (breast cancer + diabetes) |
| `e2e_parity.py` | end-to-end torch-CPU vs MLX agreement |
| `bench_mps_vs_mlx.py` | head-to-head torch/MPS vs MLX timing (mirrors `bench_tabfm_mps_vs_mlx.py`) |
| `baseline_torch.py`, `introspect*.py` | dev utilities |

## Results (M4, this repo)

- Forward parity vs torch CPU (`parity.py`, max|diff| on raw outputs):
  cls **8.97e-05**, reg **1.81e-05**, imputation **7.33e-06**
- End-to-end vs torch CPU (`e2e_parity.py`, breast cancer, 120 train / 20 test,
  3-pipeline ensemble): same accuracy 0.95, **100% class agreement**,
  proba max|diff| 2.5e-03
- Bench vs torch/MPS (`bench_mps_vs_mlx.py`, 150 train / 30 test, cls, 1 warmup
  + median of 3): **MLX 3.52s vs MPS 15.74s → 4.47x**, shared preprocessing on
  both sides. Class agreement with the MPS run is 96.7% (proba max|diff|
  2.15e-02); that gap is MPS-vs-MLX and is unchanged by the fused kernels --
  the float32 CPU comparisons above are the parity reference, not MPS.
- Imputation: NaNs fully recovered, shape-preserving

Measured on an M3 Pro (Mac15,7) with the fused `mx.fast` kernels. Before
switching RMSNorm and the attention core to `mx.fast.rms_norm` /
`mx.fast.scaled_dot_product_attention`, the same benchmark ran 3.84s (4.02x),
with byte-identical class agreement.

## Usage

```python
from limix_mlx.api import LimiXMLXPredictor

clf = LimiXMLXPredictor(
    model_path="<LimiX-2.ckpt>",                       # HF download
    inference_config="<limix_src>/config/cls_default_noretrieval_v2.json",
    seed=0)
proba = clf.predict(X_train, y_train, X_test, task_type="Classification")
y_hat = clf.predict(X_train, y_train, X_test, task_type="Regression")
imp = clf.predict(X_train, y_train, X_test, task_type="Feature_imputation")
```

First run converts the ckpt once to `~/.cache/limix_mlx/LimiX-2.npz` (1.6 GB).

## Notes

- Everything runs float32 (matches the torch CPU baseline; MPS runs mixed precision, hence slightly larger diffs there).
- The per-forward random feature-positional code is drawn with `torch.randn` under the caller's seed and fed to both backends, so ensemble members match the reference exactly.
- `LimiX-2` weights are under the StableAI non-commercial license; the port code here follows the repo's existing conventions.
