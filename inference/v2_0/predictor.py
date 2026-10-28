from .feature_view_ensemble import predict_with_feature_views
from .infer_profile import (
    attach as _profile_attach,
    enabled as _profile_enabled,
    merge_worker as _profile_merge_worker,
    reset as _profile_reset,
    span as _profile_span,
)
from .pipeline_gpu_pool import (
    PipelineGpuWorkerPool,
    normalize_pipeline_gpu_ids,
    order_pipelines_longest_first,
    warn_uneven_pipeline_gpus,
    warn_unsupported_predictor_kwargs,
)
from .preprocess import (
    FeatureShuffler,
    FilterValidFeatures,
    CategoricalFeatureEncoder,
    RebalanceFeatureDistribution,
    FingerprintFeatureEncoder,
    PolynomialInteractionGenerator,
    DatetimePreprocessor,
    FixedFeatureViewPreprocessor,
    resolve_fixed_feature_view_config,
)
from utils.loading import load_model, load_from_checkpoint
from utils.categorical_encoding import FEATURE_ENCODER_MODES, encode_categorical_features
import nvtx
import copy
import torch
from dataclasses import dataclass
from typing import List, Literal
import random
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from itertools import chain, repeat
import pandas as pd
import einops
import json
import os
import torch.nn.functional as F
from .inference_utils import predict_mean_from_logits, translate_probs_across_borders, logits_to_output
from functools import partial
from .target_transform_preprocess import TargetTransform
import hashlib
import typing
import gc
import threading
import time


NA_PLACEHOLDER = "__MISSING__"
DATETIME_PREPROCESSING_KEY = "DatetimePreprocessing"

_DEFAULT_ADAPTIVE_SVD_CONFIG = {
    "enabled": False,
    "memory_fraction": 0.65,
    "reserve_mb": 512,
    "safety_factor": 1.25,
    "minimum_components": 0,
    "retry_on_cuda_resource_error": False,
    "retry_max_components": 0,
}


def _datetime_preprocessing_enabled(
    member_config: dict[str, typing.Any],
    member_index: int,
) -> bool:
    """Functionality: Validate and read whether a pipeline member enables train-only datetime preprocessing.

    Input:
        member_config: Config dict for one pipeline member.
        member_index: Member index in the pipelines list, used in error messages.

    Output:
        bool: True when enabled, False when unset. Raises TypeError/ValueError for invalid config.
    """
    config = member_config.get(DATETIME_PREPROCESSING_KEY)
    if config is None:
        return False
    if not isinstance(config, dict):
        raise TypeError(
            f"member {member_index} {DATETIME_PREPROCESSING_KEY} must be a mapping"
        )
    unknown = set(config) - {"enabled", "detection_scope", "fit_scope"}
    if unknown:
        raise ValueError(
            f"member {member_index} {DATETIME_PREPROCESSING_KEY} has unknown keys: "
            f"{sorted(unknown)}"
        )
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(
            f"member {member_index} {DATETIME_PREPROCESSING_KEY}.enabled must be bool"
        )
    detection_scope = config.get("detection_scope", "train")
    fit_scope = config.get("fit_scope", "train")
    if detection_scope != "train" or fit_scope != "train":
        raise ValueError(
            f"member {member_index} only supports train-only datetime preprocessing: "
            "detection_scope='train', fit_scope='train'"
        )
    return enabled


def _load_v2_inference_config(inference_config: dict | str):
    """Functionality: Load and validate a v2 inference config. The top-level object must contain a non-empty pipelines list.

    Input:
        inference_config: Config dict, or a path to a JSON file.

    Output:
        tuple[dict, list]: the full config object and its pipelines list.
    """
    if isinstance(inference_config, str):
        if os.path.isfile(inference_config):
            with open(inference_config, 'r') as f:
                inference_config = json.load(f)
        else:
            raise ValueError(f"inference_config is not a config file path: {inference_config}")
    if not isinstance(inference_config, dict) or "pipelines" not in inference_config:
        raise ValueError(
            "dev LimiXPredictor only accepts v2 inference config: a JSON object with "
            "a top-level 'pipelines' list (e.g. config/cls_default_noretrieval_v2.json). "
            f"Got type={type(inference_config).__name__}."
        )
    pipelines = inference_config["pipelines"]
    if not isinstance(pipelines, list) or len(pipelines) == 0:
        raise ValueError("Invalid configuration file! the number of pipelines is 0!")
    return inference_config, pipelines


def _copy_raw_features(x):
    """Functionality: Deep-copy the raw feature table so later preprocessing cannot mutate caller data.

    Input:
        x: pandas.DataFrame, or a table convertible to a 2-D numpy array.

    Output:
        A deep-copied DataFrame (index reset) or a numpy array copy.
    """
    if isinstance(x, pd.DataFrame):
        return x.copy(deep=True).reset_index(drop=True)
    return np.array(x, copy=True)


def _resolve_feature_encoder_mode(feature_encoder_mode: str) -> str:
    """Functionality: Resolve a categorical-encoder mode name to the implementation actually used.

    Input:
        feature_encoder_mode: Mode string; 'default' is mapped to 'current'.

    Output:
        str: a legal FEATURE_ENCODER_MODES value. Unknown modes raise ValueError.
    """
    if feature_encoder_mode == "default":
        return "current"
    if feature_encoder_mode not in FEATURE_ENCODER_MODES:
        raise ValueError(
            f"Unknown feature_encoder_mode={feature_encoder_mode!r}; "
            f"expected one of {FEATURE_ENCODER_MODES} or 'default'"
        )
    return feature_encoder_mode


def _resolve_adaptive_svd_config(inference_config: dict) -> dict:
    """Functionality: Merge defaults and validate the adaptive SVD runtime config.

    Input:
        inference_config: Full inference config; may contain an adaptive_svd object.

    Output:
        dict with enabled, memory_fraction, reserve_mb, safety_factor, minimum_components, retry_on_cuda_resource_error, retry_max_components.
    """
    raw = inference_config.get("adaptive_svd", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("adaptive_svd must be a JSON object")
    config = {**_DEFAULT_ADAPTIVE_SVD_CONFIG, **raw}
    if not isinstance(config["enabled"], bool):
        raise TypeError("adaptive_svd.enabled must be boolean")
    if not 0.0 < float(config["memory_fraction"]) <= 1.0:
        raise ValueError("adaptive_svd.memory_fraction must be in (0, 1]")
    if int(config["reserve_mb"]) < 0:
        raise ValueError("adaptive_svd.reserve_mb must be non-negative")
    if float(config["safety_factor"]) <= 0:
        raise ValueError("adaptive_svd.safety_factor must be positive")
    if int(config["minimum_components"]) < 0:
        raise ValueError("adaptive_svd.minimum_components must be non-negative")
    if not isinstance(config["retry_on_cuda_resource_error"], bool):
        raise TypeError(
            "adaptive_svd.retry_on_cuda_resource_error must be boolean"
        )
    if (
        isinstance(config["retry_max_components"], bool)
        or not isinstance(config["retry_max_components"], (int, np.integer))
        or int(config["retry_max_components"]) < 0
    ):
        raise ValueError(
            "adaptive_svd.retry_max_components must be a non-negative integer"
        )
    return {
        "enabled": config["enabled"],
        "memory_fraction": float(config["memory_fraction"]),
        "reserve_mb": int(config["reserve_mb"]),
        "safety_factor": float(config["safety_factor"]),
        "minimum_components": int(config["minimum_components"]),
        "retry_on_cuda_resource_error": config[
            "retry_on_cuda_resource_error"
        ],
        "retry_max_components": int(config["retry_max_components"]),
    }


def _resolve_feature_view_runtime_config(
    inference_config: dict,
) -> tuple[str, float, dict | None]:
    """Functionality: Parse the feature-view strategy, blend weight, and fixed feature-view config.

    Input:
        inference_config: Full inference config dict.

    Output:
        tuple[str, float, dict|None]: (strategy, weight, fixed_config). Raises on illegal strategy or out-of-range weight.
    """
    strategy = inference_config.get("feature_view_strategy", "disabled")
    supported = {
        "disabled",
        "independent_view_prediction_mean",
        "merged_feature_view",
    }
    if strategy not in supported:
        raise ValueError(
            f"feature_view_strategy must be one of {sorted(supported)}, got {strategy!r}"
        )
    weight = inference_config.get("feature_view_prediction_weight", 0.5)
    if isinstance(weight, bool) or not isinstance(
        weight, (int, float, np.integer, np.floating)
    ):
        raise TypeError("feature_view_prediction_weight must be a real number")
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("feature_view_prediction_weight must be in [0, 1]")
    fixed_config = (
        resolve_fixed_feature_view_config(inference_config["fixed_feature_views"])
        if "fixed_feature_views" in inference_config
        else None
    )
    return strategy, weight, fixed_config


_FROZEN_MODEL_CACHE = {}
_FROZEN_MODEL_CACHE_LOCK = threading.Lock()


def set_deterministic(seed=0):
    """Functionality: Pin Python/NumPy/PyTorch RNGs and enable deterministic algorithms so inference is reproducible.

    Input:
        seed: Integer random seed, default 0.

    Output:
        None. Mutates global RNG state and CUBLAS_WORKSPACE_CONFIG.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def _env_flag_enabled(name: str) -> bool:
    """Functionality: Parse an optional boolean environment flag. Unknown values are not silently ignored.

    Input:
        name: Environment variable name.

    Output:
        bool: unset is False. Allowed values are 1/true/yes/on or 0/false/no/off.
    """
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {value!r}")


def _frozen_model_cache_key(
    model_path: str,
    deterministic: bool = False,
) -> tuple[str, int, int, bool]:
    """Functionality: Build a process-local cache key from the model file's real path, size, mtime, and deterministic flag.

    Input:
        model_path: Model checkpoint path.
        deterministic: Model-build deterministic flag. Templates built with
            different values are not interchangeable.

    Output:
        tuple[str, int, int, bool]: (realpath, size, mtime_ns, deterministic).
    """
    resolved_path = os.path.realpath(model_path)
    stat = os.stat(resolved_path)
    return resolved_path, stat.st_size, stat.st_mtime_ns, bool(deterministic)


def _load_model_for_predictor(
    model_path: str,
    reuse_frozen_model: bool,
    ckpt: dict | None = None,
    deterministic: bool = False,
):
    """Functionality: Load a LimiX model. Optionally reuse a process-local frozen template and deepcopy an isolated instance.

    Input:
        model_path: Model path.
        reuse_frozen_model: If True, use the process-level template cache.
        ckpt: Optional already-loaded checkpoint; skips torch.load when provided.
        deterministic: Whether to build the model in deterministic mode.

    Output:
        tuple[model, model_config].
    """
    def _load_once():
        """Functionality: Perform one actual model load from a checkpoint or disk.

        Input:
            None: Closure over outer model_path, ckpt, and deterministic.

        Output:
            tuple[model, model_config].
        """
        if ckpt is not None:
            return load_from_checkpoint(
                ckpt, mask_prediction=False, deterministic=deterministic
            )
        return load_model(model_path=model_path, deterministic=deterministic)

    if not reuse_frozen_model:
        return _load_once()

    cache_key = _frozen_model_cache_key(model_path, deterministic)
    with _FROZEN_MODEL_CACHE_LOCK:
        cached = _FROZEN_MODEL_CACHE.get(cache_key)
        if cached is None:
            started = time.perf_counter()
            model_template, model_config = _load_once()
            cached = (model_template, copy.deepcopy(model_config))
            _FROZEN_MODEL_CACHE[cache_key] = cached
            print(
                "Loaded frozen LimiX CPU template into the process-local cache in "
                f"{time.perf_counter() - started:.3f}s: {cache_key[0]}"
            )
    model_template, model_config = cached
    # A live model has mutable inference-time state outside state_dict.  Each Predictor
    # therefore gets an isolated clone while checkpoint I/O and construction are reused.
    return copy.deepcopy(model_template), copy.deepcopy(model_config)


@dataclass(frozen=True)
class PreparedClassificationMember:
    """Functionality: CPU-side preprocessing result for one no-retrieval classification ensemble member.

    Input:
        x: Numeric train+query features for this member, 2-D numpy array.
        y: Train labels already permuted by class_permutation, 1-D array.
        class_permutation: Permutation that maps this member's output columns back to the original class order.

    Output:
        Frozen dataclass instance consumed by infer_prepared_cls.
    """

    x: np.ndarray
    y: np.ndarray
    class_permutation: np.ndarray


@dataclass(frozen=True)
class PreparedClassificationInference:
    """Functionality: Complete CPU preprocessing output for one classification call, ready for a GPU thread.

    Input:
        members: Tuple of PreparedClassificationMember, length equal to the ensemble size.
        train_size: Number of training rows.
        n_classes: Number of classes.

    Output:
        Frozen dataclass instance.
    """

    members: tuple[PreparedClassificationMember, ...]
    train_size: int
    n_classes: int

class LimiXPredictor:
    """Functionality: LimiX inferencer for classification, regression, and missing-value prediction. Applies v2 pipeline preprocessing and ensembling.

    Input:
        See __init__: Device, model path, inference config, decoder and batching options.

    Output:
        The class has no return value. The public entry point is predict().
    """
    def __init__(self, 
                 device:torch.device, 
                 model_path:str, 
                 inference_config: dict|str,
                 mix_precision:bool=True,
                 outlier_remove_std: float=12,
                 softmax_temperature:float=0.9,
                 average_before_softmax: bool = True,
                 categorical_features_indices:List[int]|None=None,
                 inference_with_DDP: bool = False,
                 use_data_cache: bool = False,
                 seed:int=0,
                 regression_decoder_type: Literal["mse", "bucket", "bucket_4_tabpfn", "hierarchical"] = 'mse',
                 test_batch_mode: Literal["full_first", "fixed"] = "full_first",
                 test_batch_size: int | None = 16384,
                 ckpt: dict | None = None,
                 deterministic: bool = False,
                 enable_preprocess_parallel: bool = True,
                 preprocess_num_jobs: int = 16,
                 gpu_ids: List[int] | None = None,
                 cache_dir: str="/tmp",
                 **kwargs,
                 ):
        """Functionality: Initialize the predictor: load v2 config, build preprocess pipelines, and load the model.

        Input:
            device: Inference torch.device. CUDA is recommended; CPU disables mixed precision.
            model_path: Filesystem path to the checkpoint (.ckpt / .pt).
            inference_config: v2 config dict with a non-empty top-level 'pipelines'
                list, or a path to such a JSON file.
            mix_precision: Use autocast on GPU. Forced False on CPU.
            outlier_remove_std: Std-dev multiplier for outlier clipping in preprocess.
            softmax_temperature: Temperature applied to classification/bucket logits.
                Must be > 0.
            average_before_softmax: For bucket regression, average members in log-prob
                space before softmax when True.
            categorical_features_indices: Optional list of categorical column indices.
                Unused on the main classification/regression path.
            inference_with_DDP: Deprecated. Must stay False; DDP inference was removed.
            use_data_cache: If True, cache per-member preprocess results on disk.
            seed: RNG seed for shuffling, preprocess, and torch generators.
            regression_decoder_type: 'mse', 'bucket', 'bucket_4_tabpfn', or 'hierarchical'.
            test_batch_mode: 'full_first' tries the full query then falls back;
                'fixed' always caps at test_batch_size.
            test_batch_size: Max query rows per forward, or None for the full query.
                Must be > 0 when not None.
            ckpt: Optional already-loaded checkpoint dict; skips torch.load when set.
            deterministic: Pin RNGs and enable deterministic algorithms.
            enable_preprocess_parallel: Allow CPU-parallel QTx/SVD preprocess.
            preprocess_num_jobs: CPU worker count for parallel preprocess. Default 16.
            gpu_ids: Optional CUDA indices for pipeline-parallel ensemble members.
                None or length 1 is single-GPU. Length >= 2 spawns a worker pool.
            cache_dir: Cache directory path
            **kwargs: Unknown names are warned and ignored (forward compatibility).

        Output:
            None. Loads weights onto CPU and builds preprocess_pipelines.
            GPU H2D happens on the first predict() / member forward.
        """
        warn_unsupported_predictor_kwargs(kwargs)
        # Initialize the cache manager
        self.cache_manager = self.CacheManager(cache_dir=cache_dir)
        self.use_data_cache = use_data_cache

        inference_config, inference_pipeline_config = _load_v2_inference_config(inference_config)
        self.model_path = model_path
        self.device = device
        self.mix_precision = mix_precision
        self.categorical_features_indices = categorical_features_indices
        self.seed = seed
        if deterministic:
            set_deterministic(seed)
        self.inference_config = inference_config
        self.inference_pipeline_config = inference_pipeline_config
        self.n_estimators = len(self.inference_pipeline_config)
        self.datetime_preprocessing_enabled = tuple(
            _datetime_preprocessing_enabled(config, index)
            for index, config in enumerate(self.inference_pipeline_config)
        )
        self.model = None
        self.outlier_remove_std = outlier_remove_std
        self.class_shuffle_factor = 3
        self.min_seq_len_for_category_infer = 100
        self.max_unique_num_for_category_infer = 30
        self.min_unique_num_for_numerical_infer = 4
        self.preprocess_num = 10
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        if inference_with_DDP:
            print("Warning: inference_with_DDP has been removed; DDP inference is no longer supported")
        self.regression_decoder_type = regression_decoder_type
        if test_batch_mode not in {"full_first", "fixed"}:
            raise ValueError(f"unsupported test_batch_mode: {test_batch_mode}")
        if test_batch_size is not None and test_batch_size <= 0:
            raise ValueError("test_batch_size must be positive or None")
        self.test_batch_mode = test_batch_mode
        self.test_batch_size = test_batch_size
        self.reuse_frozen_model = _env_flag_enabled("LIMIX_REUSE_FROZEN_MODEL")
        self._reject_retrieval_config()
        if device.type == 'cpu':
            self.mix_precision = False
            print("Mixed precision is not supported for CPU inference, so it has been automatically disabled")
        self.enable_preprocess_parallel = enable_preprocess_parallel
        self.preprocess_num_jobs = preprocess_num_jobs
        self.deterministic = deterministic
        self.gpu_ids = normalize_pipeline_gpu_ids(gpu_ids)
        if self.gpu_ids is not None and len(self.gpu_ids) == 1:
            gpu_id = self.gpu_ids[0]
            if isinstance(device, torch.device) and device.type == "cuda":
                self.device = torch.device("cuda", gpu_id)
        self._pipeline_gpu_pool = None
        self._pipeline_indices = None
        self._pipeline_worker_init_kwargs = {
            "model_path": model_path,
            "inference_config": inference_config,
            "mix_precision": mix_precision,
            "outlier_remove_std": outlier_remove_std,
            "softmax_temperature": softmax_temperature,
            "average_before_softmax": average_before_softmax,
            "categorical_features_indices": categorical_features_indices,
            "inference_with_DDP": False,
            "use_data_cache": use_data_cache,
            "seed": seed,
            "regression_decoder_type": regression_decoder_type,
            "test_batch_mode": test_batch_mode,
            "test_batch_size": test_batch_size,
            "deterministic": deterministic,
            "enable_preprocess_parallel": enable_preprocess_parallel,
            "preprocess_num_jobs": preprocess_num_jobs,
        }
        if self.gpu_ids is not None and len(self.gpu_ids) >= 2:
            warn_uneven_pipeline_gpus(self.n_estimators, self.gpu_ids)

        with nvtx.annotate('model-load'):
            self.model, self.model_config = _load_model_for_predictor(
                model_path=model_path,
                reuse_frozen_model=self.reuse_frozen_model,
                ckpt=ckpt,
                deterministic=deterministic,
            )

        if self.model_config['num_buckets'] > 1:
            self.regression_decoder_type = 'bucket'

        self.preprocess_pipelines = []
        self.preprocess_configs = []

        self.build_preprocess_pipeline()
        (
            self.feature_view_strategy,
            self.feature_view_prediction_weight,
            self.fixed_feature_view_config,
        ) = _resolve_feature_view_runtime_config(inference_config)
        # Retain the legacy feature-view knobs for callers that inspect or
        # serialize predictor state, even though fixed views do not consume
        # target-aware routing or cross-fitting.
        self.route_sampling_seed = inference_config.get("route_sampling_seed", self.seed)
        self.cross_fit_seed = inference_config.get("cross_fit_seed", self.seed)
        self.feature_view_ensemble_audit = None
        self.feature_encoder_mode = _resolve_feature_encoder_mode(
            inference_config.get("feature_encoder_mode", "current")
        )
        self.adaptive_svd_config = _resolve_adaptive_svd_config(inference_config)
        self.svd_adaptation_audit = []
        self.svd_runtime_retry_audit = []
        self.polynomial_interaction_runtime_retry_audit = []
        self.cuda_pipeline_fallback_audit = []
        self._adaptive_svd_retry_component_cap = None
        self._svd_inference_call_index = 0
        self._svd_runtime_cache_signature = "adaptive_svd_unconfigured"

        # seeds = None
        self.seeds_hash = self.get_and_set_seeds(self.seed)

    def __getstate__(self):
        """Functionality: Customize pickle state so reconstructible frozen-model weights are not pickled into AutoGluon children.

        Input:
            self: Current predictor.

        Output:
            dict: a copy of __dict__. When reuse_frozen_model is set, model is None.
        """
        state = self.__dict__.copy()
        state["_pipeline_gpu_pool"] = None
        if state.get("reuse_frozen_model", False):
            state["model"] = None
        return state

    def __setstate__(self, state):
        """Functionality: Restore instance fields from pickle state and re-apply process-global deterministic settings.

        Input:
            state: Dict produced by __getstate__.

        Output:
            None. The model may be rebuilt lazily by _ensure_model_loaded.
        """
        self.__dict__.update(state)
        if getattr(self, "deterministic", False):
            set_deterministic(getattr(self, "seed", 0))

    def _ensure_model_loaded(self) -> None:
        """Functionality: In frozen-model reuse mode, clone an isolated model from the CPU template when model is missing.

        Input:
            self: Current predictor.

        Output:
            None. Raises RuntimeError if the model is missing outside reuse mode.
        """
        if self.model is not None:
            return
        if not self.reuse_frozen_model:
            raise RuntimeError("LimiX model is missing outside frozen-model reuse mode")
        self.model, _ = _load_model_for_predictor(
            model_path=self.model_path,
            reuse_frozen_model=True,
            deterministic=getattr(self, "deterministic", False),
        )

    def close(self) -> None:
        """Functionality: Shut down pipeline-parallel GPU workers if they were started and release GPU tensors.

        Input:
            self.

        Output:
            None.
        """
        pool = getattr(self, "_pipeline_gpu_pool", None)
        if pool is not None:
            pool.close()
            self._pipeline_gpu_pool = None
        if getattr(self, "model", None) is not None:
            self.model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _use_pipeline_gpu_pool(self) -> bool:
        """Functionality: Whether this predictor should shard ensemble pipelines across GPU workers.

        Input:
            self.

        Output:
            bool.
        """
        gpu_ids = getattr(self, "gpu_ids", None)
        return gpu_ids is not None and len(gpu_ids) >= 2

    def _ensure_pipeline_gpu_pool(self) -> PipelineGpuWorkerPool:
        """Functionality: Lazily spawn one worker process per GPU used for pipeline parallel.

        Input:
            self.

        Output:
            PipelineGpuWorkerPool.
        """
        if self._pipeline_gpu_pool is None:
            if getattr(self, "model", None) is not None:
                self.model.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from model.v2_0.autobatch import AutobatchConfig

            worker_init_kwargs = dict(self._pipeline_worker_init_kwargs)
            worker_init_kwargs["_enable_autobatch"] = bool(
                AutobatchConfig.ENABLE_AUTOBATCH
            )
            self._pipeline_gpu_pool = PipelineGpuWorkerPool(
                gpu_ids=self.gpu_ids,
                worker_init_kwargs=worker_init_kwargs,
                n_pipelines=self.n_estimators,
            )
        return self._pipeline_gpu_pool

    def _sync_pipeline_gpu_workers_inference_config(self) -> None:
        """Functionality: Keep lazy worker kwargs and a live GPU pool on the current inference config.

        Input:
            self: After parent set_inference_config has rebuilt local pipelines.

        Output:
            None. Workers call set_inference_config themselves; they are not respawned.
        """
        worker_kwargs = getattr(self, "_pipeline_worker_init_kwargs", None)
        if worker_kwargs is not None:
            worker_kwargs["inference_config"] = self.inference_config
            worker_kwargs["softmax_temperature"] = self.softmax_temperature
            worker_kwargs["seed"] = self.seed
        gpu_ids = getattr(self, "gpu_ids", None)
        if gpu_ids is not None and len(gpu_ids) >= 2:
            warn_uneven_pipeline_gpus(self.n_estimators, gpu_ids)
        pool = getattr(self, "_pipeline_gpu_pool", None)
        if pool is None:
            return
        pool.set_inference_config(
            inference_config=self.inference_config,
            softmax_temperature=self.softmax_temperature,
            seed=self.seed,
            n_pipelines=self.n_estimators,
        )

    def _active_pipeline_indices(self):
        """Functionality: Pipeline indices this process should run.

        Input:
            self.

        Output:
            iterable of int. Workers with a steal queue yield one index at a time.
        """
        source = getattr(self, "_pipeline_index_source", None)
        if source is not None:
            return source()
        indices = getattr(self, "_pipeline_indices", None)
        if indices is None:
            return list(range(len(self.preprocess_pipelines)))
        return list(indices)

    def _pipeline_gpu_retry_state(self) -> dict:
        """Functionality: Export the dataset-scoped SVD retry cap so GPU workers stay aligned with single-GPU predict().

        Input:
            self.

        Output:
            dict with retry_cap, call_index, and pipelines that already disabled
            polynomial interactions. Memory budgets are not included; each worker
            re-queries its own GPU in _configure_adaptive_svd_runtime.
        """
        disabled_pipelines = sorted(
            {
                int(record["pipeline_index"])
                for record in getattr(
                    self, "polynomial_interaction_runtime_retry_audit", []
                )
                if record.get("pipeline_index") is not None
            }
        )
        return {
            "retry_cap": getattr(self, "_adaptive_svd_retry_component_cap", None),
            "call_index": int(getattr(self, "_svd_inference_call_index", 0) or 0),
            "polynomial_disabled_pipelines": disabled_pipelines,
        }

    def _export_runtime_audit(self) -> dict:
        """Functionality: Copy this process's SVD adaptation and feature-width retry audits.

        Input:
            self.

        Output:
            dict consumed by merge_pipeline_runtime_audits.
        """
        return {
            "svd_adaptation_audit": copy.deepcopy(
                getattr(self, "svd_adaptation_audit", [])
            ),
            "svd_runtime_retry_audit": copy.deepcopy(
                getattr(self, "svd_runtime_retry_audit", [])
            ),
            "polynomial_interaction_runtime_retry_audit": copy.deepcopy(
                getattr(self, "polynomial_interaction_runtime_retry_audit", [])
            ),
            "cuda_pipeline_fallback_audit": copy.deepcopy(
                getattr(self, "cuda_pipeline_fallback_audit", [])
            ),
            "retry_cap": getattr(self, "_adaptive_svd_retry_component_cap", None),
            "call_index": int(getattr(self, "_svd_inference_call_index", 0) or 0),
        }

    def _ingest_pipeline_runtime_audit(self, audit: dict | None) -> None:
        """Functionality: Append worker SVD / retry audits onto this predictor, matching one single-GPU predict().

        Input:
            audit: Merged worker audit, or None.

        Output:
            None. Extends the parent audit lists and keeps any dataset-wide SVD cap.
        """
        if not audit:
            return
        self.svd_adaptation_audit.extend(audit.get("svd_adaptation_audit") or [])
        self.svd_runtime_retry_audit.extend(audit.get("svd_runtime_retry_audit") or [])
        self.polynomial_interaction_runtime_retry_audit.extend(
            audit.get("polynomial_interaction_runtime_retry_audit") or []
        )
        self.cuda_pipeline_fallback_audit.extend(
            audit.get("cuda_pipeline_fallback_audit") or []
        )
        retry_cap = audit.get("retry_cap")
        if retry_cap is not None:
            self._adaptive_svd_retry_component_cap = retry_cap
        call_index = int(audit.get("call_index") or 0)
        if call_index > int(getattr(self, "_svd_inference_call_index", 0) or 0):
            self._svd_inference_call_index = call_index

    def _submit_pipeline_gpu_collect(self, payload: dict) -> dict:
        """Functionality: Submit one collect to GPU workers, pass the current SVD retry cap, and ingest their audits.

        Input:
            payload: Task fields for PipelineGpuWorkerPool.submit, without IPC bookkeeping.

        Output:
            dict from submit(), after worker audits are copied onto this predictor.
        """
        message = dict(payload)
        message["svd_retry_state"] = self._pipeline_gpu_retry_state()
        message["pipeline_order"] = order_pipelines_longest_first(
            self.inference_pipeline_config
        )
        result = self._ensure_pipeline_gpu_pool().submit(message)
        self._ingest_pipeline_runtime_audit(result.get("runtime_audit"))
        for gpu_id, profile in (result.get("worker_profiles") or {}).items():
            _profile_merge_worker(gpu_id, profile)
        return result

    def _prepare_worker_collect_state(self, svd_retry_state: dict | None) -> None:
        """Functionality: Reset per-collect runtime fields on a GPU worker, then restore the parent's dataset SVD retry cap.

        Input:
            svd_retry_state: retry_cap, call_index, and polynomial_disabled_pipelines from the parent, or None.

        Output:
            None. Does not freeze a parent CUDA-memory snapshot; configure still queries this GPU.
        """
        self.feature_view_ensemble_audit = None
        self.svd_adaptation_audit = []
        self.svd_runtime_retry_audit = []
        self.polynomial_interaction_runtime_retry_audit = []
        self.cuda_pipeline_fallback_audit = []
        state = svd_retry_state or {}
        self._adaptive_svd_retry_component_cap = state.get("retry_cap")
        self._svd_inference_call_index = int(state.get("call_index") or 0)
        disabled_pipelines = {
            int(pipeline_index)
            for pipeline_index in (state.get("polynomial_disabled_pipelines") or [])
        }
        for pipeline_index, pipeline in enumerate(
            getattr(self, "preprocess_pipelines", ())
        ):
            enabled = pipeline_index not in disabled_pipelines
            for step in pipeline:
                if isinstance(step, PolynomialInteractionGenerator):
                    step.set_runtime_interactions_enabled(enabled)

    def _worker_collect_members(
        self,
        task_type: str,
        x_train=None,
        y_train=None,
        x_test=None,
        unique_dataset_name=None,
        svd_retry_state=None,
        prepared=None,
        member_inputs=None,
    ) -> dict:
        """Functionality: Worker-side collect of per-pipeline outputs without ensemble.

        Input:
            task_type: Classification, Regression, Feature_imputation, or prepared_cls.
            x_train/y_train/x_test: Raw arrays for the standard tasks.
            unique_dataset_name: Optional cache/audit identity.
            svd_retry_state: Parent dataset-scoped SVD retry cap, or None.
            prepared: PreparedClassificationInference for prepared_cls.
            member_inputs: Host-prepared per-pipeline numeric inputs for Regression.

        Output:
            dict with members as (pipeline_index, output) pairs plus runtime_audit.
        """
        self._ensure_model_loaded()
        self._prepare_worker_collect_state(svd_retry_state)
        if _profile_enabled():
            _profile_reset()
        if task_type == "Classification":
            members, last_failure = self._collect_cls_member_outputs(
                x_train, y_train, x_test, task_type, unique_dataset_name
            )
            result = {"members": members, "last_failure": last_failure}
        elif task_type == "Regression":
            result = self._collect_reg_member_outputs(
                x_train,
                y_train,
                x_test,
                task_type,
                unique_dataset_name,
                member_inputs=member_inputs,
            )
        elif task_type == "Feature_imputation":
            result = {
                "members": self._collect_imputation_member_outputs(
                    x_train, y_train, x_test, task_type, unique_dataset_name
                )
            }
        elif task_type == "prepared_cls":
            result = {
                "members": self._collect_prepared_cls_member_outputs(prepared)
            }
        else:
            raise ValueError(f"Unsupported worker task_type: {task_type}")
        result["runtime_audit"] = self._export_runtime_audit()
        return _profile_attach(result)

    def _ensemble_cls_outputs(self, member_outputs) -> np.ndarray:
        """Functionality: Softmax and equally average classification member logits in pipeline order.

        Input:
            member_outputs: Sequence of (pipeline_index, logits) pairs.

        Output:
            np.ndarray: row-normalized query-by-class probabilities.
        """
        outputs = [
            output
            for _, output in sorted(member_outputs, key=lambda item: item[0])
        ]
        outputs = [
            torch.nn.functional.softmax(output.float().cpu(), dim=1)
            for output in outputs
        ]
        output = torch.stack(outputs).mean(dim=0)
        output = output.float().cpu().numpy()
        return output / output.sum(axis=1, keepdims=True)

    def _raise_all_cls_pipelines_failed(self, last_cuda_pipeline_failure) -> None:
        """Functionality: Raise the historical all-pipeline CUDA resource failure.

        Input:
            last_cuda_pipeline_failure: Last failure record, or None.

        Output:
            None. Always raises RuntimeError.
        """
        final_reason = (last_cuda_pipeline_failure or {}).get("reason", "unknown")
        final_error_label = (
            "CUDA out of memory"
            if final_reason == "out_of_memory"
            else "CUDA error: invalid configuration argument"
        )
        raise RuntimeError(
            "All classification ensemble pipelines failed with CUDA resource "
            f"errors for the current dataset ({final_error_label}): "
            f"{last_cuda_pipeline_failure}"
        )

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        """Functionality: Return whether the exception is a CUDA OOM that can be retried with a smaller query batch.

        Input:
            error: Caught exception.

        Output:
            bool.
        """
        return LimiXPredictor._cuda_batch_retry_reason(error) == "out_of_memory"

    @staticmethod
    def _cuda_batch_retry_reason(error: BaseException) -> str | None:
        """Functionality: Classify CUDA failures that may be resolved by shrinking the query batch.

        Input:
            error: Caught exception and its cause/context chain.

        Output:
            str | None: 'out_of_memory', 'invalid_configuration', or None.
        """
        oom_error_type = getattr(torch.cuda, "OutOfMemoryError", ())
        current: BaseException | None = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, oom_error_type):
                return "out_of_memory"
            if isinstance(current, RuntimeError):
                message = str(current).lower()
                if (
                    "out of memory" in message
                    or "retryable cuda out_of_memory" in message
                ):
                    return "out_of_memory"
                if (
                    "cuda error: invalid configuration argument" in message
                    or "cudaerrorinvalidconfiguration" in message
                    or "retryable cuda invalid_configuration" in message
                ):
                    return "invalid_configuration"
            current = current.__cause__ or current.__context__
        return None

    @staticmethod
    def _release_cuda_after_resource_error(error: BaseException | None = None) -> None:
        """Functionality: Drop exception-held traceback references and release reclaimable CUDA cache.

        Input:
            error: Optional exception object.

        Output:
            None.
        """
        current = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            current.__traceback__ = None
            current = current.__cause__ or current.__context__
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _detach_to_cpu(output):
        """Functionality: Recursively detach nested model outputs to CPU so completed query batches free VRAM.

        Input:
            output: Tensor, or a nested list/tuple/dict.

        Output:
            A CPU structure isomorphic to the input. Non-tensor leaves are unchanged.
        """
        if torch.is_tensor(output):
            return output.detach().cpu()
        if isinstance(output, list):
            return [LimiXPredictor._detach_to_cpu(item) for item in output]
        if isinstance(output, tuple):
            return tuple(LimiXPredictor._detach_to_cpu(item) for item in output)
        if isinstance(output, dict):
            return {key: LimiXPredictor._detach_to_cpu(value) for key, value in output.items()}
        return output

    @staticmethod
    def _concat_query_outputs(outputs, sample_dim: int):
        """Functionality: Concatenate nested query-batch outputs along the sample dimension.

        Input:
            outputs: Non-empty list of batch outputs; structures must match.
            sample_dim: Concatenation dimension.

        Output:
            Concatenated Tensor, or an isomorphic list/tuple/dict.
        """
        if not outputs:
            raise ValueError("Cannot concatenate an empty list of query outputs")
        first = outputs[0]
        if torch.is_tensor(first):
            return torch.cat(outputs, dim=sample_dim)
        if isinstance(first, list):
            if any(len(output) != len(first) for output in outputs):
                raise ValueError("Query batches returned lists with different lengths")
            return [
                LimiXPredictor._concat_query_outputs([output[idx] for output in outputs], sample_dim)
                for idx in range(len(first))
            ]
        if isinstance(first, tuple):
            if any(len(output) != len(first) for output in outputs):
                raise ValueError("Query batches returned tuples with different lengths")
            return tuple(
                LimiXPredictor._concat_query_outputs([output[idx] for output in outputs], sample_dim)
                for idx in range(len(first))
            )
        if isinstance(first, dict):
            if any(output.keys() != first.keys() for output in outputs):
                raise ValueError("Query batches returned dictionaries with different keys")
            return {
                key: LimiXPredictor._concat_query_outputs([output[key] for output in outputs], sample_dim)
                for key in first
            }
        raise TypeError(f"Unsupported batched output type: {type(first)}")

    def _predict_noretrieval_in_batches(
        self,
        x: np.ndarray,
        y: np.ndarray,
        train_size: int,
        task_type: str,
        parse_output,
        sample_dim: int,
    ):
        """Functionality: Run no-retrieval inference with a fixed training context, splitting only query rows. Retryable CUDA errors shrink the batch.

        Input:
            x: Preprocessed train+query features, 2-D float array,
                shape (n_train + n_query, n_features).
            y: Train labels for this member, shape (n_train,). Length must equal train_size.
            train_size: Number of training rows; also eval_pos. Must be < len(x).
            task_type: Forward task string, e.g. 'Classification' or 'Regression'.
            parse_output: Callable mapping one model output to a concatenable tensor/array.
            sample_dim: Axis along which query chunks are concatenated.

        Output:
            Concatenated parsed output along sample_dim. Writes
            self.last_query_batch_diagnostics. First call also moves the model to GPU.
        """
        test_size = len(x) - train_size
        if test_size <= 0:
            raise ValueError("x must contain at least one test/query row")

        test_batch_mode = getattr(self, "test_batch_mode", "fixed")
        configured_batch_size = self.test_batch_size or test_size
        if test_batch_mode == "full_first" and self.test_batch_size is None:
            # Native Stage E uses a dataset-relative fallback so large query sets
            # do not jump directly to a hardware-specific constant after the
            # complete-query attempt.  The fixed training context is unchanged.
            fallback_batch_size = max(1, (test_size + 1) // 2)
        else:
            fallback_batch_size = min(configured_batch_size, test_size)
        active_batch_size = (
            test_size if test_batch_mode == "full_first" else fallback_batch_size
        )
        self.last_query_batch_diagnostics = {
            "test_batch_mode": test_batch_mode,
            "configured_test_batch_size": int(configured_batch_size),
            "fallback_test_batch_size": int(fallback_batch_size),
            "full_test_attempted": test_batch_mode == "full_first",
            "full_test_succeeded": False,
            "full_test_oom": False,
            "full_test_invalid_configuration": False,
            "minimum_test_batch_size": None,
            "num_query_batches": 0,
            "num_oom_retries": 0,
            "num_invalid_configuration_retries": 0,
            "num_cuda_batch_retries": 0,
            "train_rows": int(train_size),
            "test_rows": int(test_size),
        }
        if active_batch_size < test_size:
            print(
                f"Test/query batching enabled: {test_size} rows in batches of "
                f"at most {active_batch_size}; train rows remain fixed at {train_size}."
            )

        with nvtx.annotate('model-h2d'):
            self.model.to(self.device)
        with nvtx.annotate('build-y'):
            y_device = torch.from_numpy(np.asarray(y)).float().to(self.device).unsqueeze(0)
        completed_outputs = []
        start = 0
        while start < test_size:
            current_batch_size = min(active_batch_size, test_size - start)
            stop = start + current_batch_size
            x_device = None
            model_output = None
            parsed_output = None
            try:
                with nvtx.annotate('build-x'):
                    x_batch = np.concatenate(
                        [x[:train_size], x[train_size + start:train_size + stop]],
                        axis=0,
                    )
                    x_device = torch.from_numpy(x_batch).float().to(self.device).unsqueeze(0)
                # Reset for every forward so models with random class mappings
                # use the same mapping for all query chunks.
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)

                with (
                    nvtx.annotate('model-infer'),
                    torch.autocast(
                        device_type=self.device.type if isinstance(self.device, torch.device) else self.device,
                        enabled=self.mix_precision,
                    ),
                    torch.inference_mode(),
                ):
                    model_output = self.model(
                        x=x_device,
                        y=y_device,
                        eval_pos=train_size,
                        task_type=task_type,
                    )
                    if os.environ.get("LDM_FWD_DEBUG") == "1" and start == 0:
                        from model.v2_0.autobatch import AutobatchConfig

                        try:
                            ac_dtype = torch.get_autocast_dtype("cuda")
                        except Exception:
                            ac_dtype = torch.get_autocast_gpu_dtype()
                        free_b, total_b = (0, 0)
                        if torch.cuda.is_available() and self.device.type == "cuda":
                            free_b, total_b = torch.cuda.mem_get_info(self.device)
                        print(
                            "FWDDBG "
                            f"pid={os.getpid()} vis={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                            f"ndev={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
                            f"uuid={getattr(torch.cuda.get_device_properties(self.device), 'uuid', None) if self.device.type == 'cuda' else None} "
                            f"dev={self.device} "
                            f"name={torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'cpu'} "
                            f"mix={self.mix_precision} autobatch={AutobatchConfig.ENABLE_AUTOBATCH} "
                            f"autocast={torch.is_autocast_enabled()} ac_dtype={ac_dtype} "
                            f"x={tuple(x_device.shape)} xdtype={x_device.dtype} "
                            f"alloc={torch.cuda.memory_allocated(self.device) / 1024**3:.2f}GiB "
                            f"reserved={torch.cuda.memory_reserved(self.device) / 1024**3:.2f}GiB "
                            f"free={free_b / 1024**3:.2f}/{total_b / 1024**3:.2f}GiB",
                            flush=True,
                        )
                    parsed_output = parse_output(model_output)

                completed_outputs.append(self._detach_to_cpu(parsed_output))
                self.last_query_batch_diagnostics["num_query_batches"] += 1
                if (
                    test_batch_mode == "full_first"
                    and start == 0
                    and current_batch_size == test_size
                ):
                    self.last_query_batch_diagnostics["full_test_succeeded"] = True
                minimum_batch_size = self.last_query_batch_diagnostics["minimum_test_batch_size"]
                if minimum_batch_size is None or current_batch_size < minimum_batch_size:
                    self.last_query_batch_diagnostics["minimum_test_batch_size"] = int(current_batch_size)
                start = stop
            except Exception as error:
                retry_reason = self._cuda_batch_retry_reason(error)
                if retry_reason is None:
                    raise
                x_device = None
                model_output = None
                parsed_output = None
                error.__traceback__ = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                full_test_attempt = (
                    test_batch_mode == "full_first"
                    and start == 0
                    and current_batch_size == test_size
                )
                if full_test_attempt:
                    if retry_reason == "out_of_memory":
                        self.last_query_batch_diagnostics["full_test_oom"] = True
                    else:
                        self.last_query_batch_diagnostics[
                            "full_test_invalid_configuration"
                        ] = True
                if current_batch_size == 1:
                    if retry_reason == "out_of_memory":
                        detail = "does not fit in GPU memory"
                    else:
                        detail = "still exceeds a CUDA kernel launch limit"
                    raise RuntimeError(
                        f"Retryable CUDA {retry_reason} error with test_batch_size=1; "
                        f"the fixed training context ({train_size} rows) plus one "
                        f"query row {detail}. Reduce the training context or feature "
                        "width, or switch the offending CUDA kernel backend."
                    ) from error
                retry_counter = (
                    "num_oom_retries"
                    if retry_reason == "out_of_memory"
                    else "num_invalid_configuration_retries"
                )
                self.last_query_batch_diagnostics[retry_counter] += 1
                self.last_query_batch_diagnostics["num_cuda_batch_retries"] += 1
                error_label = (
                    "CUDA OOM"
                    if retry_reason == "out_of_memory"
                    else "CUDA invalid launch configuration"
                )
                if full_test_attempt:
                    active_batch_size = fallback_batch_size
                    if active_batch_size >= current_batch_size:
                        active_batch_size = max(1, current_batch_size // 2)
                    print(
                        f"{error_label} during full-test inference; switching to test/query "
                        f"batching with test_batch_size={active_batch_size}."
                    )
                else:
                    if test_batch_mode == "full_first" and self.test_batch_size is None:
                        active_batch_size = max(1, (current_batch_size + 1) // 2)
                    else:
                        active_batch_size = max(1, current_batch_size // 2)
                    print(
                        f"{error_label} during test/query inference; retrying the same rows "
                        f"with test_batch_size={active_batch_size}."
                    )
            finally:
                del x_device, model_output, parsed_output

        if os.environ.get("LDM_FWD_DEBUG") == "1":
            print(
                "FWDDBG "
                f"pid={os.getpid()} query_batches={self.last_query_batch_diagnostics}",
                flush=True,
            )
        return self._concat_query_outputs(completed_outputs, sample_dim=sample_dim)
    
    def get_and_set_seeds(self,seed=None):
        """Functionality: Record the seed identifier for this inference call. The current implementation returns the input and does not mutate global RNGs.

        Input:
            seed: Random seed. Historically a 6-tuple; a scalar is accepted now.

        Output:
            The input seed, later stored as seeds_hash for cache keys.
        """
        return seed
    

    class CacheManager:
        """Functionality: Multi-process-safe on-disk cache manager. File locks prevent duplicate writes.

        Input:
            cache_dir: Cache directory path.

        Output:
            Nested class providing read/write and cached_computation.
        """
        
        def __init__(self, cache_dir):
            """Functionality: Create the cache directory if it does not exist.

            Input:
                cache_dir: Cache root

            Output:
                None.
            """
            self.cache_dir = cache_dir
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
        
        def generate_cache_key(self, infe_type,id_pipe, pipe_config, unique_dataset_name, seeds_hash,extra_seed):
            """Functionality: Build a cache key from inference type, pipeline config, and dataset name.

            Input:
                infe_type: Task tag such as 'cls' or 'reg'.
                id_pipe: Pipeline index.
                pipe_config: Config dict for this member.
                unique_dataset_name: Dataset identity.
                seeds_hash: Seed hash.
                extra_seed: Extra seed (currently not included in the key string).

            Output:
                str: a stable cache key.
            """
            import json
            import hashlib
            
            pipe_str = json.dumps(pipe_config, sort_keys=True, separators=(',', ':'))
            pipe_hash = hashlib.sha256(pipe_str.encode('utf-8')).hexdigest()
            key = f"{infe_type}_{id_pipe}_{pipe_hash}_{unique_dataset_name}_{seeds_hash}"
            return key
        
        def get_cache_file_path(self, key):
            """Functionality: Map a cache key to its pickle file path.

            Input:
                key: Cache key.

            Output:
                str: {cache_dir}/{key}.pkl.
            """
            return os.path.join(self.cache_dir, f"{key}.pkl")
        
        def get_lock_file_path(self, key):
            """Functionality: Map a cache key to its lock file path.

            Input:
                key: Cache key.

            Output:
                str: {cache_dir}/{key}.lock.
            """
            return os.path.join(self.cache_dir, f"{key}.lock")
        
        def read_cache(self, key):
            """Functionality: Read the cache without locking. Corrupted files are deleted.

            Input:
                key: Cache key.

            Output:
                Deserialized object, or None if missing/corrupt.
            """
            cache_file = self.get_cache_file_path(key)
            
            if not os.path.exists(cache_file):
                return None
            
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data
            except (pickle.UnpicklingError, EOFError, Exception) as e:
                # Delete a corrupted cache file
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
                return None
        
        def write_cache(self, key, data):
            """Functionality: Write the cache under a non-blocking exclusive lock, using a temp file then an atomic rename.

            Input:
                key: Cache key.
                data: Picklable computation result.

            Output:
                bool: True on success; False on lock conflict, existing file, or failure.
            """
            import tempfile
            import fcntl
            
            cache_file = self.get_cache_file_path(key)
            lock_file = self.get_lock_file_path(key)
            
            try:
                # Try to acquire an exclusive lock (non-blocking)
                lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY)
                try:
                    # Acquire the lock without blocking
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Recheck the cache after locking to avoid duplicate writes
                    if os.path.exists(cache_file):
                        os.close(lock_fd)
                        try:
                            os.remove(lock_file)
                        except OSError:
                            pass
                        return False
                    
                    # Write a temp file, then rename atomically
                    temp_file = tempfile.NamedTemporaryFile(
                        mode='wb', 
                        dir=self.cache_dir, 
                        prefix=f"{key}_temp_", 
                        delete=False
                    )
                    
                    try:
                        import pickle
                        # Write the temp file
                        pickle.dump(data, temp_file)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                        temp_file.close()
                        
                        # Atomic rename
                        os.rename(temp_file.name, cache_file)
                        return True
                        
                    except Exception as e:
                        # Clean up the temp file
                        try:
                            os.remove(temp_file.name)
                        except OSError:
                            pass
                        return False
                    
                except BlockingIOError:
                    # Could not acquire the lock; skip the cache write
                    return False
                except Exception as e:
                    return False
                finally:
                    # Release the lock and clean up
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    except Exception:
                        pass
                    os.close(lock_fd)
                    try:
                        os.remove(lock_file)
                    except OSError:
                        pass
                        
            except OSError as e:
                # Could not create the lock file; skip
                print(f"Could not create lock file for key {key}: {e}")
                return False
        
        def cached_computation(self, key, compute_func, *args, **kwargs):
            """Functionality: Read the cache first; on miss, compute and attempt to write.

            Input:
                key: Cache key.
                compute_func: Callable invoked with *args/**kwargs.
                args/kwargs: Forwarded to compute_func.

            Output:
                Cached hit or newly computed result.
            """
            # Step 1: try reading the cache
            cached_result = self.read_cache(key)
            if cached_result is not None:
                return cached_result
            
            # Step 2: run the computation
            result = compute_func(*args, **kwargs)
            
            # Step 3: try writing the cache
            self.write_cache(key, result)
            
            return result

    def set_inference_config(self, inference_config: dict|str, softmax_temperature:float|None=None, seed:int|None=None):
        """Functionality: Hot-update the v2 inference config and rebuild preprocess pipelines and related runtime switches.

        Input:
            inference_config: v2 config dict with a non-empty 'pipelines' list, or a
                JSON file path to the same object.
            softmax_temperature: Optional new temperature (> 0). None keeps the current value.
            seed: Optional new integer seed. None keeps the current value.

        Output:
            None. Rebuilds parent preprocess_pipelines. Live GPU workers receive the
            same config without respawn; unstarted workers read
            _pipeline_worker_init_kwargs on first spawn.
        """
        inference_config, inference_pipeline_config = _load_v2_inference_config(inference_config)
        self.inference_config = inference_config
        self.inference_pipeline_config = inference_pipeline_config
        self.n_estimators = len(self.inference_pipeline_config)
        self.datetime_preprocessing_enabled = tuple(
            _datetime_preprocessing_enabled(config, index)
            for index, config in enumerate(self.inference_pipeline_config)
        )
        self._reject_retrieval_config()
        
        if softmax_temperature is not None:
            self.softmax_temperature = softmax_temperature
        if seed is not None:
            self.seed = seed
        self.build_preprocess_pipeline()
        (
            self.feature_view_strategy,
            self.feature_view_prediction_weight,
            self.fixed_feature_view_config,
        ) = _resolve_feature_view_runtime_config(inference_config)
        self.route_sampling_seed = inference_config.get("route_sampling_seed", 20260806)
        self.cross_fit_seed = inference_config.get("cross_fit_seed", 0)
        self.feature_view_ensemble_audit = None
        self.feature_encoder_mode = _resolve_feature_encoder_mode(
            inference_config.get("feature_encoder_mode", "current")
        )
        self.adaptive_svd_config = _resolve_adaptive_svd_config(inference_config)
        self.svd_adaptation_audit = []
        self.svd_runtime_retry_audit = []
        self.polynomial_interaction_runtime_retry_audit = []
        self.cuda_pipeline_fallback_audit = []
        self._adaptive_svd_retry_component_cap = None
        self._svd_inference_call_index = 0
        self._svd_runtime_cache_signature = "adaptive_svd_unconfigured"
        self._sync_pipeline_gpu_workers_inference_config()

    def _reject_retrieval_config(self) -> None:
        """Functionality: Reject any member config that still enables retrieval.

        Input:
            self: Current predictor; reads inference_pipeline_config.

        Output:
            None. Raises ValueError when use_retrieval is true.
        """
        for id_pipe, config in enumerate(self.inference_pipeline_config):
            retrieval_config = config.get("retrieval_config") or {}
            if retrieval_config.get("use_retrieval"):
                raise ValueError(
                    "retrieval has been removed from inference/dev; "
                    f"member {id_pipe} sets use_retrieval=true"
                )
    
    def build_preprocess_pipeline(self):
        """Functionality: Instantiate preprocess steps from each member config and sample a shuffle offset per pipeline.

        Input:
            self: Requires n_estimators, inference_pipeline_config, and seed.

        Output:
            None. Writes preprocess_pipelines, seeds, and all_shifts.
        """
        self.preprocess_pipelines = []
        self.preprocess_configs = []
    
        random.seed(self.seed)
        rand_gen = np.random.default_rng(self.seed)
        self.seeds = [random.randint(0, 10000) for _ in range(self.n_estimators*self.preprocess_num)]
        start_idx = rand_gen.integers(0, 1000)
        all_shifts = list(range(start_idx, start_idx + self.n_estimators))
        self.all_shifts = rand_gen.choice(all_shifts, size=self.n_estimators, replace=False)

        for idx in range(self.n_estimators):
            pipeline = []
            inference_config_item = self.inference_pipeline_config[idx]
            
            if 'PolynomialInteractionGenerator' in inference_config_item:
                pipeline.append(PolynomialInteractionGenerator(**inference_config_item['PolynomialInteractionGenerator']))

            pipeline.append(FilterValidFeatures())

            if 'RebalanceFeatureDistribution' in inference_config_item:
                RebalanceFeatureDistributionOptions = inference_config_item['RebalanceFeatureDistribution']
                RebalanceFeatureDistributionOptions['enable_parallel'] = self.enable_preprocess_parallel
                RebalanceFeatureDistributionOptions['num_jobs'] = self.preprocess_num_jobs
                pipeline.append(RebalanceFeatureDistribution(**RebalanceFeatureDistributionOptions))
            if 'CategoricalFeatureEncoder' in inference_config_item:
                pipeline.append(CategoricalFeatureEncoder(**inference_config_item['CategoricalFeatureEncoder']))
            if inference_config_item.get('FingerprintFeatureEncoder', False):
                pipeline.append(FingerprintFeatureEncoder())
            if 'FeatureShuffler' in inference_config_item:
                shuffler = FeatureShuffler(**inference_config_item['FeatureShuffler'])
                shuffler.offset = self.all_shifts[idx]
                pipeline.append(shuffler)
            if 'TargetTransform' in inference_config_item and inference_config_item['TargetTransform'] is not None:
                pipeline.append(TargetTransform(**inference_config_item['TargetTransform']))
            self.preprocess_pipelines.append(pipeline)

    def _configure_adaptive_svd_runtime(
        self,
        *,
        train_rows: int,
        query_rows: int,
        unique_dataset_name: str | None,
    ) -> None:
        """Functionality: Inject a per-forward SVD component budget into every pipeline SVD step from current free CUDA memory.

        Input:
            train_rows: Train row count.
            query_rows: Query row count.
            unique_dataset_name: Optional dataset name stored in the SVD context.

        Output:
            None. Calls RebalanceFeatureDistribution.set_svd_runtime_context.
        """
        config = getattr(
            self,
            "adaptive_svd_config",
            _DEFAULT_ADAPTIVE_SVD_CONFIG,
        )
        device_type = (
            self.device.type
            if isinstance(self.device, torch.device)
            else str(self.device).split(":", 1)[0]
        )
        context = None
        if config.get("enabled", False) and device_type == "cuda":
            try:
                free_cuda_bytes, _ = torch.cuda.mem_get_info(self.device)
                reclaimable_cache = max(
                    0,
                    torch.cuda.memory_reserved(self.device)
                    - torch.cuda.memory_allocated(self.device),
                )
                free_cuda_bytes += reclaimable_cache
            except (RuntimeError, TypeError):
                properties = torch.cuda.get_device_properties(self.device)
                free_cuda_bytes = (
                    properties.total_memory
                    - torch.cuda.memory_allocated(self.device)
                )
            model_tensors = list(self.model.parameters()) + list(self.model.buffers())
            model_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in model_tensors
            )
            model_already_resident = bool(model_tensors) and all(
                tensor.device.type == "cuda"
                for tensor in model_tensors
            )
            model_bytes_to_load = (
                0
                if model_already_resident
                else int(model_bytes * 1.10)
            )
            sequence_attention_limit = None
            if self.model_config.get("seq_attn_use_induced", False):
                sequence_attention_limit = int(
                    self.model_config.get(
                        "seq_attn_num_inducing_points",
                        192,
                    ) or 192
                )
            self._svd_inference_call_index += 1
            context = {
                **config,
                "train_rows": int(train_rows),
                "query_rows": int(query_rows),
                "free_cuda_bytes": int(free_cuda_bytes),
                "model_bytes_to_load": int(model_bytes_to_load),
                "embedding_dim": int(
                    self.model_config.get(
                        "embed_dim",
                        self.model_config.get("emsize", 192),
                    ) or 192
                ),
                "num_heads": int(self.model_config.get("nhead", 1) or 1),
                "mixed_precision": bool(self.mix_precision),
                "sequence_attention_limit": sequence_attention_limit,
                "inference_call_index": self._svd_inference_call_index,
                "unique_dataset_name": unique_dataset_name,
                "runtime_max_components": getattr(
                    self,
                    "_adaptive_svd_retry_component_cap",
                    None,
                ),
            }
            memory_bucket = free_cuda_bytes // (256 * 1024**2)
            self._svd_runtime_cache_signature = (
                f"adaptive_svd_{train_rows}_{query_rows}_{memory_bucket}_"
                f"{model_bytes_to_load}_{context['embedding_dim']}_"
                f"{context['num_heads']}_{context['runtime_max_components']}"
            )
        else:
            self._svd_runtime_cache_signature = "adaptive_svd_disabled"

        for pipeline_index, pipeline in enumerate(self.preprocess_pipelines):
            for step in pipeline:
                if isinstance(step, RebalanceFeatureDistribution):
                    step_context = (
                        {**context, "pipeline_index": pipeline_index}
                        if context is not None
                        else None
                    )
                    step.set_svd_runtime_context(step_context)

    def _record_svd_adaptation(
        self,
        *,
        pipeline_index: int,
        pipeline,
        unique_dataset_name: str | None,
    ) -> None:
        """Functionality: Append SVD-step diagnostics from one pipeline onto svd_adaptation_audit.

        Input:
            pipeline_index: Pipeline index.
            pipeline: List of preprocess steps.
            unique_dataset_name: Optional dataset name.

        Output:
            None.
        """
        for step_index, step in enumerate(pipeline):
            if not isinstance(step, RebalanceFeatureDistribution):
                continue
            diagnostics = step.last_svd_diagnostics
            if not isinstance(diagnostics, dict):
                continue
            record = copy.deepcopy(diagnostics)
            record.update(
                pipeline_index=int(pipeline_index),
                pipeline_step_index=int(step_index),
                inference_call_index=int(
                    (step.svd_runtime_context or {}).get(
                        "inference_call_index",
                        0,
                    )
                ),
                unique_dataset_name=unique_dataset_name,
            )
            self.svd_adaptation_audit.append(record)


    def _svd_adaptation_snapshot(self) -> dict:
        """Functionality: Summarize SVD adaptation, interaction retries, and CUDA pipeline-skip records for this predict() call.

        Input:
            self: Reads the various audit lists.

        Output:
            dict with enabled/adapted/records plus retry and skipped-pipeline information.
        """
        records = copy.deepcopy(
            getattr(self, "svd_adaptation_audit", [])
        )
        return {
            "enabled": bool(
                getattr(
                    self,
                    "adaptive_svd_config",
                    _DEFAULT_ADAPTIVE_SVD_CONFIG,
                ).get("enabled", False)
            ),
            "adapted": any(record.get("adapted", False) for record in records),
            "record_count": len(records),
            "records": records,
            "runtime_retry_applied": bool(
                getattr(self, "svd_runtime_retry_audit", [])
            ),
            "runtime_retry_count": len(
                getattr(self, "svd_runtime_retry_audit", [])
            ),
            "runtime_retries": copy.deepcopy(
                getattr(self, "svd_runtime_retry_audit", [])
            ),
            "polynomial_interaction_retry_applied": bool(
                getattr(
                    self,
                    "polynomial_interaction_runtime_retry_audit",
                    [],
                )
            ),
            "polynomial_interaction_retry_count": len(
                getattr(
                    self,
                    "polynomial_interaction_runtime_retry_audit",
                    [],
                )
            ),
            "polynomial_interaction_retries": copy.deepcopy(
                getattr(
                    self,
                    "polynomial_interaction_runtime_retry_audit",
                    [],
                )
            ),
            "cuda_skipped_pipeline_count": len(
                getattr(self, "cuda_pipeline_fallback_audit", [])
            ),
            "cuda_skipped_pipelines": copy.deepcopy(
                getattr(self, "cuda_pipeline_fallback_audit", [])
            ),
        }

    def _activate_polynomial_interaction_resource_retry(
        self,
        *,
        error: BaseException,
        pipeline_index: int,
        pipeline,
        train_rows: int,
        query_rows: int,
        unique_dataset_name: str | None,
    ) -> bool:
        """Functionality: After CUDA OOM, if resource retry is allowed, disable generated polynomial interaction columns for the current pipeline.

        Input:
            error: Exception that triggered the retry.
            pipeline_index: Pipeline index.
            pipeline: List of preprocess steps.
            train_rows: Train row count.
            query_rows: Query row count.
            unique_dataset_name: Optional dataset name.

        Output:
            bool: True if this fallback was activated.
        """
        retry_reason = self._cuda_batch_retry_reason(error)
        config = getattr(
            self,
            "adaptive_svd_config",
            _DEFAULT_ADAPTIVE_SVD_CONFIG,
        )
        if (
            retry_reason != "out_of_memory"
            or not config.get("retry_on_cuda_resource_error", False)
        ):
            return False

        interaction_steps = [
            step
            for step in pipeline
            if isinstance(step, PolynomialInteractionGenerator)
            and getattr(step, "runtime_interactions_enabled", True)
        ]
        if not interaction_steps:
            return False

        configured_interaction_features = sum(
            int(step.max_interactions or 0) for step in interaction_steps
        )
        for step in interaction_steps:
            step.set_runtime_interactions_enabled(False)

        retry_record = {
            "pipeline_index": int(pipeline_index),
            "reason": retry_reason,
            "configured_interaction_features": configured_interaction_features,
            "retry_interaction_features": 0,
            "train_rows": int(train_rows),
            "query_rows": int(query_rows),
            "unique_dataset_name": unique_dataset_name,
        }
        if not hasattr(self, "polynomial_interaction_runtime_retry_audit"):
            self.polynomial_interaction_runtime_retry_audit = []
        self.polynomial_interaction_runtime_retry_audit.append(retry_record)
        self._release_cuda_after_resource_error(error)
        print(
            "CUDA OOM after query batching; retrying only the current "
            "classification ensemble pipeline without generated polynomial "
            f"interaction columns (pipeline={pipeline_index})."
        )
        return True

    def _activate_next_feature_width_resource_retry(
        self,
        *,
        error: BaseException,
        pipeline_index: int,
        pipeline,
        train_rows: int,
        query_rows: int,
        unique_dataset_name: str | None,
        attempted_retries: set[str],
    ) -> str | None:
        """Functionality: Try feature-width fallbacks in order: disable polynomial interactions, then cap SVD components.

        Input:
            error: CUDA resource exception.
            pipeline_index: Pipeline index.
            pipeline: List of preprocess steps.
            train_rows: Train row count.
            query_rows: Query row count.
            unique_dataset_name: Optional dataset name.
            attempted_retries: Set of fallback names already tried.

        Output:
            str | None: 'polynomial_interactions', 'svd', or None if no further fallback remains.
        """
        if (
            "polynomial_interactions" not in attempted_retries
            and self._activate_polynomial_interaction_resource_retry(
                error=error,
                pipeline_index=pipeline_index,
                pipeline=pipeline,
                train_rows=train_rows,
                query_rows=query_rows,
                unique_dataset_name=unique_dataset_name,
            )
        ):
            return "polynomial_interactions"

        if (
            "svd" not in attempted_retries
            and self._activate_adaptive_svd_resource_retry(
                error=error,
                pipeline_index=pipeline_index,
                pipeline=pipeline,
                train_rows=train_rows,
                query_rows=query_rows,
                unique_dataset_name=unique_dataset_name,
            )
        ):
            return "svd"

        return None

    def _activate_adaptive_svd_resource_retry(
        self,
        *,
        error: BaseException,
        pipeline_index: int,
        pipeline,
        train_rows: int,
        query_rows: int,
        unique_dataset_name: str | None,
    ) -> bool:
        """Functionality: After a CUDA resource failure, cap SVD components at retry_max_components for the current dataset only.

        Input:
            error: CUDA resource exception.
            pipeline_index: Pipeline index.
            pipeline: List of preprocess steps.
            train_rows: Train row count.
            query_rows: Query row count.
            unique_dataset_name: Optional dataset name.

        Output:
            bool: whether the cap was set and SVD runtime was reconfigured.
        """
        retry_reason = self._cuda_batch_retry_reason(error)
        config = getattr(
            self,
            "adaptive_svd_config",
            _DEFAULT_ADAPTIVE_SVD_CONFIG,
        )
        if (
            retry_reason is None
            or not config.get("enabled", False)
            or not config.get("retry_on_cuda_resource_error", False)
        ):
            return False

        retry_cap = int(config.get("retry_max_components", 0))
        svd_steps = [
            step
            for step in pipeline
            if isinstance(step, RebalanceFeatureDistribution)
            and step.svd_tag == "svd"
        ]
        if not svd_steps or not any(
            int(getattr(step, "svd_n_comp", 0)) > retry_cap
            for step in svd_steps
        ):
            return False

        self._adaptive_svd_retry_component_cap = retry_cap
        retry_record = {
            "pipeline_index": int(pipeline_index),
            "reason": retry_reason,
            "retry_max_components": retry_cap,
            "train_rows": int(train_rows),
            "query_rows": int(query_rows),
            "unique_dataset_name": unique_dataset_name,
        }
        self.svd_runtime_retry_audit.append(retry_record)
        self._release_cuda_after_resource_error(error)
        self._configure_adaptive_svd_runtime(
            train_rows=train_rows,
            query_rows=query_rows,
            unique_dataset_name=unique_dataset_name,
        )
        print(
            "CUDA resource failure after query batching; retrying only the "
            f"current dataset with SVD components capped at {retry_cap} "
            f"(pipeline={pipeline_index}, reason={retry_reason})."
        )
        return True

    def _check_n_features(self, X, reset):
        """Functionality: Check that the feature count matches the previous evaluation.

        Input:
            X: 2-D feature array; uses shape[1].
            reset: If True, record n_features_in_; if False, compare against the recorded value.

        Output:
            None. Raises ValueError when the column count mismatches.
        """
        n_features = X.shape[1]
        if reset:
            self.n_features_in_ = n_features
        else:
            if self.n_features_in_ != n_features:
                raise ValueError(
                    f"X has {n_features} features, "
                    f"but this estimator is expecting {self.n_features_in_} features."
                )
    
    def validate_data(self, x=None, y=None, reset=True, validate_separately=False, **check_params):
        """Functionality: Validate features and labels with sklearn check_X_y/check_array and keep the feature count in sync.

        Input:
            x: Feature table; may be omitted.
            y: Labels; when provided, validated together with x.
            reset: Whether to reset n_features_in_.
            validate_separately: Reserved; currently unused.
            check_params: Forwarded to check_X_y/check_array, e.g. dtype and ensure_all_finite.

        Output:
            (x, y) when y is given; x when only x is given; None when both are None.
        """
        # Validate both x and y simultaneously
        if y is not None:
            x, y = check_X_y(x, y, **check_params)
            self._check_n_features(x, reset=reset)
            return x, y

        # Validate X
        if x is not None:
            x = check_array(x, **check_params)
            self._check_n_features(x, reset=reset)
            return x

        return None
    
    def convert_x_dtypes(self, x:np.ndarray, dtypes:Literal["float32", "float64"] = "float64"):
        """Functionality: Convert a numpy feature table to a DataFrame and cast numeric columns to the requested float dtype.

        Input:
            x: 2-D numpy array.
            dtypes: 'float32' or 'float64', default float64.

        Output:
            pandas.DataFrame. String dtypes are not supported.
        """
        NUMERIC_DTYPE_KINDS = "?bBiufm"
        OBJECT_DTYPE_KINDS = "OV"
        STRING_DTYPE_KINDS = "SaU"
        
        if x.dtype.kind in NUMERIC_DTYPE_KINDS:
            x = pd.DataFrame(x, copy=False, dtype=dtypes)
        elif x.dtype.kind in OBJECT_DTYPE_KINDS:
            x = pd.DataFrame(x, copy=True)
            x = x.convert_dtypes()
        else:
            raise ValueError(f"Unsupport string dtypes! {x.dtype}")

        integer_columns = x.select_dtypes(include=["number"]).columns
        if len(integer_columns) > 0:
            x[integer_columns] = x[integer_columns].astype(dtypes)
        return x
    
    def convert_category2num(self, x, dtype:np.floating=np.float64, placeholder: str = NA_PLACEHOLDER,):
        """Functionality: Ordinal-encode category/string/bool columns. Missing strings are filled with a placeholder then restored to NaN.

        Input:
            x: DataFrame with mixed dtypes.
            dtype: Floating dtype of the encoded output, default float64.
            placeholder: Missing-string placeholder, default NA_PLACEHOLDER.

        Output:
            2-D numpy array with categorical columns encoded as numbers.
        """
        ordinal_encoder = OrdinalEncoder(categories="auto",
                                        dtype=dtype,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                        encoded_missing_value=np.nan)
        col_encoder = ColumnTransformer(transformers=[("encoder", ordinal_encoder, make_column_selector(dtype_include=["category", "string", "bool"]))],
                                        remainder=FunctionTransformer(),
                                        sparse_threshold=0.0,
                                        verbose_feature_names_out=False,
                                    )
        
        string_cols = x.select_dtypes(include=["string", "object"]).columns
        if len(string_cols) > 0:
            x[string_cols] = x[string_cols].fillna(placeholder)
        
        X_encoded = col_encoder.fit_transform(x)

        string_cols_ix = [x.columns.get_loc(col) for col in string_cols]
        placeholder_mask = x[string_cols] == placeholder
        string_cols_ix_2 = list(range(len(string_cols_ix)))
        X_encoded[:, string_cols_ix_2] = np.where(
            placeholder_mask,
            np.nan,
            X_encoded[:, string_cols_ix_2],
        )

        return X_encoded

    
    def get_categorical_features_indices(self, x:np.ndarray):
        """Functionality: When the sample is long enough, treat columns with too few unique values as categorical.

        Input:
            x: 2-D numeric array.

        Output:
            list[int]: categorical column indices. Empty if the row count is below the threshold.
        """
        if x.shape[0] < self.min_seq_len_for_category_infer:
            return []
        categorical_idx = []
        for idx, col in enumerate(x.T):
            if len(np.unique(col)) < self.min_unique_num_for_numerical_infer:
                categorical_idx.append(idx)
        return categorical_idx

    @staticmethod
    def _as_classification_feature_frame(values, *, name: str) -> pd.DataFrame:
        """Functionality: Normalize classification features into a DataFrame with unique column names.

        Input:
            values: DataFrame or 2-D array.
            name: Variable name used in error messages, e.g. x_train.

        Output:
            pandas.DataFrame.
        """
        if isinstance(values, pd.DataFrame):
            frame = values.copy(deep=True).reset_index(drop=True)
        else:
            array = np.asarray(values)
            if array.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional feature table")
            frame = pd.DataFrame(array)
        if frame.columns.duplicated().any():
            raise ValueError(f"{name} contains duplicate feature names")
        return frame

    def _encode_scale_cls_features(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Functionality: Encode categorical columns with feature_encoder_mode and MinMax-scale using a scaler fit on train.

        Input:
            x_train: Train feature DataFrame.
            x_test: Test feature DataFrame; columns must align with train.

        Output:
            tuple[np.ndarray, np.ndarray]: scaled train/test numeric matrices.
        """
        x_train, x_test = encode_categorical_features(
            x_train.copy(deep=True),
            x_test.copy(deep=True),
            feature_encoder_mode=getattr(self, "feature_encoder_mode", "current"),
        )
        if x_train.shape[1] == 0:
            raise ValueError("classification preprocessing removed every feature")
        train_values = np.asarray(x_train, dtype=np.float64)
        test_values = np.asarray(x_test, dtype=np.float64)
        finite_col = ~np.isnan(train_values).all(axis=0)
        if finite_col.all():
            scaler = MinMaxScaler()
            return (
                np.asarray(scaler.fit_transform(train_values)),
                np.asarray(scaler.transform(test_values)),
            )
        scaled_train = np.full(train_values.shape, np.nan, dtype=np.float64)
        scaled_test = np.full(test_values.shape, np.nan, dtype=np.float64)
        if finite_col.any():
            scaler = MinMaxScaler()
            scaled_train[:, finite_col] = scaler.fit_transform(
                train_values[:, finite_col]
            )
            scaled_test[:, finite_col] = scaler.transform(
                test_values[:, finite_col]
            )
        return scaled_train, scaled_test

    def _classification_feature_frame_pair(
        self,
        x_train,
        x_test,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Functionality: Convert train/test features to a non-empty DataFrame pair with identical column names.

        Input:
            x_train: Train features.
            x_test: Test features.

        Output:
            tuple[DataFrame, DataFrame]. Raises if columns differ or either side is empty.
        """
        train_frame = self._as_classification_feature_frame(
            x_train, name="x_train"
        )
        test_frame = self._as_classification_feature_frame(
            x_test, name="x_test"
        )
        if tuple(train_frame.columns) != tuple(test_frame.columns):
            raise ValueError("x_train and x_test columns differ")
        if len(train_frame) == 0 or len(test_frame) == 0:
            raise ValueError("x_train and x_test must both contain rows")
        return train_frame, test_frame

    @staticmethod
    def _classification_labels(y_train, *, expected_rows: int) -> np.ndarray:
        """Functionality: Require 1-D classification labels with no missing values, matching row count, and 2 to 10 classes.

        Input:
            y_train: Train labels.
            expected_rows: Must equal the x_train row count.

        Output:
            np.ndarray: 1-D label array.
        """
        labels = np.asarray(y_train)
        if labels.ndim != 1 or len(labels) != expected_rows:
            raise ValueError(
                "y_train must be one-dimensional and match the x_train row count"
            )
        if pd.isna(labels).any():
            raise ValueError("y_train contains missing values")
        n_classes = len(np.unique(labels))
        if n_classes > 10 or n_classes < 2:
            raise ValueError(f"num_classes {n_classes} is not supported")
        return labels

    def _member_numeric_inputs(
        self,
        frame: pd.DataFrame,
        *,
        train_size: int,
        cast_float32: bool,
        task_type: Literal["Classification", "Regression"] = "Classification",
    ) -> list[tuple[np.ndarray, list[int]]]:
        """Functionality: Build task-appropriate numeric inputs for every ensemble member. Datetime detection and imputation are fit on support only so query dates cannot leak.

        Input:
            frame: Row-concatenated train+query DataFrame.
            train_size: Support row count; must be in [1, n_rows-1].
            cast_float32: Whether to cast the result to float32.
            task_type: Classification uses the native encoder/scaler; Regression preserves the legacy ordinal conversion.

        Output:
            list[tuple[np.ndarray, list[int]]]: (numeric matrix, categorical indices) per member.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("member preprocessing requires a pandas DataFrame")
        if train_size <= 0 or train_size >= len(frame):
            raise ValueError(
                f"train_size must be in [1, {len(frame) - 1}], received {train_size}"
            )
        if frame.columns.duplicated().any():
            raise ValueError("duplicate feature names are not supported")
        if task_type not in {"Classification", "Regression"}:
            raise ValueError(
                f"member preprocessing does not support task_type={task_type!r}"
            )

        enabled_policies = getattr(self, "datetime_preprocessing_enabled", None)
        if enabled_policies is None:
            configs = getattr(self, "inference_pipeline_config", ())
            enabled_policies = tuple(
                _datetime_preprocessing_enabled(config, index)
                for index, config in enumerate(configs)
            )
            self.datetime_preprocessing_enabled = enabled_policies

        cache: dict[bool, tuple[np.ndarray, list[int]]] = {}
        inputs: list[tuple[np.ndarray, list[int]]] = []
        with _profile_span("member_numeric"):
            for enabled in enabled_policies:
                if enabled not in cache:
                    converted_frame = frame.copy(deep=True).reset_index(drop=True)
                    if enabled:
                        with _profile_span("datetime"):
                            datetime_preprocessor = DatetimePreprocessor().fit(
                                converted_frame.iloc[:train_size].copy()
                            )
                            if datetime_preprocessor.columns_:
                                converted_frame = pd.concat(
                                    [
                                        datetime_preprocessor.transform(
                                            converted_frame.iloc[:train_size].copy()
                                        ),
                                        datetime_preprocessor.transform(
                                            converted_frame.iloc[train_size:].copy()
                                        ),
                                    ],
                                    ignore_index=True,
                                )
                                # Generated columns are strings while array-originated
                                # columns are integers. sklearn rejects mixed name types.
                                converted_frame.columns = pd.RangeIndex(
                                    converted_frame.shape[1]
                                )

                    with _profile_span("encode_scale"):
                        if task_type == "Classification":
                            train_values, test_values = self._encode_scale_cls_features(
                                converted_frame.iloc[:train_size].copy(),
                                converted_frame.iloc[train_size:].copy(),
                            )
                            values = np.concatenate([train_values, test_values], axis=0)
                        else:
                            # Keep datetime-disabled regression bitwise aligned with the
                            # historical single-matrix ordinal conversion path.
                            values = self.convert_category2num(converted_frame.copy())
                    if cast_float32:
                        values = values.astype(np.float32)
                    categorical_indices = self.get_categorical_features_indices(values)
                    cache[enabled] = (values, categorical_indices)

                values, categorical_indices = cache[enabled]
                inputs.append((values, categorical_indices.copy()))

        return inputs
        
    def predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        task_type: Literal["Classification", "Regression", "Feature_imputation"] = "Classification",
        unique_dataset_name: str = None,
    ) -> np.ndarray:
        """Functionality: Public inference entry point. Dispatches classification, regression, or missing-value prediction. Classification may add feature-view ensembling.

        Input:
            x_train: Train features. ndarray or pandas.DataFrame, shape (n_train, n_features).
                Rows are samples; columns must align with x_test.
            y_train: Train targets, shape (n_train,). Classification: discrete labels
                with 2–10 classes and no NaN. Regression: numeric 1-D targets.
            x_test: Query features, same type/columns as x_train, shape (n_query, n_features).
            task_type: 'Classification' (default), 'Regression', or 'Feature_imputation'.
            unique_dataset_name: Optional dataset id for preprocess cache and SVD audit.

        Output:
            np.ndarray:
                Classification: probabilities, shape (n_query, n_classes), rows sum to 1.
                Regression: predictions, shape (n_query,), original target scale.
                Feature_imputation: imputed feature matrix or None.
        """
        self._ensure_model_loaded()
        if _profile_enabled():
            _profile_reset()
        self.feature_view_ensemble_audit = None
        self.svd_adaptation_audit = []
        # This state is intentionally scoped to one public predict() call.  A
        # CUDA failure may activate it for the current dataset and its feature
        # views, but the following dataset always starts from configured
        # polynomial interactions and SVD.
        self.svd_runtime_retry_audit = []
        self.polynomial_interaction_runtime_retry_audit = []
        self.cuda_pipeline_fallback_audit = []
        self._adaptive_svd_retry_component_cap = None
        self._svd_inference_call_index = 0
        for pipeline in getattr(self, "preprocess_pipelines", ()):
            for step in pipeline:
                if isinstance(step, PolynomialInteractionGenerator):
                    step.set_runtime_interactions_enabled(True)
        if "Classification" == task_type:
            feature_view_x_train = None
            feature_view_x_test = None
            if self.feature_view_strategy != "disabled":
                feature_view_x_train = _copy_raw_features(x_train)
                feature_view_x_test = _copy_raw_features(x_test)
            # The merged classifier path preprocesses each ensemble member so
            # datetime policy can differ by member. Preserve the historical
            # extension hook when callers override _preprocess_cls_dataset.
            preprocess_cls = self._preprocess_cls_dataset
            if (
                getattr(preprocess_cls, "__func__", None)
                is not LimiXPredictor._preprocess_cls_dataset
            ):
                x_train, y_train, x_test = preprocess_cls(
                    x_train, y_train, x_test
                )
            if self.feature_view_strategy == "disabled":
                self.feature_view_ensemble_audit = {
                    "schema": "feature-view-ensemble-v1",
                    "configured": False,
                    "enabled": False,
                    "feature_view_applied": False,
                    "feature_view_strategy": "disabled",
                    "task_type": "classification",
                    "ensemble_method": "base_classification_ensemble",
                    "active_feature_views": [],
                    "feature_view_forward_count": 0,
                    "fallback_exact_base": False,
                    "fallback_reason": "feature_view_strategy_disabled",
                    "query_labels_used": False,
                    "labels_used_for_feature_generation": False,
                    "search_used": False,
                    "feature_generation_policy": "disabled",
                    "feature_selection_metric": "none",
                    "target_encoding": "none",
                    "cross_fitting": False,
                }
                prediction = self._predict_cls(
                    x_train, y_train, x_test, task_type,
                    unique_dataset_name=unique_dataset_name,
                )
                self.feature_view_ensemble_audit[
                    "svd_adaptation"
                ] = self._svd_adaptation_snapshot()
                return prediction
            return self._feature_view_ensemble_predict_cls(
                x_train, y_train, x_test, task_type,
                unique_dataset_name=unique_dataset_name,
                feature_view_x_train=feature_view_x_train,
                feature_view_x_test=feature_view_x_test,
            )
        elif "Regression" == task_type:
            configured = self.feature_view_strategy != "disabled"
            self.feature_view_ensemble_audit = {
                "schema": "feature-view-ensemble-v1",
                "configured": configured,
                "enabled": False,
                "feature_view_applied": False,
                "feature_view_strategy": self.feature_view_strategy,
                "task_type": "regression",
                "ensemble_method": "base_regression_ensemble",
                "active_feature_views": [],
                "feature_view_forward_count": 0,
                "fallback_exact_base": False,
                "fallback_reason": (
                    "feature_view_ensemble_not_supported_for_regression"
                    if configured
                    else "feature_view_strategy_disabled"
                ),
                "query_labels_used": False,
                "labels_used_for_feature_generation": False,
                "search_used": False,
                "feature_generation_policy": "classification_only",
                "feature_selection_metric": "none",
                "target_encoding": "none",
                "cross_fitting": False,
            }
            return self._predict_reg(x_train, y_train, x_test, task_type,unique_dataset_name=unique_dataset_name)
        elif "Feature_imputation" == task_type:
            return self._predict_feature_imputation(x_train, y_train, x_test, task_type,unique_dataset_name=unique_dataset_name)
        else:
            raise ValueError(f"Unsupported task type, supported tasks include Classification, Regression and Feature_imputation!")

    def _preprocess_cls_dataset(
        self,
        x_train,
        y_train,
        x_test,
    ):
        """Functionality: Encode and scale one classification feature pair. Kept as a historical extension hook that callers may override.

        Input:
            x_train: Train features.
            y_train: Train labels (validated only).
            x_test: Test features.

        Output:
            tuple: float32 trainX, original y_train, float32 testX.
        """
        x_train, x_test = self._classification_feature_frame_pair(
            x_train, x_test
        )
        train_values, test_values = self._encode_scale_cls_features(x_train, x_test)
        self._classification_labels(y_train, expected_rows=len(x_train))
        trainX = np.asarray(train_values, dtype=np.float32)
        testX = np.asarray(test_values, dtype=np.float32)
        return trainX, y_train, testX

    def _feature_view_ensemble_predict_cls(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        task_type: str,
        unique_dataset_name: str = None,
        feature_view_x_train=None,
        feature_view_x_test=None,
    ) -> np.ndarray:
        """Functionality: Delegate feature-view ensembling while keeping the core _predict_cls forward, then write back the audit.

        Input:
            x_train: Train features used by the base classifier path.
            y_train: Train labels.
            x_test: Test features used by the base classifier path.
            task_type: Task type.
            unique_dataset_name: Optional dataset name.
            feature_view_x_train: Raw train features for the view path; defaults to x_train.
            feature_view_x_test: Raw test features for the view path; defaults to x_test.

        Output:
            np.ndarray: query-by-class probability matrix.
        """
        feature_view_x_train = x_train if feature_view_x_train is None else feature_view_x_train
        feature_view_x_test = x_test if feature_view_x_test is None else feature_view_x_test
        fixed_feature_view_config = getattr(
            self, "fixed_feature_view_config", None
        )
        feature_view_preprocessor = (
            FixedFeatureViewPreprocessor(**fixed_feature_view_config)
            if fixed_feature_view_config is not None
            else None
        )
        prediction, audit = predict_with_feature_views(
            predict_cls=self._predict_cls,
            x_train=feature_view_x_train,
            y_train=y_train,
            x_test=feature_view_x_test,
            task_type=task_type,
            base_x_train=x_train,
            base_x_test=x_test,
            unique_dataset_name=unique_dataset_name,
            feature_view_strategy=self.feature_view_strategy,
            feature_view_prediction_weight=self.feature_view_prediction_weight,
            route_sampling_seed=getattr(self, "route_sampling_seed", 20260806),
            cross_fit_seed=getattr(self, "cross_fit_seed", 0),
            feature_view_preprocessor=feature_view_preprocessor,
        )
        audit["svd_adaptation"] = self._svd_adaptation_snapshot()
        self.feature_view_ensemble_audit = audit
        return prediction

    def get_xy_cls(
            self, x, id_pipe,y_train,y,pipe,categorical_idx,task_type,unique_dataset_name=None
        ):  
            # Build the cache key
        """Functionality: Run the preprocess pipeline for one classification member, optionally using the on-disk cache.

        Input:
            x: Numeric feature matrix for this member (train+query).
            id_pipe: Pipeline index.
            y_train: Original train labels (used in the cache key).
            y: Integer labels after LabelEncoder.
            pipe: List of preprocess steps.
            categorical_idx: Categorical column indices.
            task_type: Task type (unused in the computation).
            unique_dataset_name: Optional cache identity.

        Output:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (transformed x, permuted y, class_permutation).
        """
        extra_seed = id_pipe*self.preprocess_num
        key = self.cache_manager.generate_cache_key(
            'cls',
            id_pipe, 
            self.inference_pipeline_config[id_pipe], 
            unique_dataset_name, 
            self.seeds_hash,
            extra_seed
        )
        key = f"{key}_{self._svd_runtime_cache_signature}"
        
        # Define the compute function
        def compute_xy_cls():
            """Functionality: Actually run classification preprocess steps and permute labels with this member's class permutation.

            Input:
                None: Closure over x, y, pipe, categorical_idx, and id_pipe.

            Output:
                tuple: (x_, y_, class_permutations[id_pipe]).
            """
            x_ = x
            y_ = self.class_permutations[id_pipe][y]
            categorical_idx_ = categorical_idx

            with nvtx.annotate('preprocess'):
                for id_step, step in enumerate(pipe):
                    x_, categorical_idx_ = step.fit_transform(
                        x_, categorical_idx_,
                        self.seeds[extra_seed+id_step],
                        y=y_
                    )

            return (x_, y_, self.class_permutations[id_pipe])

        adaptive_svd_pipeline = (
            getattr(
                self,
                "adaptive_svd_config",
                _DEFAULT_ADAPTIVE_SVD_CONFIG,
            ).get("enabled", False)
            and any(
                isinstance(step, RebalanceFeatureDistribution)
                and step.svd_tag == "svd"
                for step in pipe
            )
        )
        polynomial_retry_pipeline = (
            getattr(
                self,
                "adaptive_svd_config",
                _DEFAULT_ADAPTIVE_SVD_CONFIG,
            ).get("retry_on_cuda_resource_error", False)
            and any(
                isinstance(step, PolynomialInteractionGenerator)
                for step in pipe
            )
        )
        if (
            self.use_data_cache
            and not adaptive_svd_pipeline
            and not polynomial_retry_pipeline
        ):
            # Run the computation through the cache manager
            return self.cache_manager.cached_computation(key, compute_xy_cls)
        else :
            return compute_xy_cls()

    def _require_prepared_cls_supported(self) -> None:
        """Functionality: Check that prepare_cls/infer_prepared_cls are usable. The split API is unsupported when the data cache is enabled.

        Input:
            self: Current predictor.

        Output:
            None. Raises ValueError when use_data_cache is True.
        """
        if self.use_data_cache:
            raise ValueError("prepared classification does not support predictor data cache")

    def prepare_cls(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> PreparedClassificationInference:
        """Functionality: Run classification validation and CPU preprocessing without touching CUDA, so CPU/GPU work can overlap.

        Input:
            x_train: Train features, shape (n_train, n_features). ndarray or DataFrame.
            y_train: Train labels, shape (n_train,). 2–10 classes, no missing values.
            x_test: Query features, shape (n_query, n_features), same columns as x_train.

        Output:
            PreparedClassificationInference: per-member host arrays (x, y, class
            permutation), plus train_size and n_classes. Does not support use_data_cache.
        """
        self._require_prepared_cls_supported()
        np_rng = np.random.default_rng(self.seed)

        train_frame, test_frame = self._classification_feature_frame_pair(
            x_train, x_test
        )
        y_train = self._classification_labels(
            y_train, expected_rows=len(train_frame)
        )
        self._configure_adaptive_svd_runtime(
            train_rows=len(y_train),
            query_rows=len(test_frame),
            unique_dataset_name=None,
        )
        frame = pd.concat([train_frame, test_frame], ignore_index=True)
        member_inputs = self._member_numeric_inputs(
            frame,
            train_size=len(y_train),
            cast_float32=True,
        )

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_train)
        n_classes = len(label_encoder.classes_)

        noise = np_rng.random((self.n_estimators * self.class_shuffle_factor, n_classes))
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)
        balance_count = self.n_estimators // len(uniqs)
        class_permutations = list(
            chain.from_iterable(repeat(elem, balance_count) for elem in uniqs)
        )
        remainder = self.n_estimators % len(uniqs)
        if remainder > 0:
            class_permutations += [
                uniqs[i] for i in np_rng.choice(len(uniqs), size=remainder)
            ]

        members = []
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            member_values, member_categorical = member_inputs[id_pipe]
            x_member = member_values.copy()
            permutation = class_permutations[id_pipe]
            y_member = permutation[y]
            categorical_member = member_categorical.copy()
            extra_seed = id_pipe * self.preprocess_num
            for id_step, step in enumerate(pipe):
                x_member, categorical_member = step.fit_transform(
                    x_member,
                    categorical_member,
                    self.seeds[extra_seed + id_step],
                    y=y_member,
                )
            self._record_svd_adaptation(
                pipeline_index=id_pipe,
                pipeline=pipe,
                unique_dataset_name=None,
            )
            members.append(
                PreparedClassificationMember(
                    x=np.array(x_member, copy=True, order="K"),
                    y=np.array(y_member, copy=True, order="K"),
                    class_permutation=np.array(permutation, copy=True, order="K"),
                )
            )

        return PreparedClassificationInference(
            members=tuple(members),
            train_size=len(y_train),
            n_classes=n_classes,
        )

    def infer_prepared_cls(
        self,
        prepared: PreparedClassificationInference,
    ) -> np.ndarray:
        """Functionality: Run the historical no-retrieval forward on prepare_cls output and equally average member probabilities.

        Input:
            prepared: Output of prepare_cls. members length must equal n_estimators.
                Each member.x is shape (n_train + n_query, n_features_i), member.y
                is shape (n_train,).

        Output:
            np.ndarray: class probabilities, shape (n_query, n_classes), rows sum to 1.
        """
        self._require_prepared_cls_supported()
        if len(prepared.members) != self.n_estimators:
            raise ValueError(
                f"prepared member count {len(prepared.members)} != {self.n_estimators}"
            )
        if self._use_pipeline_gpu_pool():
            result = self._submit_pipeline_gpu_collect(
                {
                    "task_type": "prepared_cls",
                    "prepared": prepared,
                }
            )
            return self._ensemble_cls_outputs(result["members"])
        member_outputs = self._collect_prepared_cls_member_outputs(prepared)
        return self._ensemble_cls_outputs(member_outputs)

    def _collect_prepared_cls_member_outputs(
        self,
        prepared: PreparedClassificationInference,
    ) -> list:
        """Functionality: Forward prepared classification members assigned to this process.

        Input:
            prepared: PreparedClassificationInference.

        Output:
            list of (pipeline_index, logits) pairs.
        """
        outputs = []
        for id_pipe in self._active_pipeline_indices():
            member = prepared.members[id_pipe]
            if len(member.y) != prepared.train_size:
                raise ValueError("prepared y length does not match train_size")
            if len(member.x) <= prepared.train_size:
                raise ValueError("prepared x does not contain query rows")

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            n_classes = prepared.n_classes
            class_permutation = member.class_permutation

            def parse_cls_output(output):
                """Functionality: Extract classification logits from one model output, apply temperature, and restore class order.

                Input:
                    output: Model forward result, dict or Tensor.

                Output:
                    torch.Tensor: query-by-n_classes logits.
                """
                cls_output = output['cls_output'].squeeze(0) if isinstance(output, dict) else output.squeeze(0)
                if self.model_config.get("cls_random_mapping", False):
                    cls_mapping: torch.Tensor = output['cls_process_config']['cls_head_mapping']
                    sample_num, _ = cls_output.shape
                    cls_output = torch.gather(
                        cls_output,
                        dim=-1,
                        index=cls_mapping.expand(sample_num, -1),
                    )
                if self.softmax_temperature != 1:
                    cls_output = cls_output[:, :n_classes].float() / self.softmax_temperature
                return cls_output[..., class_permutation]

            cls_output = self._predict_noretrieval_in_batches(
                x=member.x,
                y=member.y,
                train_size=prepared.train_size,
                task_type="Classification",
                parse_output=parse_cls_output,
                sample_dim=0,
            )
            outputs.append((id_pipe, cls_output))
        return outputs
        
    def _predict_cls(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:str, unique_dataset_name:str=None) -> np.ndarray:
        """Functionality: Main classification path: per-member preprocess, no-retrieval batched forward, CUDA width fallback, then averaged probabilities.

        Input:
            x_train: Train features.
            y_train: Train labels; class count must be in [2, 10].
            x_test: Test features.
            task_type: Should be Classification.
            unique_dataset_name: Optional cache/audit identity.

        Output:
            np.ndarray: row-normalized query-by-class probabilities. Raises RuntimeError if every member fails.
        """
        if self._use_pipeline_gpu_pool():
            with _profile_span("host_prepare"):
                train_frame, test_frame = self._classification_feature_frame_pair(
                    x_train, x_test
                )
                y_checked = self._classification_labels(
                    y_train, expected_rows=len(train_frame)
                )
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(y_checked)
                self.classes = self.label_encoder.classes_
                self.n_classes = len(self.classes)
            with _profile_span("pool_submit"):
                result = self._submit_pipeline_gpu_collect(
                    {
                        "task_type": "Classification",
                        "x_train": x_train,
                        "y_train": y_train,
                        "x_test": x_test,
                        "unique_dataset_name": unique_dataset_name,
                    }
                )
            if not result["members"]:
                self._raise_all_cls_pipelines_failed(result.get("last_failure"))
            with _profile_span("ensemble"):
                return self._ensemble_cls_outputs(result["members"])

        member_outputs, last_cuda_pipeline_failure = self._collect_cls_member_outputs(
            x_train, y_train, x_test, task_type, unique_dataset_name
        )
        if not member_outputs:
            self._raise_all_cls_pipelines_failed(last_cuda_pipeline_failure)
        with _profile_span("ensemble"):
            return self._ensemble_cls_outputs(member_outputs)

    def _collect_cls_member_outputs(
        self,
        x_train,
        y_train,
        x_test,
        task_type,
        unique_dataset_name=None,
    ):
        """Functionality: Run assigned classification pipelines and return per-member logits without ensemble.

        Input:
            x_train: Train features.
            y_train: Train labels.
            x_test: Test features.
            task_type: Should be Classification.
            unique_dataset_name: Optional cache/audit identity.

        Output:
            tuple: (list of (pipeline_index, logits), last CUDA failure record or None).
        """
        np_rng = np.random.default_rng(self.seed)

        train_frame, test_frame = self._classification_feature_frame_pair(
            x_train, x_test
        )
        y_train = self._classification_labels(
            y_train, expected_rows=len(train_frame)
        )
        self._configure_adaptive_svd_runtime(
            train_rows=len(y_train),
            query_rows=len(test_frame),
            unique_dataset_name=unique_dataset_name,
        )
        frame = pd.concat([train_frame, test_frame], ignore_index=True)
        member_inputs = self._member_numeric_inputs(
            frame,
            train_size=len(y_train),
            cast_float32=True,
        )
        
        # Encode y_train
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_train)
        self.classes = self.label_encoder.classes_
        self.n_classes = len(self.classes)
        
        # shuffle y
        noise = np_rng.random((self.n_estimators * self.class_shuffle_factor, self.n_classes))
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)
        balance_count = self.n_estimators // len(uniqs)
        self.class_permutations = list(chain.from_iterable(repeat(elem, balance_count) for elem in uniqs))
        cout = self.n_estimators%len(uniqs)
        if self.n_estimators%len(uniqs) > 0:
            self.class_permutations += [uniqs[i] for i in np_rng.choice(len(uniqs), size=cout)]
        
        outputs = []
        last_cuda_pipeline_failure = None
        
        for id_pipe in self._active_pipeline_indices():
            pipe = self.preprocess_pipelines[id_pipe]
            member_values, member_categorical = member_inputs[id_pipe]
            attempted_feature_width_retries: set[str] = set()
            while True:
                with nvtx.annotate('get-xy'):
                    with _profile_span(f"preprocess.p{id_pipe}"):
                        x_,y_,pipe_class_permutations = self.get_xy_cls(
                            member_values.copy(),
                            id_pipe,
                            y_train.copy(),
                            y.copy(),
                            pipe,
                            member_categorical.copy(),
                            task_type,
                            unique_dataset_name=unique_dataset_name,
                        )
                with nvtx.annotate('svd'):
                    self._record_svd_adaptation(
                        pipeline_index=id_pipe,
                        pipeline=pipe,
                        unique_dataset_name=unique_dataset_name,
                    )
                cur_pipe_class_permutations = pipe_class_permutations

                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                def parse_cls_output(output):
                    """Functionality: Parse classification logits for the current pipeline and restore column order with this member's class permutation.

                    Input:
                        output: Model forward result.

                    Output:
                        torch.Tensor: query-by-n_classes logits.
                    """
                    cls_output = output['cls_output'].squeeze(0) if isinstance(output, dict) else output.squeeze(0)
                    if self.model_config.get("cls_random_mapping", False):
                        cls_mapping: torch.Tensor = output['cls_process_config']['cls_head_mapping']
                        sample_num, _ = cls_output.shape
                        cls_output = torch.gather(cls_output, dim=-1, index=cls_mapping.expand(sample_num, -1))
                    if self.softmax_temperature != 1:
                        cls_output = (cls_output[:, :self.n_classes].float() / self.softmax_temperature)

                    return cls_output[..., cur_pipe_class_permutations]

                try:
                    with _profile_span(f"forward.p{id_pipe}"):
                        cls_output = self._predict_noretrieval_in_batches(
                            x=np.asarray(x_),
                            y=np.asarray(y_),
                            train_size=len(y_train),
                            task_type=task_type,
                            parse_output=parse_cls_output,
                            sample_dim=0,
                        )
                except Exception as error:
                    activated_retry = self._activate_next_feature_width_resource_retry(
                        error=error,
                        pipeline_index=id_pipe,
                        pipeline=pipe,
                        train_rows=len(y_train),
                        query_rows=len(test_frame),
                        unique_dataset_name=unique_dataset_name,
                        attempted_retries=attempted_feature_width_retries,
                    )
                    if activated_retry is not None:
                        attempted_feature_width_retries.add(activated_retry)
                        continue

                    retry_reason = self._cuda_batch_retry_reason(error)
                    allow_partial_ensemble = bool(
                        self.adaptive_svd_config.get(
                            "retry_on_cuda_resource_error",
                            False,
                        )
                    )
                    if retry_reason is None or not allow_partial_ensemble:
                        raise
                    failure_record = {
                        "pipeline_index": int(id_pipe),
                        "reason": retry_reason,
                        "polynomial_interaction_retry_attempted": (
                            "polynomial_interactions"
                            in attempted_feature_width_retries
                        ),
                        "svd_retry_attempted": (
                            "svd" in attempted_feature_width_retries
                        ),
                        "train_rows": int(len(y_train)),
                        "query_rows": int(len(test_frame)),
                        "unique_dataset_name": unique_dataset_name,
                    }
                    self.cuda_pipeline_fallback_audit.append(failure_record)
                    last_cuda_pipeline_failure = failure_record
                    self._release_cuda_after_resource_error(error)
                    print(
                        "CUDA resource failure persisted for one classification "
                        "ensemble pipeline; skipping only that pipeline for the "
                        f"current dataset (pipeline={id_pipe}, "
                        f"reason={retry_reason})."
                    )
                    break
                outputs.append((id_pipe, cls_output))
                break

        return outputs, last_cuda_pipeline_failure
    
    def get_xy_reg(
            self, x, id_pipe,y_train,pipe,categorical_idx,task_type,unique_dataset_name=None
        ):  
        # Build the cache key
        """Functionality: Run feature preprocessing and optional target transforms for one regression member. May use the cache.

        Input:
            x: Concatenated train+test features.
            id_pipe: Pipeline index.
            y_train: Standardized train targets.
            pipe: List of preprocess steps.
            categorical_idx: Categorical column indices.
            task_type: Task type (unused in the computation).
            unique_dataset_name: Optional cache identity.

        Output:
            tuple: (transformed x, transformed y, target_transforms list).
        """
        extra_seed = id_pipe*self.preprocess_num
        key = self.cache_manager.generate_cache_key(
            'reg',
            id_pipe, 
            self.inference_pipeline_config[id_pipe], 
            unique_dataset_name, 
            self.seeds_hash,
            extra_seed
        )
        
        # Define the compute function
        def compute_xy_reg():
            """Functionality: Actually run regression preprocessing. TargetTransform applies to y; other steps apply to x.

            Input:
                None: Closure over x, y_train, pipe, categorical_idx, and id_pipe.

            Output:
                tuple: (x_, y_, target_transforms).
            """
            x_ = x
            y_ = y_train
            categorical_idx_ = categorical_idx
            target_transforms = [None] * len(pipe)
            for id_step, step in enumerate(pipe):
                if isinstance(step, TargetTransform):
                    y_ = step.fit_transform(y_.reshape(-1, 1), categorical_features=None, seed=None).squeeze()
                    target_transforms[id_pipe] = step
                else:
                    x_, categorical_idx_ = step.fit_transform(
                        x_, categorical_idx_, 
                        self.seeds[extra_seed+id_step], 
                        y=y_
                    )
            
            return (x_, y_, target_transforms)
        
        if self.use_data_cache:
            # Run the computation through the cache manager
            return self.cache_manager.cached_computation(key, compute_xy_reg)
        else:
            # Run the computation without caching
            return compute_xy_reg()
    
    def _predict_reg(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:str,unique_dataset_name:str=None) -> np.ndarray:
        """Functionality: Main regression path: validate, standardize targets, preprocess, batched forward, then decode and invert standardization.

        Input:
            x_train: Train features.
            y_train: Train targets.
            x_test: Test features.
            task_type: Should be Regression.
            unique_dataset_name: Optional cache identity.

        Output:
            np.ndarray: 1-D regression predictions aligned with x_test rows, on the original target scale.
        """
        raw_x_train, raw_y_train, raw_x_test = x_train, y_train, x_test
        if self._use_pipeline_gpu_pool():
            with _profile_span("host_prepare"):
                member_inputs, y_train = self._prepare_reg_context(
                    raw_x_train, raw_y_train, raw_x_test
                )
            with _profile_span("pool_submit"):
                collected = self._submit_pipeline_gpu_collect(
                    {
                        "task_type": "Regression",
                        "y_train": y_train,
                        "member_inputs": member_inputs,
                        "unique_dataset_name": unique_dataset_name,
                    }
                )
        else:
            collected = self._collect_reg_member_outputs(
                raw_x_train, raw_y_train, raw_x_test, task_type, unique_dataset_name
            )
        with _profile_span("decode_ensemble"):
            return self._decode_reg_member_outputs(
                collected,
                x_train=self._reg_decode_x_train,
                y_train=self._reg_decode_y_train,
                y_train_ori=self._reg_decode_y_train_ori,
            )

    def _prepare_reg_context(self, x_train, y_train, x_test):
        """Functionality: Validate and standardize regression inputs; store decode fields on self.

        Input:
            x_train: Train features.
            y_train: Train targets.
            x_test: Test features.

        Output:
            tuple: (member_inputs, standardized y_train).
        """
        self.y_mean = float(np.asarray(y_train, dtype=np.float64).mean())
        self.y_std = float(np.asarray(y_train, dtype=np.float64).std(ddof=1))
        if self.y_std == 0:
            self.y_std = 1.0
        x_train, y_train = self.validate_data(x_train, y_train, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        x_test = self.validate_data(x_test, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        y_train = (np.asarray(y_train, dtype=np.float64) - self.y_mean) / self.y_std
        self._reg_decode_x_train = x_train
        self._reg_decode_y_train = y_train
        self._reg_decode_y_train_ori = y_train
        frame = self.convert_x_dtypes(np.concatenate([x_train, x_test], axis=0))
        member_inputs = self._member_numeric_inputs(
            frame,
            train_size=len(y_train),
            cast_float32=False,
            task_type="Regression",
        )
        return member_inputs, y_train

    def _collect_reg_member_outputs(
        self,
        x_train,
        y_train,
        x_test,
        task_type,
        unique_dataset_name=None,
        member_inputs=None,
    ) -> dict:
        """Functionality: Run assigned regression pipelines and return per-member decoder inputs.

        Input:
            x_train: Train features. Unused when member_inputs is provided.
            y_train: Train targets, or host-standardized targets when member_inputs is set.
            x_test: Test features. Unused when member_inputs is provided.
            task_type: Should be Regression.
            unique_dataset_name: Optional cache identity.
            member_inputs: Host-prepared per-pipeline numeric inputs. When set, skip
                _prepare_reg_context so GPU workers do not redo datetime/encode.

        Output:
            dict with members, target_transforms, and target_transforms_pipeline_index.
        """
        if member_inputs is None:
            member_inputs, y_train = self._prepare_reg_context(
                x_train, y_train, x_test
            )
        outputs = []
        target_transforms = None
        last_id_pipe = -1
        for id_pipe in self._active_pipeline_indices():
            pipe = self.preprocess_pipelines[id_pipe]
            member_values, member_categorical = member_inputs[id_pipe]
            with _profile_span(f"preprocess.p{id_pipe}"):
                x_, y_, pipe_target_transforms = self.get_xy_reg(
                    member_values.copy(), id_pipe, y_train.copy(), pipe,
                    member_categorical.copy(), task_type,
                    unique_dataset_name=unique_dataset_name,
                )
            if id_pipe >= last_id_pipe:
                last_id_pipe = id_pipe
                target_transforms = pipe_target_transforms

            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            def parse_reg_output(output):
                # TODO: keep output structure consistent to simplify later maintenance
                """Functionality: Extract regression tensors or bucket logits from a model output, supporting several decoder field names.

                Input:
                    output: Model forward result, dict or list.

                Output:
                    The first list element or the whole list, for later regression decoding.
                """
                if isinstance(output, dict) and 'reg_output' in output:
                    output = output['reg_output']
                elif isinstance(output, dict) and self.model_config['num_buckets'] > 1:  # Bucket-based regression
                    if 'bucket' == self.regression_decoder_type:
                        output = output['output_4_fbar']
                    else:
                        raise ValueError(f"Unknown regression_decoder_type: {self.regression_decoder_type}")
                elif isinstance(output, dict):
                    raise KeyError("Failed to parse regression inference result")

                assert isinstance(output, list), f"regression output should be a list, but got {type(output)}"
                return output[0] if len(output) == 1 else output

            with _profile_span(f"forward.p{id_pipe}"):
                output = self._predict_noretrieval_in_batches(
                    x=np.asarray(x_),
                    y=np.asarray(y_),
                    train_size=len(y_train),
                    task_type=task_type,
                    parse_output=parse_reg_output,
                    sample_dim=1,
                )
            outputs.append((id_pipe, output))
        return {
            "members": outputs,
            "target_transforms": target_transforms,
            "target_transforms_pipeline_index": last_id_pipe,
        }

    def _decode_reg_member_outputs(
        self,
        collected: dict,
        x_train,
        y_train,
        y_train_ori,
    ) -> np.ndarray:
        """Functionality: Decode gathered regression member outputs and invert target standardization.

        Input:
            collected: Dict with members and optional target_transforms.
            x_train: Standardized train features used by some decoders.
            y_train: Standardized train targets.
            y_train_ori: Same standardized targets, historical name.

        Output:
            np.ndarray: 1-D predictions on the original target scale.
        """
        outputs = [
            output
            for _, output in sorted(collected["members"], key=lambda item: item[0])
        ]
        
        if self.model_config['num_buckets'] > 1:
            if 'bucket' == self.regression_decoder_type:
                output = self.get_reg_pred_result(outputs, y_train_ori)
            else:
                raise ValueError(f"Unknown regression_decoder_type: {self.regression_decoder_type}")
            
        # Convert torch.Tensor outputs to numpy.ndarray
        if isinstance(output, torch.Tensor):
            output = output.float().cpu().numpy()

        output = np.asarray(output, dtype=np.float64) * self.y_std + self.y_mean
        return output

    def get_reg_pred_result(self, inputs, y_train:np.ndarray):
        """Functionality: Align each member's bucket logits onto full-support borders, average them, and decode the expectation.

        Input:
            inputs: List of bucket outputs from each member.
            y_train: Train targets in standardized space; used in quantile-border mode.

        Output:
            Regression prediction in standardized space.
        """
        tmp_inputs = [ele[0].squeeze(0) for ele in inputs]
        bucket_borders = self.model._reg_borders
        bucket_widths = bucket_borders[1:] - bucket_borders[:-1]
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        borders_es = [bucket_borders] * len(tmp_inputs)
        inputs = []
        for input_ in tmp_inputs:
            if self.softmax_temperature != 1:
                input_ = input_.float() / self.softmax_temperature
            inputs.append(input_)

        transformed_logits = []
        for logits, borders_t in zip(inputs, borders_es):
            trans_logites = translate_probs_across_borders(
                        logits.squeeze(0),
                        frm=borders_t.to(logits.device),  # torch.as_tensor(borders_t, device=logits.device),
                        to=borders_t.to(logits.device),
                    )
            transformed_logits.append(trans_logites)

        stacked_logits = torch.stack(transformed_logits, dim=0)
        logits = stacked_logits.mean(dim=0)

        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()
            
        output = logits_to_output(logits=logits,
                                  quantiles=quantiles,
                                  output_type='mean',
                                  borders=bucket_borders,
                                  bucket_widths=bucket_widths,
                                )
        return output
    

    def _predict_feature_imputation(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:str,unique_dataset_name:str=None) -> np.ndarray:
        """Functionality: Missing-value prediction: preprocess, forward each member, then invert transforms to recover imputed features.

        Input:
            x_train: Train features.
            y_train: Train labels, used for class permutation.
            x_test: Features on the side being imputed.
            task_type: Should be Feature_imputation.
            unique_dataset_name: Reserved parameter.

        Output:
            np.ndarray | None: member-averaged imputed matrix, or None if there is no prediction.
        """
        if self._use_pipeline_gpu_pool():
            result = self._submit_pipeline_gpu_collect(
                {
                    "task_type": "Feature_imputation",
                    "x_train": x_train,
                    "y_train": y_train,
                    "x_test": x_test,
                    "unique_dataset_name": unique_dataset_name,
                }
            )
            member_outputs = result["members"]
        else:
            member_outputs = self._collect_imputation_member_outputs(
                x_train, y_train, x_test, task_type, unique_dataset_name
            )
        mask_predictions = [
            prediction
            for _, prediction in sorted(member_outputs, key=lambda item: item[0])
        ]
        return np.stack(mask_predictions).mean(axis=0) if mask_predictions != [] else None

    def _collect_imputation_member_outputs(
        self,
        x_train,
        y_train,
        x_test,
        task_type,
        unique_dataset_name=None,
    ) -> list:
        """Functionality: Run assigned missing-value pipelines and return per-member imputed matrices.

        Input:
            x_train: Train features.
            y_train: Train labels.
            x_test: Features on the side being imputed.
            task_type: Should be Feature_imputation.
            unique_dataset_name: Reserved parameter.

        Output:
            list of (pipeline_index, imputed matrix) pairs.
        """
        np_rng = np.random.default_rng(self.seed)
        self._feature_imputation_check_pipeline()
        
        x_train, y_train = self.validate_data(x_train, y_train, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        x_test = self.validate_data(x_test, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        
        # "Concatenate x_train and x_test to ensure the preprocessing logic is completely consistent.
        x = np.concatenate([x_train, x_test], axis=0)
        
        # Encode y_train
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_train)
        self.classes = self.label_encoder.classes_
        self.n_classes = len(self.classes)
        
        # shuffle y
        noise = np_rng.random((self.n_estimators * self.class_shuffle_factor, self.n_classes))
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)
        balance_count = self.n_estimators // len(uniqs)
        self.class_permutations = list(chain.from_iterable(repeat(elem, balance_count) for elem in uniqs))
        cout = self.n_estimators%len(uniqs)
        if self.n_estimators%len(uniqs) > 0:
            self.class_permutations += [uniqs[i] for i in np_rng.choice(len(uniqs), size=cout)]
        
        # Preprocess x
        x = self.convert_x_dtypes(x)
        x = self.convert_category2num(x)
        x = x.astype(np.float32)
        categorical_idx = self.get_categorical_features_indices(x)
        mask_predictions = []
        for id_pipe in self._active_pipeline_indices():
            pipe = self.preprocess_pipelines[id_pipe]
            x_ = x.copy()
            y_ = self.class_permutations[id_pipe][y.copy()]
            categorical_idx_ = categorical_idx.copy()
            for id_step, step in enumerate(pipe):
                x_, categorical_idx_ = step.fit_transform(x_, categorical_idx_, self.seeds[id_pipe*self.preprocess_num+id_step], y=y_)
            
            x_ = torch.from_numpy(x_[:, :]).float().to(self.device)
            y_ = torch.from_numpy(y_).float().to(self.device)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.model.to(self.device)
            with(torch.autocast(device_type=self.device.type if isinstance(self.device, torch.device) else self.device, enabled=self.mix_precision), torch.inference_mode()):
                x_=x_.unsqueeze(0)
                y_ = y_.unsqueeze(0)
                output=self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type=task_type)
                member_preds = []
                self._construct_feature(output, pipe, member_preds, y_)  # Impute missing features
            if member_preds:
                mask_predictions.append((id_pipe, member_preds[0]))
        return mask_predictions
    

    def _feature_imputation_check_pipeline(self):
        """Functionality: Check missing-value preprocess config: warn and replace power transforms, and force discrete_flag=True.

        Input:
            self: Reads and may mutate inference_pipeline_config in place.

        Output:
            None.
        """
        for inference_config_item in self.inference_pipeline_config:
            if len(inference_config_item['RebalanceFeatureDistribution']['worker_tags']) > 0:
                for i, v in enumerate(inference_config_item['RebalanceFeatureDistribution']['worker_tags']):
                    if v == 'power':
                        print("WARNING: Missing value imputation does not currently support the preprocessing method of power! Using the default worker_tags method")
                        inference_config_item['RebalanceFeatureDistribution']['worker_tags'].pop(i)
                        inference_config_item['RebalanceFeatureDistribution']['worker_tags'].append(None)
            inference_config_item['RebalanceFeatureDistribution']['discrete_flag'] = True


    def _construct_feature(self, model_output, pipe, mask_predictions, y_:torch.Tensor):
        """Functionality: Take missing-value-head predictions, invert in-model transforms on numeric/categorical branches, and append to the result list.

        Input:
            model_output: Model output containing feature_pred and related fields.
            pipe: This member's preprocess pipeline, used for later inversion.
            mask_predictions: List that is appended in place.
            y_: Current member label tensor; may be used by bucket decoding.

        Output:
            None. Appends one imputed matrix to mask_predictions.
        """
        process_config = model_output['feature_process_config']
        decoder_config = model_output.get('decoder_config', {})
        if 'x_categorical_mask' not in process_config:
            output_feature_pred = self._feature_imputation_postprocess_in_model(model_output['feature_pred'], process_config)
            output_feature_pred = self._feature_imputation_postprocess(output_feature_pred, pipe, process_config)
        else:
            categorical_feature_pred = model_output['categorical_feature_pred']
            numerical_feature_pred = model_output['numerical_feature_pred']
            tmp_numerical_feature_pred = self._feature_imputation_postprocess_in_model(numerical_feature_pred, process_config)
            tmp_categorical_feature_pred = self._feature_imputation_postprocess_in_model_4_categorical_feature(categorical_feature_pred, process_config)
            x_categorical_mask = process_config['x_categorical_mask'].cpu().numpy()
            if process_config['n_x_padding'] > 0:
                x_categorical_mask = x_categorical_mask[:, :-process_config['n_x_padding']]
            output_feature_pred = np.where(x_categorical_mask, tmp_categorical_feature_pred, tmp_numerical_feature_pred)
            output_feature_pred = self._feature_imputation_postprocess(output_feature_pred, pipe, process_config)
        mask_predictions.append(output_feature_pred)
        

    def _feature_imputation_postprocess_in_model(self, feature_pred:torch.tensor, config: dict) -> torch.tensor:
        # Revert preprocess in model forward
        """Functionality: Undo in-model feature normalization, group scaling, and padding.

        Input:
            feature_pred: Numeric feature-prediction tensor.
            config: feature_process_config.

        Output:
            np.ndarray: sample-by-feature numeric predictions.
        """
        feature_pred = feature_pred / torch.sqrt(config['features_per_group'] / config['num_used_features'].to(self.device))
        feature_pred = feature_pred*config['std_for_normalization'] + config['mean_for_normalization']
        feature_pred = einops.rearrange(feature_pred, "b s f n -> s b (f n)").squeeze(1).float().cpu().numpy()
        if config['n_x_padding'] > 0:
            feature_pred = feature_pred[:,:-config['n_x_padding']]
        return feature_pred

    def _feature_imputation_postprocess_in_model_4_categorical_feature(self, categorical_feature_pred:torch.tensor, config: dict) -> torch.tensor:
        # Revert preprocess in model forward
        """Functionality: Take argmax over categorical feature predictions and drop model padding.

        Input:
            categorical_feature_pred: Categorical logit tensor.
            config: feature_process_config.

        Output:
            np.ndarray: sample-by-feature category indices.
        """
        categorical_feature_pred = categorical_feature_pred.argmax(dim=-1)
        categorical_feature_pred = einops.rearrange(categorical_feature_pred, "b s f n -> s b (f n)").squeeze(1).float().cpu().numpy()
        if config['n_x_padding'] > 0:
            categorical_feature_pred = categorical_feature_pred[:,:-config['n_x_padding']]
        return categorical_feature_pred
    
    def _feature_imputation_postprocess(self, feature_pred:np.ndarray, pipeline:List, config: dict, gt=False) -> np.ndarray:        
        # Revert preprocess in the Classifier
        """Functionality: Invert shuffle/encoding/distribution-rebalancing/invalid-column deletion in reverse pipeline order.

        Input:
            feature_pred: Imputed matrix after in-model inversion.
            pipeline: List of preprocess steps.
            config: Model-side process_config.
            gt: Reserved; currently unused.

        Output:
            np.ndarray: imputed result restored to the original feature layout.
        """
        for id_step, step in enumerate(reversed(pipeline)):
            if isinstance(step, FeatureShuffler):
                if step.mode == "shuffle":
                    inv_p = np.argsort(step.feature_indices)
                    feature_pred = feature_pred[:, inv_p]
                else:
                    raise NotImplementedError
            elif isinstance(step, CategoricalFeatureEncoder):
                if step.encoding_strategy != 'onehot':
                    if step.category_mappings is not None:
                        categorical_indices = list(step.category_mappings.keys())
                        feature_pred[:, categorical_indices] = np.round(feature_pred[:, categorical_indices])
                    if step.transformer is not None:
                        for idx, p in step.category_mappings.items():
                            feature_pred[:, idx] = np.clip(feature_pred[:, idx], a_min=0, a_max=max(p))
                            inv_p = np.argsort(p)
                            feature_pred[:, idx] = inv_p[feature_pred[:, idx].astype(int)].astype(feature_pred.dtype)
                        inv_col = np.argsort(step.feature_indices)
                        feature_pred = feature_pred[:, inv_col]
                else:
                    if len(step.categorical_features) == 0 or step.transformer is None:
                        continue
                    cont_features_indices = [idx for idx in range(feature_pred.shape[1]) if idx not in step.categorical_features]
                    
                    assert np.array_equal(step.categorical_features, np.arange(len(step.categorical_features)))
                    start_idx = 0
                    for idx, out_category in enumerate(step.transformer.named_transformers_['one_hot_encoder'].categories_):
                        assert len(out_category) >= 2
                        if not np.any(np.isnan(out_category)):
                            if len(out_category) == 2: # e.g. [3, 5.5]
                                feature_pred[:,start_idx] = np.round(np.clip(feature_pred[:,start_idx], a_min=0, a_max=1))
                                start_idx += 1
                            else:
                                arr = feature_pred[:, start_idx:start_idx+len(out_category)]
                                feature_pred[:, start_idx:start_idx+len(out_category)] = (arr == arr.max(axis=1, keepdims=True)).astype(float)
                                start_idx += len(out_category)
                        else:
                            if len(out_category) == 2: # e.g. [0, nan]
                                feature_pred[:,start_idx] = 0
                                start_idx += 1
                            else:
                                arr = feature_pred[:, start_idx:start_idx+len(out_category)-1]
                                feature_pred[:, start_idx:start_idx+len(out_category)-1] = (arr == arr.max(axis=1, keepdims=True)).astype(float)
                                feature_pred[:, start_idx+len(out_category)-1] = 0
                                start_idx += len(out_category)
                    feature_pred = np.column_stack([step.transformer.named_transformers_['one_hot_encoder'].inverse_transform(feature_pred[:, step.categorical_features]), feature_pred[:, cont_features_indices]])
                    
            elif isinstance(step, RebalanceFeatureDistribution):
                if step.svd_tag == 'svd' and step.svd_n_comp > 0:
                    feature_pred = feature_pred[:, :-step.svd_n_comp]
                if step.worker_tags[0] in ["quantile_uniform_10", "quantile_uniform_5", "quantile_uniform_all_data"] and step.n_quantile_features > 0:
                    feature_pred = feature_pred[:, :-step.n_quantile_features]
                elif step.worker_tags[0] == "power":
                    raise ValueError(f"Missing value imputation does not currently support the preprocessing method of power!")
                    # reverse feature order
                if step.feature_indices is not None:
                    inv_p = np.argsort(step.feature_indices)
                    feature_pred = feature_pred[:, inv_p]

                    
            elif isinstance(step, FilterValidFeatures):
                deleted_indices = np.where(step.invalid_indices)[0]
                if len(deleted_indices) > 0:
                    original_cols = len(deleted_indices) + feature_pred.shape[1]
                    restored = np.zeros((feature_pred.shape[0], original_cols))                
                    all_indices = set(range(original_cols))
                    kept_indices = list(all_indices - set(deleted_indices)) 
                    for i, idx in enumerate(kept_indices):
                        restored[:, idx] = feature_pred[:, i]                
                    for i, idx in enumerate(deleted_indices):
                        restored[:, idx] = step.invalid_features[:, i]
                    feature_pred = restored.copy()
        return feature_pred
