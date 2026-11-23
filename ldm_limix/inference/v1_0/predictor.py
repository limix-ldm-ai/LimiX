from .inference_method import InferenceAttentionMap, InferenceResultWithRetrieval
from .preprocess import (
    FeatureShuffler, 
    FilterValidFeatures, 
    CategoricalFeatureEncoder, 
    RebalanceFeatureDistribution, 
    FingerprintFeatureEncoder,
    PolynomialInteractionGenerator,
    SubSampleData)
from ldm_limix.utils.loading import load_model, load_from_checkpoint
import torch
from typing import List, Literal
import random
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
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


NA_PLACEHOLDER = "__MISSING__"


def set_deterministic(seed=0):
    """Pin Python, NumPy, and PyTorch RNGs and enable deterministic algorithms.

    Input:
        seed: Integer seed applied to random, numpy, and torch (CPU and all CUDA devices).

    Output:
        None. Also sets CUBLAS_WORKSPACE_CONFIG=:4096:8.
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


class LimiXPredictor:
    """LimiX inferencer for classification, regression, and missing-value prediction.

    Public entry points are __init__, predict, and set_inference_config.
    """
    def __init__(self, 
                 device:torch.device, 
                 model_path:str, 
                 inference_config: list|str,
                 mix_precision:bool=True,
                 outlier_remove_std: float=12,
                 softmax_temperature:float=0.9,
                 average_before_softmax: bool = True,
                 categorical_features_indices:List[int]|None=None,
                 inference_with_DDP: bool = False,
                 use_data_cache: bool = False,
                 seed:int=0,
                 regression_decoder_type: Literal["mse", "bucket", "bucket_4_tabpfn", "hierarchical"] = 'mse',
                 deterministic: bool = False,
                 ckpt: dict | None = None,
                 **kwargs,
                 ):
        """Load the V1.0 predictor: config, preprocess pipelines, and checkpoint.

        Input:
            device: Inference torch.device. CUDA is recommended; CPU forbids retrieval
                and disables mixed precision.
            model_path: Filesystem path to the checkpoint.
            inference_config: list of pipeline dicts, or a JSON path to that list.
                Length must be > 0.
            mix_precision: Use autocast on GPU. Forced False on CPU.
            outlier_remove_std: Std-dev multiplier for outlier clipping.
            softmax_temperature: Temperature for classification/bucket logits. Must be > 0.
            average_before_softmax: Average bucket members before softmax when True.
            categorical_features_indices: Optional categorical column indices; unused
                on the main path.
            inference_with_DDP: If True, use the DDP retrieval inference path.
            use_data_cache: Cache per-member preprocess results on disk when True.
            seed: RNG seed.
            regression_decoder_type: 'mse', 'bucket', 'bucket_4_tabpfn', or 'hierarchical'.
            deterministic: Pin RNGs and enable deterministic algorithms.
            ckpt: Optional already-loaded checkpoint dict; skips torch.load when set.
            **kwargs: Unknown names are warned and ignored.

        Output:
            None. Weights are loaded; GPU transfer happens on the first predict().
        """
        if kwargs:
            print(
                "WARNING: ignoring unsupported predictor kwargs: "
                f"{sorted(kwargs)}"
            )
        # Initialize the cache manager
        self.cache_manager = self.CacheManager()
        self.use_data_cache = use_data_cache

        if isinstance(inference_config, str):
            if os.path.isfile(inference_config):
                with open(inference_config, 'r') as f:
                    inference_config = json.load(f)
            else:
                raise ValueError(f"inference_config is not a config file path: {inference_config}")
        self.model_path = model_path
        self.device = device
        self.mix_precision = mix_precision
        self.categorical_features_indices = categorical_features_indices
        self.seed = seed
        if deterministic: set_deterministic(seed)
        self.inference_config = inference_config
        n_estimators = len(inference_config)
        assert n_estimators > 0, f"Invalid configuration file! the number of pipelines is 0!"
        self.n_estimators = n_estimators
        self.model = None
        self.outlier_remove_std = outlier_remove_std
        self.class_shuffle_factor = 3
        self.min_seq_len_for_category_infer = 100
        self.max_unique_num_for_category_infer = 30
        self.min_unique_num_for_numerical_infer = 4
        self.preprocess_num = 10
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.inference_with_DDP=inference_with_DDP
        self.regression_decoder_type = regression_decoder_type
        if device.type == 'cpu':
            if self.inference_config[0]["retrieval_config"]["use_retrieval"]:
                raise ValueError("Retrieval is not supported for CPU inference! Please use the noretrieval configuration when running on a CPU device!")
            self.mix_precision = False
            print("Mixed precision is not supported for CPU inference, so it has been automatically disabled")
            
        if ckpt is not None:
            self.model, self.model_config = load_from_checkpoint(
                ckpt, mask_prediction=False, deterministic=deterministic
            )
        else:
            self.model, self.model_config = load_model(model_path=model_path, deterministic=deterministic)

        if self.model_config['num_buckets'] > 1:
            self.regression_decoder_type = 'bucket'

        self.preprocess_pipelines = []
        self.preprocess_configs = []

        self.build_preprocess_pipeline()

        # seeds = None
        self.seeds_hash = self.get_and_set_seeds(self.seed)
    
    def get_and_set_seeds(self,seed=None):
        """Record the seed used as a cache-key fragment.

        Input:
            seed: Integer seed, or a historical 6-tuple. The current implementation
                does not mutate global RNGs.

        Output:
            The same seed object, stored later as seeds_hash.
        """
        return seed
    

    class CacheManager:
        """Multi-process-safe on-disk cache. File locks prevent duplicate writes."""
        
        def __init__(self, cache_dir="/mnt/public/infe_cache"):
            """Create the cache directory if it does not exist.

            Input:
                cache_dir: Root directory for pickle cache and lock files.

            Output:
                None.
            """
            self.cache_dir = cache_dir
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
        
        def generate_cache_key(self, infe_type,id_pipe, pipe_config, unique_dataset_name, seeds_hash,extra_seed):
            """Build a cache key from task type, pipeline config, and dataset id.

            Input:
                infe_type: Task tag such as 'cls' or 'reg'.
                id_pipe: Pipeline index, integer >= 0.
                pipe_config: JSON-serializable pipeline dict.
                unique_dataset_name: Dataset identity string.
                seeds_hash: Seed fragment from get_and_set_seeds.
                extra_seed: Unused in the key string; kept for the historical signature.

            Output:
                str: cache key used as the pickle filename stem.
            """
            import json
            import hashlib
            
            pipe_str = json.dumps(pipe_config, sort_keys=True, separators=(',', ':'))
            pipe_hash = hashlib.sha256(pipe_str.encode('utf-8')).hexdigest()
            key = f"{infe_type}_{id_pipe}_{pipe_hash}_{unique_dataset_name}_{seeds_hash}"
            return key
        
        def get_cache_file_path(self, key):
            """Map a cache key to its pickle path, '{cache_dir}/{key}.pkl'."""
            return os.path.join(self.cache_dir, f"{key}.pkl")
        
        def get_lock_file_path(self, key):
            """Map a cache key to its lock path, '{cache_dir}/{key}.lock'."""
            return os.path.join(self.cache_dir, f"{key}.lock")
        
        def read_cache(self, key):
            """Read a cached pickle without locking.

            Input:
                key: Cache key from generate_cache_key.

            Output:
                Unpickled object, or None if missing or corrupted (file is deleted).
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
            """Write a cache entry under a non-blocking exclusive lock.

            Input:
                key: Cache key from generate_cache_key.
                data: Picklable object to store.

            Output:
                bool: True if written, False if the lock could not be taken or I/O failed.
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
            """Return a cached value, or compute and try to write it.

            Input:
                key: Cache key from generate_cache_key.
                compute_func: Zero-or-more-arg callable that produces the value.
                *args, **kwargs: Forwarded to compute_func on a cache miss.

            Output:
                The cached or freshly computed result.
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

    def set_inference_config(self, inference_config: list|str, softmax_temperature:float|None=None, seed:int|None=None):
        """Replace the pipeline config and rebuild preprocess pipelines.

        Input:
            inference_config: list of pipeline dicts, or a JSON path. Length > 0.
            softmax_temperature: Optional new temperature (> 0). None keeps current.
            seed: Optional new integer seed. None keeps current.

        Output:
            None. Updates inference_config, n_estimators, and preprocess_pipelines.
        """
        if isinstance(inference_config, str):
            if os.path.isfile(inference_config):
                with open(inference_config, 'r') as f:
                    inference_config = json.load(f)
            else:
                raise ValueError(f"inference_config is not a config file path: {inference_config}")
        self.inference_config = inference_config
        n_estimators = len(inference_config)
        assert n_estimators > 0, f"Invalid configuration file! the number of pipelines is 0!"
        self.n_estimators = n_estimators
        
        if softmax_temperature is not None:
            self.softmax_temperature = softmax_temperature
        if seed is not None:
            self.seed = seed
        self.build_preprocess_pipeline()
    
    def build_preprocess_pipeline(self):
        """Instantiate preprocess steps from each member config and sample shuffle offsets.

        Input:
            self: Uses n_estimators, inference_config, and seed.

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
            inference_config_item = self.inference_config[idx]
            retrieval_config = inference_config_item["retrieval_config"]
            if retrieval_config["use_retrieval"] and retrieval_config["retrieval_before_preprocessing"]:
                if retrieval_config["subsample_type"] == "sample":
                    assert retrieval_config[
                        "calculate_sample_attention"], "Retrieval on sample level must calculate sample attention score before."
                    if retrieval_config["use_type"] == "mixed":
                        assert retrieval_config[
                            "calculate_feature_attention"], "Retrieval on mixed type must calculate sample and feature attention score before."
                if retrieval_config["subsample_type"] == "feature":
                    assert retrieval_config[
                        "calculate_feature_attention"], "Retrieval on sample level must calculate feature attention score before."
                pipeline.append(
                    InferenceAttentionMap(self.model_path, retrieval_config["calculate_feature_attention"],
                                          retrieval_config["calculate_sample_attention"]))
                pipeline.append(SubSampleData(retrieval_config["subsample_type"], retrieval_config["use_type"]))
            
            if 'PolynomialInteractionGenerator' in inference_config_item:
                pipeline.append(PolynomialInteractionGenerator(**inference_config_item['PolynomialInteractionGenerator']))

            pipeline.append(FilterValidFeatures())

            if 'RebalanceFeatureDistribution' in inference_config_item:
                pipeline.append(RebalanceFeatureDistribution(**inference_config_item['RebalanceFeatureDistribution']))
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
            
            if retrieval_config["use_retrieval"] and not retrieval_config["retrieval_before_preprocessing"]:
                if retrieval_config["subsample_type"] == "sample":
                    assert retrieval_config[
                        "calculate_sample_attention"], "Retrieval on sample level must calculate sample attention score before."
                    if retrieval_config["use_type"] == "mixed":
                        assert retrieval_config[
                            "calculate_feature_attention"], "Retrieval on mixed type must calculate sample and feature attention score before."
                if retrieval_config["subsample_type"] == "feature":
                    assert retrieval_config[
                        "calculate_feature_attention"], "Retrieval on sample level must calculate feature attention score before."
                pipeline.append(
                    InferenceAttentionMap(self.model_path, retrieval_config["calculate_feature_attention"],
                                          retrieval_config["calculate_sample_attention"]))
                pipeline.append(SubSampleData(retrieval_config["subsample_type"], retrieval_config["use_type"]))
            self.preprocess_pipelines.append(pipeline)

    def _check_n_features(self, X, reset):
        """Check that X.shape[1] matches the feature count from the last reset.

        Input:
            X: 2-D array, shape (n_samples, n_features).
            reset: If True, store n_features as n_features_in_; else compare.

        Output:
            None. Raises ValueError on a mismatch when reset is False.
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
        """Validate features and optional labels with sklearn check_X_y / check_array.

        Input:
            x: Feature table, shape (n_samples, n_features), or None.
            y: Optional labels, shape (n_samples,). If set, x and y are checked together.
            reset: Forwarded to _check_n_features.
            validate_separately: Unused; kept for sklearn-style compatibility.
            **check_params: Passed to check_X_y / check_array (e.g. dtype, ensure_all_finite).

        Output:
            (x, y) if y is not None, else x, else None.
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
        """Wrap a numpy feature table as a DataFrame and cast numeric columns.

        Input:
            x: ndarray, shape (n_samples, n_features). Numeric or object dtypes only.
            dtypes: Target float dtype for numeric columns, 'float32' or 'float64'.

        Output:
            pandas.DataFrame with numeric columns cast to dtypes.
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
        """Ordinal-encode category/string/bool columns; restore missing strings to NaN.

        Input:
            x: DataFrame, shape (n_samples, n_features).
            dtype: Floating dtype of the encoded output. Default np.float64.
            placeholder: Temporary fill for missing strings before encoding.

        Output:
            ndarray, shape (n_samples, n_features), numeric with NaN for missing strings.
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
        """Treat low-cardinality columns as categorical when the sample is long enough.

        Input:
            x: Numeric array, shape (n_samples, n_features). Heuristic is skipped
                when n_samples < min_seq_len_for_category_infer.

        Output:
            list[int]: column indices whose unique count is below the numeric threshold.
        """
        if x.shape[0] < self.min_seq_len_for_category_infer:
            return []
        categorical_idx = []
        for idx, col in enumerate(x.T):
            if len(np.unique(col)) < self.min_unique_num_for_numerical_infer:
                categorical_idx.append(idx)
        return categorical_idx
        
    def predict(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:Literal["Classification", "Regression", "Feature_imputation"] = 'Classification',unique_dataset_name:str=None) -> np.ndarray:
        """Run classification, regression, or missing-value imputation.

        Input:
            x_train: Train features, shape (n_train, n_features).
            y_train: Train targets, shape (n_train,).
            x_test: Query features, shape (n_query, n_features), same columns as x_train.
            task_type: 'Classification' (default), 'Regression', or 'Feature_imputation'.
            unique_dataset_name: Optional dataset id for the preprocess cache.

        Output:
            np.ndarray:
                Classification: probabilities, shape (n_query, n_classes), rows sum to 1.
                Regression: predictions, shape (n_query,).
                Feature_imputation: imputed feature matrix.
        """
        if "Classification" == task_type:
            return self._predict_cls(x_train, y_train, x_test, task_type,unique_dataset_name=unique_dataset_name)
        elif "Regression" == task_type:
            return self._predict_reg(x_train, y_train, x_test, task_type,unique_dataset_name=unique_dataset_name)
        elif "Feature_imputation" == task_type:
            return self._predict_feature_imputation(x_train, y_train, x_test, task_type,unique_dataset_name=unique_dataset_name)
        else:
            raise ValueError(f"Unsupported task type, supported tasks include Classification, Regression and Feature_imputation!")

    def get_xy_cls(
            self, x, id_pipe,y_train,y,pipe,categorical_idx,task_type,unique_dataset_name=None
        ):
        """Run the preprocess pipeline for one classification member.

        Input:
            x: Concatenated train+query features, shape (n_train + n_query, n_features).
            id_pipe: Pipeline index in [0, n_estimators).
            y_train: Original train labels, shape (n_train,); used by retrieval steps.
            y: LabelEncoder-transformed train labels, shape (n_train,).
            pipe: List of preprocess steps for this member.
            categorical_idx: Categorical column indices for this table.
            task_type: Forwarded to retrieval attention, typically 'Classification'.
            unique_dataset_name: Optional cache identity.

        Output:
            tuple: (x_processed, y_permuted, attention_score_or_None, class_permutation).
        """
        # Build the cache key
        extra_seed = id_pipe*self.preprocess_num
        key = self.cache_manager.generate_cache_key(
            'cls',
            id_pipe, 
            self.inference_config[id_pipe], 
            unique_dataset_name, 
            self.seeds_hash,
            extra_seed
        )
        
        # Define the compute function
        def compute_xy_cls():
            """Apply classification preprocess steps and permute labels for this member."""
            x_ = x
            y_ = self.class_permutations[id_pipe][y]
            categorical_idx_ = categorical_idx
            attention_score = None
            
            for id_step, step in enumerate(pipe):
                if isinstance(step, InferenceAttentionMap):
                    feature_attention_score, sample_attention_score = step.inference(
                        X_train=x_[:len(y_train)].astype(np.float32),
                        y_train=y_train.astype(np.float32),
                        X_test=x_[len(y_train):].astype(np.float32),
                        task_type=task_type, device=self.device
                    )
                elif isinstance(step, SubSampleData):
                    step.fit(
                        torch.from_numpy(x_[:len(y_train)]), 
                        torch.from_numpy(y_train),
                        feature_attention_score=feature_attention_score,
                        sample_attention_score=sample_attention_score,
                        subsample_ratio=self.inference_config[id_pipe]["retrieval_config"].get("sub_feature_ratio", 0.5)
                    )
                    if self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "feature":
                        x_ = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                        categorical_idx_ = self.get_categorical_features_indices(x_)
                    else:
                        attention_score = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                else:
                    x_, categorical_idx_ = step.fit_transform(
                        x_, categorical_idx_, 
                        self.seeds[extra_seed+id_step], 
                        y=y_
                    )
            
            return (x_, y_, attention_score,self.class_permutations[id_pipe])
        if self.use_data_cache:
            # Run the computation through the cache manager
            return self.cache_manager.cached_computation(key, compute_xy_cls)
        else :
            return compute_xy_cls()
        
    def _predict_cls(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:str, unique_dataset_name:str=None) -> np.ndarray:
        """Classification path: preprocess each member, forward, then average probabilities.

        Input:
            x_train: Train features, shape (n_train, n_features).
            y_train: Train labels, shape (n_train,).
            x_test: Query features, shape (n_query, n_features).
            task_type: Should be 'Classification'.
            unique_dataset_name: Optional preprocess-cache id.

        Output:
            np.ndarray: probabilities, shape (n_query, n_classes), rows sum to 1.
        """
        np_rng = np.random.default_rng(self.seed)
        
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
        outputs = []
        
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):  
            x_,y_,attention_score,pipe_class_permutations = self.get_xy_cls(x.copy(), id_pipe,y_train.copy(),y.copy(),pipe,categorical_idx,task_type,unique_dataset_name=unique_dataset_name)
            cur_pipe_class_permutations = pipe_class_permutations
            
            x_ = torch.from_numpy(x_[:, :]).float().to(self.device)
            y_ = torch.from_numpy(y_).float().to(self.device)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if self.inference_config[id_pipe]["retrieval_config"]["use_retrieval"] and \
                    self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "sample":
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="AM")
                output = inference.inference(x_[:len(y_train)], y_,
                                             x_[len(y_train):],
                                             attention_score=attention_score,
                                             retrieval_len=self.inference_config[id_pipe]["retrieval_config"]["retrieval_len"],
                                             dynamic_ratio=self.inference_config[id_pipe]["retrieval_config"].get("dynamic_ratio", None),
                                             use_cluster=self.inference_config[id_pipe]["retrieval_config"].get("use_cluster", False),
                                             cluster_num=self.inference_config[id_pipe]["retrieval_config"].get("cluster_num", 20),
                                             task_type=task_type,
                                             use_threshold=self.inference_config[id_pipe]["retrieval_config"].get("use_threshold", False),
                                             threshold=self.inference_config[id_pipe]["retrieval_config"].get("threshold", 1),
                                             mixed_method=self.inference_config[id_pipe]["retrieval_config"].get("mixed_method", "max"),
                                             device=self.device)
                if self.softmax_temperature != 1:
                    output = (output[:, :self.n_classes].float() / self.softmax_temperature)

                # output = output[..., self.class_permutations[id_pipe]]
                output = output[..., cur_pipe_class_permutations]
                outputs.append(output)
            elif self.inference_with_DDP:
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="DDP")
                output = inference.inference(x_[:len(y_train)].squeeze(1), y_, x_[len(y_train):].squeeze(1),
                                             task_type=task_type)
                if self.softmax_temperature != 1:
                    output = (output[:, :self.n_classes].float() / self.softmax_temperature)

                # output = output[..., self.class_permutations[id_pipe]]
                output = output[..., cur_pipe_class_permutations]
                outputs.append(output)
            else:
                self.model.to(self.device)
                with(torch.autocast(device_type=self.device.type if isinstance(self.device, torch.device) else self.device, enabled=self.mix_precision), torch.inference_mode()):
                    x_=x_.unsqueeze(0)
                    y_ = y_.unsqueeze(0)
                    output=self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type=task_type)
                    cls_output = output['cls_output'].squeeze(0) if isinstance(output, dict) else output.squeeze(0)
                    if self.model_config.get("cls_random_mapping", False):
                        cls_mapping: torch.Tensor = output['cls_process_config']['cls_head_mapping']
                        sample_num, _ = cls_output.shape
                        cls_output = torch.gather(cls_output, dim=-1, index=cls_mapping.expand(sample_num, -1))
                    if self.softmax_temperature != 1:
                        cls_output = (cls_output[:, :self.n_classes].float() / self.softmax_temperature)

                    # output = output[..., self.class_permutations[id_pipe]]
                    
                    cls_output = cls_output[..., cur_pipe_class_permutations]
                outputs.append(cls_output)
            
        outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
        output = torch.stack(outputs).mean(dim=0)
        output = output.float().cpu().numpy()

        return output / output.sum(axis=1, keepdims=True)
    
    def get_xy_reg(
            self, x, id_pipe,y_train,pipe,categorical_idx,task_type,unique_dataset_name=None
        ):
        """Run feature (and optional target) preprocessing for one regression member.

        Input:
            x: Concatenated train+query features, shape (n_train + n_query, n_features).
            id_pipe: Pipeline index in [0, n_estimators).
            y_train: Train targets, shape (n_train,).
            pipe: List of preprocess steps for this member.
            categorical_idx: Categorical column indices.
            task_type: Typically 'Regression'.
            unique_dataset_name: Optional cache identity.

        Output:
            tuple: (x_processed, y_processed, attention_score_or_None, target_transforms).
        """
        # Build the cache key
        extra_seed = id_pipe*self.preprocess_num
        key = self.cache_manager.generate_cache_key(
            'reg',
            id_pipe, 
            self.inference_config[id_pipe], 
            unique_dataset_name, 
            self.seeds_hash,
            extra_seed
        )
        
        # Define the compute function
        def compute_xy_reg():
            """Apply regression preprocess steps. TargetTransform applies to y; others to x."""
            x_ = x
            y_ = y_train
            categorical_idx_ = categorical_idx
            attention_score = None
            target_transforms = [None] * len(pipe)
            for id_step, step in enumerate(pipe):
                if isinstance(step, InferenceAttentionMap):
                    feature_attention_score, sample_attention_score = step.inference(
                        X_train=x_[:len(y_train)].astype(np.float32),
                        y_train=y_.astype(np.float32),
                        X_test=x_[len(y_train):].astype(np.float32),
                        task_type=task_type, device=self.device
                    )
                elif isinstance(step, SubSampleData):
                    step.fit(
                        torch.from_numpy(x_[:len(y_train)]), 
                        torch.from_numpy(y_train),
                        feature_attention_score=feature_attention_score,
                        sample_attention_score=sample_attention_score,
                        subsample_ratio=self.inference_config[id_pipe]["retrieval_config"].get("sub_feature_ratio", 0.5)
                    )
                    if self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "feature":
                        x_ = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                        categorical_idx_ = self.get_categorical_features_indices(x_)
                    else:
                        attention_score = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                elif isinstance(step, TargetTransform):
                    y_ = step.fit_transform(y_.reshape(-1, 1), categorical_features=None, seed=None).squeeze()
                    target_transforms[id_pipe] = step
                else:
                    x_, categorical_idx_ = step.fit_transform(
                        x_, categorical_idx_, 
                        self.seeds[extra_seed+id_step], 
                        y=y_
                    )
            
            return (x_, y_, attention_score, target_transforms)
        
        if self.use_data_cache:
            # Run the computation through the cache manager
            return self.cache_manager.cached_computation(key, compute_xy_reg)
        else:
            # Run the computation without caching
            return compute_xy_reg()
    
    def _predict_reg(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:str,unique_dataset_name:str=None) -> np.ndarray:
        """Regression path: standardize y, preprocess, forward, decode, invert standardization.

        Input:
            x_train: Train features, shape (n_train, n_features).
            y_train: Train targets, shape (n_train,).
            x_test: Query features, shape (n_query, n_features).
            task_type: Should be 'Regression'.
            unique_dataset_name: Optional preprocess-cache id.

        Output:
            np.ndarray: predictions, shape (n_query,), original target scale.
        """
        np_rng = np.random.default_rng(self.seed)
        y_train_ori = y_train
        x_train, y_train = self.validate_data(x_train, y_train, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        x_test = self.validate_data(x_test, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        
        # "Concatenate x_train and x_test to ensure the preprocessing logic is completely consistent.
        x = np.concatenate([x_train, x_test], axis=0)
    
        # preprocess x
        x = self.convert_x_dtypes(x)
        x = self.convert_category2num(x)
        categorical_idx = self.get_categorical_features_indices(x)
    
        outputs = []
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            x_, y_, attention_score, target_transforms = self.get_xy_reg(x.copy(), id_pipe, y_train.copy(),  pipe, categorical_idx.copy(), task_type, unique_dataset_name=unique_dataset_name)
            
            x_ = torch.from_numpy(x_[:, :]).float().to(self.device)
            y_ = torch.from_numpy(y_).float().to(self.device)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if self.inference_config[id_pipe]["retrieval_config"]["use_retrieval"] and \
                    self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "sample":
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="AM")
                output = inference.inference(x_[:len(y_train)], y_,
                                             x_[len(y_train):],
                                             attention_score=attention_score,
                                             retrieval_len=self.inference_config[id_pipe]["retrieval_config"]["retrieval_len"],
                                             dynamic_ratio=self.inference_config[id_pipe]["retrieval_config"].get("dynamic_ratio", None),
                                             use_cluster=self.inference_config[id_pipe]["retrieval_config"].get("use_cluster", False),
                                             cluster_num=self.inference_config[id_pipe]["retrieval_config"].get("cluster_num", 20),
                                             task_type=task_type,
                                             use_threshold=self.inference_config[id_pipe]["retrieval_config"].get("use_threshold", False),
                                             threshold=self.inference_config[id_pipe]["retrieval_config"].get("threshold", 1),
                                             mixed_method=self.inference_config[id_pipe]["retrieval_config"].get("mixed_method", "max"),
                                             device=self.device,
                                             num_buckets=self.model_config['num_buckets'],
                                             regression_decoder_type=self.regression_decoder_type)
                outputs.append(output)
            elif self.inference_with_DDP:
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="DDP")
                output = inference.inference(x_[:len(y_train)].squeeze(1), y_, x_[len(y_train):].squeeze(1),
                                             task_type=task_type,
                                             num_buckets=self.model_config['num_buckets'],
                                             regression_decoder_type=self.regression_decoder_type)
                outputs.append(output)
            else:
                self.model.to(self.device)
                with(torch.autocast(device_type=self.device.type if isinstance(self.device, torch.device) else self.device, enabled=self.mix_precision), torch.inference_mode()):
                    x_=x_.unsqueeze(0)
                    y_ = y_.unsqueeze(0)
                    output=self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type=task_type)

                # TODO: keep output structure consistent to simplify later maintenance
                if isinstance(output, dict):
                    if 'reg_output' in output:
                        output = output['reg_output']
                    elif self.model_config['num_buckets'] > 1:  # Bucket-based regression
                        if 'mse' == self.regression_decoder_type:
                            output = output['output_4_mse']
                        elif 'bucket' == self.regression_decoder_type:
                            output = output['output_4_fbar']
                        else:
                            raise ValueError(f"Unknown regression_decoder_type: {self.regression_decoder_type}")
                    else:
                        raise KeyError("Failed to parse regression inference result")

                assert isinstance(output, list), f"regression output should be a list, but got {type(output)}"
                if 1 == len(output):
                    output = output[0]
                outputs.append(output)
        
        if self.model_config['num_buckets'] > 1:
            if 'mse' == self.regression_decoder_type:
                output = self.reg_pred_result_from_joint_reg_model(outputs)
            elif 'bucket' == self.regression_decoder_type:
                output = self.get_reg_pred_result(outputs, y_train_ori)
            else:
                raise ValueError(f"Unknown regression_decoder_type: {self.regression_decoder_type}")
        else:
            output = self.reg_pred_result_from_mse_model(outputs)
            
        # Convert torch.Tensor outputs to numpy.ndarray
        if isinstance(output, torch.Tensor):
            output = output.float().cpu().numpy()

        return output

    def get_reg_pred_result(self, inputs, y_train:np.ndarray):
        """Average member bucket logits on full-support borders and decode the mean.

        Input:
            inputs: Per-member model outputs; each ele[0] holds bucket logits.
            y_train: Standardized train targets, shape (n_train,). Used when
                borders come from training-set quantiles.

        Output:
            Prediction in standardized space, shape (n_query,) or a tensor of that rank.
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


    def reg_pred_result_from_joint_reg_model(self, inputs):
        """Mean of per-member MSE/joint regression heads.

        Input:
            inputs: Per-member tensors after ele[0], typically (1, n_query, 1).

        Output:
            Tensor of shape (n_query,).
        """
        inputs = [ele[0].squeeze(0) for ele in inputs]
        output = torch.stack(inputs).squeeze(2).mean(dim=0)
        return output
    
    def reg_pred_result_from_mse_model(self, inputs):
        """Mean of per-member MSE regression scalars.

        Input:
            inputs: Per-member tensors after ele[0], typically (1, n_query, 1).

        Output:
            Tensor of shape (n_query,).
        """
        inputs = [ele[0].squeeze(0) for ele in inputs]
        output = torch.stack(inputs).squeeze(2).mean(dim=0)
        return output

    def _predict_feature_imputation(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:str,unique_dataset_name:str=None) -> np.ndarray:
        """Impute missing entries in the concatenated train+query feature table.

        Input:
            x_train: Train features, shape (n_train, n_features).
            y_train: Train labels/targets, shape (n_train,); used as context.
            x_test: Query features, shape (n_query, n_features).
            task_type: Should be 'Feature_imputation'.
            unique_dataset_name: Optional preprocess-cache id.

        Output:
            np.ndarray: imputed features for the concatenated table.
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
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            x_ = x.copy()
            y_ = self.class_permutations[id_pipe][y.copy()]
            categorical_idx_ = categorical_idx.copy()
            for id_step, step in enumerate(pipe):
                if isinstance(step, InferenceAttentionMap):
                    feature_attention_score, sample_attention_score = step.inference(X_train=x_[:len(y_train)].astype(np.float32),
                                                                                     y_train=y_train.astype(np.float32),
                                                                                     X_test=x_[len(y_train):].astype(np.float32),
                                                                                     task_type=task_type,
                                                                                     device=self.device)
                   
                elif isinstance(step, SubSampleData):
                    step.fit(torch.from_numpy(x_[:len(y_train)]), torch.from_numpy(y_train),
                             feature_attention_score=feature_attention_score,
                             sample_attention_score=sample_attention_score,
                             subsample_ratio=self.inference_config[id_pipe]["retrieval_config"].get("sub_feature_ratio", 0.5))
                    if self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "feature":
                        x_ = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                        categorical_idx_ = self.get_categorical_features_indices(x_)
                    else:
                        attention_score = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                else:
                    x_, categorical_idx_ = step.fit_transform(x_, categorical_idx_, self.seeds[id_pipe*self.preprocess_num+id_step], y=y_)
            
            x_ = torch.from_numpy(x_[:, :]).float().to(self.device)
            y_ = torch.from_numpy(y_).float().to(self.device)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if self.inference_config[id_pipe]["retrieval_config"]["use_retrieval"] and \
                    self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "sample":
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="AM")
                output = inference.inference(x_[:len(y_train)], y_,
                                             x_[len(y_train):],
                                             attention_score=attention_score,
                                             retrieval_len=self.inference_config[id_pipe]["retrieval_config"]["retrieval_len"],
                                             dynamic_ratio=self.inference_config[id_pipe]["retrieval_config"].get("dynamic_ratio", None),
                                             use_cluster=self.inference_config[id_pipe]["retrieval_config"].get("use_cluster", False),
                                             cluster_num=self.inference_config[id_pipe]["retrieval_config"].get("cluster_num", 20),
                                             task_type=task_type,
                                             use_threshold=self.inference_config[id_pipe]["retrieval_config"].get("use_threshold", False),
                                             threshold=self.inference_config[id_pipe]["retrieval_config"].get("threshold", 1),
                                             mixed_method=self.inference_config[id_pipe]["retrieval_config"].get("mixed_method", "max"),
                                             device=self.device)
            elif self.inference_with_DDP:
                inference = InferenceResultWithRetrieval(model=self.model, 
                                                         sample_selection_type="DDP")
                output = inference.inference(x_[:len(y_train)].squeeze(1), y_, x_[len(y_train):].squeeze(1),
                                             task_type=task_type)
            else:
                self.model.to(self.device)
                with(torch.autocast(device_type=self.device.type if isinstance(self.device, torch.device) else self.device, enabled=self.mix_precision), torch.inference_mode()):
                    x_=x_.unsqueeze(0)
                    y_ = y_.unsqueeze(0)
                    output=self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type=task_type)
                    self._construct_feature(output, pipe, mask_predictions, y_)  # Impute missing features
            
        mask_prediction = np.stack(mask_predictions).mean(axis=0) if mask_predictions != [] else None
        
        return mask_prediction
    

    def _feature_imputation_check_pipeline(self):
        """Drop unsupported 'power' QTx tags and force discrete_flag for imputation.

        Input:
            self: Mutates each member's RebalanceFeatureDistribution in place.

        Output:
            None.
        """
        for inference_config_item in self.inference_config:
            if len(inference_config_item['RebalanceFeatureDistribution']['worker_tags']) > 0:
                for i, v in enumerate(inference_config_item['RebalanceFeatureDistribution']['worker_tags']):
                    if v == 'power':
                        print("WARNING: Missing value imputation does not currently support the preprocessing method of power! Using the default worker_tags method")
                        inference_config_item['RebalanceFeatureDistribution']['worker_tags'].pop(i)
                        inference_config_item['RebalanceFeatureDistribution']['worker_tags'].append(None)
            inference_config_item['RebalanceFeatureDistribution']['discrete_flag'] = True


    def _construct_feature(self, model_output, pipe, mask_predictions, y_:torch.Tensor):
        """Invert in-model and pipeline preprocess and append one member's imputations.

        Input:
            model_output: Dict with feature predictions and feature_process_config.
            pipe: Preprocess steps to invert.
            mask_predictions: List that this call appends to.
            y_: Train targets on device; unused in the current body.

        Output:
            None. Appends a numpy feature matrix to mask_predictions.
        """
        process_config = model_output['feature_process_config']
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
        """Undo model-side feature scaling and packing for numeric imputations.

        Input:
            feature_pred: Packed model tensor, layout (b, s, f, n).
            config: feature_process_config with std/mean, padding, features_per_group.

        Output:
            ndarray, shape (n_samples, n_features) on CPU, padding columns dropped.
        """
        # Revert preprocess in model forward
        feature_pred = feature_pred / torch.sqrt(config['features_per_group'] / config['num_used_features'].to(self.device))
        feature_pred = feature_pred*config['std_for_normalization'] + config['mean_for_normalization']
        feature_pred = einops.rearrange(feature_pred, "b s f n -> s b (f n)").squeeze(1).float().cpu().numpy()
        if config['n_x_padding'] > 0:
            feature_pred = feature_pred[:,:-config['n_x_padding']]
        return feature_pred

    def _feature_imputation_postprocess_in_model_4_categorical_feature(self, categorical_feature_pred:torch.tensor, config: dict) -> torch.tensor:
        """Argmax categorical imputations and undo packing/padding.

        Input:
            categorical_feature_pred: Logits over category ids, last dim = class.
            config: feature_process_config with n_x_padding.

        Output:
            ndarray, shape (n_samples, n_features) of predicted category ids.
        """
        # Revert preprocess in model forward
        categorical_feature_pred = categorical_feature_pred.argmax(dim=-1)
        categorical_feature_pred = einops.rearrange(categorical_feature_pred, "b s f n -> s b (f n)").squeeze(1).float().cpu().numpy()
        if config['n_x_padding'] > 0:
            categorical_feature_pred = categorical_feature_pred[:,:-config['n_x_padding']]
        return categorical_feature_pred
    
    def _feature_imputation_postprocess(self, feature_pred:np.ndarray, pipeline:List, config: dict, gt=False) -> np.ndarray:
        """Invert pipeline feature shuffle / categorical encoding / rebalancing.

        Input:
            feature_pred: Imputed table, shape (n_samples, n_features).
            pipeline: Preprocess steps in forward order; inverted here.
            config: feature_process_config; used when undoing SVD/rebalance.
            gt: Unused; kept for the historical signature.

        Output:
            ndarray of the same rank, columns restored to the original feature order.
        """
        # Revert preprocess in the Classifier
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
