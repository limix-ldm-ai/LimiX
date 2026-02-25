import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import warnings
import scipy
from typing_extensions import override
from typing import Literal, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
    PowerTransformer,
    StandardScaler,
    QuantileTransformer, 
    MinMaxScaler,
    RobustScaler
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted
from utils.data_utils import TabularInferenceDataset
from torch.cuda import OutOfMemoryError

import hashlib
from kditransform import KDITransformer

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)

class SelectiveInversePipeline(Pipeline):
    def __init__(self, steps, skip_inverse=None):
        super().__init__(steps)
        self.skip_inverse = skip_inverse or []
    
    def inverse_transform(self, X):
        """跳过指定步骤的inverse_transform"""
        if X.shape[1] == 0:
            return X
        for step_idx in range(len(self.steps) - 1, -1, -1):
            name, transformer = self.steps[step_idx]
            try:
                check_is_fitted(transformer)
            except Exception:
                continue
            
            if name in self.skip_inverse:
                continue
                
            if hasattr(transformer, 'inverse_transform'):
                X = transformer.inverse_transform(X)
                if np.any(np.isnan(X)):
                    print(f"After reverse RebalanceFeatureDistribution of {name}, there is nan")
        return X

class RobustPowerTransformer(PowerTransformer):
    """PowerTransformer with automatic feature reversion when variance or value constraints fail."""

    def __init__(self, var_tolerance: float = 1e-3,
                 max_abs_value: float = 100,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.var_tolerance = var_tolerance
        self.max_abs_value = max_abs_value
        self.restore_indices_: np.ndarray | None = None


    def fit(self, X, y=None):
        fitted = super().fit(X, y)
        self.restore_indices_ = np.array([], dtype=int)
        return fitted

    def fit_transform(self, X, y=None):
        Z = super().fit_transform(X,y)
        self.restore_indices_ = self._should_revert(Z)
        return Z

    def _should_revert(self, Z: np.ndarray) -> np.ndarray:
        """Determine which columns to revert to their original values."""
        variances = np.nanvar(Z, axis=0)
        bad_var = np.flatnonzero(np.abs(variances - 1.0) > self.var_tolerance)

        bad_large = np.flatnonzero(np.any(Z > self.max_abs_value, axis=0))

        return np.unique(np.concatenate([bad_var, bad_large]))

    def _apply_reversion(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.restore_indices_.size > 0:
            Z[:, self.restore_indices_] = X[:, self.restore_indices_]
        return Z

    def transform(self, X):
        Z = super().transform(X)
        # self.restore_indices_ = self._should_revert(Z)
        return self._apply_reversion(Z, X)

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        "Overload_yeo_johnson_optimize to avoid crashes caused by values such as NaN and Inf."
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message=r"overflow encountered",
                                        category=RuntimeWarning)
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except Exception as e:
            return np.nan

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        "_yeo_johnson_transform to avoid crashes caused by NaN"
        if np.isnan(lmbda):
            return x
        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore

class BasePreprocess:
    """Abstract base class for preprocessing class"""

    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs)->list[int]:
        """Fit the preprocessing model to the data"""
        raise NotImplementedError
    
    def transform(self, x:np.ndarray, **kwargs)->tuple[np.ndarray, list[int]]:
        """Transform the data using the fitted preprocessing model"""
        raise NotImplementedError
    
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs)->tuple[np.ndarray, list[int]]:
        """Fit the preprocessing model to the data and transform the data"""
        self.fit(x, categorical_features, seed, **kwargs)
        return self.transform(x, **kwargs)

def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer the random state and return the seed and generator"""
    if random_state is None:
        np_rng = np.random.default_rng()
        return int(np_rng.integers(0, MAXINT_RANDOM_SEED)), np_rng
        
    if isinstance(random_state, (int, np.integer)):
        return int(random_state), np.random.default_rng(random_state)
        
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, MAXINT_RANDOM_SEED))
        return seed, np.random.default_rng(seed)
        
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, MAXINT_RANDOM_SEED)), random_state
        
    raise ValueError(f"Invalid random_state {random_state}")

class FilterValidFeatures(BasePreprocess):
    def __init__(self):
        self.valid_features: list[bool] | None = None
        self.categorical_idx: list[int] | None = None
        self.invalid_indices: list[int] | None = None
        self.invalid_features: list[int] | None = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, y:np.ndarray | None = None, **kwargs) -> list[int]:
        self.categorical_idx = categorical_features
        self.valid_features = ((x[0:1, :] == x).mean(axis=0) < 1.0).tolist()
        self.invalid_indices = ((x[0:1, :] == x).mean(axis=0) == 1.0).tolist()

        if y is not None:
            eval_pos = len(y)
            nan_train = np.isnan(x[:eval_pos, :])
            all_nan_train = np.all(nan_train, axis=0)
            nan_test = np.isnan(x[eval_pos:, :])
            all_nan_test = np.all(nan_test, axis=0)
            
            features_nan = all_nan_train | all_nan_test
            self.valid_features = self.valid_features & ~features_nan
            self.invalid_indices = self.invalid_indices | features_nan

        if not any(self.valid_features):
            raise ValueError("All features are constant! Please check your data.")

        self.categorical_idx = [
            index
            for index, idx in enumerate(np.where(self.valid_features)[0])
            if idx in categorical_features
        ]

        return self.categorical_idx
    
    @override
    def transform(self, x:np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        assert self.valid_features is not None, "You must call fit first to get effective_features"
        self.invalid_features = x[:, self.invalid_indices]
        return x[:, self.valid_features], self.categorical_idx

class FeatureShuffler(BasePreprocess):
    """
    Feature column reordering preprocessor
    """

    def __init__(
        self,
        mode: Literal['rotate', 'shuffle'] | None = "shuffle",
        offset: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.offset = offset
        self.random_seed = None
        self.feature_indices = None
        self.categorical_indices = None
    
    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        n_features = x.shape[1]
        self.random_seed = seed
        
        indices = np.arange(n_features)
        
        if self.mode == "rotate":
            self.feature_indices = np.roll(indices, self.offset)
        elif self.mode == "shuffle":
            _, rng = infer_random_state(self.random_seed)
            self.feature_indices = rng.permutation(indices)
        elif self.mode is None:
            self.feature_indices = np.arange(n_features)
        else:
            raise ValueError(f"Unsupported reordering mode: {self.mode}")

        is_categorical = np.isin(np.arange(n_features), categorical_features)
        self.categorical_indices = np.where(is_categorical[self.feature_indices])[0].tolist()
        
        return self.categorical_indices

    @override
    def transform(self, x:np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        if self.feature_indices is None:
            raise RuntimeError("Please call the fit method first to initialize")
        if len(self.feature_indices) != x.shape[1]:
            raise ValueError("The number of features in the input data does not match the training data")
            
        return x[:, self.feature_indices], self.categorical_indices or []

class CategoricalFeatureEncoder(BasePreprocess):
    """
    Categorical feature encoder
    """

    def __init__(
        self,
        encoding_strategy: Literal['ordinal', 'ordinal_strict_feature_shuffled', 'ordinal_shuffled', 'onehot', 'numeric', 'none']|None = "ordinal",
    ):
        super().__init__()
        self.encoding_strategy = encoding_strategy
        self.random_seed = None
        self.transformer = None
        self.category_mappings = None
        self.categorical_features = None

    @override
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> tuple[np.ndarray, list[int]]:
        self.random_seed = seed
        return self._fit_transform(x, categorical_features)

    def _fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        # print(f"encoding_strategy: {self.encoding_strategy}")
        ct, categorical_features = self._create_transformer(X, categorical_features)
        if ct is None:
            self.transformer = None
            return X, categorical_features

        _, rng = infer_random_state(self.random_seed)

        if self.encoding_strategy.startswith("ordinal"):       
            Xt = ct.fit_transform(X)
            categorical_features = list(range(len(categorical_features)))

            if self.encoding_strategy.endswith("_shuffled"):
                self.category_mappings = {}
                for col_ix in categorical_features:
                    col_cats = len(
                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
                    )
                    perm = rng.permutation(col_cats)
                    self.category_mappings[col_ix] = perm
                    
                    col_data = Xt[:, col_ix]
                    valid_mask = ~np.isnan(col_data)
                    col_data[valid_mask] = perm[col_data[valid_mask].astype(int)].astype(col_data.dtype)

        elif self.encoding_strategy == "onehot":
            Xt = ct.fit_transform(X)
            if Xt.size >= 1_000_000:
                ct = None
                Xt = X
            else:
                categorical_features = list(range(Xt.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
        else:
            raise ValueError(
                f"Unknown categorical transform {self.encoding_strategy}",
            )

        self.transformer = ct
        self.categorical_features = categorical_features
        return Xt, categorical_features

    @staticmethod
    def get_least_common_category_count(column: np.ndarray) -> int:
        """Retrieve the smallest count value among categorical features"""
        if len(column) == 0:
            return 0
        return int(np.unique(column, return_counts=True)[1].min())

    def _create_transformer(self, data: np.ndarray, categorical_columns: list[int]) -> tuple[ColumnTransformer | None, list[int]]:
        """Create an appropriate column transformer"""
        if self.encoding_strategy.startswith("ordinal"):
            suffix = self.encoding_strategy[len("ordinal"):]
            
            if "feature_shuffled" in suffix:
                categorical_columns = [
                    idx for idx in categorical_columns 
                    if self._is_valid_common_category(data[:, idx], suffix)
                ]
            remainder_columns = [idx for idx in range(data.shape[1]) if idx not in categorical_columns]
            self.feature_indices = categorical_columns + remainder_columns
                
            return ColumnTransformer(
                [("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), categorical_columns)],
                remainder="passthrough"
            ), categorical_columns
            
        elif self.encoding_strategy == "onehot":
            return ColumnTransformer(
                [("one_hot_encoder", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"), categorical_columns)],
                remainder="passthrough"
            ), categorical_columns
            
        elif self.encoding_strategy in ("numeric", "none"):
            return None, categorical_columns
            
        raise ValueError(f"Unsupported encoding strategy: {self.encoding_strategy}")

    def _is_valid_common_category(self, column: np.ndarray, suffix: str) -> bool:
        """Check whether the input data meets the common category conditions"""
        min_count = self.get_least_common_category_count(column)
        unique_count = len(np.unique(column))
        
        if "strict_feature_shuffled" in suffix:
            return min_count >= 10 and unique_count < (len(column) // 10)
        return min_count >= 10

class QTx(QuantileTransformer):
    """
    Works like QuantileTransformer, but quietly fixes n_quantiles > n_samples.
    """

    def __init__(self, *, n_quantiles: int = 1000, **kwargs: Any) -> None:
        # tuck away the original request
        self._preferred = n_quantiles
        # pass a placeholder to parent (will be overwritten later anyway)
        super().__init__(n_quantiles=n_quantiles, **kwargs)

    def fit(self, X, y=None):
        # sample count
        m = getattr(X, "shape", [0])[0]

        # pick the actual quantiles we’ll use (safe value)
        q = [self._preferred, m, self.subsample]
        q = max(1, min(*q))

        # overwrite parent attr just-in-time
        object.__setattr__(self, "n_quantiles", q)

        # random_state adjustments
        rs = getattr(self, "random_state", None)
        if isinstance(rs, np.random.Generator):
            rs = np.random.RandomState(int(rs.integers(0, 2**32)))
        elif hasattr(rs, "bit_generator"):
            raise ValueError(
                f"Unsupported random_state type: {type(rs)}"
            )
        self.random_state = rs

        # delegate to parent
        return super().fit(X, y)

class KDIX(KDITransformer):
    """
    Variant of KDITransformer that won't crash on NaNs.
    """

    def _more_tags(self):
        # obscure way of saying "NaNs are okay"
        d = {}
        d.update(allow_nan=True)
        return d

    def fit(self, X, y=None):
        # accept both numpy and torch
        if hasattr(X, "detach"):   # torch.Tensor case
            base = X.cpu().numpy()
        else:
            base = np.asarray(X)

        # replace NaNs with col means for training
        means = np.nanmean(base, axis=0)
        cleaned = np.where(np.isnan(base), means, base)

        return super().fit(cleaned, y)  # type: ignore

    def transform(self, X):
        # lazy conversion
        if isinstance(X, torch.Tensor):
            mat = X.cpu().numpy()
        else:
            mat = np.array(X, copy=False)

        # track NaNs
        nan_pos = np.isnan(mat)

        # impute with column means (zero fallback)
        col_means = np.nanmean(mat, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        filled = np.where(np.isnan(mat), col_means, mat)

        # apply KDI
        res = super().transform(filled)

        # put NaNs back in
        np.putmask(res, nan_pos, np.nan)
        return res  # type: ignore


class RebalanceFeatureDistribution(BasePreprocess):
    def __init__(
            self,
            *,
            worker_tags: list[str] | None = ["quantile"],
            discrete_flag: bool = False,
            original_flag: bool = False,
            svd_tag: Literal['svd'] | None = None,
            joined_svd_feature: bool = True,
            joined_log_normal: bool = True,
    ):
        super().__init__()
        self.worker_tags = worker_tags
        self.discrete_flag = discrete_flag
        self.original_flag = original_flag
        self.random_state = None
        self.svd_tag = svd_tag
        self.worker: Pipeline | ColumnTransformer | None = None
        self.joined_svd_feature = joined_svd_feature
        self.joined_log_normal = joined_log_normal
        self.feature_indices = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        self.random_state = seed
        n_samples, n_features = x.shape
        worker, self.dis_ix = self._set(n_samples,n_features,categorical_features)
        worker.fit(x)
        self.worker = worker
        return self.dis_ix

    @override
    def transform(self, x:np.ndarray, **kwargs) -> np.ndarray:
        assert self.worker is not None
        return self.worker.transform(x), self.dis_ix  # type: ignore

    @override
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, *, y:np.ndarray, **kwargs)->tuple[np.ndarray, list[int]]:
        """Fit the preprocessing model to the data and transform the data"""
        assert y is not None, "The input y cannot be None"
        x_train_ = x[:len(y)]
        x_test_ = x[len(y):]
        if x_train_.shape[1] != x_test_.shape[1]:
            x_test_ = x_test_[:, :x_train_.shape[1]]
        categorical_idx_ = self.fit(x_train_, categorical_features, seed)
        x_train_, categorical_idx_ = self.transform(x_train_)
        x_test_, categorical_idx_ = self.transform(x_test_)
        x_ = np.concatenate([x_train_, x_test_], axis=0)

        return (x_, categorical_idx_)

    def _set(self,n_samples: int,
        n_features: int,
        categorical_features: list[int],
        ):
        static_seed, rng = infer_random_state(self.random_state)
        all_ix = list(range(n_features))
        workers = []
        cont_ix = [i for i in all_ix if i not in categorical_features]
        if self.original_flag:
            trans_ixs = categorical_features + cont_ix if self.discrete_flag else cont_ix
            workers.append(("original", "passthrough", all_ix))
            dis_ix = categorical_features
        elif self.discrete_flag:
            # trans_ixs = all_ix
            # dis_ix = categorical_features
            trans_ixs = categorical_features + cont_ix
            self.feature_indices = categorical_features + cont_ix
            dis_ix = []
        else:
            workers.append(("discrete", "passthrough", categorical_features))
            trans_ixs, dis_ix = cont_ix, list(range(len(categorical_features)))
        for worker_tag in self.worker_tags:
            # print(f"== worker_tag: \033[31m{worker_tag}\033[0m")
            if worker_tag == "logNormal":
                sworker = Pipeline(steps=[
                                        ("save_standard", Pipeline(steps=[
                                            ("i2n_pre",
                                             FunctionTransformer(
                                                 func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan,
                                                                              posinf=np.nan),
                                                 inverse_func=lambda x: x, check_inverse=False)),
                                            ("fill_missing_pre",
                                             SimpleImputer(missing_values=np.nan, strategy="mean",
                                                           keep_empty_features=True)),
                                            ("feature_shift",
                                             FunctionTransformer(func=lambda x: x + np.abs(np.nanmin(x)))),
                                            ("add_epsilon", FunctionTransformer(func=lambda x: x + 1e-10)),
                                            ("logNormal", FunctionTransformer(np.log, validate=False)),
                                            ("i2n_post",
                                             FunctionTransformer(
                                                 func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan,
                                                                              posinf=np.nan),
                                                 inverse_func=lambda x: x, check_inverse=False)),
                                            ("fill_missing_post",
                                             SimpleImputer(missing_values=np.nan, strategy="mean",
                                                           keep_empty_features=True))])),
                                        ])
                # trans_ixs = cont_ix
            elif worker_tag == "quantile_uniform_10":
                sworker = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 10, 2),
                    random_state=static_seed,
                )
            elif worker_tag == "quantile_uniform_5":
                sworker = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                )
            elif worker_tag == "quantile_uniform_all_data":
                sworker = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    subsample=n_samples,
                )
            elif worker_tag == 'power':
                self.feature_indices = categorical_features+cont_ix
                self.dis_ix = dis_ix
                nan_to_mean_transformer = SimpleImputer(
                                                    missing_values=np.nan,
                                                    strategy="mean",
                                                    keep_empty_features=True,
                                                )
            
                sworker = SelectiveInversePipeline(
                                steps=[
                                    ("power_transformer", RobustPowerTransformer(standardize=False)),
                                    ("inf_to_nan_1", FunctionTransformer(
                                                        func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                                                        inverse_func=lambda x: x,
                                                        check_inverse=False,
                                                    )),
                                    ("nan_to_mean_1", nan_to_mean_transformer),
                                    ("scaler", StandardScaler()),
                                    ("inf_to_nan_2", FunctionTransformer(
                                                        func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                                                        inverse_func=lambda x: x,
                                                        check_inverse=False,
                                                    )),
                                    ("nan_to_mean_2", nan_to_mean_transformer),
                                ],
                        skip_inverse=['nan_to_mean_1', 'nan_to_mean_2']
                )

            elif worker_tag=="quantile_norm_10":
                sworker = QTx(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 10, 2),
                    random_state=static_seed,
                )
            elif worker_tag=="quantile_norm_5":
                sworker = QTx(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                )
            elif worker_tag == "quantile_norm_all_data":
                sworker = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    subsample=n_samples,
                )
            elif worker_tag=="norm_and_kdi":
                sworker = FeatureUnion(
                    [
                        (
                            "norm",
                            QTx(
                                output_distribution="normal",
                                n_quantiles=max(n_samples // 10, 2),
                                random_state=static_seed,
                            ),
                        ),
                        (
                            "kdi",
                            KDIX(alpha=1.0, output_distribution="uniform"),
                        ),
                    ],
                )

            elif worker_tag=="robust":
                sworker = RobustScaler(unit_variance=True)
            elif worker_tag=="kdi_uni":
                sworker = KDIX(alpha=1.0, output_distribution="uniform")
            elif worker_tag is None:
                sworker = FunctionTransformer(lambda x: x)
            elif worker_tag.startswith("kdi_uni_alpha_"):
                alpha = float(worker_tag.split("_")[-1])
                sworker = KDIX(alpha=alpha, output_distribution="uniform")
            elif worker_tag.startswith("kdi_norm_alpha_"):
                alpha = float(worker_tag.split("_")[-1])
                sworker = KDIX(alpha=alpha, output_distribution="normal")
            elif worker_tag=="kdi_norm":
                sworker = KDIX(alpha=1.0, output_distribution="normal")
            else:
                sworker = FunctionTransformer(lambda x: x)
            if worker_tag in ["quantile_uniform_10", "quantile_uniform_5", "quantile_uniform_all_data"]:
                self.n_quantile_features = len(trans_ixs)
            workers.append((f"feat_transform_{worker_tag}", sworker, trans_ixs))

        CT_worker = ColumnTransformer(workers,remainder="drop",sparse_threshold=0.0)
        if self.svd_tag == "svd" and n_features >= 2:
            svd_worker = FeatureUnion([
                    ("default", FunctionTransformer(func=lambda x: x)),
                    ("svd",Pipeline(steps=[
                                    ("save_standard",Pipeline(steps=[
                                    ("i2n_pre", FunctionTransformer(func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),inverse_func=lambda x: x, check_inverse=False)),
                                    ("fill_missing_pre", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)),
                                    ("standard", StandardScaler(with_mean=False)) ,
                                    ("i2n_post", FunctionTransformer(func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),inverse_func=lambda x: x, check_inverse=False)),
                                    ("fill_missing_post", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True))])),
                                    ("svd",TruncatedSVD(algorithm="arpack",n_components=max(1,min(n_samples // 10 + 1,n_features // 2)),random_state=static_seed))]))
                    ])
            self.svd_n_comp = max(1,min(n_samples // 10 + 1,n_features // 2))
            worker = Pipeline([("worker", CT_worker), ("svd_worker", svd_worker)])
        else:   
            self.svd_n_comp = 0
            worker = CT_worker

        self.worker = worker
        return worker, dis_ix


class SubSampleData():
    def __init__(
            self,
            subsample_type: Literal["feature", "sample"] = "sample",
            use_type: Literal["mixed", "only_sample"] = "mixed",
    ):
        super().__init__()
        self.subsample_type = subsample_type
        self.use_type = use_type

    def fit(self,
            x: torch.Tensor=None,
            y: torch.Tensor = None,
            feature_attention_score: torch.Tensor = None,
            sample_attention_score: torch.Tensor = None,
            subsample_ratio: float | int = 200,
            subsample_idx:list[int] | np.ndarray[int] = None,
            ):
        if isinstance(subsample_ratio, float):
            if self.subsample_type == "sample":
                self.subsample_num = int(subsample_ratio * x.shape[0])
            else:
                self.subsample_num = int(subsample_ratio * x.shape[1])
        else:
            self.subsample_num = subsample_ratio
        if self.subsample_type == "sample":
            if self.use_type == "mixed":
                y_feature_attention_score = feature_attention_score[:, -1, :].squeeze().permute(1, 0).unsqueeze(
                    -1) # shape [features,test_sample_lens,1] broadcast to [features,test_sample_lens,train_sample_lens]
                #TODO jianshengli may cause OOM
                try:
                    self.attention_score = torch.mean(sample_attention_score.to("cuda") * y_feature_attention_score.to("cuda"),
                                                      dim=0).cpu()  # shape [test_sample_lens,train_sample_lens]
                except OutOfMemoryError as e:
                    print("calculate attention score OOM, use cpu")
                    self.attention_score = torch.mean(
                        sample_attention_score.cpu() * y_feature_attention_score.cpu(),
                        dim=0)
                del sample_attention_score,y_feature_attention_score

            else:
                self.attention_score = sample_attention_score[-1, :, :]
            self.X_train = x
            self.y_train = y
        else:
            y_feature_attention_score = torch.mean(feature_attention_score[:, -1, :].squeeze(),dim=0)  # shape [test_sample_lens,features]
            if subsample_idx is None:
                self.subsample_idx = torch.argsort(y_feature_attention_score)[-min(self.subsample_num, x.shape[0]):]
            else:
                self.subsample_idx = subsample_idx
            self.X_train = x

    def transform(self, x: torch.Tensor=None) -> np.ndarray |torch.Tensor | TabularInferenceDataset:
        if self.subsample_type == "feature":
            return torch.cat([self.X_train, x], dim=0)[:, self.subsample_idx].numpy()
        else:
            return self.attention_score



# Large constant for hash normalization
_HASH_MODULUS = 10**12

def float_hash_arr(input_array: np.ndarray) -> float:
    """
    Generate a normalized floating-point hash value from a numpy array.
    
    This function computes a SHA256 hash of the array's byte representation,
    converts it to an integer, and normalizes it to a float between 0 and 1.
    
    Args:
        input_array: Input numpy array to be hashed
        
    Returns:
        Normalized hash value in the range [0, 1)
    """
    # Convert array to bytes and compute SHA256 hash
    array_bytes = input_array.tobytes()
    hash_hex = hashlib.sha256(array_bytes).hexdigest()
    
    # Convert hex digest to integer
    hash_int = int(hash_hex, 16)
    
    # Normalize to [0, 1) range using modulus operation
    normalized_hash = (hash_int % _HASH_MODULUS) / _HASH_MODULUS
    
    return normalized_hash


class FingerprintFeatureEncoder(BasePreprocess):
    """
    Appends a fingerprint column derived from row-wise hashing of input data.
    
    For test data: Uses first computed hash even if collisions occur.
    For training data: Resolves hash collisions through iterative rehashing.
    """
    
    def __init__(self, rng_seed: int | np.random.Generator | None = None):
        super().__init__()
        # self.rng_seed = rng_seed
        self.salt_value = None
        self.categorical_features = None
    
    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Initialize random salt and return categorical feature indices."""
        _, rng = infer_random_state(seed)
        self.salt_value = int(rng.integers(0, 65536))  # 2^16 range
        self.categorical_features = categorical_features
        return categorical_features.copy()
    
    @override
    def transform(self, x:np.ndarray, is_test:bool=False, **kwargs) -> tuple[np.ndarray, list[int]]:
        """
        Transform input by appending fingerprint column.
        
        Args:
            X_data: Input array of shape (n_samples, n_features)
            is_test: Whether processing test data (affects collision handling)
            
        Returns:
            Transformed array with fingerprint column and updated categorical indices
        """
        # print(f"add finger")
        if self.salt_value is None:
            raise RuntimeError("Must call fit() before transform()")
        
        n_samples = x.shape[0]
        fingerprint_col = np.zeros(n_samples, dtype=x.dtype)
        
        # Apply salt to input data
        salted_data = x + self.salt_value
        
        if is_test:
            # Test mode: use first hash regardless of collisions
            for idx in range(n_samples):
                row_hash = float_hash_arr(salted_data[idx] + self.salt_value)
                fingerprint_col[idx] = row_hash
        else:
            # Training mode: resolve hash collisions
            existing_hashes = set()
            for idx in range(n_samples):
                current_row = salted_data[idx]
                hash_val = float_hash_arr(current_row)
                increment = 0
                
                # Handle collisions by incrementing and rehashing
                while hash_val in existing_hashes:
                    increment += 1
                    hash_val = float_hash_arr(current_row + increment)
                
                fingerprint_col[idx] = hash_val
                existing_hashes.add(hash_val)
        
        # Append fingerprint column and update categorical indices
        transformed = np.column_stack([x, fingerprint_col.reshape(-1, 1)])
        # cat_indices_updated = list(range(x.shape[1]))  # Original features remain categorical
        
        return transformed, self.categorical_features

class PolynomialInteractionGenerator(BasePreprocess):
    """
    Generates polynomial interaction features through randomized pairwise combinations
    with standardized preprocessing and memory-efficient implementation.
    """
    
    def __init__(
        self, 
        *, 
        max_interaction_features: int | None = None,
        random_generator: int | np.random.Generator | None = None
    ):
        super().__init__()
        self.max_interactions = max_interaction_features
        # self.rng_config = random_generator
        # print(f"max_interactions: {self.max_interactions}")
        if self.max_interactions:
            assert max_interaction_features > 0, "max_interaction_features must be greater than 0"
        else:
            self.max_interactions = 100
        
        self.primary_factor_indices: np.ndarray | None = None
        self.secondary_factor_indices: np.ndarray | None = None
        self.feature_normalizer = StandardScaler(with_mean=False)
        self.categorical_features = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Configure polynomial feature generation parameters from training data."""
        assert x.ndim == 2, "Input matrix must be 2-dimensional"
        
        _, random_engine = infer_random_state(seed)
        
        # Handle empty dataset scenarios
        if x.size == 0:
            return categorical_features.copy()
        
        feature_count = x.shape[1]
        
        # Calculate maximum possible interaction combinations
        max_possible_combinations = (feature_count * (feature_count + 1)) // 2
        
        # print(f"max_possible_combinations: {max_possible_combinations}")
        # Determine actual interaction count with constraint
        actual_interaction_count = (
            min(self.max_interactions, max_possible_combinations) 
            if self.max_interactions is not None 
            else max_possible_combinations
        )
        
        # Standardize features before interaction generation
        normalized_data = self.feature_normalizer.fit_transform(x)
        
        # Generate randomized factor pairs efficiently
        self._generate_interaction_pairs(feature_count, actual_interaction_count, random_engine)
        self.categorical_features = categorical_features
        return categorical_features
    
    def _generate_interaction_pairs(
        self, 
        total_features: int, 
        required_pairs: int, 
        rng: np.random.Generator
    ) -> None:
        """Efficiently generate unique factor pairs for polynomial feature creation."""
        self.primary_factor_indices = rng.choice(
            np.arange(total_features),
            size=required_pairs,
            replace=True,
        )

        self.secondary_factor_indices = np.full_like(self.primary_factor_indices, -1)

        for i in range(required_pairs):
            while self.secondary_factor_indices[i] == -1:
                a = self.primary_factor_indices[i]
                used_b = self.secondary_factor_indices[self.primary_factor_indices == a]
                allowed_b = [b for b in range(a, total_features) if b not in used_b]

                if len(allowed_b) == 0:
                    self.primary_factor_indices[i] = rng.choice(np.arange(total_features))
                    continue
                else:
                    self.secondary_factor_indices[i] = rng.choice(allowed_b)

    @override
    def transform(self, x:np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Apply polynomial feature transformation to input data."""
        assert x.ndim == 2, "Input matrix must be 2-dimensional"
        
        if x.size == 0:
            return x, []
        
        # Standardize input features
        standardized_features = self.feature_normalizer.transform(x)
        
        # Generate polynomial interaction features
        interaction_features = (
            standardized_features[:, self.primary_factor_indices] * 
            standardized_features[:, self.secondary_factor_indices]
        )
        
        # Combine original and interaction features
        transformed_output = np.column_stack([standardized_features, interaction_features])
        
        return transformed_output, self.categorical_features