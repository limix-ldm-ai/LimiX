import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations
import nvtx
import re
import torch
import unicodedata
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import resample

import hashlib
from joblib import Parallel, delayed
from kditransform import KDITransformer
from .svd_memory import choose_svd_components

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)
DATETIME_FEATURES = ("unix_ns", "year", "month", "day", "dayofweek")
FIXED_FEATURE_VIEW_NAMES = (
    "numeric_tail_interactions",
    "structured_string_statistics",
    "low_cardinality_numeric_states",
)
DEFAULT_FIXED_FEATURE_VIEW_CONFIG = {
    "enabled": True,
    "views": list(FIXED_FEATURE_VIEW_NAMES),
    "max_numeric_columns": 8,
    "max_numeric_pairs": 6,
    "max_string_columns": 6,
    "max_low_cardinality_columns": 6,
    "low_cardinality_max_states": 10,
}


def resolve_fixed_feature_view_config(raw_config: dict | None) -> dict:
    """Functionality: Validate and complete deterministic feature-view preprocessing config. These options only bound fixed, column-order transforms and never trigger label-based search.

    Input:
        raw_config: JSON object or None. None is treated as an empty object and uses defaults.

    Output:
        dict: legal config merged with DEFAULT_FIXED_FEATURE_VIEW_CONFIG.
    """
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise TypeError("fixed_feature_views must be a JSON object")
    unknown = set(raw_config) - set(DEFAULT_FIXED_FEATURE_VIEW_CONFIG)
    if unknown:
        raise ValueError(
            f"fixed_feature_views has unknown keys: {sorted(unknown)}"
        )
    config = {**DEFAULT_FIXED_FEATURE_VIEW_CONFIG, **raw_config}
    if not isinstance(config["enabled"], bool):
        raise TypeError("fixed_feature_views.enabled must be boolean")

    views = config["views"]
    if not isinstance(views, (list, tuple)) or any(
        not isinstance(view, str) for view in views
    ):
        raise TypeError("fixed_feature_views.views must be a list of strings")
    if len(set(views)) != len(views):
        raise ValueError("fixed_feature_views.views contains duplicates")
    unsupported = set(views) - set(FIXED_FEATURE_VIEW_NAMES)
    if unsupported:
        raise ValueError(
            f"unsupported fixed feature views: {sorted(unsupported)}"
        )

    integer_keys = (
        "max_numeric_columns",
        "max_numeric_pairs",
        "max_string_columns",
        "max_low_cardinality_columns",
        "low_cardinality_max_states",
    )
    for key in integer_keys:
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"fixed_feature_views.{key} must be a non-negative integer")
        config[key] = int(value)
    if config["low_cardinality_max_states"] == 1:
        raise ValueError(
            "fixed_feature_views.low_cardinality_max_states must be 0 or at least 2"
        )
    config["views"] = list(views)
    return config


def _pandas_mixed_datetime_format_works() -> bool:
    parsed = pd.to_datetime(
        pd.Series(["2020-01-01", "01/02/2020"]),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    return bool(parsed.notna().all())


# pandas 2+ treats format="mixed" as a per-element parser. pandas 1.x treats it
# as a literal strftime pattern and returns all NaT, so it still needs the
# format-less fallback.
_PANDAS_MIXED_DATETIME = _pandas_mixed_datetime_format_works()


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Functionality: Parse mixed-format datetime values to UTC timestamps, compatible with pandas 1.x and 2.x.

    Input:
        series: pandas.Series to parse.

    Output:
        pandas.Series of UTC datetime64; unparseable values are NaT.
    """
    if _PANDAS_MIXED_DATETIME:
        return pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
            format="mixed",
        )
    try:
        parsed = pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, utc=True, errors="coerce")

    # pandas 1.x accepts ``format="mixed"`` as a literal format and silently
    # turns every otherwise-valid value into NaT. Retry with its legacy parser
    # in that case.
    if series.notna().any() and parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed


class SkipAllNaNStandardScaler(StandardScaler):
    """StandardScaler that does not reduce all-NaN columns.

    sklearn's fit still computes mean/var on empty columns and warns
    (divide by zero).  Those columns already become NaN statistics; this
    subclass skips the empty-column reduction and writes the same NaN stats.
    """

    def fit(self, X, y=None, sample_weight=None):
        values = np.asarray(X)
        if values.ndim != 2 or values.size == 0:
            return super().fit(X, y, sample_weight)
        finite_col = ~np.isnan(values).all(axis=0)
        if finite_col.all():
            return super().fit(X, y, sample_weight)

        filled = np.array(values, dtype=np.float64, copy=True)
        empty_col = ~finite_col
        filled[:, empty_col] = 0.0
        super().fit(filled, y, sample_weight)
        if self.mean_ is not None:
            self.mean_[empty_col] = np.nan
        if self.var_ is not None:
            self.var_[empty_col] = np.nan
        if self.scale_ is not None:
            self.scale_[empty_col] = np.nan
        n_seen = getattr(self, "n_samples_seen_", None)
        if isinstance(n_seen, np.ndarray) and n_seen.shape == empty_col.shape:
            n_seen[empty_col] = 0
        return self


class SelectiveInversePipeline(Pipeline):
    """Functionality: sklearn Pipeline subclass whose inverse_transform can skip named steps such as mean imputation.

    Input:
        steps: List of (name, transformer) pairs, same as Pipeline.
        skip_inverse: Step names skipped during inverse transform.

    Output:
        Pipeline instance.
    """
    def __init__(self, steps, skip_inverse=None):
        """Functionality: Store skip_inverse and delegate to the parent initializer.

        Input:
            steps: Preprocess step list.
            skip_inverse: Step names skipped on inverse transform; default empty.

        Output:
            None.
        """
        super().__init__(steps)
        self.skip_inverse = skip_inverse or []
    
    def inverse_transform(self, X):
        """Functionality: Inverse-transform from last step to first, skipping unfitted steps and names in skip_inverse.

        Input:
            X: 2-D array. Returned unchanged when it has 0 columns.

        Output:
            np.ndarray: inverse-transformed features.
        """
        if X.shape[1] == 0:
            return X
        for step_idx in range(len(self.steps) - 1, -1, -1):
            name, transformer = self.steps[step_idx]
            try:
                check_is_fitted(transformer)
            except:
                continue
            
            if name in self.skip_inverse:
                continue
                
            if hasattr(transformer, 'inverse_transform'):
                X = transformer.inverse_transform(X)
                if np.any(np.isnan(X)):
                    print(f"After reverse RebalanceFeatureDistribution of {name}, there is nan")
        return X

class RobustPowerTransformer(PowerTransformer):
    """Functionality: PowerTransformer that can revert columns with bad variance or magnitude, and clip overflow to finite training bounds.

    Input:
        method: 'yeo-johnson' or 'box-cox'.
        standardize: Whether to standardize after the transform.
        copy: Whether to copy the input.
        var_tolerance: Tolerance for variance departing from 1.
        max_abs_value: Maximum allowed absolute value after transform.

    Output:
        Estimator instance.
    """

    def __init__(
        self,
        method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
        standardize: bool = True,
        copy: bool = True,
        var_tolerance: float = 1e-3,
        max_abs_value: float = 100,
    ) -> None:
        # Keep all PowerTransformer parameters explicit. ColumnTransformer
        # clones child estimators via get_params(); hiding these values in
        # **kwargs made clone() silently restore standardize=True even when
        # this project requested standardize=False.
        """Functionality: Store every PowerTransformer parameter explicitly so ColumnTransformer.clone cannot silently restore defaults.

        Input:
            method: Transform family.
            standardize: Whether to standardize.
            copy: Whether to copy.
            var_tolerance: Variance tolerance.
            max_abs_value: Absolute-value cap.

        Output:
            None.
        """
        super().__init__(method=method, standardize=standardize, copy=copy)
        self.var_tolerance = var_tolerance
        self.max_abs_value = max_abs_value
        self.restore_indices_: np.ndarray | None = None
        self.finite_min_: np.ndarray | None = None
        self.finite_max_: np.ndarray | None = None
        self.range_clip_fallback_count_: int = 0
        self.last_range_clip_count_: int = 0


    def _record_finite_bounds(self, X: np.ndarray) -> None:
        """Functionality: Record per-column finite min/max for later overflow clipping.

        Input:
            X: 2-D array used at fit time.

        Output:
            None. Writes finite_min_ and finite_max_.
        """
        data = np.asarray(X)
        self.finite_min_ = np.full(data.shape[1], np.nan, dtype=np.float64)
        self.finite_max_ = np.full(data.shape[1], np.nan, dtype=np.float64)
        for index in range(data.shape[1]):
            finite_values = data[np.isfinite(data[:, index]), index]
            if finite_values.size:
                self.finite_min_[index] = np.min(finite_values)
                self.finite_max_[index] = np.max(finite_values)
        self.range_clip_fallback_count_ = 0
        self.last_range_clip_count_ = 0

    def _clip_to_fitted_bounds(self, X: np.ndarray) -> tuple[np.ndarray, int]:
        """Functionality: Clip the input to the finite range recorded at fit time.

        Input:
            X: 2-D array whose column count must match fit.

        Output:
            tuple[np.ndarray, int]: (clipped array, number of finite values rewritten).
        """
        if self.finite_min_ is None or self.finite_max_ is None:
            raise RuntimeError("finite training bounds are unavailable")
        data = np.asarray(X).copy()
        if data.ndim != 2 or data.shape[1] != self.finite_min_.shape[0]:
            raise ValueError("input feature count does not match fitted bounds")
        clipped_count = 0
        for index, (lower, upper) in enumerate(zip(self.finite_min_, self.finite_max_)):
            if not np.isfinite(lower) or not np.isfinite(upper):
                continue
            original = data[:, index].copy()
            np.clip(data[:, index], lower, upper, out=data[:, index])
            changed = ~np.isnan(original) & (original != data[:, index])
            clipped_count += int(np.count_nonzero(changed))
        return data, clipped_count

    @staticmethod
    def _is_float_overflow_error(error: ValueError) -> bool:
        """Functionality: Return whether a ValueError was caused by inf or values too large for float32.

        Input:
            error: ValueError raised by PowerTransformer.

        Output:
            bool.
        """
        message = str(error)
        return (
            "contains infinity" in message
            or "too large for dtype('float32')" in message
        )

    def fit(self, X, y=None):
        """Functionality: Record finite bounds and fit the parent PowerTransformer.

        Input:
            X: 2-D features.
            y: Ignored.

        Output:
            self. restore_indices_ is initialized empty.
        """
        self._record_finite_bounds(X)
        fitted = super().fit(X, y)
        self.restore_indices_ = np.array([], dtype=int)
        return fitted

    def fit_transform(self, X, y=None):
        """Functionality: Fit and transform, marking columns that should revert based on variance or magnitude.

        Input:
            X: 2-D features.
            y: Ignored.

        Output:
            np.ndarray: transformed result. restore_indices_ records columns to restore.
        """
        self._record_finite_bounds(X)
        Z = super().fit_transform(X,y)
        self.restore_indices_ = self._should_revert(Z)
        return Z

    def _should_revert(self, Z: np.ndarray) -> np.ndarray:
        """Functionality: Find columns whose transformed variance is not near 1 or that contain overly large values.

        Input:
            Z: Transformed 2-D array.

        Output:
            np.ndarray: column indices that should be restored.
        """
        variances = np.nanvar(Z, axis=0)
        bad_var = np.flatnonzero(np.abs(variances - 1.0) > self.var_tolerance)

        bad_large = np.flatnonzero(np.any(Z > self.max_abs_value, axis=0))

        return np.unique(np.concatenate([bad_var, bad_large]))

    def _apply_reversion(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Functionality: Replace marked columns with their pre-transform values.

        Input:
            Z: Transformed array.
            X: Pre-transform array, column-aligned.

        Output:
            np.ndarray: result after optional column restoration.
        """
        if self.restore_indices_.size > 0:
            Z[:, self.restore_indices_] = X[:, self.restore_indices_]
        return Z

    def transform(self, X):
        """Functionality: Apply the power transform and restore columns in restore_indices_. On overflow, clip then retry.

        Input:
            X: 2-D features; column count must match fit.

        Output:
            np.ndarray: robust transformed result.
        """
        try:
            Z = super().transform(X)
            # self.restore_indices_ = self._should_revert(Z)
            return self._apply_reversion(Z, X)
        except ValueError as error:
            if not self._is_float_overflow_error(error):
                raise
            safe_X, clipped_count = self._clip_to_fitted_bounds(X)
            if clipped_count == 0:
                raise
            Z = super().transform(safe_X)
            self.range_clip_fallback_count_ += 1
            self.last_range_clip_count_ = clipped_count
            warnings.warn(
                "RobustPowerTransformer clipped "
                f"{clipped_count} values to finite training bounds after "
                f"PowerTransformer overflow: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._apply_reversion(Z, safe_X)

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        """Functionality: Optimize the Yeo-Johnson lambda while swallowing NaN/Inf crashes and ignoring overflow warnings.

        Input:
            x: 1-D single column.

        Output:
            float: optimal lambda, or nan on failure.
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message=r"overflow encountered",
                                        category=RuntimeWarning)
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except Exception as e:
            return np.nan

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Functionality: Apply the Yeo-Johnson transform. If lambda is nan, return the original column.

        Input:
            x: 1-D single column.
            lmbda: Transform parameter.

        Output:
            np.ndarray: transformed column.
        """
        if np.isnan(lmbda):
            return x
        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore


def looks_like_datetime(series: pd.Series) -> bool:
    """Functionality: Return whether a string/object column looks like datetime (sampled parse, missing rate at most 80%).

    Input:
        series: pandas.Series.

    Output:
        bool. Columns that convert wholly to numeric return False.
    """
    if not (
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.StringDtype)
    ):
        return False
    if series.isnull().all():
        return False
    try:
        pd.to_numeric(series)
    except (ValueError, TypeError):
        try:
            sample = (
                series.sample(n=500, random_state=0)
                if len(series) > 500
                else series
            )
            parsed = _parse_datetime_series(sample)
            return bool(parsed.isnull().mean() <= 0.8)
        except Exception:
            return False
    return False


@dataclass(frozen=True)
class DatetimeColumnState:
    """Functionality: State for one detected datetime column: original position, missing fill timestamp, and generated output names.

    Input:
        position: Column position in the input DataFrame.
        fill_value: Timestamp used to fill NaT, from the train mean.
        output_columns: Tuple of expanded feature names.

    Output:
        Frozen dataclass.
    """
    position: int
    fill_value: pd.Timestamp
    output_columns: tuple[str, ...]


class DatetimePreprocessor:
    """Functionality: Detect datetime columns on train and expand them into unix_ns/year/month/day/dayofweek numeric features.

    Input:
        None: State is written during fit.

    Output:
        Preprocessor instance.
    """

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        """Functionality: Require a DataFrame with unique column names.

        Input:
            X: Table to validate.

        Output:
            None. Raises on illegal type or duplicate names.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("datetime preprocessing requires a DataFrame")
        if X.columns.duplicated().any():
            raise ValueError("duplicate feature names are not supported")

    @staticmethod
    def _parse(series: pd.Series) -> pd.Series:
        """Functionality: Parse one column via _parse_datetime_series.

        Input:
            series: pandas.Series.

        Output:
            Parsed UTC datetime Series.
        """
        return _parse_datetime_series(series)

    def fit(self, X: pd.DataFrame) -> "DatetimePreprocessor":
        """Functionality: Detect datetime columns on the given table and record fill values and output names. Labels are not used.

        Input:
            X: Usually the x_train DataFrame.

        Output:
            self. Writes feature_names_in_ and columns_.
        """
        self._validate_frame(X)
        states: list[DatetimeColumnState] = []
        existing_names = {str(column) for column in X.columns}
        generated_names: set[str] = set()
        for position in range(X.shape[1]):
            series = X.iloc[:, position]
            if not (
                pd.api.types.is_datetime64_any_dtype(series.dtype)
                or looks_like_datetime(series)
            ):
                continue

            parsed = self._parse(series)
            fill_value = parsed.mean()
            if pd.isna(fill_value):
                raise ValueError(
                    f"datetime column at position {position} has no valid value"
                )
            output_columns = tuple(
                f"__datetime_{position}__{feature}"
                for feature in DATETIME_FEATURES
            )
            collisions = (existing_names | generated_names).intersection(
                output_columns
            )
            if collisions:
                raise ValueError(
                    f"generated datetime feature names collide: {sorted(collisions)}"
                )
            generated_names.update(output_columns)
            states.append(
                DatetimeColumnState(
                    position=position,
                    fill_value=fill_value,
                    output_columns=output_columns,
                )
            )

        self.feature_names_in_ = tuple(X.columns)
        self.columns_ = tuple(states)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Functionality: Drop detected datetime source columns and append expanded numeric date features. Missing values use the fitted fill_value.

        Input:
            X: DataFrame whose column names must match fit.

        Output:
            pandas.DataFrame: non-datetime columns plus expanded date features.
        """
        if not hasattr(self, "columns_"):
            raise RuntimeError("DatetimePreprocessor is not fitted")
        self._validate_frame(X)
        if tuple(X.columns) != self.feature_names_in_:
            raise ValueError("transform columns differ from fitted columns")
        detected_positions = {state.position for state in self.columns_}
        result = X.iloc[
            :,
            [
                position
                for position in range(X.shape[1])
                if position not in detected_positions
            ],
        ].copy()
        for state in self.columns_:
            parsed = self._parse(X.iloc[:, state.position]).fillna(
                state.fill_value
            )
            result[state.output_columns[0]] = pd.to_numeric(parsed)
            result[state.output_columns[1]] = parsed.dt.year.astype(np.int64)
            result[state.output_columns[2]] = parsed.dt.month.astype(np.int64)
            result[state.output_columns[3]] = parsed.dt.day.astype(np.int64)
            result[state.output_columns[4]] = parsed.dt.dayofweek.astype(np.int64)
        return result


@dataclass(frozen=True)
class _FixedNumericColumnState:
    """Functionality: Robust location/scale for one column in the fixed numeric view: median and MAD scale.

    Input:
        position: Column position.
        median: Median of finite train values.
        scale: MAD; falls back to 1 when non-positive.

    Output:
        Frozen dataclass.
    """
    position: int
    median: float
    scale: float


@dataclass(frozen=True)
class _FixedFrequencyColumnState:
    """Functionality: Train-set state-frequency table for a low-cardinality numeric column.

    Input:
        position: Column position.
        frequencies: Map from state key to train frequency.

    Output:
        Frozen dataclass.
    """
    position: int
    frequencies: dict[float | str, float]


class FixedFeatureViewPreprocessor:
    """Functionality: Build label-independent deterministic classification feature views. fit accepts features only, visits columns in original order, and is bounded by config caps.

    Input:
        enabled: Whether the preprocessor is enabled.
        views: View names to generate.
        max_numeric_columns: Max numeric columns used by the numeric-tail view.
        max_numeric_pairs: Max numeric interaction pairs.
        max_string_columns: Max string-statistic columns.
        max_low_cardinality_columns: Max low-cardinality frequency columns.
        low_cardinality_max_states: Max low-cardinality state count; 0 or at least 2.

    Output:
        Preprocessor instance.
    """

    _TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
    _NONFINITE_KEY = "__nonfinite__"

    def __init__(
        self,
        *,
        enabled: bool = True,
        views: list[str] | tuple[str, ...] = FIXED_FEATURE_VIEW_NAMES,
        max_numeric_columns: int = 8,
        max_numeric_pairs: int = 6,
        max_string_columns: int = 6,
        max_low_cardinality_columns: int = 6,
        low_cardinality_max_states: int = 10,
    ) -> None:
        """Functionality: Validate config and store it on self.config.

        Input:
            kwargs: Same fields as resolve_fixed_feature_view_config.

        Output:
            None.
        """
        self.config = resolve_fixed_feature_view_config(
            {
                "enabled": enabled,
                "views": list(views),
                "max_numeric_columns": max_numeric_columns,
                "max_numeric_pairs": max_numeric_pairs,
                "max_string_columns": max_string_columns,
                "max_low_cardinality_columns": max_low_cardinality_columns,
                "low_cardinality_max_states": low_cardinality_max_states,
            }
        )

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        """Functionality: Require a non-empty DataFrame with unique column names.

        Input:
            X: Feature table.

        Output:
            None.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("fixed feature views require a pandas DataFrame")
        if X.columns.duplicated().any():
            raise ValueError("duplicate feature names are not supported")
        if len(X) == 0:
            raise ValueError("fixed feature views require at least one row")

    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        """Functionality: Return whether a series is a non-boolean numeric column.

        Input:
            series: pandas.Series.

        Output:
            bool.
        """
        return (
            not pd.api.types.is_bool_dtype(series.dtype)
            and pd.api.types.is_numeric_dtype(series.dtype)
        )

    @staticmethod
    def _numeric_values(series: pd.Series) -> np.ndarray:
        """Functionality: Convert a column to float64. Unparseable values become NaN.

        Input:
            series: pandas.Series.

        Output:
            np.ndarray.
        """
        return pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

    @classmethod
    def _frequency_key(cls, value: float) -> float | str:
        """Functionality: Map a numeric value to a frequency-table key: finite values as float, non-finite as a sentinel string.

        Input:
            value: Scalar numeric value.

        Output:
            float or '__nonfinite__'.
        """
        return float(value) if np.isfinite(value) else cls._NONFINITE_KEY

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Functionality: NFKC-normalize, casefold, and collapse whitespace. Missing values become an empty string.

        Input:
            value: Arbitrary scalar.

        Output:
            str.
        """
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return " ".join(
            unicodedata.normalize("NFKC", str(value)).casefold().split()
        )

    @staticmethod
    def _signed_log1p(values: np.ndarray) -> np.ndarray:
        """Functionality: Apply sign(x)*log1p(|x|) to finite values. Non-finite values stay NaN.

        Input:
            values: Numeric array.

        Output:
            np.ndarray.
        """
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = np.sign(values) * np.log1p(np.abs(values))
        return np.where(np.isfinite(values), transformed, np.nan)

    def fit(self, X: pd.DataFrame) -> "FixedFeatureViewPreprocessor":
        """Functionality: Select numeric/string/low-cardinality columns in order and record medians, scales, and train frequencies.

        Input:
            X: x_train only; no labels.

        Output:
            self. Writes numeric_states_, string_positions_, frequency_states_, active_views_, and related state.
        """
        self._validate_frame(X)
        self.feature_names_in_ = tuple(X.columns)
        self.n_features_in_ = X.shape[1]

        numeric_positions = [
            position
            for position in range(X.shape[1])
            if self._is_numeric(X.iloc[:, position])
        ]
        string_positions = [
            position
            for position in range(X.shape[1])
            if not self._is_numeric(X.iloc[:, position])
            and not pd.api.types.is_bool_dtype(X.iloc[:, position].dtype)
        ]

        numeric_states: list[_FixedNumericColumnState] = []
        for position in numeric_positions[: self.config["max_numeric_columns"]]:
            values = self._numeric_values(X.iloc[:, position])
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if finite.size else 0.0
            absolute_deviation = np.abs(finite - median)
            scale = (
                float(np.median(absolute_deviation))
                if absolute_deviation.size
                else 0.0
            )
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            numeric_states.append(
                _FixedNumericColumnState(position, median, scale)
            )
        self.numeric_states_ = tuple(numeric_states)
        self.numeric_pairs_ = tuple(
            list(combinations(self.numeric_states_, 2))[
                : self.config["max_numeric_pairs"]
            ]
        )
        self.string_positions_ = tuple(
            string_positions[: self.config["max_string_columns"]]
        )

        frequency_states: list[_FixedFrequencyColumnState] = []
        max_states = self.config["low_cardinality_max_states"]
        if max_states >= 2:
            for position in numeric_positions:
                values = self._numeric_values(X.iloc[:, position])
                keys = [self._frequency_key(value) for value in values]
                counts: dict[float | str, int] = {}
                for key in keys:
                    counts[key] = counts.get(key, 0) + 1
                if not 2 <= len(counts) <= max_states:
                    continue
                frequencies = {
                    key: float(count) / len(values)
                    for key, count in counts.items()
                }
                frequency_states.append(
                    _FixedFrequencyColumnState(position, frequencies)
                )
                if (
                    len(frequency_states)
                    >= self.config["max_low_cardinality_columns"]
                ):
                    break
        self.frequency_states_ = tuple(frequency_states)

        counts = {
            "numeric_tail_interactions": (
                2 * len(self.numeric_states_) + 2 * len(self.numeric_pairs_)
            ),
            "structured_string_statistics": 4 * len(self.string_positions_),
            "low_cardinality_numeric_states": len(self.frequency_states_),
        }
        configured = set(self.config["views"])
        self.derived_feature_count_ = {
            view: count for view, count in counts.items() if view in configured
        }
        self.active_views_ = tuple(
            view
            for view in self.config["views"]
            if self.config["enabled"] and counts[view] > 0
        )
        return self

    def _validate_transform_input(self, X: pd.DataFrame) -> None:
        """Functionality: Require a fitted preprocessor and transform columns that match fit.

        Input:
            X: DataFrame.

        Output:
            None.
        """
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("FixedFeatureViewPreprocessor is not fitted")
        self._validate_frame(X)
        if tuple(X.columns) != self.feature_names_in_:
            raise ValueError("transform columns differ from fitted columns")

    def _raw_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Functionality: Copy the input and rename columns to raw_0000 form as the view's original prefix.

        Input:
            X: DataFrame.

        Output:
            pandas.DataFrame.
        """
        result = X.copy(deep=True).reset_index(drop=True)
        result.columns = [f"raw_{position:04d}" for position in range(X.shape[1])]
        return result

    def _numeric_tail_features(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Functionality: Build signed_log1p/robust_tail columns and log difference/product for numeric pairs.

        Input:
            X: DataFrame aligned with fitted columns.

        Output:
            dict[str, np.ndarray]: derived feature name to column vector.
        """
        features: dict[str, np.ndarray] = {}
        transformed: dict[int, np.ndarray] = {}
        for state in self.numeric_states_:
            values = self._numeric_values(X.iloc[:, state.position])
            signed_log = self._signed_log1p(values)
            robust = self._signed_log1p(
                (values - state.median) / state.scale
            )
            transformed[state.position] = signed_log
            prefix = f"fixed_numeric_{state.position:04d}"
            features[f"{prefix}__signed_log1p"] = signed_log
            features[f"{prefix}__robust_tail"] = robust
        for left, right in self.numeric_pairs_:
            left_values = transformed[left.position]
            right_values = transformed[right.position]
            prefix = f"fixed_pair_{left.position:04d}_{right.position:04d}"
            features[f"{prefix}__log_difference"] = left_values - right_values
            features[f"{prefix}__log_product"] = left_values * right_values
        return features

    def _string_features(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Functionality: Build length, token count, digit ratio, and punctuation ratio for selected string columns.

        Input:
            X: DataFrame.

        Output:
            dict[str, np.ndarray].
        """
        features: dict[str, np.ndarray] = {}
        for position in self.string_positions_:
            values = [self._normalize_text(value) for value in X.iloc[:, position]]
            lengths = np.asarray([len(value) for value in values], dtype=np.float32)
            token_counts = np.asarray(
                [len(self._TOKEN_PATTERN.findall(value)) for value in values],
                dtype=np.float32,
            )
            digit_counts = np.asarray(
                [sum(character.isdigit() for character in value) for value in values],
                dtype=np.float32,
            )
            punctuation_counts = np.asarray(
                [
                    sum(
                        not character.isalnum() and not character.isspace()
                        for character in value
                    )
                    for value in values
                ],
                dtype=np.float32,
            )
            denominator = np.maximum(lengths, 1.0)
            prefix = f"fixed_string_{position:04d}"
            features[f"{prefix}__length"] = lengths
            features[f"{prefix}__token_count"] = token_counts
            features[f"{prefix}__digit_ratio"] = digit_counts / denominator
            features[f"{prefix}__punctuation_ratio"] = (
                punctuation_counts / denominator
            )
        return features

    def _frequency_features(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Functionality: Look up low-cardinality numeric-state frequencies from the train table. Unseen states are 0.

        Input:
            X: DataFrame.

        Output:
            dict[str, np.ndarray].
        """
        features: dict[str, np.ndarray] = {}
        for state in self.frequency_states_:
            values = self._numeric_values(X.iloc[:, state.position])
            features[
                f"fixed_low_cardinality_{state.position:04d}__train_frequency"
            ] = np.asarray(
                [
                    state.frequencies.get(self._frequency_key(value), 0.0)
                    for value in values
                ],
                dtype=np.float32,
            )
        return features

    def _derived_features(
        self,
        X: pd.DataFrame,
        view: str,
    ) -> dict[str, np.ndarray]:
        """Functionality: Dispatch to the derived-feature builder for one view name.

        Input:
            X: DataFrame.
            view: numeric_tail_interactions / structured_string_statistics / low_cardinality_numeric_states.

        Output:
            dict[str, np.ndarray].
        """
        if view == "numeric_tail_interactions":
            return self._numeric_tail_features(X)
        if view == "structured_string_statistics":
            return self._string_features(X)
        if view == "low_cardinality_numeric_states":
            return self._frequency_features(X)
        raise ValueError(f"unsupported fixed feature view: {view!r}")

    def transform(self, X: pd.DataFrame, *, view: str) -> pd.DataFrame:
        """Functionality: Build one fixed view: the raw-column prefix plus that view's derived columns.

        Input:
            X: DataFrame whose column names match fit.
            view: Configured view name.

        Output:
            pandas.DataFrame.
        """
        self._validate_transform_input(X)
        if view not in self.config["views"]:
            raise ValueError(f"feature view {view!r} is not configured")
        result = self._raw_frame(X)
        for name, values in self._derived_features(X, view).items():
            result[name] = values
        return result

    def transform_merged(self, X: pd.DataFrame) -> pd.DataFrame:
        """Functionality: Concatenate derived columns from every active view onto one table.

        Input:
            X: DataFrame whose column names match fit.

        Output:
            pandas.DataFrame: raw prefix plus all derived columns.
        """
        self._validate_transform_input(X)
        result = self._raw_frame(X)
        for view in self.active_views_:
            for name, values in self._derived_features(X, view).items():
                if name in result.columns:
                    raise ValueError(f"generated feature name collision: {name}")
                result[name] = values
        return result

    def audit(self) -> dict:
        """Functionality: Return audit metadata for the fixed-view policy, stating that labels, search, and cross-fitting are unused.

        Input:
            self: Must already be fitted.

        Output:
            dict with policy, active_feature_views, derived_feature_count, config, and related fields.
        """
        if not hasattr(self, "active_views_"):
            raise RuntimeError("FixedFeatureViewPreprocessor is not fitted")
        return {
            "policy": "fixed_no_search",
            "fit_scope": "x_train_only",
            "column_order": "original",
            "labels_accepted": False,
            "target_encoding": "none",
            "candidate_scoring": "none",
            "cross_fitting": False,
            "active_feature_views": list(self.active_views_),
            "derived_feature_count": {
                view: self.derived_feature_count_[view]
                for view in self.active_views_
            },
            "config": dict(self.config),
        }


class BasePreprocess:
    """Functionality: Abstract base for inference preprocess steps. Defines fit/transform/fit_transform.

    Input:
        None: Subclasses implement the actual transform.

    Output:
        The base class cannot be used for transforms by itself.
    """

    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs)->list[int]:
        """Functionality: Fit preprocess state from data.

        Input:
            x: 2-D feature array.
            categorical_features: Categorical column indices.
            seed: Random seed.
            kwargs: Subclass extension arguments.

        Output:
            list[int]: updated categorical indices. Unimplemented subclasses raise NotImplementedError.
        """
        raise NotImplementedError
    
    def transform(self, x:np.ndarray, **kwargs)->tuple[np.ndarray, list[int]]:
        """Functionality: Transform data with fitted state.

        Input:
            x: 2-D feature array.
            kwargs: Subclass extension arguments.

        Output:
            tuple[np.ndarray, list[int]]: (transformed features, categorical indices).
        """
        raise NotImplementedError
    
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs)->tuple[np.ndarray, list[int]]:
        """Functionality: Fit, then transform.

        Input:
            x: 2-D feature array.
            categorical_features: Categorical column indices.
            seed: Random seed.
            kwargs: Forwarded to both fit and transform.

        Output:
            tuple[np.ndarray, list[int]].
        """
        self.fit(x, categorical_features, seed, **kwargs)
        return self.transform(x, **kwargs)

def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Functionality: Normalize several random-source types into (integer seed, numpy Generator).

    Input:
        random_state: None, int, RandomState, or Generator.

    Output:
        tuple[int, np.random.Generator].
    """
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
    """Functionality: Drop invalid feature columns that are constant everywhere or all-NaN on either the train or test side.

    Input:
        None: State is recorded during fit.

    Output:
        Preprocessor instance.
    """
    def __init__(self):
        """Functionality: Initialize valid-column masks and related state to empty.

        Input:
            None: No extra arguments.

        Output:
            None.
        """
        self.valid_features: list[bool] | None = None
        self.categorical_idx: list[int] | None = None
        self.invalid_indices: list[int] | None = None
        self.invalid_features: list[int] | None = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, y:np.ndarray | None = None, **kwargs) -> list[int]:
        """Functionality: Mark non-constant columns. When y is provided, also drop columns that are all-NaN on either the train or test split.

        Input:
            x: Concatenated train+test features.
            categorical_features: Original categorical indices.
            seed: Unused.
            y: Optional train labels used to determine eval_pos.

        Output:
            list[int]: categorical indices that remain after filtering. Raises ValueError if every column is invalid.
        """
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
        """Functionality: Slice by valid_features and keep dropped-column values for missing-value inversion.

        Input:
            x: Feature matrix whose column count matches fit.

        Output:
            tuple[np.ndarray, list[int]]: (kept columns, updated categorical indices).
        """
        assert self.valid_features is not None, "You must call fit first to get effective_features"
        self.invalid_features = x[:, self.invalid_indices]
        return x[:, self.valid_features], self.categorical_idx

class FeatureShuffler(BasePreprocess):
    """Functionality: Reorder columns with rotate/shuffle and keep categorical indices in sync.

    Input:
        mode: 'rotate', 'shuffle', or None to keep the original order.
        offset: Roll offset used in rotate mode.

    Output:
        Preprocessor instance.
    """

    def __init__(
        self,
        mode: Literal['rotate', 'shuffle'] | None = "shuffle",
        offset: int = 0,
    ):
        """Functionality: Store the reorder mode and offset.

        Input:
            mode: Reorder mode.
            offset: Rotation offset, default 0.

        Output:
            None.
        """
        super().__init__()
        self.mode = mode
        self.offset = offset
        self.random_seed = None
        self.feature_indices = None
        self.categorical_indices = None
    
    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Functionality: Build feature_indices from the mode and compute categorical indices after reordering.

        Input:
            x: 2-D features; only the column count is used.
            categorical_features: Original categorical indices.
            seed: Random seed for shuffle mode.

        Output:
            list[int]: categorical indices after reordering.
        """
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
        """Functionality: Reorder features with the fitted column permutation.

        Input:
            x: Column count must match fit.

        Output:
            tuple[np.ndarray, list[int]].
        """
        if self.feature_indices is None:
            raise RuntimeError("Please call the fit method first to initialize")
        if len(self.feature_indices) != x.shape[1]:
            raise ValueError("The number of features in the input data does not match the training data")
            
        return x[:, self.feature_indices], self.categorical_indices or []

class CategoricalFeatureEncoder(BasePreprocess):
    """Functionality: Encode categorical columns with ordinal, shuffled-ordinal, or one-hot encoding, or leave them numeric.

    Input:
        encoding_strategy: ordinal family, onehot, numeric, or none.
        onehot_size_fallback: Fall back to no encoding when one-hot output is too large.

    Output:
        Preprocessor instance.
    """

    def __init__(
        self,
        encoding_strategy: Literal['ordinal', 'ordinal_strict_feature_shuffled', 'ordinal_shuffled', 'onehot', 'numeric', 'none']|None = "ordinal",
        onehot_size_fallback: bool = True,
    ):
        """Functionality: Store the encoding strategy and one-hot fallback switch.

        Input:
            encoding_strategy: Encoding strategy name.
            onehot_size_fallback: Default True.

        Output:
            None.
        """
        super().__init__()
        self.encoding_strategy = encoding_strategy
        self.onehot_size_fallback = onehot_size_fallback
        self.random_seed = None
        self.transformer = None
        self.category_mappings = None
        self.categorical_features = None
        self.onehot_fallback_triggered = False
        self.onehot_input_shape = None
        self.onehot_output_shape = None

    @override
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Functionality: Record the random seed and run encoding.

        Input:
            x: 2-D features.
            categorical_features: Categorical column indices.
            seed: Random seed used for ordinal shuffling.

        Output:
            tuple[np.ndarray, list[int]]: (encoded features, new categorical indices).
        """
        self.random_seed = seed
        return self._fit_transform(x, categorical_features)

    def _fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        # print(f"encoding_strategy: {self.encoding_strategy}")
        """Functionality: Build a ColumnTransformer and fit_transform. One-hot may fall back when the result is too large.

        Input:
            X: 2-D features.
            categorical_features: Categorical column indices.

        Output:
            tuple[np.ndarray, list[int]].
        """
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
            self.onehot_fallback_triggered = False
            self.onehot_input_shape = tuple(X.shape)
            Xt = ct.fit_transform(X)
            self.onehot_output_shape = tuple(Xt.shape)
            if self.onehot_size_fallback and Xt.size >= 1_000_000:
                self.onehot_fallback_triggered = True
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
        """Functionality: Return the occurrence count of the rarest category in one column.

        Input:
            column: 1-D array.

        Output:
            int. Empty columns return 0.
        """
        if len(column) == 0:
            return 0
        return int(np.unique(column, return_counts=True)[1].min())

    def _create_transformer(self, data: np.ndarray, categorical_columns: list[int]) -> tuple[ColumnTransformer | None, list[int]]:
        """Functionality: Build an OrdinalEncoder/OneHotEncoder ColumnTransformer, or return None for numeric/none.

        Input:
            data: 2-D features used to filter shufflable categorical columns.
            categorical_columns: Candidate categorical indices.

        Output:
            tuple[ColumnTransformer | None, list[int]].
        """
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
        """Functionality: Return whether a column meets the common-category rule used by feature_shuffled strategies.

        Input:
            column: 1-D array.
            suffix: ordinal strategy suffix; stricter when it contains strict_feature_shuffled.

        Output:
            bool: min count >= 10. Strict mode also requires unique count < n/10.
        """
        min_count = self.get_least_common_category_count(column)
        unique_count = len(np.unique(column))
        
        if "strict_feature_shuffled" in suffix:
            return min_count >= 10 and unique_count < (len(column) // 10)
        return min_count >= 10

class QTx(QuantileTransformer):
    """Functionality: QuantileTransformer variant that silently lowers n_quantiles when it exceeds the sample count.

    Input:
        kwargs: Same keywords as sklearn QuantileTransformer: n_quantiles, output_distribution, subsample, etc.

    Output:
        Estimator instance.
    """

    def __init__(
        self,
        *,
        n_quantiles: int = 1000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
        ignore_implicit_zeros: bool = False,
        subsample: int = 10_000,
        random_state: int | np.random.RandomState | None = None,
        copy: bool = True,
        enable_parallel: bool = True,
        num_jobs: int = 16, # assumption: 128 (cpus) / 8 (gpus)
        parallel_threashold: int = 150_000, # assumption: 5000 (n_samples) * 30 (n_features)
    ) -> None:
        # tuck away the original request
        """Functionality: Expand the sklearn signature explicitly and remember the originally requested n_quantiles.

        Input:
            n_quantiles: Requested quantile count.
            output_distribution: 'uniform' or 'normal'.
            ignore_implicit_zeros: Sparse-matrix option.
            subsample: Fit subsample cap.
            random_state: Random source.
            copy: Whether to copy.

        Output:
            None.
        """
        self._preferred = n_quantiles
        self.enable_parallel = enable_parallel
        self.parallel_threashold = parallel_threashold
        self.num_jobs = num_jobs

        # Keep the complete sklearn signature explicit. ColumnTransformer
        # clones child estimators via get_params(); parameters hidden in
        # **kwargs would otherwise silently revert to sklearn defaults.
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )

    # NOTE: this is an override version of QuantileTransformer _dense_fit
    # (for original implmentation, refer to sklearn/preprocessing/_data.py)
    # here replace the bottleneck step 'np.nanpercentile' with parallel implementation
    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn(
                "'ignore_implicit_zeros' takes effect only with"
                " sparse matrix. This parameter has no effect."
            )

        n_samples, n_features = X.shape
        references = self.references_ * 100

        if self.subsample is not None and self.subsample < n_samples:
            # Take a subsample of `X`
            X = resample(
                X, replace=False, n_samples=self.subsample, random_state=random_state
            )

        if not self.enable_parallel:
            self.quantiles_ = np.nanpercentile(X, references, axis=0)
        else:
            assert X.ndim == 2, f'X must be 2D, got {X.ndim}D with shape {X.shape}'
            if n_samples * n_features < self.parallel_threashold:
                self.quantiles_ = np.nanpercentile(X, references, axis=0)
            else:
                n_jobs = min(self.num_jobs, X.shape[1])
                quantiles = Parallel(n_jobs=n_jobs, prefer='threads')(
                    delayed(np.nanpercentile)(X[:, i], references) for i in range(X.shape[1])
                )
                self.quantiles_ = np.column_stack(quantiles)

        # Due to floating-point precision error in `np.nanpercentile`,
        # make sure that quantiles are monotonically increasing.
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def fit(self, X, y=None):
        # sample count
        """Functionality: Clamp n_quantiles to [1, min(preferred, n_samples, subsample)] then fit the parent.

        Input:
            X: 2-D features.
            y: Ignored.

        Output:
            self. A Generator random_state is converted to RandomState.
        """
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
    """Functionality: KDITransformer variant that fills NaNs with column means for fit/transform, then restores NaN locations after transform.

    Input:
        Inherited KDITransformer args: For example alpha and output_distribution.

    Output:
        Estimator instance.
    """

    def _more_tags(self):
        # obscure way of saying "NaNs are okay"
        """Functionality: Tell sklearn that this estimator allows NaNs.

        Input:
            self: Current estimator.

        Output:
            dict containing allow_nan=True.
        """
        d = {}
        d.update(allow_nan=True)
        return d

    def fit(self, X, y=None):
        # accept both numpy and torch
        """Functionality: Accept numpy or torch, fill NaNs with column means, then fit the parent.

        Input:
            X: 2-D features or Tensor.
            y: Ignored.

        Output:
            self.
        """
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
        """Functionality: Fill NaNs, apply KDI, then write original NaN locations back.

        Input:
            X: 2-D array or Tensor.

        Output:
            np.ndarray: transformed result with the same shape as the input.
        """
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
    """Functionality: Apply distribution transforms from worker_tags to continuous (optionally discrete) columns, and optionally append TruncatedSVD components.

    Input:
        worker_tags: Transformer tags such as quantile, power, kdi_uni.
        discrete_flag: If True, discrete columns also enter the transform.
        original_flag: If True, keep original columns.
        svd_tag: When 'svd', append SVD features.
        svd_max_components: Optional SVD component cap.
        joined_svd_feature: Whether SVD is joined (retained field).
        joined_log_normal: Whether logNormal is joined (retained field).

    Output:
        Preprocessor instance.
    """
    def __init__(
            self,
            *,
            worker_tags: list[str] | None = ["quantile"],
            discrete_flag: bool = False,
            original_flag: bool = False,
            svd_tag: Literal['svd'] | None = None,
            svd_max_components: int | None = None,
            joined_svd_feature: bool = True,
            joined_log_normal: bool = True,
            enable_parallel: bool = True,
            num_jobs: int = 16,
    ):
        """Functionality: Store distribution-rebalancing and SVD options.

        Input:
            See class docstring: worker_tags, discrete_flag, svd_tag, and related fields.

        Output:
            None.
        """
        super().__init__()
        self.worker_tags = worker_tags
        self.discrete_flag = discrete_flag
        self.original_flag = original_flag
        self.random_state = None
        self.svd_tag = svd_tag
        if svd_max_components is not None and svd_max_components <= 0:
            raise ValueError("svd_max_components must be positive when provided")
        self.svd_max_components = svd_max_components
        self.worker: Pipeline | ColumnTransformer | None = None
        self.joined_svd_feature = joined_svd_feature
        self.joined_log_normal = joined_log_normal
        self.enable_parallel = enable_parallel
        self.num_jobs = num_jobs
        self.feature_indices = None
        self.svd_runtime_context: dict[str, Any] | None = None
        self.last_svd_diagnostics: dict[str, Any] | None = None

    @staticmethod
    def _identity_transform(x):
        return x

    @staticmethod
    def _nan_to_num_keep_nan(x):
        return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)

    @staticmethod
    def _shift_by_abs_min(x):
        return x + np.abs(np.nanmin(x))

    @staticmethod
    def _add_epsilon(x):
        return x + 1e-10

    def set_svd_runtime_context(self, context: dict[str, Any] | None) -> None:
        """Functionality: Inject this forward's CUDA memory budget so SVD components can be capped before the model runs.

        Input:
            context: Dict with train_rows, free_cuda_bytes, etc., or None to disable.

        Output:
            None. Also clears last_svd_diagnostics.
        """
        self.svd_runtime_context = context
        self.last_svd_diagnostics = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Functionality: Build a ColumnTransformer (and optional SVD) from sample/feature counts and fit on x only.

        Input:
            x: Usually train features only.
            categorical_features: Categorical column indices.
            seed: Transformer random seed.

        Output:
            list[int]: dis_ix, the output columns treated as discrete.
        """
        self.random_state = seed
        n_samples, n_features = x.shape
        with nvtx.annotate('set'):
            worker, self.dis_ix = self._set(n_samples,n_features,categorical_features)
        with nvtx.annotate('inner-fit'):
            worker.fit(x)
        self.worker = worker
        return self.dis_ix

    @override
    def transform(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Functionality: Apply the fitted distribution transform and optional SVD.

        Input:
            x: 2-D features.

        Output:
            tuple[np.ndarray, list[int]]: (transformed result, discrete indices).
        """
        assert self.worker is not None
        return self.worker.transform(x), self.dis_ix  # type: ignore

    @override
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, *, y:np.ndarray, **kwargs)->tuple[np.ndarray, list[int]]:
        """Functionality: Split train/test by len(y), fit on train only, transform each side, then concatenate.

        Input:
            x: Concatenated train+test features.
            categorical_features: Categorical column indices.
            seed: Random seed.
            y: Train labels; required, used to locate the split.

        Output:
            tuple[np.ndarray, list[int]].
        """
        assert y is not None, "The input y cannot be None"
        x_train_ = x[:len(y)]
        x_test_ = x[len(y):]
        if x_train_.shape[1] != x_test_.shape[1]:
            x_test_ = x_test_[:, :x_train_.shape[1]]
        with nvtx.annotate('fit'):
            categorical_idx_ = self.fit(x_train_, categorical_features, seed)
        with nvtx.annotate('transform'):
            x_train_, categorical_idx_ = self.transform(x_train_)
            x_test_, categorical_idx_ = self.transform(x_test_)
        x_ = np.concatenate([x_train_, x_test_], axis=0)

        return (x_, categorical_idx_)

    def _set(self,n_samples: int,
        n_features: int,
        categorical_features: list[int],
        ):
        """Functionality: Build the sklearn pipeline from worker_tags and SVD config, possibly lowering svd_n_comp to the memory budget.

        Input:
            n_samples: Fit sample count.
            n_features: Input feature count.
            categorical_features: Categorical column indices.

        Output:
            tuple[Pipeline | ColumnTransformer, list[int]]: (worker, discrete indices).
        """
        static_seed, rng = infer_random_state(self.random_state)
        all_ix = list(range(n_features))
        workers = []
        cont_ix = [i for i in all_ix if i not in categorical_features]
        if self.original_flag:
            trans_ixs = categorical_features + cont_ix if self.discrete_flag else cont_ix
            workers.append(("original", "passthrough", all_ix))
            dis_ix = categorical_features
        elif self.discrete_flag:
            trans_ixs = categorical_features + cont_ix
            self.feature_indices = categorical_features + cont_ix
            dis_ix = []
        else:
            workers.append(("discrete", "passthrough", categorical_features))
            trans_ixs, dis_ix = cont_ix, list(range(len(categorical_features)))
        for worker_tag in self.worker_tags:
            if worker_tag == "logNormal":
                sworker = Pipeline(steps=[
                                        ("save_standard", Pipeline(steps=[
                                            ("i2n_pre",
                                             FunctionTransformer(
                                                 func=self._nan_to_num_keep_nan,
                                                 inverse_func=self._identity_transform, check_inverse=False)),
                                            ("fill_missing_pre",
                                             SimpleImputer(missing_values=np.nan, strategy="mean",
                                                           keep_empty_features=True)),
                                            ("feature_shift",
                                             FunctionTransformer(func=self._shift_by_abs_min)),
                                            ("add_epsilon", FunctionTransformer(func=self._add_epsilon)),
                                            ("logNormal", FunctionTransformer(np.log, validate=False)),
                                            ("i2n_post",
                                             FunctionTransformer(
                                                 func=self._nan_to_num_keep_nan,
                                                 inverse_func=self._identity_transform, check_inverse=False)),
                                            ("fill_missing_post",
                                             SimpleImputer(missing_values=np.nan, strategy="mean",
                                                           keep_empty_features=True))])),
                                        ])
                # trans_ixs = cont_ix
            elif worker_tag == "quantile_uniform_10":
                sworker = QTx(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 10, 2),
                    random_state=static_seed,
                    enable_parallel=self.enable_parallel,
                    num_jobs=self.num_jobs,
                )
            elif worker_tag == "quantile_uniform_5":
                sworker = QTx(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    enable_parallel=self.enable_parallel,
                    num_jobs=self.num_jobs,
                )
            elif worker_tag == "quantile_uniform_all_data":
                sworker = QTx(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    subsample=n_samples,
                    enable_parallel=self.enable_parallel,
                    num_jobs=self.num_jobs,
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
                                    ("power_transformer", RobustPowerTransformer(standardize=True)),
                                    ("inf_to_nan_1", FunctionTransformer(
                                                        func=self._nan_to_num_keep_nan,
                                                        inverse_func=self._identity_transform,
                                                        check_inverse=False,
                                                    )),
                                    ("nan_to_mean_1", nan_to_mean_transformer),
                                    ("scaler", SkipAllNaNStandardScaler()),
                                    ("inf_to_nan_2", FunctionTransformer(
                                                        func=self._nan_to_num_keep_nan,
                                                        inverse_func=self._identity_transform,
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
                    enable_parallel=self.enable_parallel,
                    num_jobs=self.num_jobs,
                )
            elif worker_tag=="quantile_norm_5":
                sworker = QTx(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    enable_parallel=self.enable_parallel,
                    num_jobs=self.num_jobs,
                )
            elif worker_tag == "quantile_norm_all_data":
                sworker = QTx(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    subsample=n_samples,
                    enable_parallel=self.enable_parallel,
                    num_jobs=self.num_jobs,
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
                                enable_parallel=self.enable_parallel,
                                num_jobs=self.num_jobs,
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
                sworker = FunctionTransformer(self._identity_transform)
            elif worker_tag.startswith("kdi_uni_alpha_"):
                alpha = float(worker_tag.split("_")[-1])
                sworker = KDIX(alpha=alpha, output_distribution="uniform")
            elif worker_tag.startswith("kdi_norm_alpha_"):
                alpha = float(worker_tag.split("_")[-1])
                sworker = KDIX(alpha=alpha, output_distribution="normal")
            elif worker_tag=="kdi_norm":
                sworker = KDIX(alpha=1.0, output_distribution="normal")
            else:
                sworker = FunctionTransformer(self._identity_transform)
            if worker_tag in ["quantile_uniform_10", "quantile_uniform_5", "quantile_uniform_all_data"]:
                self.n_quantile_features = len(trans_ixs)
            workers.append((f"feat_transform_{worker_tag}", sworker, trans_ixs))

        CT_worker = ColumnTransformer(workers,remainder="drop",sparse_threshold=0.0)
        if self.svd_tag == "svd" and n_features >= 2:
            svd_limits = [n_samples // 10 + 1, n_features // 2]
            if self.svd_max_components is not None:
                svd_limits.append(self.svd_max_components)
            requested_svd_n_comp = max(1, min(svd_limits))
            base_output_features = (
                (n_features if self.original_flag else 0)
                + (
                    len(categorical_features)
                    if not self.original_flag and not self.discrete_flag
                    else 0
                )
                + len(self.worker_tags) * len(trans_ixs)
            )
            self.svd_n_comp = requested_svd_n_comp
            context = self.svd_runtime_context
            if context is not None and context.get("enabled", False):
                runtime_max_components = context.get("runtime_max_components")
                effective_requested_svd_n_comp = requested_svd_n_comp
                if runtime_max_components is not None:
                    effective_requested_svd_n_comp = min(
                        effective_requested_svd_n_comp,
                        max(0, int(runtime_max_components)),
                    )
                diagnostics = choose_svd_components(
                    requested_components=effective_requested_svd_n_comp,
                    base_output_features=max(1, base_output_features),
                    train_rows=int(context["train_rows"]),
                    query_rows=int(context["query_rows"]),
                    free_cuda_bytes=int(context["free_cuda_bytes"]),
                    model_bytes_to_load=int(context["model_bytes_to_load"]),
                    embedding_dim=int(context["embedding_dim"]),
                    num_heads=int(context["num_heads"]),
                    mixed_precision=bool(context["mixed_precision"]),
                    memory_fraction=float(context["memory_fraction"]),
                    reserve_mb=int(context["reserve_mb"]),
                    safety_factor=float(context["safety_factor"]),
                    minimum_components=int(context["minimum_components"]),
                    sequence_attention_limit=context.get(
                        "sequence_attention_limit"
                    ),
                )
                if effective_requested_svd_n_comp < requested_svd_n_comp:
                    diagnostics.update(
                        requested_components=int(requested_svd_n_comp),
                        memory_budget_requested_components=int(
                            effective_requested_svd_n_comp
                        ),
                        adapted=True,
                        reason="cuda_resource_retry_component_cap",
                    )
                diagnostics.update(
                    configured_max_components=self.svd_max_components,
                    runtime_max_components=runtime_max_components,
                    input_features=int(n_features),
                )
                self.last_svd_diagnostics = diagnostics
                self.svd_n_comp = diagnostics["selected_components"]
            else:
                self.last_svd_diagnostics = {
                    "requested_components": int(requested_svd_n_comp),
                    "selected_components": int(requested_svd_n_comp),
                    "adapted": False,
                    "reason": "adaptive_svd_disabled",
                    "configured_max_components": self.svd_max_components,
                    "input_features": int(n_features),
                    "base_output_features": int(base_output_features),
                    "selected_output_features": int(
                        base_output_features + requested_svd_n_comp
                    ),
                }
            if self.svd_n_comp <= 0:
                worker = CT_worker
                self.worker = worker
                return worker, dis_ix
            svd_worker = FeatureUnion([
                    ("default", FunctionTransformer(func=self._identity_transform)),
                    ("svd",Pipeline(steps=[
                                    ("save_standard",Pipeline(steps=[
                                    ("i2n_pre", FunctionTransformer(func=self._nan_to_num_keep_nan,inverse_func=self._identity_transform, check_inverse=False)),
                                    ("fill_missing_pre", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)),
                                    ("standard", SkipAllNaNStandardScaler(with_mean=False)) ,
                                    ("i2n_post", FunctionTransformer(func=self._nan_to_num_keep_nan,inverse_func=self._identity_transform, check_inverse=False)),
                                    ("fill_missing_post", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True))])),
                                    ("svd",TruncatedSVD(algorithm="arpack",n_components=self.svd_n_comp,random_state=static_seed))]))
                    ])
            worker = Pipeline([("worker", CT_worker), ("svd_worker", svd_worker)])
        else:   
            self.svd_n_comp = 0
            worker = CT_worker

        self.worker = worker
        return worker, dis_ix


# Large constant for hash normalization
_HASH_MODULUS = 10**12

def float_hash_arr(input_array: np.ndarray) -> float:
    """Functionality: SHA256-hash an array's bytes and normalize to a float fingerprint in [0, 1).

    Input:
        input_array: Arbitrary numpy array.

    Output:
        float in [0, 1).
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
    """Functionality: Append a per-row hash fingerprint column. Train resolves collisions by rehashing; test uses the first hash.

    Input:
        rng_seed: Historical argument, currently unused. The salt is generated from seed at fit.

    Output:
        Preprocessor instance.
    """
    
    def __init__(self, rng_seed: int | np.random.Generator | None = None):
        """Functionality: Initialize salt and categorical indices to empty.

        Input:
            rng_seed: Reserved; unused.

        Output:
            None.
        """
        super().__init__()
        # self.rng_seed = rng_seed
        self.salt_value = None
        self.categorical_features = None
    
    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Functionality: Sample a 16-bit salt and remember categorical indices.

        Input:
            x: Feature matrix; only the random source is used.
            categorical_features: Categorical indices, returned unchanged.
            seed: Used to generate the salt.

        Output:
            list[int]: copy of the categorical indices.
        """
        _, rng = infer_random_state(seed)
        self.salt_value = int(rng.integers(0, 65536))  # 2^16 range
        self.categorical_features = categorical_features
        return categorical_features.copy()
    
    @override
    def transform(self, x:np.ndarray, is_test:bool=False, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Functionality: Append a fingerprint column on the right.

        Input:
            x: 2-D features.
            is_test: If True, resolve collisions against a copy of the state left by the train transform,
                so test rows get the same hashes as when appended after train in one matrix.

        Output:
            tuple[np.ndarray, list[int]]: (matrix with fingerprint column, original categorical indices).
        """
        # print(f"add finger")
        if self.salt_value is None:
            raise RuntimeError("Must call fit() before transform()")
        
        n_samples = x.shape[0]
        fingerprint_col = np.zeros(n_samples, dtype=x.dtype)
        
        # Apply salt to input data
        salted_data = x + self.salt_value
        
        if is_test:
            existing_hashes = set(self.train_hashes_)
            next_increment_by_row: dict[bytes, int] = dict(self.train_increments_)
        else:
            existing_hashes = set()
            next_increment_by_row = {}
        for idx in range(n_samples):
            current_row = salted_data[idx]
            row_key = current_row.tobytes()
            increment = next_increment_by_row.get(row_key, 0)
            hash_val = float_hash_arr(current_row if increment == 0 else current_row + increment)
            rehash_useless = False
            
            # Handle collisions by incrementing and rehashing.
            # Guard: for a row that is entirely non-finite (all NaN/inf),
            # `current_row + increment` leaves the bytes unchanged, so the
            # rehash is constant and this loop would spin forever. Break as
            # soon as the rehash stops changing. Rows with any finite value
            # keep the original behavior exactly, so results are unchanged
            # for every dataset that previously completed.
            while hash_val in existing_hashes:
                increment += 1
                new_hash = float_hash_arr(current_row + increment)
                if new_hash == hash_val:
                    rehash_useless = True
                    break
                hash_val = new_hash
            
            fingerprint_col[idx] = hash_val
            existing_hashes.add(hash_val)
            if rehash_useless:
                # NOTE: The 1st all-NaN/inf row creates a {row_key, hash} dict item
                #       the 2nd all-NaN/inf row detects that rehashing is useless and removes its row_key
                #       the resulting hash will still collide anyway
                next_increment_by_row.pop(row_key, None)
            else:
                next_increment_by_row[row_key] = increment + 1
        if not is_test:
            self.train_hashes_ = existing_hashes
            self.train_increments_ = next_increment_by_row
        
        # Append fingerprint column and update categorical indices
        transformed = np.column_stack([x, fingerprint_col.reshape(-1, 1)])
        # cat_indices_updated = list(range(x.shape[1]))  # Original features remain categorical
        
        return transformed, self.categorical_features

class PolynomialInteractionGenerator(BasePreprocess):
    """Functionality: Sample feature pairs and generate standardized pairwise product interaction columns.

    Input:
        max_interaction_features: Maximum interaction columns, default 100.
        random_generator: Reserved; the actual RNG comes from the fit seed.

    Output:
        Preprocessor instance.
    """
    
    def __init__(
        self, 
        *, 
        max_interaction_features: int | None = None,
        random_generator: int | np.random.Generator | None = None
    ):
        """Functionality: Store the interaction-column cap and enable runtime interaction generation by default.

        Input:
            max_interaction_features: Positive int, or None which falls back to 100.
            random_generator: Unused.

        Output:
            None.
        """
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
        self.feature_normalizer = SkipAllNaNStandardScaler(with_mean=False)
        self.categorical_features = None
        # Runtime CUDA fallback state.  Keep the configured interaction count
        # intact so the next public predict() call can restore normal behavior.
        self.runtime_interactions_enabled = True

    def set_runtime_interactions_enabled(self, enabled: bool) -> None:
        """Functionality: Enable or disable appending interaction columns for this inference call without changing configured max_interactions.

        Input:
            enabled: If False, transform returns only standardized original features.

        Output:
            None.
        """
        self.runtime_interactions_enabled = bool(enabled)

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Functionality: Fit StandardScaler and sample factor pairs. Factor indices are empty when runtime generation is disabled.

        Input:
            x: 2-D features.
            categorical_features: Categorical indices, kept as-is.
            seed: Random seed used to sample factor pairs.

        Output:
            list[int]: categorical indices.
        """
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
        
        # Fit only here; transform() materializes the standardized values once.
        self.feature_normalizer.fit(x)

        if not getattr(self, "runtime_interactions_enabled", True):
            self.primary_factor_indices = np.empty(0, dtype=np.int64)
            self.secondary_factor_indices = np.empty(0, dtype=np.int64)
            self.categorical_features = categorical_features
            return categorical_features
        
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
        """Functionality: Randomly generate unique (i, j) factor pairs with j >= i.

        Input:
            total_features: Total feature count.
            required_pairs: Number of interaction pairs needed.
            rng: numpy Generator.

        Output:
            None. Writes primary_factor_indices and secondary_factor_indices.
        """
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
        """Functionality: Standardize the input and, when enabled, concatenate pairwise product interaction columns.

        Input:
            x: 2-D features; column count must match fit.

        Output:
            tuple[np.ndarray, list[int]]: (standardized[+interaction] features, categorical indices).
        """
        assert x.ndim == 2, "Input matrix must be 2-dimensional"
        
        if x.size == 0:
            return x, []
        
        # Standardize input features
        standardized_features = self.feature_normalizer.transform(x)
        
        if not getattr(self, "runtime_interactions_enabled", True):
            return standardized_features, self.categorical_features

        # Generate polynomial interaction features
        interaction_features = (
            standardized_features[:, self.primary_factor_indices] * 
            standardized_features[:, self.secondary_factor_indices]
        )
        
        # Combine original and interaction features
        transformed_output = np.column_stack([standardized_features, interaction_features])
        
        return transformed_output, self.categorical_features
