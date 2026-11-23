"""Feature View Ensemble inference built on the core classifier forward."""

from __future__ import annotations
import time
import nvtx

import gc
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from .infer_profile import round_name as _profile_round, span as _profile_span
from .preprocess import (
    FixedFeatureViewPreprocessor,
    resolve_fixed_feature_view_config,
)


FeatureViewStrategy = Literal[
    "independent_view_prediction_mean",
    "merged_feature_view",
]


class FeatureViewEnsembleError(ValueError):
    """Functionality: Raised when the fixed feature-view contract is violated, e.g. misaligned rows/columns or an illegal probability matrix.

    Input:
        Same as ValueError: Error message string.

    Output:
        Exception type for callers to catch.
    """


class _FixedFeatureViewEnsembleRunner:
    """Functionality: Run deterministic fixed feature-view ensembling with an injected classifier forward.

    Input:
        predict_cls: Callable compatible with (x_train, y_train, x_test, task_type, unique_dataset_name=...).

    Output:
        Runner instance. predict() returns probabilities; the audit is stored on feature_view_ensemble_audit.
    """

    def __init__(self, predict_cls):
        """Functionality: Store the classifier forward callable.

        Input:
            predict_cls: Core classification predict function.

        Output:
            None.
        """
        self._predict_cls = predict_cls
        self.feature_view_ensemble_audit: dict | None = None

    @staticmethod
    def _as_feature_frame(values, *, columns=None, name: str) -> pd.DataFrame:
        """Functionality: Normalize input to a DataFrame with unique column names and at least one row. Query may be forced to match given columns.

        Input:
            values: DataFrame or 2-D array.
            columns: Optional; query must match support column names/count.
            name: Variable name used in error messages.

        Output:
            pandas.DataFrame.
        """
        if isinstance(values, pd.DataFrame):
            frame = values.copy(deep=True).reset_index(drop=True)
            if columns is not None and tuple(frame.columns) != tuple(columns):
                raise FeatureViewEnsembleError("support and query columns differ")
        else:
            array = np.asarray(values)
            if array.ndim != 2:
                raise FeatureViewEnsembleError(f"{name} must be two-dimensional")
            frame = pd.DataFrame(array)
            frame.columns = (
                [f"feature_{index:04d}" for index in range(frame.shape[1])]
                if columns is None
                else list(columns)
            )
        if frame.columns.duplicated().any():
            raise FeatureViewEnsembleError(f"{name} contains duplicate columns")
        if columns is not None and frame.shape[1] != len(columns):
            raise FeatureViewEnsembleError("support and query feature counts differ")
        if len(frame) == 0:
            raise FeatureViewEnsembleError(f"{name} must contain at least one row")
        return frame

    @staticmethod
    def _probability_matrix(probabilities, *, rows: int) -> np.ndarray:
        """Functionality: Validate classifier output as a query-by-class matrix of finite non-negative probabilities and row-normalize it.

        Input:
            probabilities: 2-D probabilities or array-like output.
            rows: Expected query row count.

        Output:
            np.ndarray[float32]: probability matrix whose rows sum to 1.
        """
        values = np.asarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != rows or values.shape[1] < 2:
            raise FeatureViewEnsembleError(
                "classifier output must be a query-row-by-class probability matrix"
            )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise FeatureViewEnsembleError("classifier probabilities are invalid")
        row_sums = values.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise FeatureViewEnsembleError(
                "classifier probability rows must have positive sums"
            )
        return np.ascontiguousarray(values / row_sums, dtype=np.float32)

    @staticmethod
    def _cuda_resource_failure_reason(error: BaseException) -> str | None:
        """Functionality: Walk the exception chain and classify CUDA OOM or illegal kernel configuration.

        Input:
            error: Caught exception.

        Output:
            str | None: 'out_of_memory', 'invalid_configuration', or None.
        """
        cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        current: BaseException | None = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(current, cuda_oom_type):
                return "out_of_memory"
            if isinstance(current, RuntimeError):
                message = str(current).casefold()
                if "out of memory" in message or "retryable cuda out_of_memory" in message:
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
    def _release_cuda_after_resource_error(error: BaseException) -> None:
        """Functionality: Drop exception traceback references and empty the CUDA cache.

        Input:
            error: Resource-failure exception.

        Output:
            None.
        """
        current: BaseException | None = error
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            current.__traceback__ = None
            current = current.__cause__ or current.__context__
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _is_fatal_feature_view_error(error: BaseException) -> bool:
        """Functionality: Return whether a feature-view failure must be re-raised (interrupts, system exit, severe CUDA errors, etc.).

        Input:
            error: Exception.

        Output:
            bool: True means the caller must not fall back to the base prediction.
        """
        if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit, MemoryError)):
            return True
        if not isinstance(error, RuntimeError):
            return False
        message = str(error).casefold()
        return any(
            marker in message
            for marker in (
                "cuda error",
                "cuda out of memory",
                "cudnn",
                "cublas",
                "device-side assert",
                "illegal memory access",
                "misaligned address",
            )
        )

    def _set_audit(self, *, strategy: str, task_type: str, **details) -> None:
        """Functionality: Write the fixed-view ensemble audit dict, stating that query labels and search were not used.

        Input:
            strategy: Feature-view strategy name.
            task_type: 'binary' or 'multiclass'.
            details: Audit fields to overlay or add.

        Output:
            None. Result is stored on self.feature_view_ensemble_audit.
        """
        audit = {
            "schema": "feature-view-ensemble-v1",
            "configured": True,
            "enabled": True,
            "feature_view_strategy": strategy,
            "task_type": task_type,
            "query_labels_used": False,
            "labels_used_for_feature_generation": False,
            "search_used": False,
            "feature_generation_policy": "fixed_no_search",
            "feature_selection_metric": "none_fixed_rules",
            "target_encoding": "none",
            "cross_fitting": False,
        }
        audit.update(details)
        self.feature_view_ensemble_audit = audit

    def predict(
        self,
        x_train,
        y_train,
        x_test,
        task_type: str,
        *,
        base_x_train=None,
        base_x_test=None,
        unique_dataset_name: str | None = None,
        feature_view_strategy: FeatureViewStrategy = "independent_view_prediction_mean",
        feature_view_prediction_weight: float = 0.5,
        fixed_feature_views: dict | None = None,
        feature_view_preprocessor: FixedFeatureViewPreprocessor | None = None,
    ) -> np.ndarray:
        """Functionality: Run the base classifier, then independent or merged fixed-view forwards, and blend probabilities by weight.

        Input:
            x_train: Support features.
            y_train: 1-D labels with no missing values and at least two classes.
            x_test: Query features; columns must match support.
            task_type: Task type forwarded to predict_cls.
            base_x_train: Optional train features for the base path.
            base_x_test: Optional test features for the base path.
            unique_dataset_name: Optional cache/name prefix.
            feature_view_strategy: independent_view_prediction_mean or merged_feature_view.
            feature_view_prediction_weight: View weight in [0, 1].
            fixed_feature_views: Fixed-view config; mutually exclusive with preprocessor.
            feature_view_preprocessor: Already constructed FixedFeatureViewPreprocessor.

        Output:
            np.ndarray[float32]: query-by-class blended probabilities. Falls back to base on failure or zero weight.
        """
        supported_strategies = {
            "independent_view_prediction_mean",
            "merged_feature_view",
        }
        if feature_view_strategy not in supported_strategies:
            raise ValueError(
                "feature_view_strategy must be one of "
                f"{sorted(supported_strategies)}, got {feature_view_strategy!r}"
            )
        if isinstance(feature_view_prediction_weight, bool) or not isinstance(
            feature_view_prediction_weight,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("feature_view_prediction_weight must be a real number")
        view_weight = float(feature_view_prediction_weight)
        if not 0.0 <= view_weight <= 1.0:
            raise ValueError("feature_view_prediction_weight must be in [0, 1]")
        base_weight = 1.0 - view_weight

        support = self._as_feature_frame(x_train, name="x_train")
        query = self._as_feature_frame(
            x_test,
            columns=support.columns,
            name="x_test",
        )
        labels = np.asarray(y_train)
        if labels.ndim != 1 or len(labels) != len(support):
            raise FeatureViewEnsembleError(
                "support labels must be one-dimensional and row aligned"
            )
        if pd.isna(labels).any():
            raise FeatureViewEnsembleError("support labels contain missing values")
        class_count = len(pd.unique(labels))
        if class_count < 2:
            raise FeatureViewEnsembleError(
                "classification requires at least two support classes"
            )
        audit_task_type = "multiclass" if class_count >= 3 else "binary"

        # print(f'get-base inputs | train = {base_x_train.shape}, test = {base_x_test.shape}')
        with nvtx.annotate('base'):
            with _profile_round("base"):
                base = self._probability_matrix(
                    self._predict_cls(
                        support if base_x_train is None else base_x_train,
                        labels,
                        query if base_x_test is None else base_x_test,
                        task_type,
                        unique_dataset_name=unique_dataset_name,
                    ),
                    rows=len(query),
                )

        if base.shape[1] != class_count:
            raise FeatureViewEnsembleError(
                "base probability class count differs from support labels"
            )

        active_views: tuple[str, ...] = ()
        completed_predictions: list[np.ndarray] = []
        current_view: str | None = None
        preprocess_audit: dict | None = None

        def return_base(reason: str, **details) -> np.ndarray:
            """Functionality: Record fallback audit and return the unblended base probabilities.

            Input:
                reason: Fallback reason string.
                details: Extra audit fields.

            Output:
                np.ndarray: base probability matrix.
            """
            self._set_audit(
                strategy=feature_view_strategy,
                task_type=audit_task_type,
                class_count=class_count,
                active_feature_views=list(active_views),
                feature_view_forward_count=len(completed_predictions),
                feature_view_prediction_count=len(completed_predictions),
                base_prediction_weight=base_weight,
                feature_view_prediction_weight=view_weight,
                ensemble_method="base_classification_ensemble",
                feature_view_applied=False,
                fallback_exact_base=True,
                fallback_reason=reason,
                preprocess=preprocess_audit,
                **details,
            )
            return base

        if view_weight == 0.0:
            return return_base("zero_feature_view_weight")

        try:
            if feature_view_preprocessor is not None and fixed_feature_views is not None:
                raise ValueError(
                    "pass fixed_feature_views or feature_view_preprocessor, not both"
                )
            if feature_view_preprocessor is None:
                config = resolve_fixed_feature_view_config(fixed_feature_views)
                feature_view_preprocessor = FixedFeatureViewPreprocessor(**config)
            with _profile_span("feature_view.fit"):
                feature_view_preprocessor.fit(support)
                preprocess_audit = feature_view_preprocessor.audit()
                active_views = feature_view_preprocessor.active_views_
            if not feature_view_preprocessor.config["enabled"]:
                return return_base("fixed_feature_views_disabled")
            if not active_views:
                return return_base("no_applicable_fixed_feature_view")

            if feature_view_strategy == "independent_view_prediction_mean":
                with _profile_span("feature_view.transform"):
                    view_pairs = [
                        (
                            view,
                            feature_view_preprocessor.transform(support, view=view),
                            feature_view_preprocessor.transform(query, view=view),
                        )
                        for view in active_views
                    ]
            else:
                with _profile_span("feature_view.transform"):
                    view_pairs = [
                        (
                            "merged",
                            feature_view_preprocessor.transform_merged(support),
                            feature_view_preprocessor.transform_merged(query),
                        )
                    ]

            for idx, (current_view, view_support, view_query) in enumerate(view_pairs):
                with nvtx.annotate('prob'):
                    with _profile_round(f"view.{current_view}"):
                        probabilities = self._probability_matrix(
                            self._predict_cls(
                                view_support,
                                labels,
                                view_query,
                                task_type,
                                unique_dataset_name=(
                                    f"{unique_dataset_name or 'anonymous'}"
                                    f"__feature_view__{current_view}"
                                ),
                            ),
                            rows=len(query),
                        )

                if probabilities.shape != base.shape:
                    raise FeatureViewEnsembleError(
                        f"{current_view} probability shape differs from base"
                    )
                completed_predictions.append(probabilities)

            view_mean = np.mean(np.stack(completed_predictions), axis=0)
            blended = base_weight * base + view_weight * view_mean
            blended /= blended.sum(axis=1, keepdims=True)
            self._set_audit(
                strategy=feature_view_strategy,
                task_type=audit_task_type,
                class_count=class_count,
                active_feature_views=list(active_views),
                feature_view_forward_count=len(completed_predictions),
                feature_view_prediction_count=len(completed_predictions),
                base_prediction_weight=base_weight,
                feature_view_prediction_weight=view_weight,
                ensemble_method=feature_view_strategy,
                feature_view_applied=True,
                formula=(
                    "normalize(base_weight * P_base + "
                    "view_weight * mean(P_fixed_feature_views))"
                ),
                fallback_exact_base=False,
                fallback_reason=None,
                preprocess=preprocess_audit,
            )
            return np.asarray(blended, dtype=np.float32)
        except Exception as error:
            resource_reason = self._cuda_resource_failure_reason(error)
            if resource_reason is not None:
                self._release_cuda_after_resource_error(error)
                print(
                    "CUDA resource failure in an optional fixed feature view; "
                    "returning the completed base prediction "
                    f"(view={current_view}, reason={resource_reason})."
                )
                return return_base(
                    f"feature_view_cuda_{resource_reason}",
                    feature_view_attempted_count=len(completed_predictions) + 1,
                    failed_feature_view=current_view,
                    feature_view_error_type=type(error).__name__,
                    feature_view_error=str(error),
                )
            if self._is_fatal_feature_view_error(error):
                raise
            return return_base(
                "feature_view_failure",
                failed_feature_view=current_view,
                feature_view_error_type=type(error).__name__,
                feature_view_error=str(error),
            )


def _predict_with_fixed_feature_views(
    *,
    predict_cls,
    x_train,
    y_train,
    x_test,
    task_type: str,
    base_x_train=None,
    base_x_test=None,
    unique_dataset_name: str | None = None,
    feature_view_strategy: FeatureViewStrategy = "independent_view_prediction_mean",
    feature_view_prediction_weight: float = 0.5,
    fixed_feature_views: dict | None = None,
    feature_view_preprocessor: FixedFeatureViewPreprocessor | None = None,
) -> tuple[np.ndarray, dict]:
    """Functionality: Construct the fixed-view runner and return its prediction and audit.

    Input:
        predict_cls: Classifier forward.
        x_train: Support features.
        y_train: Support labels.
        x_test: Query features.
        task_type: Task type.
        base_x_train: Optional base train features.
        base_x_test: Optional base test features.
        unique_dataset_name: Optional dataset name.
        feature_view_strategy: View strategy.
        feature_view_prediction_weight: View weight.
        fixed_feature_views: Fixed-view config.
        feature_view_preprocessor: Optional already constructed preprocessor.

    Output:
        tuple[np.ndarray, dict]: (probabilities, audit).
    """
    runner = _FixedFeatureViewEnsembleRunner(predict_cls)
    prediction = runner.predict(
        x_train,
        y_train,
        x_test,
        task_type,
        base_x_train=base_x_train,
        base_x_test=base_x_test,
        unique_dataset_name=unique_dataset_name,
        feature_view_strategy=feature_view_strategy,
        feature_view_prediction_weight=feature_view_prediction_weight,
        fixed_feature_views=fixed_feature_views,
        feature_view_preprocessor=feature_view_preprocessor,
    )
    assert runner.feature_view_ensemble_audit is not None
    return prediction, runner.feature_view_ensemble_audit



class _FeatureViewEnsembleRunner:
    """Functionality: Legacy feature-view runner: may select views from labels, build derived features, and blend with base. Multiclass uses a prior adjustment.

    Input:
        predict_cls: Core classifier forward.

    Output:
        Runner instance.
    """

    def __init__(self, predict_cls):
        """Functionality: Store the classifier forward.

        Input:
            predict_cls: Callable.

        Output:
            None.
        """
        self._predict_cls = predict_cls
        self.feature_view_ensemble_audit = None

    def predict(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        task_type: str,
        base_x_train=None,
        base_x_test=None,
        unique_dataset_name: str = None,
        feature_view_strategy: Literal[
            "independent_view_prediction_mean",
            "merged_feature_view",
        ] = "independent_view_prediction_mean",
        feature_view_prediction_weight: float = 0.5,
        route_sampling_seed: int = 20260806,
        cross_fit_seed: int = 0,
    ) -> np.ndarray:
        """Functionality: Run support-side view selection, feature construction, classifier forwards, and probability blending. Query labels are neither accepted nor used.

        Input:
            x_train: Support features.
            y_train: Support labels.
            x_test: Query features.
            task_type: Task type.
            base_x_train: Optional base train features.
            base_x_test: Optional base test features.
            unique_dataset_name: Optional dataset name.
            feature_view_strategy: Independent-view mean or merged view.
            feature_view_prediction_weight: View weight in [0, 1].
            route_sampling_seed: View-routing / sampling seed.
            cross_fit_seed: Target-encoding cross-fit seed.

        Output:
            np.ndarray: query-by-class probabilities. Multiclass returns prior-adjusted base; binary may blend views.
        """
        import math
        import re
        import unicodedata
        from collections import Counter, defaultdict

        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import MinMaxScaler

        supported_strategies = {
            "independent_view_prediction_mean",
            "merged_feature_view",
        }
        if feature_view_strategy not in supported_strategies:
            raise ValueError(
                "feature_view_strategy must be one of "
                f"{sorted(supported_strategies)}, got {feature_view_strategy!r}"
            )
        if isinstance(feature_view_prediction_weight, bool) or not isinstance(
            feature_view_prediction_weight, (int, float, np.integer, np.floating)
        ):
            raise TypeError("feature_view_prediction_weight must be a real number")
        feature_view_prediction_weight = float(feature_view_prediction_weight)
        if not 0.0 <= feature_view_prediction_weight <= 1.0:
            raise ValueError("feature_view_prediction_weight must be in [0, 1]")
        if isinstance(route_sampling_seed, bool) or not isinstance(
            route_sampling_seed, (int, np.integer)
        ):
            raise TypeError("route_sampling_seed must be an integer")
        if isinstance(cross_fit_seed, bool) or not isinstance(
            cross_fit_seed, (int, np.integer)
        ):
            raise TypeError("cross_fit_seed must be an integer")
        route_sampling_seed = int(route_sampling_seed)
        cross_fit_seed = int(cross_fit_seed)
        base_prediction_weight = 1.0 - feature_view_prediction_weight

        derived_feature_budget = 12
        pair_pool_columns = 24
        route_sample_capacity = 2048
        target_folds = 2
        target_smoothing = 12.0
        low_cardinality_max_states = 10
        low_cardinality_max_selected_columns = 6
        multiclass_adjustment_epsilon = 1e-8
        token_pattern = re.compile(r"[^\W_]+", flags=re.UNICODE)
        nonfinite_state = "__nonfinite_state__"

        class FeatureViewEnsembleError(ValueError):
            """Functionality: Contract-error type used inside the legacy feature-view path.

            Input:
                message: Which view constraint was violated.

            Output:
                Exception type.
            """
            pass

        def _as_feature_frame(values, columns=None):
            """Functionality: Convert an array or DataFrame into a table with feature_xxxx column names, optionally applying support column names.

            Input:
                values: DataFrame or 2-D array.
                columns: Optional column-name sequence.

            Output:
                pandas.DataFrame.
            """
            if isinstance(values, pd.DataFrame):
                frame = values.copy(deep=True).reset_index(drop=True)
            else:
                array = np.asarray(values)
                if array.ndim != 2:
                    raise FeatureViewEnsembleError(
                        "support and query features must be two-dimensional"
                    )
                frame = pd.DataFrame(array)
                frame = frame.infer_objects()
            if columns is None:
                frame.columns = [f"feature_{index:04d}" for index in range(frame.shape[1])]
            else:
                if frame.shape[1] != len(columns):
                    raise FeatureViewEnsembleError(
                        "support and query feature counts differ"
                    )
                frame.columns = list(columns)
            return frame

        def _validate_feature_pair(support, query):
            """Functionality: Require support/query to be non-empty DataFrames with unique, identical column names.

            Input:
                support: Train feature table.
                query: Test feature table.

            Output:
                tuple[DataFrame, DataFrame]: deep-copied tables with reset index.
            """
            if not isinstance(support, pd.DataFrame) or not isinstance(query, pd.DataFrame):
                raise TypeError("support and query must be pandas DataFrames")
            if support.columns.has_duplicates or query.columns.has_duplicates:
                raise FeatureViewEnsembleError(
                    "support and query columns must be unique"
                )
            if tuple(support.columns) != tuple(query.columns):
                raise FeatureViewEnsembleError("support and query columns differ")
            if len(support) == 0 or len(query) == 0:
                raise FeatureViewEnsembleError(
                    "support and query must be non-empty"
                )
            return (
                support.copy(deep=True).reset_index(drop=True),
                query.copy(deep=True).reset_index(drop=True),
            )

        def _encode_binary_labels(values, rows):
            """Functionality: Encode binary labels as {0,1}. Require 1-D, no missing values, and exactly two classes.

            Input:
                values: Label array.
                rows: Must equal the support row count.

            Output:
                np.ndarray[int64]: 0/1 labels.
            """
            labels = np.asarray(values)
            if labels.ndim != 1 or len(labels) != rows:
                raise FeatureViewEnsembleError(
                    "support labels must be one-dimensional and row aligned"
                )
            if pd.isna(labels).any():
                raise FeatureViewEnsembleError(
                    "support labels contain missing values"
                )
            unique = np.unique(labels)
            if len(unique) != 2:
                raise FeatureViewEnsembleError(
                    "binary feature views require exactly two support classes"
                )
            encoder = {value: index for index, value in enumerate(unique.tolist())}
            return np.asarray([encoder[value] for value in labels], dtype=np.int64)

        def _normalize_text(value):
            """Functionality: Normalize a scalar to NFKC lowercased text. Missing becomes __missing__; empty becomes __empty__.

            Input:
                value: Arbitrary scalar.

            Output:
                str.
            """
            if value is None:
                return "__missing__"
            try:
                if bool(pd.isna(value)):
                    return "__missing__"
            except (TypeError, ValueError):
                pass
            normalized = " ".join(
                unicodedata.normalize("NFKC", str(value)).casefold().split()
            )
            return normalized or "__empty__"

        def _tokens(value):
            """Functionality: Extract unicode word tokens from normalized text.

            Input:
                value: Raw scalar; normalized internally first.

            Output:
                tuple[str, ...].
            """
            return tuple(token_pattern.findall(_normalize_text(value)))

        def _is_numeric_series(series):
            """Functionality: Return whether a Series is a non-boolean numeric column.

            Input:
                series: pandas.Series.

            Output:
                bool.
            """
            return (
                not pd.api.types.is_bool_dtype(series.dtype)
                and pd.api.types.is_numeric_dtype(series.dtype)
            )

        def _column_types(frame):
            """Functionality: Build a per-column numeric/categorical type map.

            Input:
                frame: DataFrame.

            Output:
                dict[str, str].
            """
            return {
                str(column): (
                    "numeric" if _is_numeric_series(frame[column]) else "categorical"
                )
                for column in frame.columns
            }

        def _is_numeric_type(value):
            """Functionality: Return whether a type annotation denotes a numeric column.

            Input:
                value: Type string.

            Output:
                bool.
            """
            return str(value).lower() in {"numeric", "columntype.numeric"}

        def _structured_string_statistics_selected(support):
            """Functionality: Decide whether to enable the string-statistics view from heuristics such as non-empty rate and token diversity.

            Input:
                support: Support DataFrame.

            Output:
                bool.
            """
            row_count = len(support)
            for column in support.columns:
                series = support[column]
                if _is_numeric_series(series) or pd.api.types.is_bool_dtype(series.dtype):
                    continue
                normalized = []
                for value in series:
                    missing = value is None
                    if not missing and not isinstance(value, (list, tuple, dict)):
                        try:
                            missing = bool(pd.isna(value))
                        except (TypeError, ValueError):
                            missing = False
                    if not missing:
                        normalized.append(_normalize_text(value))
                normalized = [
                    value
                    for value in normalized
                    if value not in {"__empty__", "__missing__"}
                ]
                nonempty = len(normalized)
                if nonempty < 128 or nonempty / max(1, row_count) < 0.5:
                    continue
                token_rows = [tuple(token_pattern.findall(value)) for value in normalized]
                exact_unique_rate = len(set(normalized)) / nonempty
                first_unique_rate = len(
                    {row[0] if row else "__none__" for row in token_rows}
                ) / nonempty
                last_unique_rate = len(
                    {row[-1] if row else "__none__" for row in token_rows}
                ) / nonempty
                token_count = sum(len(row) for row in token_rows)
                token_unique_rate = len(
                    {token for row in token_rows for token in row}
                ) / max(1, token_count)
                mean_characters = float(np.mean([len(value) for value in normalized]))
                mean_tokens = float(np.mean([len(row) for row in token_rows]))
                parseable_numeric_rate = float(
                    np.mean(
                        [
                            pd.notna(pd.to_numeric(value, errors="coerce"))
                            for value in normalized
                        ]
                    )
                )
                if (
                    0.25 <= exact_unique_rate <= 0.95
                    and mean_characters <= 32.0
                    and mean_tokens <= 4.0
                    and parseable_numeric_rate < 0.5
                    and min(
                        first_unique_rate,
                        last_unique_rate,
                        token_unique_rate,
                    )
                    <= 0.8 * exact_unique_rate
                ):
                    return True
            return False

        def _moment_skew(values):
            """Functionality: Compute third-moment skewness of finite values.

            Input:
                values: 1-D numeric array.

            Output:
                float. Returns 0 when there are too few valid points or variance is 0.
            """
            finite = values[np.isfinite(values)]
            if len(finite) < 2:
                return 0.0
            centered = finite - finite.mean()
            second = float(np.sum(centered * centered))
            if second <= 0.0:
                return 0.0
            third = float(np.sum(centered * centered * centered))
            return math.sqrt(len(finite)) * third / (second ** 1.5)

        def _sampled_extreme_rate(values):
            """Functionality: Estimate the extreme-value rate with a 3x IQR rule. Oversized samples are subsampled with route_sampling_seed.

            Input:
                values: 1-D numeric array.

            Output:
                float: fraction of extreme values.
            """
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                return 0.0
            if len(finite) > route_sample_capacity:
                indices = np.random.RandomState(route_sampling_seed).choice(
                    len(finite),
                    route_sample_capacity,
                    replace=False,
                )
                finite = finite[indices]
            q25, q75 = np.quantile(finite, (0.25, 0.75))
            iqr = q75 - q25
            return float(
                np.mean(
                    (finite < q25 - 3.0 * iqr)
                    | (finite > q75 + 3.0 * iqr)
                )
            )

        def _numeric_tail_interactions_selected(support):
            """Functionality: Enable the numeric-tail interaction view when at least two numeric columns show high skew or extremes.

            Input:
                support: Support DataFrame.

            Output:
                bool.
            """
            signals = []
            for column in support.columns:
                series = support[column]
                if not _is_numeric_series(series):
                    continue
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    continue
                signals.append(
                    (abs(_moment_skew(finite)), _sampled_extreme_rate(finite))
                )
            return len(signals) >= 2 and (
                max(skew for skew, _ in signals) >= 1.0
                or max(extreme for _, extreme in signals) >= 0.01
            )

        def _low_cardinality_numeric_states_selected(support):
            """Functionality: Enable the low-cardinality state view when a numeric column has state count in [1, low_cardinality_max_states].

            Input:
                support: Support DataFrame.

            Output:
                bool.
            """
            for column in support.columns:
                series = support[column]
                if not _is_numeric_series(series):
                    continue
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                unique = np.unique(values[np.isfinite(values)])
                if 1 <= len(unique) <= low_cardinality_max_states:
                    return True
            return False

        def _select_feature_views(support, support_y):
            """Functionality: After confirming binary labels, select enabled view names from heuristics.

            Input:
                support: Support features.
                support_y: Support labels.

            Output:
                tuple[str, ...]: active view names; may be empty.
            """
            _encode_binary_labels(support_y, len(support))
            selectors = (
                ("structured_string_statistics", _structured_string_statistics_selected),
                ("numeric_tail_interactions", _numeric_tail_interactions_selected),
                ("low_cardinality_numeric_states", _low_cardinality_numeric_states_selected),
            )
            return tuple(name for name, is_selected in selectors if is_selected(support))

        def _smoothed_map(fit_keys, labels, values, smoothing):
            """Functionality: Fit an additively smoothed target-mean map by key and apply it to query keys.

            Input:
                fit_keys: Fit-key sequence.
                labels: 0/1 labels aligned with fit_keys.
                values: Keys to score.
                smoothing: Smoothing strength.

            Output:
                np.ndarray[float32]: smoothed target mean per query key.
            """
            sums = defaultdict(float)
            counts = Counter()
            for key, label in zip(fit_keys, labels):
                sums[key] += float(label)
                counts[key] += 1
            prior = float(np.mean(labels))
            return np.asarray(
                [
                    (sums[key] + smoothing * prior)
                    / (counts[key] + smoothing)
                    for key in values
                ],
                dtype=np.float32,
            )

        def _cross_fitted_map(
            fit_keys,
            labels,
            predict_keys,
            *,
            folds,
            smoothing,
            random_state,
        ):
            """Functionality: Cross-fit target means for train keys with stratified folds, and map query keys with a full-data smoothed map.

            Input:
                fit_keys: Train keys.
                labels: 0/1 labels.
                predict_keys: Query keys.
                folds: Maximum fold count.
                smoothing: Smoothing strength.
                random_state: Seed for stratified fold splits.

            Output:
                tuple[np.ndarray, np.ndarray]: (train OOF values, query mapped values).
            """
            keys = np.asarray(fit_keys, dtype=object)
            fold_count = min(folds, int(np.bincount(labels).min()))
            oof = np.empty(len(labels), dtype=np.float32)
            if fold_count < 2:
                prior = float(np.mean(labels))
                sums = defaultdict(float)
                counts = Counter(keys)
                for key, label in zip(keys, labels):
                    sums[key] += float(label)
                oof[:] = [
                    (
                        sums[key]
                        - float(label)
                        + smoothing * prior
                    )
                    / (counts[key] - 1 + smoothing)
                    for key, label in zip(keys, labels)
                ]
                return oof, _smoothed_map(
                    keys,
                    labels,
                    predict_keys,
                    smoothing,
                )
            splitter = StratifiedKFold(
                n_splits=fold_count,
                shuffle=True,
                random_state=random_state,
            )
            for fit, holdout in splitter.split(np.zeros(len(labels)), labels):
                oof[holdout] = _smoothed_map(
                    keys[fit],
                    labels[fit],
                    keys[holdout],
                    smoothing,
                )
            return oof, _smoothed_map(
                keys,
                labels,
                predict_keys,
                smoothing,
            )

        def _token_aggregates(fit_rows, labels, rows, smoothing):
            """Functionality: Pool smoothed target means over token sets and emit pooled/max/min per sample.

            Input:
                fit_rows: Token sequences for fit rows.
                labels: 0/1 labels.
                rows: Token sequences to score.
                smoothing: Smoothing strength.

            Output:
                np.ndarray of shape (n, 3).
            """
            sums = defaultdict(float)
            counts = Counter()
            for tokens, label in zip(fit_rows, labels):
                for token in set(tokens):
                    sums[token] += float(label)
                    counts[token] += 1
            prior = float(np.mean(labels))
            result = np.empty((len(rows), 3), dtype=np.float32)
            for index, tokens in enumerate(rows):
                known = [token for token in set(tokens) if counts[token] > 0]
                if not known:
                    result[index] = prior
                    continue
                pooled = (
                    sum(sums[token] for token in known)
                    + smoothing * prior
                ) / (
                    sum(counts[token] for token in known)
                    + smoothing
                )
                individual = [
                    (sums[token] + smoothing * prior)
                    / (counts[token] + smoothing)
                    for token in known
                ]
                result[index] = (pooled, max(individual), min(individual))
            return result

        def _cross_fitted_token_aggregates(
            fit_rows,
            labels,
            predict_rows,
            *,
            folds,
            smoothing,
            random_state,
        ):
            """Functionality: Cross-fit token-aggregate features. Falls back to non-OOF stats when there are too few folds.

            Input:
                fit_rows: Train token rows.
                labels: 0/1 labels.
                predict_rows: Query token rows.
                folds: Maximum fold count.
                smoothing: Smoothing strength.
                random_state: Fold-split seed.

            Output:
                tuple[np.ndarray, np.ndarray]: train and query (n, 3) aggregate features.
            """
            rows = np.asarray(fit_rows, dtype=object)
            fold_count = min(folds, int(np.bincount(labels).min()))
            if fold_count < 2:
                return (
                    _token_aggregates(
                        fit_rows,
                        labels,
                        fit_rows,
                        smoothing,
                    ),
                    _token_aggregates(
                        fit_rows,
                        labels,
                        predict_rows,
                        smoothing,
                    ),
                )
            oof = np.empty((len(labels), 3), dtype=np.float32)
            splitter = StratifiedKFold(
                n_splits=fold_count,
                shuffle=True,
                random_state=random_state,
            )
            for fit, holdout in splitter.split(np.zeros(len(labels)), labels):
                oof[holdout] = _token_aggregates(
                    rows[fit].tolist(),
                    labels[fit],
                    rows[holdout].tolist(),
                    smoothing,
                )
            return oof, _token_aggregates(
                fit_rows,
                labels,
                predict_rows,
                smoothing,
            )

        def _feature_score(labels, values):
            """Functionality: Score a derived feature's binary discrimination with |AUC-0.5|.

            Input:
                labels: 0/1 labels.
                values: 1-D feature.

            Output:
                float. Returns 0 when there are fewer than 2 unique values.
            """
            if len(np.unique(values)) < 2:
                return 0.0
            return abs(float(roc_auc_score(labels, values)) - 0.5)

        def _structured_string_statistics_candidates(
            support,
            labels,
            query,
            column_types,
        ):
            """Functionality: Build cross-fitted target-encoding candidates from string or binned numeric columns and truncate to the feature budget.

            Input:
                support: Support table.
                labels: Binary 0/1 labels.
                query: Query table.
                column_types: Column-type map.

            Output:
                list[tuple]: (score, name, support_series, query_series), at most derived_feature_budget items.
            """
            if len(labels) < 4 or set(np.unique(labels)) != {0, 1}:
                return []
            candidates = []
            pairable = []
            for index, column in enumerate(map(str, support.columns)):
                if _is_numeric_type(column_types.get(column)):
                    fit_values = pd.to_numeric(
                        support[column],
                        errors="coerce",
                    )
                    predict_values = pd.to_numeric(
                        query[column],
                        errors="coerce",
                    )
                    finite = fit_values[np.isfinite(fit_values)]
                    if finite.nunique() < 3:
                        continue
                    median = float(finite.median())
                    edges = np.unique(
                        np.quantile(finite, np.linspace(0.1, 0.9, 9))
                    )
                    blocks = (
                        (
                            "bin",
                            np.digitize(
                                fit_values.fillna(median),
                                edges,
                            ).tolist(),
                            np.digitize(
                                predict_values.fillna(median),
                                edges,
                            ).tolist(),
                        ),
                    )
                else:
                    fit_normalized = [
                        _normalize_text(value)
                        for value in support[column]
                    ]
                    predict_normalized = [
                        _normalize_text(value)
                        for value in query[column]
                    ]
                    fit_tokens = [
                        _tokens(value)
                        for value in fit_normalized
                    ]
                    predict_tokens = [
                        _tokens(value)
                        for value in predict_normalized
                    ]
                    blocks = (
                        (
                            "exact",
                            fit_normalized,
                            predict_normalized,
                        ),
                        (
                            "first_token",
                            [
                                row[0] if row else "__none__"
                                for row in fit_tokens
                            ],
                            [
                                row[0] if row else "__none__"
                                for row in predict_tokens
                            ],
                        ),
                        (
                            "last_token",
                            [
                                row[-1] if row else "__none__"
                                for row in fit_tokens
                            ],
                            [
                                row[-1] if row else "__none__"
                                for row in predict_tokens
                            ],
                        ),
                    )
                    if 2 <= len(set(fit_normalized)) <= 256:
                        pairable.append(
                            (
                                index,
                                column,
                                fit_normalized,
                                predict_normalized,
                            )
                        )
                for block_index, (
                    operation,
                    fit_keys,
                    predict_keys,
                ) in enumerate(blocks):
                    fit, predict = _cross_fitted_map(
                        fit_keys,
                        labels,
                        predict_keys,
                        folds=target_folds,
                        smoothing=target_smoothing,
                        random_state=(
                            cross_fit_seed
                            + 101
                            + index * 17
                            + block_index
                        ),
                    )
                    name = (
                        "__structured_string_statistics_"
                        f"{index}_{operation}"
                    )
                    candidates.append(
                        (
                            _feature_score(labels, fit),
                            name,
                            pd.Series(fit),
                            pd.Series(predict),
                        )
                    )
                if not _is_numeric_type(column_types.get(column)):
                    fit_tokens = [
                        _tokens(value)
                        for value in support[column]
                    ]
                    predict_tokens = [
                        _tokens(value)
                        for value in query[column]
                    ]
                    fit, predict = _cross_fitted_token_aggregates(
                        fit_tokens,
                        labels,
                        predict_tokens,
                        folds=target_folds,
                        smoothing=target_smoothing,
                        random_state=(
                            cross_fit_seed
                            + 108
                            + index * 19
                        ),
                    )
                    operations = (
                        "token_mean",
                        "token_max",
                        "token_min",
                    )
                    for token_index, operation in enumerate(operations):
                        name = (
                            "__structured_string_statistics_"
                            f"{index}_{operation}"
                        )
                        candidates.append(
                            (
                                _feature_score(
                                    labels,
                                    fit[:, token_index],
                                ),
                                name,
                                pd.Series(fit[:, token_index]),
                                pd.Series(predict[:, token_index]),
                            )
                        )

            pairs = []
            for left, (_, _, left_fit, _) in enumerate(pairable):
                for right in range(left + 1, len(pairable)):
                    right_fit = pairable[right][2]
                    joint = [
                        f"{left_value}||{right_value}"
                        for left_value, right_value
                        in zip(left_fit, right_fit)
                    ]
                    repeated = (
                        1.0
                        - len(set(joint))
                        / max(1, len(joint))
                    )
                    if repeated >= 0.05:
                        pairs.append((repeated, left, right))
            pairs.sort(reverse=True)
            for pair_index, (_, left, right) in enumerate(pairs[:4]):
                _, _, left_fit, left_predict = pairable[left]
                _, _, right_fit, right_predict = pairable[right]
                fit_keys = [
                    f"{left_value}||{right_value}"
                    for left_value, right_value
                    in zip(left_fit, right_fit)
                ]
                predict_keys = [
                    f"{left_value}||{right_value}"
                    for left_value, right_value
                    in zip(left_predict, right_predict)
                ]
                fit, predict = _cross_fitted_map(
                    fit_keys,
                    labels,
                    predict_keys,
                    folds=target_folds,
                    smoothing=target_smoothing,
                    random_state=(
                        cross_fit_seed
                        + 502
                        + pair_index
                    ),
                )
                candidates.append(
                    (
                        _feature_score(labels, fit),
                        (
                            "__structured_string_statistics_z_pair_"
                            f"{pair_index}_exact"
                        ),
                        pd.Series(fit),
                        pd.Series(predict),
                    )
                )
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[:derived_feature_budget]

        def _numeric_tail_interaction_candidates(
            support,
            labels,
            query,
            column_types,
        ):
            """Functionality: Select high-scoring derived features from numeric transforms, missingness indicators, text stats, and pairwise difference/ratio/product.

            Input:
                support: Support table.
                labels: Binary labels.
                query: Query table.
                column_types: Column-type map.

            Output:
                list[tuple]: candidates truncated by score.
            """
            candidates = []
            numeric = []

            def _add_candidate(name, fit, predict):
                """Functionality: Sanitize non-finite values, score the feature, and append it to the candidate list.

                Input:
                    name: Derived column name.
                    fit: Train derived values.
                    predict: Query derived values.

                Output:
                    float: feature score for this candidate.
                """
                fit_array = np.nan_to_num(
                    fit,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                ).astype(np.float32)
                predict_array = np.nan_to_num(
                    predict,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                ).astype(np.float32)
                score = _feature_score(labels, fit_array)
                candidates.append(
                    (
                        score,
                        name,
                        pd.Series(fit_array),
                        pd.Series(predict_array),
                    )
                )
                return score

            for index, column in enumerate(map(str, support.columns)):
                declared_numeric = _is_numeric_type(
                    column_types.get(column)
                )
                fit_numeric = pd.to_numeric(
                    support[column],
                    errors="coerce",
                ).to_numpy(float)
                predict_numeric = pd.to_numeric(
                    query[column],
                    errors="coerce",
                ).to_numpy(float)
                if (
                    declared_numeric
                    or np.isfinite(fit_numeric).mean() >= 0.8
                ):
                    finite = fit_numeric[np.isfinite(fit_numeric)]
                    median = (
                        float(np.median(finite))
                        if finite.size
                        else 0.0
                    )
                    fit_filled = np.where(
                        np.isfinite(fit_numeric),
                        fit_numeric,
                        median,
                    )
                    predict_filled = np.where(
                        np.isfinite(predict_numeric),
                        predict_numeric,
                        median,
                    )
                    reference = (
                        np.sort(finite)
                        if finite.size
                        else np.asarray([0.0])
                    )
                    prefix = f"__numeric_tail_interactions_{index}"
                    scores = (
                        _add_candidate(
                            f"{prefix}_log",
                            np.sign(fit_filled)
                            * np.log1p(np.abs(fit_filled)),
                            np.sign(predict_filled)
                            * np.log1p(np.abs(predict_filled)),
                        ),
                        _add_candidate(
                            f"{prefix}_rank",
                            np.searchsorted(
                                reference,
                                fit_filled,
                                side="right",
                            )
                            / len(reference),
                            np.searchsorted(
                                reference,
                                predict_filled,
                                side="right",
                            )
                            / len(reference),
                        ),
                        _add_candidate(
                            f"{prefix}_sqrt",
                            np.sqrt(np.abs(fit_filled)),
                            np.sqrt(np.abs(predict_filled)),
                        ),
                        _add_candidate(
                            f"{prefix}_missing",
                            ~np.isfinite(fit_numeric),
                            ~np.isfinite(predict_numeric),
                        ),
                    )
                    numeric.append(
                        (
                            max(scores),
                            index,
                            fit_filled,
                            predict_filled,
                        )
                    )
                    continue
                fit_text = [
                    _normalize_text(value)
                    for value in support[column]
                ]
                predict_text = [
                    _normalize_text(value)
                    for value in query[column]
                ]
                counts = Counter(fit_text)
                fit_tokens = [
                    _tokens(value)
                    for value in fit_text
                ]
                predict_tokens = [
                    _tokens(value)
                    for value in predict_text
                ]
                token_counts = Counter()
                for row in fit_tokens:
                    token_counts.update(set(row))
                rows = max(1, len(fit_text))
                prefix = f"__numeric_tail_interactions_{index}"
                _add_candidate(
                    f"{prefix}_length",
                    np.log1p([len(value) for value in fit_text]),
                    np.log1p([len(value) for value in predict_text]),
                )
                _add_candidate(
                    f"{prefix}_tokens",
                    np.log1p([len(row) for row in fit_tokens]),
                    np.log1p([len(row) for row in predict_tokens]),
                )
                _add_candidate(
                    f"{prefix}_frequency",
                    [counts[value] / rows for value in fit_text],
                    [counts[value] / rows for value in predict_text],
                )
                _add_candidate(
                    f"{prefix}_token_frequency",
                    [
                        (
                            np.mean(
                                [
                                    token_counts[token]
                                    for token in set(row)
                                ]
                            )
                            / rows
                            if row
                            else 0.0
                        )
                        for row in fit_tokens
                    ],
                    [
                        (
                            np.mean(
                                [
                                    token_counts[token]
                                    for token in set(row)
                                ]
                            )
                            / rows
                            if row
                            else 0.0
                        )
                        for row in predict_tokens
                    ],
                )
                _add_candidate(
                    f"{prefix}_digits",
                    [
                        sum(char.isdigit() for char in value)
                        for value in fit_text
                    ],
                    [
                        sum(char.isdigit() for char in value)
                        for value in predict_text
                    ],
                )

            pair_numeric = sorted(
                numeric,
                key=lambda item: (-item[0], item[1]),
            )[: min(len(numeric), pair_pool_columns)]
            pair_numeric.sort(key=lambda item: item[1])
            for left_position, (
                _,
                left_index,
                left_fit,
                left_query,
            ) in enumerate(pair_numeric):
                for (
                    _,
                    right_index,
                    right_fit,
                    right_query,
                ) in pair_numeric[left_position + 1:]:
                    scale = max(
                        float(np.median(np.abs(right_fit))) * 1e-3,
                        1e-3,
                    )
                    prefix = (
                        "__numeric_tail_interactions_pair_"
                        f"{left_index}_{right_index}"
                    )
                    _add_candidate(
                        f"{prefix}_difference",
                        left_fit - right_fit,
                        left_query - right_query,
                    )
                    _add_candidate(
                        f"{prefix}_ratio",
                        left_fit / (np.abs(right_fit) + scale),
                        left_query / (np.abs(right_query) + scale),
                    )
                    _add_candidate(
                        f"{prefix}_product",
                        np.sign(left_fit * right_fit)
                        * np.log1p(np.abs(left_fit * right_fit)),
                        np.sign(left_query * right_query)
                        * np.log1p(np.abs(left_query * right_query)),
                    )
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[:derived_feature_budget]

        def _append_derived_features(support, query, candidates):
            """Functionality: Append candidate derived columns to support/query. Names must not collide with raw columns or each other.

            Input:
                support: Original support table.
                query: Original query table.
                candidates: List of (score, name, support_series, query_series).

            Output:
                tuple[DataFrame, DataFrame]: view pair with derived columns.
            """
            support_view = support.copy(deep=True)
            query_view = query.copy(deep=True)
            raw_names = set(map(str, support.columns))
            for _, name, support_values, query_values in candidates:
                if name in raw_names or name in support_view.columns:
                    raise FeatureViewEnsembleError(
                        f"derived feature collision: {name}"
                    )
                support_view[name] = support_values.to_numpy()
                query_view[name] = query_values.to_numpy()
            return support_view, query_view

        def _state_key(value):
            """Functionality: Normalize a numeric state to a hashable key. Non-finite values use a sentinel; 0 is unified to 0.0 hex.

            Input:
                value: Scalar numeric value.

            Output:
                str.
            """
            if not math.isfinite(float(value)):
                return nonfinite_state
            normalized = 0.0 if float(value) == 0.0 else float(value)
            return normalized.hex()

        def _state_keys(values):
            """Functionality: Build a state key for every element of a numeric column.

            Input:
                values: 1-D numeric sequence.

            Output:
                tuple[str, ...].
            """
            return tuple(_state_key(value) for value in values)

        def _fit_target_map(keys, labels):
            """Functionality: Fit an additively smoothed state-to-target-mean map.

            Input:
                keys: Train state keys.
                labels: 0/1 labels.

            Output:
                tuple[dict, float]: (mapping, global prior).
            """
            prior = float(np.mean(labels))
            sums = defaultdict(float)
            counts = Counter()
            for key, label in zip(keys, labels):
                sums[key] += float(label)
                counts[key] += 1
            mapping = {
                key: (
                    sums[key] + target_smoothing * prior
                ) / (
                    counts[key] + target_smoothing
                )
                for key in counts
            }
            return mapping, prior

        def _apply_target_map(keys, mapping, prior):
            """Functionality: Score keys with a mapping. Unseen keys fall back to the prior.

            Input:
                keys: State-key sequence.
                mapping: State to target mean.
                prior: Fallback prior.

            Output:
                np.ndarray[float64].
            """
            return np.asarray(
                [mapping.get(key, prior) for key in keys],
                dtype=np.float64,
            )

        def _cross_fitted_target(keys, labels):
            """Functionality: Build cross-fitted target means for train state keys. Uses leave-one-out smoothing when there are too few folds.

            Input:
                keys: Train state keys.
                labels: 0/1 labels.

            Output:
                np.ndarray[float64]: OOF target means, same length as keys.
            """
            minimum = int(
                np.bincount(labels, minlength=2).min()
            )
            fold_count = min(target_folds, minimum)
            if fold_count >= 2:
                output = np.empty(len(labels), dtype=np.float64)
                fold_ids = np.empty(len(labels), dtype=np.int64)
                random_state = np.random.RandomState(cross_fit_seed)
                for label in (0, 1):
                    indices = np.flatnonzero(labels == label)
                    shuffled = indices[
                        random_state.permutation(len(indices))
                    ]
                    fold_ids[shuffled] = (
                        np.arange(len(shuffled)) % fold_count
                    )
                keys_array = np.asarray(keys, dtype=object)
                for fold in range(fold_count):
                    fit = np.flatnonzero(fold_ids != fold)
                    holdout = np.flatnonzero(fold_ids == fold)
                    mapping, prior = _fit_target_map(
                        keys_array[fit],
                        labels[fit],
                    )
                    output[holdout] = _apply_target_map(
                        keys_array[holdout],
                        mapping,
                        prior,
                    )
                return output
            total_positive = float(labels.sum())
            counts = Counter(keys)
            sums = defaultdict(float)
            for key, label in zip(keys, labels):
                sums[key] += float(label)
            output = np.empty(len(labels), dtype=np.float64)
            for index, (key, label) in enumerate(zip(keys, labels)):
                remaining = len(labels) - 1
                prior = (
                    (total_positive - float(label)) / remaining
                    if remaining
                    else 0.5
                )
                output[index] = (
                    sums[key]
                    - float(label)
                    + target_smoothing * prior
                ) / (
                    counts[key] - 1 + target_smoothing
                )
            return output

        def _build_low_cardinality_numeric_states_view(
            support,
            labels,
            query,
        ):
            """Functionality: Select low-cardinality numeric columns and append train frequency plus cross-fitted target-mean columns.

            Input:
                support: Support table.
                labels: 0/1 labels.
                query: Query table.

            Output:
                tuple[DataFrame, DataFrame, dict]: view pair and column-selection audit.
            """
            eligible = []
            for index, column in enumerate(support.columns):
                series = support.iloc[:, index]
                if not _is_numeric_series(series):
                    continue
                values = series.to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )
                finite = values[np.isfinite(values)]
                unique = np.unique(finite)
                if not 2 <= len(unique) <= low_cardinality_max_states:
                    continue
                keys = _state_keys(values)
                coverage = len(finite) / max(1, len(values))
                finite_keys = tuple(
                    key
                    for key in keys
                    if key != nonfinite_state
                )
                counts = Counter(finite_keys)
                entropy = -sum(
                    (
                        count / max(1, len(finite_keys))
                    )
                    * math.log(
                        count / max(1, len(finite_keys))
                    )
                    for count in counts.values()
                )
                normalized_entropy = entropy / math.log(len(unique))
                repeated_rate = (
                    1.0
                    - len(unique) / max(1, len(finite))
                )
                eligible.append(
                    {
                        "index": index,
                        "column": column,
                        "finite_states": len(unique),
                        "score": (
                            coverage
                            * normalized_entropy
                            * repeated_rate
                        ),
                        "keys": keys,
                    }
                )
            eligible.sort(
                key=lambda item: (-item["score"], item["index"])
            )
            states = []
            for item in eligible[
                :low_cardinality_max_selected_columns
            ]:
                keys = item["keys"]
                counts = Counter(keys)
                frequency = {
                    key: count / max(1, len(support))
                    for key, count in counts.items()
                }
                target_mean, prior = _fit_target_map(keys, labels)
                states.append(
                    {
                        "index": item["index"],
                        "column": item["column"],
                        "finite_states": item["finite_states"],
                        "frequency": frequency,
                        "target_mean": target_mean,
                        "target_prior": prior,
                        "support_target_oof": _cross_fitted_target(
                            keys,
                            labels,
                        ),
                    }
                )
            support_view = support.copy(deep=True)
            query_view = query.copy(deep=True)
            raw_names = set(map(str, support.columns))
            for state in states:
                support_keys = _state_keys(
                    support.iloc[:, state["index"]].to_numpy(
                        dtype=np.float64,
                        na_value=np.nan,
                    )
                )
                query_keys = _state_keys(
                    query.iloc[:, state["index"]].to_numpy(
                        dtype=np.float64,
                        na_value=np.nan,
                    )
                )
                prefix = (
                    "__low_cardinality_numeric_states_"
                    f"{state['index']:04d}"
                )
                frequency_name = f"{prefix}_frequency"
                target_name = f"{prefix}_target_mean"
                collisions = raw_names.intersection(
                    {frequency_name, target_name}
                )
                if collisions:
                    raise FeatureViewEnsembleError(
                        "derived feature names collide with raw columns: "
                        f"{sorted(collisions)}"
                    )
                support_view[frequency_name] = np.asarray(
                    [
                        state["frequency"][key]
                        for key in support_keys
                    ],
                    dtype=np.float64,
                )
                support_view[target_name] = state[
                    "support_target_oof"
                ].copy()
                query_view[frequency_name] = np.asarray(
                    [
                        state["frequency"].get(key, 0.0)
                        for key in query_keys
                    ],
                    dtype=np.float64,
                )
                query_view[target_name] = _apply_target_map(
                    query_keys,
                    state["target_mean"],
                    state["target_prior"],
                )
            return support_view, query_view, {
                "eligible_column_count": len(eligible),
                "selected_column_count": len(states),
                "derived_feature_count": 2 * len(states),
            }

        def _normalize_bool_pair(support, query):
            """Functionality: Cast boolean columns to 0/1 float while keeping support/query aligned.

            Input:
                support: Support table.
                query: Query table.

            Output:
                tuple[DataFrame, DataFrame].
            """
            support, query = _validate_feature_pair(
                support,
                query,
            )
            for column in support.columns:
                if not (
                    pd.api.types.is_bool_dtype(
                        support[column].dtype
                    )
                    or pd.api.types.is_bool_dtype(
                        query[column].dtype
                    )
                ):
                    continue
                support[column] = support[column].to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )
                query[column] = query[column].to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )
            return support, query

        def _build_independent_feature_views(
            support,
            labels,
            query,
            active_feature_views,
        ):
            """Functionality: Build an independent support/query feature pair for each active view.

            Input:
                support: Original support.
                labels: Encoded binary labels.
                query: Original query.
                active_feature_views: Active view-name sequence.

            Output:
                tuple[dict, dict]: view name to (support, query), plus derived-column audit.
            """
            support, query = _validate_feature_pair(
                support,
                query,
            )
            normalized_support, normalized_query = (
                _normalize_bool_pair(support, query)
            )
            column_types = _column_types(normalized_support)
            views = {}
            audit = {
                "active_feature_views": list(active_feature_views),
                "derived_feature_count": {},
            }
            candidate_builders = (
                ("structured_string_statistics", _structured_string_statistics_candidates),
                ("numeric_tail_interactions", _numeric_tail_interaction_candidates),
            )
            for view_name, build_candidates in candidate_builders:
                if view_name not in active_feature_views:
                    continue
                candidates = build_candidates(
                    normalized_support,
                    labels,
                    normalized_query,
                    column_types,
                )
                if not candidates:
                    raise FeatureViewEnsembleError(f"{view_name} produced no features")
                views[view_name] = _append_derived_features(
                    normalized_support, normalized_query, candidates
                )
                audit["derived_feature_count"][view_name] = len(candidates)
            if (
                "low_cardinality_numeric_states"
                in active_feature_views
            ):
                (
                    low_card_support,
                    low_card_query,
                    low_card_audit,
                ) = _build_low_cardinality_numeric_states_view(
                    support,
                    labels,
                    query,
                )
                low_card_support, low_card_query = (
                    _normalize_bool_pair(
                        low_card_support,
                        low_card_query,
                    )
                )
                views["low_cardinality_numeric_states"] = (
                    low_card_support,
                    low_card_query,
                )
                audit["derived_feature_count"][
                    "low_cardinality_numeric_states"
                ] = low_card_audit["derived_feature_count"]
                audit["low_cardinality_numeric_states"] = (
                    low_card_audit
                )
            return views, audit

        def _build_merged_feature_view(
            support,
            query,
            independent_views,
            active_feature_views,
        ):
            """Functionality: Concatenate each independent view's derived suffix onto a shared raw prefix.

            Input:
                support: Original support.
                query: Original query.
                independent_views: Dict from view name to feature pair.
                active_feature_views: Active view names.

            Output:
                tuple[DataFrame, DataFrame, dict]: merged tables and column-count audit.
            """
            merged_support, merged_query = _normalize_bool_pair(
                support,
                query,
            )
            raw_feature_count = merged_support.shape[1]
            per_view_count = {}
            for view_name in active_feature_views:
                view_support, view_query = independent_views[view_name]
                if (
                    tuple(view_support.columns[:raw_feature_count])
                    != tuple(merged_support.columns[:raw_feature_count])
                    or tuple(view_query.columns[:raw_feature_count])
                    != tuple(merged_query.columns[:raw_feature_count])
                ):
                    raise FeatureViewEnsembleError(
                        f"{view_name} changed the raw feature prefix"
                    )
                suffix_support = view_support.iloc[
                    :,
                    raw_feature_count:,
                ]
                suffix_query = view_query.iloc[
                    :,
                    raw_feature_count:,
                ]
                if tuple(suffix_support.columns) != tuple(
                    suffix_query.columns
                ):
                    raise FeatureViewEnsembleError(
                        f"{view_name} support/query schemas differ"
                    )
                per_view_count[view_name] = suffix_support.shape[1]
                for column in suffix_support.columns:
                    if column in merged_support.columns:
                        raise FeatureViewEnsembleError(
                            f"derived feature collision: {column}"
                        )
                    merged_support[column] = (
                        suffix_support[column].to_numpy()
                    )
                    merged_query[column] = (
                        suffix_query[column].to_numpy()
                    )
            return merged_support, merged_query, {
                "active_feature_views": list(active_feature_views),
                "derived_feature_count": per_view_count,
                "merged_feature_count": int(
                    merged_support.shape[1]
                ),
            }

        def _preprocess_feature_view_pair(support, query):
            """Functionality: Label-encode, median-impute, and MinMax-scale view features for the core classifier.

            Input:
                support: View support table.
                query: View query table.

            Output:
                tuple[np.ndarray, np.ndarray, dict]: float32 matrix pair and preprocess audit.
            """
            support, query = _validate_feature_pair(
                support,
                query,
            )
            encoded = []
            dropped = []
            for column in list(support.columns):
                dtype = support[column].dtype
                if not (
                    pd.api.types.is_object_dtype(dtype)
                    or pd.api.types.is_string_dtype(dtype)
                ):
                    continue
                encoder = LabelEncoder()
                try:
                    support[column] = encoder.fit_transform(
                        support[column]
                    )
                    query[column] = encoder.transform(
                        query[column]
                    )
                except Exception:
                    support = support.drop(columns=[column])
                    query = query.drop(columns=[column])
                    dropped.append(str(column))
                    continue
                encoded.append(str(column))
            if support.shape[1] < 1:
                raise FeatureViewEnsembleError(
                    "feature-view preprocessing dropped every feature"
                )
            replaced_support = 0
            replaced_query = 0
            for column in support.columns:
                train_values = support[column].to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )
                query_values = query[column].to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )
                train_finite = np.isfinite(train_values)
                query_finite = np.isfinite(query_values)
                finite_support = train_values[train_finite]
                fill = (
                    float(np.median(finite_support))
                    if finite_support.size
                    else 0.0
                )
                replaced_support += int(
                    (~train_finite).sum()
                )
                replaced_query += int(
                    (~query_finite).sum()
                )
                support[column] = np.where(
                    train_finite,
                    train_values,
                    fill,
                )
                query[column] = np.where(
                    query_finite,
                    query_values,
                    fill,
                )
            scaler = MinMaxScaler()
            train_array = np.ascontiguousarray(
                scaler.fit_transform(support),
                dtype=np.float32,
            )
            query_array = np.ascontiguousarray(
                scaler.transform(query),
                dtype=np.float32,
            )
            if (
                not np.isfinite(train_array).all()
                or not np.isfinite(query_array).all()
            ):
                raise FeatureViewEnsembleError(
                    "feature-view preprocessing produced non-finite values"
                )
            return train_array, query_array, {
                "encoded_object_columns": encoded,
                "dropped_columns": dropped,
                "support_nonfinite_replaced": replaced_support,
                "query_nonfinite_replaced": replaced_query,
                "feature_count": int(train_array.shape[1]),
            }

        def _probability_matrix(
            probabilities,
            *,
            rows=None,
            dtype=np.float32,
        ):
            """Functionality: Validate a classification probability matrix and optionally row-normalize it.

            Input:
                probabilities: 2-D probabilities.
                rows: Optional expected row count.
                dtype: Output dtype, default float32.

            Output:
                np.ndarray: legal probability matrix.
            """
            values = np.asarray(probabilities, dtype=dtype)
            if (
                values.ndim != 2
                or values.shape[0] == 0
                or values.shape[1] < 2
            ):
                raise FeatureViewEnsembleError(
                    "probabilities must be a non-empty N-by-C matrix"
                )
            if rows is not None and values.shape[0] != rows:
                raise FeatureViewEnsembleError(
                    "probability row count differs"
                )
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
            ):
                raise FeatureViewEnsembleError(
                    "probabilities must be finite and non-negative"
                )
            if not np.allclose(
                values.sum(axis=1),
                1.0,
                rtol=1e-6,
                atol=1e-8,
            ):
                raise FeatureViewEnsembleError(
                    "probability rows must sum to one"
                )
            return np.ascontiguousarray(values)

        def _support_prior(labels, class_count):
            """Functionality: Compute the support class prior from integer labels. Every output class must occur at least once.

            Input:
                labels: Integer-encoded labels.
                class_count: Number of classes.

            Output:
                np.ndarray[float64]: prior of length class_count.
            """
            counts = np.bincount(
                labels.astype(np.int64),
                minlength=class_count,
            )
            if (
                len(counts) != class_count
                or (counts == 0).any()
            ):
                raise FeatureViewEnsembleError(
                    "every output class must occur in support labels"
                )
            return counts.astype(np.float64) / counts.sum()

        def _apply_support_prior_multiclass_adjustment(
            probabilities,
            labels,
        ):
            """Functionality: Apply a temperature-style support-prior adjustment to multiclass probabilities. Binary probabilities are returned unchanged.

            Input:
                probabilities: Query-by-class probabilities.
                labels: Integer support labels.

            Output:
                np.ndarray[float32]: adjusted, row-normalized probabilities.
            """
            matrix = _probability_matrix(
                probabilities,
                dtype=np.float64,
            )
            if matrix.shape[1] == 2:
                return np.asarray(matrix, dtype=np.float32)
            prior = _support_prior(labels, matrix.shape[1])
            temperatures = -np.sum(
                matrix
                * np.log(
                    prior.reshape(1, -1)
                    + multiclass_adjustment_epsilon
                ),
                axis=1,
            )
            if (
                not np.isfinite(temperatures).all()
                or (temperatures <= 0).any()
            ):
                raise FeatureViewEnsembleError(
                    "support-prior multiclass temperature is invalid"
                )
            shifted = matrix / temperatures.reshape(-1, 1)
            shifted -= shifted.max(axis=1, keepdims=True)
            exponentials = np.exp(
                np.clip(shifted, -745.0, 80.0)
            )
            adjustment = (
                exponentials
                / exponentials.sum(axis=1, keepdims=True)
            )
            adjusted = matrix * (
                adjustment
                / (
                    prior.reshape(1, -1)
                    + multiclass_adjustment_epsilon
                )
            )
            row_sums = adjusted.sum(axis=1, keepdims=True)
            if (
                not np.isfinite(adjusted).all()
                or (row_sums <= 0).any()
            ):
                raise FeatureViewEnsembleError(
                    "support-prior multiclass adjustment is invalid"
                )
            adjusted /= row_sums
            if not np.isfinite(adjusted).all():
                raise FeatureViewEnsembleError(
                    "support-prior multiclass adjustment is non-finite"
                )
            return np.ascontiguousarray(
                adjusted,
                dtype=np.float32,
            )

        def _blend_base_and_view_predictions(
            base_probabilities,
            feature_view_probabilities,
        ):
            """Functionality: Blend base probabilities with the mean of view probabilities using base_weight and view_weight, then row-normalize.

            Input:
                base_probabilities: Base probabilities.
                feature_view_probabilities: List of view probabilities; empty returns base only.

            Output:
                np.ndarray[float32].
            """
            base = _probability_matrix(base_probabilities)
            if not feature_view_probabilities:
                return base
            checked = [
                _probability_matrix(
                    probabilities,
                    rows=len(base),
                )
                for probabilities in feature_view_probabilities
            ]
            if any(
                values.shape != base.shape
                for values in checked
            ):
                raise FeatureViewEnsembleError(
                    "feature-view/base probability shapes differ"
                )
            view_mean = sum(checked) / len(checked)
            output = (
                np.float32(base_prediction_weight) * base
                + np.float32(feature_view_prediction_weight)
                * view_mean
            )
            output /= output.sum(axis=1, keepdims=True)
            return np.asarray(output, dtype=np.float32)

        def _is_fatal_feature_view_error(error):
            """Functionality: Return whether a failure on the legacy view path is fatal (including CUDA OOM and device asserts).

            Input:
                error: Exception.

            Output:
                bool.
            """
            if isinstance(
                error,
                (
                    KeyboardInterrupt,
                    SystemExit,
                    GeneratorExit,
                    MemoryError,
                    torch.cuda.OutOfMemoryError,
                ),
            ):
                return True
            if not isinstance(error, RuntimeError):
                return False
            message = str(error).casefold()
            fatal_markers = (
                "cuda error",
                "cuda out of memory",
                "cudnn",
                "cublas",
                "device-side assert",
                "illegal memory access",
                "misaligned address",
            )
            return any(marker in message for marker in fatal_markers)

        def _set_audit(task_type, **details):
            """Functionality: Write the legacy-view audit dict and force query_labels_used=False.

            Input:
                task_type: 'binary' or 'multiclass'.
                details: Remaining audit fields.

            Output:
                None. Writes self.feature_view_ensemble_audit.
            """
            audit = {
                "schema": "feature-view-ensemble-v1",
                "enabled": True,
                "feature_view_strategy": feature_view_strategy,
                "task_type": task_type,
            }
            audit.update(details)
            audit["query_labels_used"] = False
            self.feature_view_ensemble_audit = audit

        def _return_binary_fallback(base, reason, active_feature_views=(), **details):
            """Functionality: When a binary view fails or is disabled, record the fallback reason and return base probabilities.

            Input:
                base: Base probabilities.
                reason: Fallback reason.
                active_feature_views: Views that were active at the time.
                details: Extra audit fields.

            Output:
                np.ndarray: base probabilities.
            """
            _set_audit(
                "binary",
                active_feature_views=list(active_feature_views),
                feature_view_forward_count=0,
                base_prediction_weight=base_prediction_weight,
                feature_view_prediction_weight=feature_view_prediction_weight,
                fallback_exact_base=True,
                fallback_reason=reason,
                **details,
            )
            return base

        support_frame = _as_feature_frame(x_train)
        query_frame = _as_feature_frame(
            x_test,
            columns=support_frame.columns,
        )
        support_labels = np.asarray(y_train)
        if (
            support_labels.ndim != 1
            or len(support_labels) != len(support_frame)
        ):
            raise FeatureViewEnsembleError(
                "support labels must be one-dimensional and row aligned"
            )
        retained_support = ~pd.isna(support_labels)
        if not retained_support.all():
            support_frame = support_frame.loc[
                retained_support
            ].reset_index(drop=True)
            support_labels = support_labels[retained_support]
        if len(support_frame) == 0:
            raise FeatureViewEnsembleError(
                "no labeled support rows remain"
            )

        encoded_support_labels = LabelEncoder().fit_transform(
            support_labels
        )
        class_count = len(np.unique(encoded_support_labels))
        if class_count < 2:
            raise FeatureViewEnsembleError(
                "classification requires at least two support classes"
            )

        base_probabilities = _probability_matrix(
            self._predict_cls(
                support_frame if base_x_train is None else base_x_train,
                support_labels,
                query_frame if base_x_test is None else base_x_test,
                task_type,
                unique_dataset_name=unique_dataset_name,
            ),
            rows=len(query_frame),
            dtype=np.float64,
        )

        if class_count >= 3:
            adjusted = _apply_support_prior_multiclass_adjustment(
                base_probabilities,
                encoded_support_labels,
            )
            _set_audit(
                "multiclass",
                method="support_prior_multiclass_adjustment",
                active_feature_views=[],
                feature_view_forward_count=0,
            )
            return adjusted

        base = _probability_matrix(base_probabilities)
        if feature_view_prediction_weight == 0.0:
            return _return_binary_fallback(base, "zero_feature_view_weight")

        def _predict_feature_view(view_name, view_support, view_query, shape_name=None):
            """Functionality: Preprocess one view feature pair, call the core classifier, and require the output shape to match base.

            Input:
                view_name: View name used as a unique_dataset_name suffix.
                view_support: View support table.
                view_query: View query table.
                shape_name: Name used in shape-mismatch errors.

            Output:
                tuple[np.ndarray, dict]: (probabilities, preprocess audit).
            """
            train_array, query_array, audit = _preprocess_feature_view_pair(
                view_support, view_query
            )
            probabilities = _probability_matrix(
                self._predict_cls(
                    train_array,
                    support_labels,
                    query_array,
                    task_type,
                    unique_dataset_name=(
                        f"{unique_dataset_name or 'anonymous'}"
                        f"__feature_view__{view_name}"
                    ),
                ),
                rows=len(query_frame),
            )
            if probabilities.shape != base.shape:
                raise FeatureViewEnsembleError(
                    f"{shape_name or view_name} class shape differs from Base"
                )
            return probabilities, audit

        try:
            active_feature_views = _select_feature_views(
                support_frame,
                encoded_support_labels,
            )
            if not active_feature_views:
                return _return_binary_fallback(base, "no_feature_view_selected")

            independent_views, view_audit = (
                _build_independent_feature_views(
                    support_frame,
                    encoded_support_labels,
                    query_frame,
                    active_feature_views,
                )
            )
            preprocess_audit = {}
            view_probabilities = []
            merge_audit = None

            if feature_view_strategy == "independent_view_prediction_mean":
                for view_name in active_feature_views:
                    probabilities, audit = _predict_feature_view(
                        view_name, *independent_views[view_name]
                    )
                    view_probabilities.append(probabilities)
                    preprocess_audit[view_name] = audit
            else:
                (
                    merged_support,
                    merged_query,
                    merge_audit,
                ) = _build_merged_feature_view(
                    support_frame,
                    query_frame,
                    independent_views,
                    active_feature_views,
                )
                derived_feature_count = (
                    merged_support.shape[1]
                    - support_frame.shape[1]
                )
                if derived_feature_count == 0:
                    return _return_binary_fallback(
                        base,
                        "no_derived_feature_in_merged_view",
                        active_feature_views,
                        view=view_audit,
                        merged_feature_view=merge_audit,
                    )
                probabilities, audit = _predict_feature_view(
                    "merged",
                    merged_support,
                    merged_query,
                    "merged feature view",
                )
                view_probabilities.append(probabilities)
                preprocess_audit["merged_feature_view"] = audit

            blended = _blend_base_and_view_predictions(base, view_probabilities)
            _set_audit(
                "binary",
                active_feature_views=list(active_feature_views),
                feature_view_prediction_count=len(view_probabilities),
                feature_view_forward_count=len(view_probabilities),
                base_prediction_weight=base_prediction_weight,
                feature_view_prediction_weight=feature_view_prediction_weight,
                formula=(
                    "normalize(base_weight * P_base + "
                    "view_weight * P_feature_view)"
                ),
                fallback_exact_base=False,
                fallback_reason=None,
                view=view_audit,
                preprocess=preprocess_audit,
                merged_feature_view=merge_audit,
            )
            return blended
        except Exception as error:
            if _is_fatal_feature_view_error(error):
                raise
            return _return_binary_fallback(
                base,
                "feature_view_failure",
                feature_view_error_type=type(error).__name__,
                feature_view_error=str(error),
            )


def predict_with_feature_views(
    *,
    predict_cls,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task_type: str,
    base_x_train=None,
    base_x_test=None,
    unique_dataset_name: str = None,
    feature_view_strategy: Literal[
        "independent_view_prediction_mean",
        "merged_feature_view",
    ] = "independent_view_prediction_mean",
    feature_view_prediction_weight: float = 0.5,
    route_sampling_seed: int = 20260806,
    cross_fit_seed: int = 0,
    fixed_feature_views: dict | None = None,
    feature_view_preprocessor=None,
) -> tuple[np.ndarray, dict]:
    """Functionality: Feature-view ensemble entry point. Multiclass or missing fixed-view config uses the legacy path; otherwise the deterministic fixed views are used.

    Input:
        predict_cls: Core classifier forward.
        x_train: Support features.
        y_train: Support labels.
        x_test: Query features.
        task_type: Task type.
        base_x_train: Optional base train features.
        base_x_test: Optional base test features.
        unique_dataset_name: Optional dataset name.
        feature_view_strategy: View strategy.
        feature_view_prediction_weight: View weight.
        route_sampling_seed: Legacy-path routing seed.
        cross_fit_seed: Legacy-path cross-fit seed.
        fixed_feature_views: Fixed-view config.
        feature_view_preprocessor: Optional already constructed fixed-view preprocessor.

    Output:
        tuple[np.ndarray, dict]: (query-by-class probabilities, audit dict).
    """
    labels = np.asarray(y_train)
    labeled = labels[~pd.isna(labels)] if labels.ndim == 1 else labels
    use_legacy_multiclass = (
        labels.ndim == 1 and len(pd.unique(labeled)) >= 3
    )
    if (
        not use_legacy_multiclass
        and (fixed_feature_views is not None or feature_view_preprocessor is not None)
    ):
        return _predict_with_fixed_feature_views(
            predict_cls=predict_cls,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            task_type=task_type,
            base_x_train=base_x_train,
            base_x_test=base_x_test,
            unique_dataset_name=unique_dataset_name,
            feature_view_strategy=feature_view_strategy,
            feature_view_prediction_weight=feature_view_prediction_weight,
            fixed_feature_views=fixed_feature_views,
            feature_view_preprocessor=feature_view_preprocessor,
        )
    runner = _FeatureViewEnsembleRunner(predict_cls)
    prediction = runner.predict(
        x_train,
        y_train,
        x_test,
        task_type,
        base_x_train=base_x_train,
        base_x_test=base_x_test,
        unique_dataset_name=unique_dataset_name,
        feature_view_strategy=feature_view_strategy,
        feature_view_prediction_weight=feature_view_prediction_weight,
        route_sampling_seed=route_sampling_seed,
        cross_fit_seed=cross_fit_seed,
    )
    return prediction, runner.feature_view_ensemble_audit
