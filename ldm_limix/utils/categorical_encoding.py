import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_ENCODER_MODES = (
    "current",
    "train_test",
    "train_unknown_nan",
)


def _is_label_encode_column(series: pd.Series) -> bool:
    """Functionality: Decide whether a column should be LabelEncoded before MinMaxScaler.

    Input:
        series: One train-side feature column.

    Output:
        bool. True for object, pandas StringDtype (pandas 3 CSV `str`), and category.
    """
    dtype = series.dtype
    if pd.api.types.is_object_dtype(dtype):
        return True
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_numeric_dtype(
        dtype
    )


def encode_categorical_features(
    X_train,
    X_test,
    feature_encoder_mode="current",
):
    """Encode object/string/category columns according to the selected feature encoding mode.

    Modes:
        current:
            Fit on train and transform train/test. If test contains an unseen
            category (or any encoding error occurs), drop that entire column.
        train_test:
            Fit on the concatenation of train and test, then transform both.
        train_unknown_nan:
            Fit only on train. Values in test that are absent from train are
            encoded as NaN instead of causing the column to be dropped.
    """
    if feature_encoder_mode not in FEATURE_ENCODER_MODES:
        raise ValueError(
            f"Unknown feature_encoder_mode={feature_encoder_mode!r}; "
            f"expected one of {FEATURE_ENCODER_MODES}"
        )

    for col in list(X_train.columns):
        if not _is_label_encode_column(X_train[col]):
            continue

        try:
            feature_encoder = LabelEncoder()

            if feature_encoder_mode == "train_test":
                combined = pd.concat(
                    [X_train[col], X_test[col]],
                    axis=0,
                    ignore_index=True,
                )
                feature_encoder.fit(combined)
                X_train[col] = feature_encoder.transform(X_train[col])
                X_test[col] = feature_encoder.transform(X_test[col])
            elif feature_encoder_mode == "train_unknown_nan":
                X_train[col] = feature_encoder.fit_transform(X_train[col])
                category_to_code = {
                    category: code
                    for code, category in enumerate(feature_encoder.classes_)
                }
                X_test[col] = X_test[col].map(category_to_code).astype(float)
            else:
                X_train[col] = feature_encoder.fit_transform(X_train[col])
                X_test[col] = feature_encoder.transform(X_test[col])
        except Exception:
            # Preserve the original classifier behavior for columns that cannot
            # be encoded for reasons other than an unseen test category.
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    return X_train, X_test
