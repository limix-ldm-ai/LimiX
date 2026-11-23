import torch
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import hf_hub_download
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from ldm_limix import LimiXPredictor

def mask_prediction_eval(x_pred_:np.ndarray, x_true_:np.ndarray, mask:np.ndarray, categories: dict):
    # There are categorical variables that are not treated as categorical during inference      
    x_pred = x_pred_.copy()
    x_true = x_true_.copy()
    mask_pred_cls_error = 0
    mask_pred_reg_error = 0

    categorical_idx = list(categories.keys())
    for idx in categorical_idx:
        distances = np.abs(x_pred[:,idx][:,np.newaxis]-categories[idx])
        nearest_indices = np.argmin(distances, axis=1)  
        x_pred[:,idx] = categories[idx][nearest_indices]

    cls_error_list = []
    reg_error_list = []
    for idx in range(x_true.shape[1]):
        mask_col = mask[:,idx]
        if idx in categorical_idx:
            cls_error_list.append(x_pred[:,idx][mask_col] != x_true[:,idx][mask_col])
        else:
            reg_error_list.append(x_pred[:,idx][mask_col] - x_true[:,idx][mask_col])

    if cls_error_list != []:
        mask_pred_cls_error = np.concatenate(cls_error_list).mean()
    if reg_error_list != []:
        mask_pred_reg_error = np.sqrt(np.mean(np.concatenate(reg_error_list)**2))

    return mask_pred_cls_error, mask_pred_reg_error  


def gen_nan(x:np.ndarray, drop_percentage):
    if drop_percentage <= 0:
        return x, x, None
    rng = np.random.default_rng(42)
    existing_nan_mask = np.isnan(x)
    valid_positions = np.where(~existing_nan_mask)
    n_valid = len(valid_positions[0])
    assert n_valid > 0
    n_new_missing = int(n_valid * drop_percentage)
    
    indices_to_missing = rng.choice(
        len(valid_positions[0]), 
        size=n_new_missing,       
        replace=False             
    )

    rows_to_missing = valid_positions[0][indices_to_missing]
    cols_to_missing = valid_positions[1][indices_to_missing] 
    nan_mask = np.zeros_like(x, dtype=np.bool_)
    nan_mask[rows_to_missing, cols_to_missing] = True
    x_original = x.copy()
    x[nan_mask] = np.nan

    return x, x_original, nan_mask


def get_categorical_features_indices(x_:np.ndarray):
    min_unique_num_for_numerical_infer = 10
    categories = {}
    x = x_.values if isinstance(x_, pd.DataFrame) else x_ 
    for idx, col in enumerate(x.T):
        if len(np.unique(col)) < min_unique_num_for_numerical_infer:
            categories[idx] = np.unique(col)
    return categories


scaler = MinMaxScaler()
le = LabelEncoder()

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        try:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
        except Exception as e:
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test) 

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)

categories = get_categorical_features_indices(X_train)

data_device = f'cuda:0'

model_path = hf_hub_download(repo_id="stable-ai/LimiX-2-1016", filename="LimiX-2-1016.ckpt", local_dir="./cache")

testX, testX_original, nan_mask = gen_nan(X_test, 0.3)

model = LimiXPredictor(
    device=torch.device(data_device),
    model_path=model_path,
    inference_config=os.path.join(ROOT_DIR, "ldm_limix", "config", "reg_default_noretrieval_MVI_v2.json"),
)
reconstructed_X = model.predict(X_train, y_train, testX, task_type="Feature_imputation")
mask_prediction_ = reconstructed_X[-X_test.shape[0]:].astype(X_test.dtype)

mask_pred_cls_error, mask_pred_reg_error = mask_prediction_eval(mask_prediction_, testX_original, nan_mask, categories)

print(f'Feature Reconstruction done!')
print(f'Categorical feature predict error: {mask_pred_cls_error}')
print(f'Numerical feature predict error: {mask_pred_reg_error}')
