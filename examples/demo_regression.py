from functools import partial

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from huggingface_hub import hf_hub_download
import torch

try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    mean_squared_error = partial(mean_squared_error, squared=False)
import os, sys

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from inference.predictor import LimiXPredictor

house_data = fetch_california_housing()
X, y = house_data.data, house_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y_mean = y_train.mean()
y_std = y_train.std()
y_train_normalized = (y_train - y_mean) / y_std
y_test_normalized = (y_test - y_mean) / y_std

model_path = hf_hub_download(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", local_dir="./cache")
model = LimiXPredictor(device=torch.device('cuda'), model_path=model_path, inference_config='config/reg_default_16M_retrieval.json')  # config/reg_default_noretrieval.json
y_pred = model.predict(X_train, y_train_normalized, X_test, task_type="Regression")    

# Compute RMSE and R²
y_pred = y_pred.to('cpu').numpy()
rmse = mean_squared_error(y_test_normalized, y_pred)
r2 = r2_score(y_test_normalized, y_pred)

print(f'RMSE: {rmse}')
print(f'R2: {r2}')