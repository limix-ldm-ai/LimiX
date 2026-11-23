from functools import partial

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from huggingface_hub import hf_hub_download
import torch

try:
    from sklearn.metrics import root_mean_squared_error as mean_squared_error
except:
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
from ldm_limix import LimiXPredictor


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_path = hf_hub_download(repo_id="stable-ai/LimiX-2-1016", filename="LimiX-2-1016.ckpt", local_dir="./cache")

model = LimiXPredictor(
    device=torch.device('cuda'),
    model_path=model_path,
    inference_config=os.path.join(ROOT_DIR, "ldm_limix", "config", "reg_default_noretrieval_v2.json"),
)
y_pred = model.predict(X_train, y_train, X_test, task_type="Regression")

rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R2: {r2}')
