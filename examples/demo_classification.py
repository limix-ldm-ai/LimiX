from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
import numpy as np
import os, sys
import torch

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from ldm_limix import LimiXPredictor

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model_file = hf_hub_download(repo_id="stable-ai/LimiX-2-1016", filename="LimiX-2-1016.ckpt", local_dir="./cache")

clf = LimiXPredictor(
    device=torch.device('cuda'),
    model_path=model_file,
    inference_config=os.path.join(ROOT_DIR, "ldm_limix", "config", "cls_default_noretrieval_v2.json"),
)
prediction = clf.predict(X_train, y_train, X_test, task_type="Classification")

auc = roc_auc_score(y_test, prediction[:, 1])
acc = accuracy_score(y_test, np.argmax(prediction, axis=1))
print(f"auc: {auc}")
print(f"acc: {acc}")
