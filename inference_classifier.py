import json
import os
import sys
import time

from inference.predictor import LimiXPredictor
from inference.inference_method import setup
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import torch
import argparse
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import roc_auc_score
from pathlib import Path
import torch.distributed as dist
from utils.inference_utils import  generate_infenerce_config, sample_inferece_params

os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from utils.utils import  download_datset, download_model

if not torch.cuda.is_available():
    print('GPU device not found. For fast training, please enable GPU.', file=sys.stderr)

def auc_metric(target, pred, multi_class='ovo', numpy=False):
    lib = np if numpy else torch
    try:
        if not numpy:
            target = torch.tensor(target) if not torch.is_tensor(target) else target
            pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
        if len(lib.unique(target)) > 2:
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred, multi_class=multi_class))
            return roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred))
            return roc_auc_score(target, pred)
    except ValueError as e:
        print(e)
        return np.nan if numpy else torch.tensor(np.nan)

# --- ECE (Expected Calibration Error) ---
def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) implementation"""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        predictions = (confidences >= 0.5).astype(int)

    accuracies = (predictions == y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return ece

def inference_dataset(classifier, le, scaler, X_train, y_train, X_test, y_test):
    for col in X_train.columns:
        if X_train[col].dtype == 'object':  # Check whether it is a string column.
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
    num_classes = len(le.classes_)

    trainX, trainy = X_train, y_train
    
    trainX = np.asarray(trainX, dtype=np.float32)
    trainy = np.asarray(trainy, dtype=np.int64)
    
    # Datasets with too many or too few categories are not supported yet
    if len(np.unique(trainy)) > 10 or len(np.unique(trainy)) < 2:
        return None, None, None
    # When seq_len is greater than 50,000, skip due to GPU memory limitations
    if len(trainX) >= 50000:
        return None, None, None
    
    testX, testy = X_test, y_test
    testX = np.asarray(testX, dtype=np.float32)
    testy = np.asarray(testy, dtype=np.int64)

    prediction_ = classifier.predict(trainX, trainy, testX, task_type="Classification")
    prediction_label = np.argmax(prediction_, axis=1)
    
    roc = auc_metric(testy, prediction_)
    acc = accuracy_score(testy, prediction_label)
    f1 = f1_score(testy, prediction_label, average='macro' if num_classes > 2 else 'binary')
    ce = log_loss(testy, prediction_)
    ece = compute_ece(testy, prediction_, n_bins=10)

    rst = {
            'num_data_train': len(trainX),
            'num_data_test': len(testX),
            'num_feat': len(trainX[0]), 
            'num_class': len(np.unique(trainy)),
            'acc': float(acc),
            'f1': float(f1),
            'logloss': float(ce),
            'ece': float(ece),
            'auc': float(roc),
        }
    return rst,prediction_,testy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LimiX inference')
    parser.add_argument('--data_dir', type=str, default=None, help='Specify the local storage directory of the dataset')
    parser.add_argument('--save_name', default=None, type=str, help="path to save result")
    parser.add_argument('--inference_config_path', type=str, default="./config/cls_default_retrieval.json", help="path to example config")
    parser.add_argument('--model_path',type=str, default=None, help="path to you model")
    parser.add_argument('--inference_with_DDP', default=False, action='store_true', help="Inference with DDP")
    parser.add_argument('--debug', default=False, action='store_true', help="debug mode")
    parser.add_argument('--search_space_sample_num', type=int, default=0, help="number of samples to search in the search space")
    args = parser.parse_args()

    model_file = args.model_path
    data_root = args.data_dir
    search_space_sample_num = args.search_space_sample_num
    
    if data_root is None:
        download_datset(repo_id="stableai-org/bcco_cls", revision="main", save_dir="./cache")
        data_root = "./cache/bcco_cls"
    if model_file is None:
        model_file = download_model(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", save_path="./cache")
	
    if args.save_name is None:
        # Dynamically generate the save path
        args.save_name = time.strftime("%Y%m%d-%H%M%S")

    save_root = f"./result/{args.save_name}"
    os.makedirs(save_root, exist_ok=True)

    if not os.path.exists(args.inference_config_path):
        generate_infenerce_config(args)

    with open(args.inference_config_path, 'r') as f:
        inference_config = json.load(f)

    save_result_path = os.path.join(save_root, f"all_rst.csv")
    save_config_path = os.path.join(save_root, "config.json")
    with open(save_config_path, "w") as f:
        json.dump(inference_config, f)

    setup()
    scaler = MinMaxScaler()
    le = LabelEncoder()

    rng = np.random.default_rng(42)

    classifier = LimiXPredictor(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path=model_file,inference_config=inference_config,
                               inference_with_DDP=args.inference_with_DDP)
    rsts = []
    aucs = {}
    # Iterate through all datasets and perform inference
    for idx, folder in tqdm(enumerate(os.listdir(data_root))):
        X_train, X_test, y_train, y_test = None, None, None, None
        folder_path = os.path.join(data_root, folder)
        
        if os.path.isfile(folder_path):
            continue
        try:
            # start_time_pre = time.time()
            train_path = os.path.join(folder_path, folder+'_train.csv')
            test_path = os.path.join(folder_path, folder+'_test.csv')
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)
                
                if os.path.exists(test_path):
                    test_df = pd.read_csv(test_path)
                else:
                    # If there is no test.csv, split train.csv into training and testing sets
                    train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=42)
                    
            dataset_name = folder  # Use the folder name as the dataset name.

            # The last column is the target variable
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            sample_index = 0
            aucs['dataset'] = folder
            aucs['default_auc'] = 0
            aucs['sample_auc'] = []
            while sample_index == 0 or sample_index < search_space_sample_num:
                if search_space_sample_num > 0:
                    if sample_index > 0:
                        hyperopt_config, base_config = sample_inferece_params(rng, 2, 2)
                        classifier.set_inference_config(inference_config=hyperopt_config, **base_config)
                        print(f"{sample_index}/{search_space_sample_num}", end="\r")
                    else:
                        classifier.set_inference_config(inference_config, 0.9, 0)

                try:
                    rst, prediction_,testy = inference_dataset(classifier, le, scaler, X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
                    assert rst is not None, f'Error processing {folder} with sample_index {sample_index}: rst is None. seq_len({len(X_train)}) is greater than 50,000, skip due to GPU memory limitations'
                except Exception as e:
                    print(f"Error processing {folder} with sample_index {sample_index}: {e}")
                    sample_index += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                class_num = prediction_.shape[1]
                rst['dataset name'] = folder
                rst['search_space_sample_index'] = sample_index
                if sample_index == 0:
                    aucs['default_auc'] = rst['auc']
                aucs['sample_auc'].append(rst['auc'])

                sample_index += 1

                if not(int(os.environ.get('WORLD_SIZE', -1)) > 0 and dist.get_rank() != 0):
                    output_df = {'label':testy}
                    for i in range(class_num):
                        output_df[f'pred_{i}'] = prediction_[:,i]
                    pd.DataFrame(output_df).to_csv(os.path.join(save_root, rst['dataset name']+'_pred_LimiX.csv'), index=False)
                    del prediction_

                    rsts.append(rst)
                    if args.debug and search_space_sample_num <= 0:
                        print(f"[{idx}] {folder} -> {rst['auc']}")
            
            if args.debug and search_space_sample_num > 0 and len(aucs['sample_auc']) > 0:
                aucs_list = np.array(aucs['sample_auc'], dtype=float)
                print(f"[{idx}] {folder} -> default_auc: {aucs['default_auc']:.6f}, sample_auc: max: {np.max(aucs_list):.6f}, mean: {np.mean(aucs_list):.6f},  min: {np.min(aucs_list):.6f}")

        except Exception as e:
            print(f"Error processing: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not(int(os.environ.get('WORLD_SIZE', -1))>0 and dist.get_rank() != 0):
        rstsdf = pd.DataFrame(rsts)
        rstsdf.to_csv(os.path.join(save_root, 'all_rst.csv'), index=False)

