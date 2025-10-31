from inference.inference_method import InferenceAttentionMap, InferenceResultWithRetrieval
from inference.preprocess import (
    FeatureShuffler, 
    FilterValidFeatures, 
    CategoricalFeatureEncoder, 
    RebalanceFeatureDistribution, 
    FingerprintFeatureEncoder,
    PolynomialInteractionGenerator,
    SubSampleData)
from utils.loading import load_model
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


NA_PLACEHOLDER = "__MISSING__"

class LimiXPredictor:
    """"LimiX model inferencer, supporting tasks such as classification, regression, and missing value prediction."""
    def __init__(self, 
                 device:torch.device, 
                 model_path:str, 
                 inference_config: list|str,
                 mix_precision:bool=True,
                 outlier_remove_std: float=12,
                 softmax_temperature:float=0.9,
                #  task_type: Literal['Classification', 'Regression']='Classification',
                 mask_prediction:bool=False,
                 categorical_features_indices:List[int]|None=None,
                 inference_with_DDP: bool = False,
                 seed:int=0):
        """
        init LimiXPredictor
        
        Args:
            device: The device for performing inference; GPU is recommended
            model_path: The model path of LimiX
            mix_precision: Whether to use mixed precision inference
            outlier_remove_std: Standard deviation used for removing outliers
            softmax_temperature: Softmax temperature coefficient
            task_type:  task type
            mask_prediction: Whether to enable missing value prediction
            categorical_features_indices: Index numbers of categorical features, currently not in use
            inference_config: inference_config_setting,
            inference_with_DDP: If using DDP to inference,
            seed: Random seed
        """
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
        # self.task_type = task_type
        self.mask_prediction = mask_prediction        
        self.inference_with_DDP=inference_with_DDP
        self.model=load_model(model_path=model_path,mask_prediction=mask_prediction)

        self.preprocess_pipelines = []
        self.preprocess_configs = []

        self.build_preprocess_pipeline()

    def set_inference_config(self, inference_config: list|str, softmax_temperature:float|None=None, seed:int|None=None):
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
        self.preprocess_pipelines = []
        self.preprocess_configs = []
    
        random.seed(self.seed)
        rand_gen = np.random.default_rng(self.seed)
        self.seeds = [random.randint(0, 10000) for _ in range(self.n_estimators*self.preprocess_num)]
        start_idx = rand_gen.integers(0, 1000)
        all_shifts = list(range(start_idx, start_idx + self.n_estimators))
        self.all_shifts = rand_gen.choice(all_shifts, size=self.n_estimators, replace=False)
    
        if self.mask_prediction:
            for inference_config_item in self.inference_config:
                if len(inference_config_item['RebalanceFeatureDistribution']['worker_tags']) > 0:
                    for i, v in enumerate(inference_config_item['RebalanceFeatureDistribution']['worker_tags']):
                        if v == 'power':
                            print("WARNING: Missing value imputation does not currently support the preprocessing method of power! Using the default worker_tags method")
                            inference_config_item['RebalanceFeatureDistribution']['worker_tags'].pop(i)
                            inference_config_item['RebalanceFeatureDistribution']['worker_tags'].append(None)
                inference_config_item['RebalanceFeatureDistribution']['discrete_flag'] = True

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
        """Check whether the number of features matches the previous evaluation"""
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
        """
        {'accept_sparse': False, 'dtype': None, 'ensure_all_finite': 'allow-nan'}
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
        ordinal_encoder = OrdinalEncoder(categories="auto",
                                        dtype=dtype,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                        encoded_missing_value=np.nan)
        col_encoder = ColumnTransformer(transformers=[("encoder", ordinal_encoder, make_column_selector(dtype_include=["category", "string"]))],
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
        if x.shape[0] < self.min_seq_len_for_category_infer:
            return []
        categorical_idx = []
        for idx, col in enumerate(x.T):
            if len(np.unique(col)) < self.min_unique_num_for_numerical_infer:
                categorical_idx.append(idx)
        return categorical_idx
        
    
    def predict(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, task_type:Literal['Classification', 'Regression']='Classification') -> np.ndarray:
        """
        Perform inference using the LimiX model
        
        Args:
        x_train: Training data x
        y_train: Training data y
        x_test:  Testing data x
        """
        if task_type == "Classification":
            return self._predict_cls(x_train, y_train, x_test)
        elif task_type == "Regression":
            return self._predict_reg(x_train, y_train, x_test)
        else:
            raise ValueError(f"Unsupported task type, supported tasks include classification and regression!")
        
    def _predict_cls(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray) -> np.ndarray:
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
        categorical_idx = self.get_categorical_features_indices(x)
        outputs = []
        mask_predictions = []
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            x_ = x.copy()
            y_ = self.class_permutations[id_pipe][y.copy()]
            categorical_idx_ = categorical_idx.copy()
            for id_step, step in enumerate(pipe):
                if isinstance(step, InferenceAttentionMap):
                    feature_attention_score, sample_attention_score = step.inference(X_train=x_[:len(y_train)],
                                                                                     y_train=y_train,
                                                                                     X_test=x_[len(y_train):],
                                                                                     task_type="cls")
                   
                elif isinstance(step, SubSampleData):
                    step.fit(torch.from_numpy(x_[:len(y_train)]), torch.from_numpy(y_train),
                             feature_attention_score=feature_attention_score,
                             sample_attention_score=sample_attention_score,
                             subsample_ratio=self.inference_config[id_pipe]["retrieval_config"]["subsample_ratio"])
                    if self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "feature":
                        x_ = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                        categorical_idx_ = self.get_categorical_features_indices(x_)
                    else:
                        attention_score = step.transform(torch.from_numpy(x_[len(y_train):]).float())
                else:
                    x_, categorical_idx_ = step.fit_transform(x_, categorical_idx_, self.seeds[id_pipe*self.preprocess_num+id_step], y=y_)
                    # print(f"step {id_step} categorical_idx_ {categorical_idx_}")
            
            x_ = torch.from_numpy(x_[:, :]).float().to(self.device)
            y_ = torch.from_numpy(y_).float().to(self.device)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if self.inference_config[id_pipe]["retrieval_config"]["use_retrieval"] and \
                    self.inference_config[id_pipe]["retrieval_config"]["subsample_type"] == "sample":
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="AM")
                output = inference.inference(x_[:len(y_train)].squeeze(), y_.squeeze(),
                                             x_[len(y_train):].squeeze(),
                                             attention_score=attention_score,
                                             retrieval_len=self.inference_config[id_pipe]["retrieval_config"][
                                                 "subsample_ratio"],
                                             dynamic_ratio=self.inference_config[id_pipe]["retrieval_config"][
                                                 "dynamic_ratio"] if "dynamic_ratio" in self.inference_config[id_pipe][
                                                 "retrieval_config"] else None,
                                             task_type="cls")
                if self.softmax_temperature != 1:
                    output = (output[:, :self.n_classes].float() / self.softmax_temperature)

                output = output[..., self.class_permutations[id_pipe]]
                outputs.append(output)
            elif self.inference_with_DDP:
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="DDP")
                output = inference.inference(x_[:len(y_train)].squeeze(1), y_, x_[len(y_train):].squeeze(1),
                                             task_type="cls")
                if self.softmax_temperature != 1:
                    output = (output[:, :self.n_classes].float() / self.softmax_temperature)

                output = output[..., self.class_permutations[id_pipe]]
                outputs.append(output)
            else:
                self.model.to(self.device)
                with(torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                    enabled=(self.mix_precision and torch.cuda.is_available())),
                     torch.inference_mode()):
                    x_=x_.unsqueeze(0)
                    y_ = y_.unsqueeze(0)
                    output=self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type='cls')

                    if self.mask_prediction:
                        process_config = output['process_config']
                        output_feature_pred = self.PostProcessInModel(output['feature_pred'], process_config)
                        output_feature_pred = self.PostProcess(output_feature_pred, pipe, process_config)
                        mask_predictions.append(output_feature_pred)
                        output = output['cls_output']

                    output = output if isinstance(output, dict) else output.squeeze(0)

                    if self.softmax_temperature != 1:
                        output = (output[:, :self.n_classes].float() / self.softmax_temperature)

                    output = output[..., self.class_permutations[id_pipe]]
                outputs.append(output)
            
        outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
        output = torch.stack(outputs).mean(dim=0)
        mask_prediction = np.stack(mask_predictions).mean(axis=0) if mask_predictions != [] and self.mask_prediction else None
        output = output.float().cpu().numpy()

        if self.mask_prediction:
            return output / output.sum(axis=1, keepdims=True), mask_prediction
        else:
            return output / output.sum(axis=1, keepdims=True)
    
    def PostProcessInModel(self, feature_pred:torch.tensor, config: dict) -> torch.tensor:
        # Revert preprocess in model forward
        feature_pred = feature_pred / torch.sqrt(config['features_per_group'] / config['num_used_features'].to(self.device))
        feature_pred = feature_pred*config['std_for_normalization'] + config['mean_for_normalization']
        feature_pred = einops.rearrange(feature_pred, "b s f n -> s b (f n)").squeeze(1).float().cpu().numpy()
        if config['n_x_padding'] > 0:
            feature_pred = feature_pred[:,:-config['n_x_padding']]
        return feature_pred
    
    def PostProcess(self, feature_pred:np.ndarray, pipeline:List, config: dict, gt=False) -> np.ndarray:        
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
                    cont_features_indices = [idx for idx in range(feature_pred.shape[1]) if idx not in step.dis_ix]
                    feature_pred[:, cont_features_indices] = step.worker.named_transformers_['feat_transform'].inverse_transform(feature_pred[:, cont_features_indices])
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
        
    def _predict_reg(self, x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray) -> np.ndarray:
        np_rng = np.random.default_rng(self.seed)
        
        x_train, y_train = self.validate_data(x_train, y_train, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        x_test = self.validate_data(x_test, reset=True, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False)
        
        # "Concatenate x_train and x_test to ensure the preprocessing logic is completely consistent.
        x = np.concatenate([x_train, x_test], axis=0)
    
        # preprocess x
        x = self.convert_x_dtypes(x)
        x = self.convert_category2num(x)
        x = x.astype(float)
        categorical_idx = self.get_categorical_features_indices(x)
    
        outputs = []
        mask_predictions = []
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            x_ = x.copy()
            y_ = y_train.copy()
            categorical_idx_ = categorical_idx.copy()
            for id_step, step in enumerate(pipe):
                if isinstance(step, InferenceAttentionMap):

                    feature_attention_score, sample_attention_score = step.inference(X_train=x_[:len(y_train)],
                                                                                     y_train=y_train,
                                                                                     X_test=x_[len(y_train):],
                                                                                     task_type="reg")
                    
                elif isinstance(step, SubSampleData):
                    step.fit(torch.from_numpy(x_[:len(y_train)]), torch.from_numpy(y_train),
                             feature_attention_score=feature_attention_score,
                             sample_attention_score=sample_attention_score,
                             subsample_ratio=self.inference_config[id_pipe]["retrieval_config"]["subsample_ratio"])
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
                output = inference.inference(x_[:len(y_train)].squeeze(), y_.squeeze(),
                                             x_[len(y_train):].squeeze(),
                                             attention_score=attention_score,
                                             retrieval_len=self.inference_config[id_pipe]["retrieval_config"][
                                                 "subsample_ratio"], task_type="reg")
                outputs.append(output)
            elif self.inference_with_DDP:
                inference = InferenceResultWithRetrieval(model=self.model,
                                                         sample_selection_type="DDP")
                output = inference.inference(x_[:len(y_train)].squeeze(1), y_, x_[len(y_train):].squeeze(1),
                                             task_type="reg")
                outputs.append(output)
            else:
                self.model.to(self.device)
                with(torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                    enabled=(self.mix_precision and torch.cuda.is_available())),
                     torch.inference_mode()):
                    x_=x_.unsqueeze(0)
                    y_ = y_.unsqueeze(0)

                    output=self.model(x=x_, y=y_, eval_pos=y_.shape[1], task_type='reg')

                if self.mask_prediction:
                    process_config = output['process_config']
                    output_feature_pred = self.PostProcessInModel(output['feature_pred'], process_config)
                    output_feature_pred = self.PostProcess(output_feature_pred, pipe, process_config)
                    mask_predictions.append(output_feature_pred)
                    output = output['reg_output']

                output = output if isinstance(output, dict) else output.squeeze(0)
            outputs.append(output)
            
        output = torch.stack(outputs).squeeze(2).mean(dim=0)
        mask_prediction = np.stack(mask_predictions).mean(axis=0) if mask_predictions != [] else None
        
        if self.mask_prediction:
            return output, mask_prediction
        else:
            return output
