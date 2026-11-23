import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from torch.utils.data import DistributedSampler
import torch.nn.functional as F


def auc_metric(target, pred, multi_class='ovo', numpy=False):
    """Compute ROC-AUC for binary or multiclass classification scores.

    Input:
        target: Ground-truth class labels, shape (N,). Tensor or array-like of
            integer class ids.
        pred: Predicted scores. Shape (N,) for binary scores, or (N, C) class
            probabilities. For binary (N, 2), column 1 is used.
        multi_class: sklearn multiclass strategy when C > 2. Default 'ovo'.
        numpy: If True, return a NumPy/Python scalar; if False, return a 0-dim tensor.

    Output:
        Scalar AUC. NaN (matching numpy=) when sklearn raises ValueError.
    """
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


def calculate_result(y_test_encoded, y_pred_proba):
    """Print classification metrics and return them as a tuple.

    Input:
        y_test_encoded: Integer labels, shape (N,). Must already be encoded
            (0 .. C-1), not raw class names.
        y_pred_proba: Class probabilities, shape (N, C), rows should sum to 1.
            C == 2 uses the positive-class column for AUC; C > 2 uses ovo AUC.

    Output:
        tuple: (accuracy, auc, macro-or-binary f1, log_loss, ece). All floats.
            Also prints each metric. ECE uses 10 equal-width confidence bins.
    """
    y_pred_label = np.argmax(y_pred_proba, axis=1)
    if len(np.unique(y_test_encoded)) == 2:
        final_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
    else:
        final_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class="ovo")
    print(f"✅ AUC = {final_auc:.4f}")

    # --- Accuracy ---
    acc = accuracy_score(y_test_encoded, y_pred_label)
    print(f"✅ Accuracy = {acc:.4f}")

    # --- F1 Score ---
    f1 = f1_score(y_test_encoded, y_pred_label, average='macro' if len(np.unique(y_test_encoded)) > 2 else 'binary')
    print(f"✅ F1 Score = {f1:.4f}")

    # --- Cross Entropy / LogLoss ---
    ce = log_loss(y_test_encoded, y_pred_proba)
    print(f"✅ LogLoss (Cross Entropy) = {ce:.4f}")

    # --- ECE (Expected Calibration Error) ---
    def compute_ece(y_true, y_prob, n_bins=10):
        """Expected Calibration Error over equal-width confidence bins.

        Input:
            y_true: Integer labels, shape (N,).
            y_prob: Probabilities, shape (N,) or (N, C). For (N, C) with C > 1,
                confidence is max over classes and the predicted class is argmax.
            n_bins: Number of bins in [0, 1]. Must be a positive integer.

        Output:
            float: weighted mean of |accuracy - confidence| over non-empty bins.
        """
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

    ece = compute_ece(y_test_encoded, y_pred_proba, n_bins=10)
    print(f"✅ ECE (Expected Calibration Error, 10 bins) = {ece:.4f}")

    return acc, final_auc, f1, ce, ece





def generate_infenerce_config(args):
    """Write a default 4-member no-retrieval inference config JSON.

    Input:
        args: Namespace with inference_config_path (str). That path is created
            or overwritten. Used when a caller did not supply a config file.

    Output:
        None. The JSON is a list of 4 pipeline dicts (quantile/svd and
            numeric/no-svd pairs, each duplicated).
    """
    retrieval_config = dict(
        use_retrieval=False,
        retrieval_before_preprocessing=False,
        calculate_feature_attention=False,
        calculate_sample_attention=False,
        subsample_ratio=1,
        subsample_type=None,
        use_type=None,
    )

    config_list = [
        dict(RebalanceFeatureDistribution=dict(worker_tags=["quantile"], discrete_flag=False, original_flag=True,
                                               svd_tag="svd"),
             CategoricalFeatureEncoder=dict(encoding_strategy="ordinal_strict_feature_shuffled"),
             FeatureShuffler=dict(mode="shuffle"),
             retrieval_config=retrieval_config,
             ),
        dict(RebalanceFeatureDistribution=dict(worker_tags=["quantile"], discrete_flag=False, original_flag=True,
                                               svd_tag="svd"),
             CategoricalFeatureEncoder=dict(encoding_strategy="ordinal_strict_feature_shuffled"),
             FeatureShuffler=dict(mode="shuffle"), retrieval_config=retrieval_config,
             ),
        dict(RebalanceFeatureDistribution=dict(worker_tags=[None], discrete_flag=True, original_flag=False,
                                               svd_tag=None),
             CategoricalFeatureEncoder=dict(encoding_strategy="numeric"),
             FeatureShuffler=dict(mode="shuffle"),
             retrieval_config=retrieval_config,
             ),
        dict(RebalanceFeatureDistribution=dict(worker_tags=[None], discrete_flag=True, original_flag=False,
                                               svd_tag=None),
             CategoricalFeatureEncoder=dict(encoding_strategy="numeric"),
             FeatureShuffler=dict(mode="shuffle"),
             retrieval_config=retrieval_config)
    ]

    with open(args.inference_config_path, 'w') as f:
        json.dump(config_list, f)


def sample_inferece_params(rng:np.random.Generator, sample_num:int=2, repeat_num:int=2):
    """Sample pipeline and base hyperparameters from the hyperopt search space.

    Input:
        rng: NumPy Generator used by hyperopt.stochastic.sample.
        sample_num: Number of distinct pipeline configs to draw. Must be >= 1.
        repeat_num: How many times to repeat each drawn pipeline config in the
            returned list. Must be >= 1.

    Output:
        tuple:
            hyperopt_configs: list[dict] of length sample_num * repeat_num,
                each a pipeline member config (preprocess + retrieval_config).
            base_config: dict with softmax_temperature and seed for the run.
    """
    from hyperopt import hp
    from hyperopt.pyll import stochastic

    search_space = {
        "RebalanceFeatureDistribution":{
            "worker_tags": hp.choice("worker_tags", [["logNormal"], 
                                                     ["quantile_uniform_10"],
                                                     ["quantile_uniform_5"],
                                                     ["quantile_uniform_all_data"],
                                                     ["power"],
                                                     ["quantile_norm_10"],
                                                     ["quantile_norm_5"],
                                                     ["quantile_norm_all_data"],
                                                     ["norm_and_kdi"],
                                                     ["none"],
                                                     ["robust"],
                                                     ["kdi_uni"],
                                                     ["kdi_alpha_0.3"],
                                                     ["kdi_alpha_3.0"],
                                                     ["kdi_norm"],
                                                     ["power", "quantile_uniform_5"],
                                                     ["kdi", "quantile_uniform_5"]]),
            "discrete_flag": hp.choice("discrete_flag", [True, False]),
            "original_flag": hp.choice("original_flag", [True, False]),
            "svd_tag": hp.choice("svd_tag", ["svd", None])
        },

        "CategoricalFeatureEncoder": {
            "encoding_strategy": hp.choice("encoding_strategy", ["ordinal_strict_feature_shuffled", 
                                                                 "ordinal",
                                                                 "ordinal_strict_feature_shuffled",
                                                                 "ordinal_shuffled",
                                                                 "onehot",
                                                                 "numeric",
                                                                 "none",]),
        },
        "FeatureShuffler": {
            "mode": hp.choice("mode", ["shuffle", "rotate"])
        },
        "FingerprintFeatureEncoder": hp.choice("FingerprintFeatureEncoder", [True, False]),
        "PolynomialInteractionGenerator":{
            "max_interaction_features": hp.choice("max_interaction_features", [None, 50])
        },
        "retrieval_config": {
            "use_retrieval": False,
            "retrieval_before_preprocessing": False,
            "calculate_feature_attention": False,
            "calculate_sample_attention": False,
            "subsample_ratio": 0.7,
            "subsample_type": "sample",
            "use_type": "mixed"
        }
    }
    if rng.random() > 0.5:
        search_space["PolynomialInteractionGenerator"] = {
            "max_interaction_features": hp.choice("max_interaction_features", [None, 50])
        }
    
    base_search_space = {
        "softmax_temperature": hp.choice("softmax_temperature", [0.75, 0.8, 0.9, 0.95, 1.0]),
        "seed": hp.uniformint("seed", 0, 1000000)
    }

    hyperopt_configs = []
    for _ in range(sample_num):
        config = stochastic.sample(search_space, rng=rng)
        for _ in range(repeat_num):
            hyperopt_configs.append(config)

    base_config = stochastic.sample(base_search_space, rng=rng)

    return hyperopt_configs, base_config

class NonPaddingDistributedSampler(DistributedSampler):
    """DistributedSampler that does not pad the dataset to a multiple of world size.

    Input (constructor):
        dataset: Sequence-like dataset; len(dataset) is the global sample count.
        num_replicas: World size. Default torch.distributed.get_world_size().
        rank: This process rank. Default torch.distributed.get_rank().
        shuffle: Unused here; iteration is always sequential strided slices.

    Output:
        Iterator of local integer indices: range(rank, N, num_replicas).
        num_samples may differ across ranks when N is not divisible by world size.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.num_samples = len(range(rank, len(dataset), num_replicas))
        self.total_size = len(dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

def swap_rows_back(tensor, indices):
    """Undo a row permutation produced by a distributed sampler.

    Input:
        tensor: Rank-2+ tensor whose dim 0 was gathered in sampler order,
            shape (N, ...).
        indices: Length-N list or 1-D tensor of original row ids. indices[i]
            is the original index of gathered row i.

    Output:
        Tensor of shape (N, ...), rows restored to original index order.
    """
    inverse_indices = [0] * len(indices)
    for i, idx in enumerate(indices):
        inverse_indices[idx] = i
    return tensor[inverse_indices]


# ================ BinnedRegression utils start ==================
def init_borders(start, end, steps):
    """Build equally spaced bucket borders on [start, end].

    Input:
        start: Left endpoint, scalar float.
        end: Right endpoint, scalar float; should be > start.
        steps: Number of border points (buckets = steps - 1). Integer > 1.

    Output:
        Tensor of shape (steps,), dtype float32 by default.
    """
    return torch.linspace(start, end, steps)


def get_bucket_centers(borders:torch.Tensor):
    """Midpoints of consecutive bucket borders.

    Input:
        borders: Sorted 1-D borders, shape (B+1,) for B buckets.

    Output:
        Tensor of shape (B,), (borders[:-1] + borders[1:]) / 2.
    """
    return (borders[:-1] + borders[1:]) / 2.0  # (num_bars,)


def predict_mean_from_logits(input:torch.Tensor, borders:torch.Tensor):
    """Softmax-weighted expectation of bucket centers.

    Input:
        input: Logits over buckets, shape (..., B).
        borders: Sorted 1-D borders, shape (B+1,), B matching input's last dim.

    Output:
        Tensor of shape (...,), the expected value in border units.
    """
    probs = F.softmax(input, dim=-1)  # (..., num_bars)
    centers = get_bucket_centers(borders).to(probs.device)  # (num_bars,)
    return torch.sum(probs * centers, dim=-1)


def get_bucket_limits(
    num_outputs: int,
    full_range: tuple | None = None,
    ys: torch.Tensor | None = None,
    *,
    verbose: bool = False,  # noqa: ARG001
    widen_bucket_limits_factor: float | None = None,
) -> torch.Tensor:
    """Choose bucket border locations from a target range or an empirical y sample.

    Input:
        num_outputs: Number of buckets B. Must be >= 1. When ys is set, len(ys)
            after dropping NaNs must be > num_outputs.
        full_range: (ymin, ymax) used when ys is None, or as a clamp when ys is
            set. Exactly one of full_range and ys must be provided.
        ys: Optional 1-D or flattened targets. NaNs are dropped. Do not pass
            full_range unless it contains the min/max of ys.
        verbose: Unused; kept for call-site compatibility.
        widen_bucket_limits_factor: If set and not 1.0, multiply all borders by
            this factor to widen the support.

    Output:
        Tensor of shape (B+1,), strictly covering the requested range.
    """
    assert (ys is None) != (
        full_range is None
    ), "Either full_range or ys must be passed."

    if ys is not None:
        ys = ys.flatten()
        ys = ys[~torch.isnan(ys)]
        assert (
            len(ys) > num_outputs
        ), f"Number of ys :{len(ys)} must be larger than num_outputs: {num_outputs}"
        if len(ys) % num_outputs:
            ys = ys[: -(len(ys) % num_outputs)]
        ys_per_bucket = len(ys) // num_outputs
        if full_range is None:
            full_range = (ys.min(), ys.max())
        else:
            assert full_range[0] <= ys.min()
            assert full_range[1] >= ys.max()
            full_range = torch.tensor(full_range)  # type: ignore

        ys_sorted, ys_order = ys.sort(0)  # type: ignore
        bucket_limits = (
            ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
            + ys_sorted[ys_per_bucket::ys_per_bucket]
        ) / 2
        bucket_limits = torch.cat(
            [full_range[0].unsqueeze(0), bucket_limits, full_range[1].unsqueeze(0)],  # type: ignore
            0,
        )
        if widen_bucket_limits_factor is not None:
            bucket_limits = bucket_limits * widen_bucket_limits_factor

    else:
        class_width = (full_range[1] - full_range[0]) / num_outputs  # type: ignore
        bucket_limits = torch.cat(
            [
                full_range[0] + torch.arange(num_outputs).float() * class_width,  # type: ignore
                torch.tensor(full_range[1]).unsqueeze(0),  # type: ignore
            ],
            0,
        )

    assert len(bucket_limits) - 1 == num_outputs, (
        f"len(bucket_limits) - 1 == {len(bucket_limits) - 1}"
        f" != {num_outputs} == num_outputs"
    )

    if not widen_bucket_limits_factor or widen_bucket_limits_factor == 1.0:
        assert (
            full_range[0] == bucket_limits[0]  # type: ignore
        ), f"{full_range[0]} != {bucket_limits[0]}"  # type: ignore
        assert (
            full_range[-1] == bucket_limits[-1]  # type: ignore
        ), f"{full_range[-1]} != {bucket_limits[-1]}"  # type: ignore

    return bucket_limits
# ================ BinnedRegression utils end ==================

# ================ bar distribution utils start ==================
def cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Evaluate the piecewise-linear CDF of a bucket distribution at ys.

    Input:
        logits: Unnormalized bucket logits, shape (..., B).
        borders: Sorted 1-D borders, shape (B+1,).
        ys: Query values, broadcastable to logits[..., :1] after repeating
            along the batch dims. Values outside [borders[0], borders[-1]]
            map to CDF 0 or 1.

    Output:
        Tensor of shape (..., n_ys), clipped to [0, 1].
    """
    ys = ys.repeat(logits.shape[:-1] + (1,))
    n_bars = len(borders) - 1
    y_buckets = map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = (ys - borders[y_buckets]) / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)

    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)
    

def map_to_bucket_ix( y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Map each y to the index of the bucket that contains it.

    Input:
        y: Query values, any shape.
        borders: Sorted 1-D borders, shape (B+1,). y == borders[0] maps to 0;
            y == borders[-1] maps to B-1.

    Output:
        Long tensor, same shape as y, with values in [0, B-1].
    """
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix

def translate_probs_across_borders(
        logits: torch.Tensor,
        frm: torch.Tensor,
        to: torch.Tensor,
    ) -> torch.Tensor:
    """Rebin softmax probabilities from one border grid onto another.

    Input:
        logits: Source logits, shape (..., B_from).
        frm: Source borders, shape (B_from+1,).
        to: Destination borders, shape (B_to+1,). Must be sorted.

    Output:
        Tensor of shape (..., B_to), non-negative bucket masses on `to`.
        Endpoint CDF is forced to 0 and 1.
    """
    prob_left = cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0

    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)
    
def logits_to_output(
        output_type: str,
        logits: torch.Tensor,
        quantiles: list[float],
        borders: torch.Tensor, 
        bucket_widths: torch.Tensor
    ) -> np.ndarray | list[np.ndarray]:
    """Convert bucket logits to a detached prediction tensor.

    Input:
        output_type: Currently only 'mean' is supported.
        logits: Bucket logits, shape (..., B).
        quantiles: Unused for output_type='mean'; kept for the historical signature.
        borders: Sorted borders, shape (B+1,).
        bucket_widths: Widths, shape (B,), typically borders[1:] - borders[:-1].

    Output:
        Detached tensor, shape (...,) for output_type='mean'. Raises ValueError
        for any other output_type.
    """
    # TODO: support
    #   "pi": criterion.pi(logits, np.max(self.y)),
    #   "ei": criterion.ei(logits),
    if output_type == "mean":
        output = mean(logits, borders, bucket_widths)
    else:
        raise ValueError(f"Invalid output type: {output_type}")
        
    return output.detach()

def mean(logits: torch.Tensor, borders: torch.Tensor, bucket_widths: torch.Tensor) -> torch.Tensor:
    """Expectation of a bucket distribution, with half-normal tails on the ends.

    Input:
        logits: Bucket logits, shape (..., B).
        borders: Sorted borders, shape (B+1,).
        bucket_widths: Widths, shape (B,).

    Output:
        Tensor of shape (...,), same dtype/device as logits.
    """
    bucket_means = borders[:-1] + bucket_widths / 2
    p = torch.softmax(logits, -1)
    side_normals = (
        halfnormal_with_p_weight_before(bucket_widths[0]),
        halfnormal_with_p_weight_before(bucket_widths[-1]),
    )
    bucket_means[0] = -side_normals[0].mean + borders[1]
    bucket_means[-1] = side_normals[1].mean + borders[-2]
    return p @ bucket_means.to(logits.device).type(logits.dtype)

def halfnormal_with_p_weight_before(
        range_max: float,
        p: float = 0.5,
    ) -> torch.distributions.HalfNormal:
    """HalfNormal whose CDF at range_max equals p.

    Input:
        range_max: Positive scale location where CDF(range_max) = p.
        p: CDF target in (0, 1). Default 0.5.

    Output:
        torch.distributions.HalfNormal with scale fitted to (range_max, p).
    """
    s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
        torch.tensor(p),
    )
    return torch.distributions.HalfNormal(s + 1e-8)

# ================ bar distribution utils end ==================

if __name__ == "__main__":
    args = init_args()
    generate_infenerce_config(args)
