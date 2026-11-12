import gc
import torch
import torch.nn as nn
from typing import Any, Literal
import numpy as np
from .utils import SetRandomSeed
from .autobatch import autobatch
from .utils import create_mlp_layer
from .operators.rmsnorm import build_rmsnorm


def calc_mean(x: torch.Tensor, dim: int):
    num = torch.sum(~torch.isnan(x), dim=dim).clip(min=1.0)
    return torch.nansum(x, dim=dim) / num, num


def calc_std(x: torch.Tensor, dim: int, mean_v: torch.Tensor | None = None, value_num: torch.Tensor | None = None):
    if mean_v is None or value_num is None:
        mean_v, value_num = calc_mean(x, dim)
    mean_broadcast = torch.repeat_interleave(mean_v.unsqueeze(dim), x.shape[dim], dim=dim, )
    return torch.sqrt(torch.nansum(torch.square(mean_broadcast - x), dim=dim) / (value_num - 1))


def drop_outliers(
        x: torch.Tensor,
        std_sigma: float = 4,
        eval_pos: int = -1,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
        dim: int = 1
):
    # assert len(x.shape)==3, "x.shape must be B,S,F"  

    if lower is None:
        data = x[:, :eval_pos].clone()
        data_mean, value_num = calc_mean(data, dim=dim)
        data_std = calc_std(data, dim=dim, mean_v=data_mean, value_num=value_num)
        cut_off = data_std * std_sigma
        lower, upper = (data_mean - cut_off).unsqueeze(1), (data_mean + cut_off).unsqueeze(1)
        data[torch.logical_or(data > upper, data < lower)] = np.nan

        data_mean, value_num = calc_mean(data, dim=dim)
        data_std = calc_std(data, dim=dim, mean_v=data_mean, value_num=value_num)
        cut_off = data_std * std_sigma
        lower, upper = (data_mean - cut_off).unsqueeze(1), (data_mean + cut_off).unsqueeze(1)

    x = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
    x = torch.minimum(torch.log(1 + torch.abs(x)) + upper, x)

    return x, lower, upper


def normalize_mean0_std1(
        x: torch.Tensor,
        eval_pos: int = -1,
        clip: bool = True,
        dim: int = 1,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None
):
    '''Normalize along dim=1 (sample axis). For x of shape (1, 1391, 2, 2), mean/std have shape (2, 2).'''
    if mean is None:
        mean, value_num = calc_mean(x[:, :eval_pos], dim=dim)
        std = calc_std(x[:, :eval_pos], dim=dim, mean_v=mean, value_num=value_num) + 1e-20

        if x.shape[1] == 1 or eval_pos == 1:
            std[:] = 1.0
    x = (x - mean.unsqueeze(1).expand_as(x)) / std.unsqueeze(1).expand_as(x)
    if clip:
        x = torch.clip(x, min=-100, max=100)
    return x, mean, std


class YNoneEmbeddingEncoder(nn.Module):
    """Embed valid and invalid Y values separately. Invalid values are mainly None and +/- Inf."""

    def __init__(
            self,
            num_features: int,
            emsize: int,
            numeric_embed_type: str = 'linear',
            bias: bool = True,
            in_keys: list[str] = ['data'],
            out_key: str = 'data',
            **kwargs,
    ):
        """Initialize the YNoneEmbeddingDecoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_key = out_key
        self.emb_mlp_hidden_dim_ratio = kwargs.get("emb_mlp_hidden_dim_ratio", 0.5)
        self.embedding_dim = emsize
        self.mask_token_emb_type = kwargs.get("mask_token_emb_type", "mask_emb")

        if self.mask_token_emb_type in ("mask_emb", "uniform_mask_emb_type"):
            self.none_embedding = nn.Embedding(1, self.embedding_dim)
        else:
            raise ValueError(f"Unknown mask_token_emb_type: {self.mask_token_emb_type}")

        uniform_encoder = kwargs.get('uniform_encoder', None)
        if uniform_encoder is not None:
            self.numeric_encoder = uniform_encoder
        elif 'MLP_' in numeric_embed_type:
            self.numeric_encoder = create_mlp_layer(
                mlp_type=numeric_embed_type,
                input_dim_size=1,
                hidden_size=int(emsize * self.emb_mlp_hidden_dim_ratio),
                output_dim_size=emsize,
                dropout=0.0,
                bias=bias,
                norm_impl=kwargs.get('rmsnorm_impl', 'triton'),
                recompute=kwargs.get('layer_recompute', False),
            )
        else:
            raise ValueError(f"Invalid reg_y_numeric_embed_type: {numeric_embed_type}")

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        assert 'data' in input

        none_mask = torch.isnan(input['data']) | torch.isinf(input['data'])
        no_none_data = torch.where(none_mask, torch.zeros_like(input['data']), input['data'])

        non_none_data_emb = self.numeric_encoder(no_none_data.unsqueeze(-1))
        none_emb = self.none_embedding(torch.zeros_like(none_mask, dtype=torch.int64))

        combined_emb = torch.where(none_mask.unsqueeze(-1), none_emb, non_none_data_emb)
        # combined_emb = self._add_mask_indicator(combined_emb, none_mask.unsqueeze(-1))

        input[self.out_key] = combined_emb

        return input


class MaskEmbEncoder(nn.Module):
    """
    For masked features, use the mask vector to obtain their representations;
    for numerical features, use a nonlinear network to obtain their representations
    """

    def __init__(
            self,
            num_features: int,
            emsize: int,
            mask_embedding_size: int,
            numeric_embed_type: str = "linear",
            RBF_config: dict | None = None,
            nan_to_zero: bool = False,
            bias: bool = True,
            in_keys: list[str] = ['data'],
            out_key: str = 'data',
            feature_fusion_network_type: str = "original",
            **kwargs
    ):
        """Initialize the MaskEmbEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.embedding_dim = emsize
        self.mask_embedding_size = mask_embedding_size
        self.in_keys = in_keys
        self.out_key = out_key
        self.use_feature_mask_indicator = kwargs.get("use_feature_mask_indicator", False)
        self.use_feature_group_encoder = kwargs.get("use_feature_group_encoder", True)
        self.mask_token_emb_type = kwargs.get("mask_token_emb_type", "mask_emb")
        uniform_fusion_network = kwargs.get("uniform_fusion_network", None)

        if "uniform_mask_emb_type" == self.mask_token_emb_type:
            self.mask_embedding = nn.Embedding(1, self.embedding_dim)
        else:
            raise ValueError(f"Unknown mask_token_emb_type: {self.mask_token_emb_type}")

        self.numeric_embed_type = numeric_embed_type
        norm_impl = kwargs.get('rmsnorm_impl', 'triton')
        recompute = kwargs.get('layer_recompute', False)
        if numeric_embed_type == "linear":
            self.numeric_mlp = nn.Sequential(
                nn.Linear(1, self.embedding_dim // 2),
                build_rmsnorm(
                    self.embedding_dim // 2, eps=1e-5, elementwise_affine=True,
                    norm_impl=norm_impl, recompute=recompute,
                ),
                nn.ReLU(),
                nn.Linear(self.embedding_dim // 2, self.embedding_dim),
                build_rmsnorm(
                    self.embedding_dim, eps=1e-5, elementwise_affine=True,
                    norm_impl=norm_impl, recompute=recompute,
                ),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Invalid numeric_embed_type: {numeric_embed_type}")

        if self.use_feature_group_encoder:
            # Merging layer: maps the concatenated feature vectors back to embedding_dim.
            if uniform_fusion_network is not None:
                self.fusion_network = uniform_fusion_network
            elif 'original' == feature_fusion_network_type:
                self.fusion_network = nn.Sequential(
                    nn.Linear(num_features * self.embedding_dim, self.embedding_dim, bias=bias),
                    build_rmsnorm(
                        self.embedding_dim, eps=1e-5, elementwise_affine=True,
                        norm_impl=norm_impl, recompute=recompute,
                    ),
                    nn.ReLU(),
                    nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
                    build_rmsnorm(
                        self.embedding_dim, eps=1e-5, elementwise_affine=True,
                        norm_impl=norm_impl, recompute=recompute,
                    ),
                )
            else:
                raise ValueError(f"Invalid feature_fusion_network_type: {feature_fusion_network_type}")

        self.nan_to_zero = nan_to_zero

    @autobatch(batch_dim=1)
    def _cal_numeric_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.numeric_mlp(x)

    @autobatch(batch_dim=1)
    def _cal_fusion_network(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_network(x)

    @autobatch(batch_dim=1)
    def _apply_mask_emb(self, x_emb: torch.Tensor, is_mask: torch.Tensor) -> torch.Tensor:
        """Apply mask merge in seq_len batches, matching _cal_numeric_mlp."""
        mask_vec = self.mask_embedding.weight[0]
        mask_emb = mask_vec.view(*([1] * (x_emb.dim() - 1)), -1).expand_as(x_emb)
        combined_emb = torch.where(is_mask, mask_emb, x_emb)
        return self._add_mask_indicator(combined_emb, is_mask)

    def _add_mask_indicator(self, combined_emb: torch.Tensor, is_mask: torch.Tensor) -> torch.Tensor:
        """Add mask-indicator encoding."""
        if not self.use_feature_mask_indicator:
            return combined_emb

        is_mask = is_mask.to(combined_emb.dtype)
        mask_indicator = self.mask_indicator_mlp(is_mask)
        result = self.mask_emb_fusion(torch.cat([combined_emb, mask_indicator], dim=-1))
        return result

    def _make_feature_group_emb(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Get the embedding of a feature group."""
        try:
            feature_group = self._cal_fusion_network(x_emb)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            feature_group = self._cal_fusion_network(x_emb)
        return feature_group

    @autobatch(batch_dim=1)
    def _chunk_encode_per_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        is_mask = torch.isnan(x)
        x = x.masked_fill(is_mask, 0.0)

        x_emb = self._cal_numeric_mlp(x)  # 1 -> [1, 192]

        mask_vec = self.mask_embedding.weight[0]
        mask_emb = mask_vec.view(*([1] * (x_emb.dim() - 1)), -1).expand_as(x_emb)
        combined_emb = torch.where(is_mask, mask_emb, x_emb)
        combined_emb = self._add_mask_indicator(combined_emb, is_mask)

        return combined_emb

    def encode_per_feature(self, input: dict[str, torch.Tensor | int]) -> torch.Tensor:
        """Encode each scalar feature without applying feature-group fusion."""
        assert 'data' in input and 'nan_encoding' in input
        x = [input[key] for key in self.in_keys]
        x: torch.Tensor = torch.cat(x, dim=-1)  # type: ignore

        return self._chunk_encode_per_feature(x)

    def group_per_feature(self, per_feature_emb: torch.Tensor) -> torch.Tensor:
        """Apply the existing feature-group fusion to per-feature embeddings."""
        batch_size, seq_len, group = per_feature_emb.shape[:3]
        if self.use_feature_group_encoder:
            combined_emb = per_feature_emb.flatten(3)
            sample_representation = self._make_feature_group_emb(combined_emb)
        else:
            sample_representation = per_feature_emb
        return sample_representation.view(batch_size, seq_len, group, -1)

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        input[self.out_key] = self.group_per_feature(self.encode_per_feature(input))
        return input


class NanEncoder(nn.Module):
    """Encoder stage that deals with NaN and infinite values in the input"""

    def __init__(
            self,
            nan_value: float = -2.0,
            inf_value: float = 2.0,
            neg_info_value: float = 4.0,
            in_keys: list[str] = ['data'],
            out_key: str = 'nan_encoding'
    ):
        """Initialize the NanEncoder.

        Args:
            keep_nans: Flag to maintain NaN values as individual indicators.
        """
        super().__init__()
        self.nan_value = nan_value
        self.inf_value = inf_value
        self.neg_info_value = neg_info_value
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        x = input[self.in_keys[0]]  # type: ignore
        eval_pos = input['eval_pos']
        # x = x.contiguous()

        mean_value, _ = calc_mean(x[:, :eval_pos], dim=1)
        mean_value = mean_value.unsqueeze(1)

        x_nan, x_inf = x.isnan(), x.isinf()
        nans_indicator = (
                x_nan * self.nan_value +
                torch.logical_and(x_inf, x > 0) * self.inf_value +
                torch.logical_and(x_inf, x < 0) * self.neg_info_value
        ).to(x.dtype)
        nan_mask = torch.logical_or(x_nan, x_inf)
        x_new = torch.where(nan_mask, mean_value, x)

        input[self.in_keys[0]] = x_new
        input[self.out_key] = nans_indicator
        return input


class ValidFeatureEncoder(nn.Module):
    """Valid feature encoder"""

    def __init__(
            self,
            num_features: int,
            nan_normalize: bool = True,
            sqrt_normalize: bool = True,
            in_keys: list[str] = ['data'],
            out_key: str = 'data'
    ):
        """Initialize the ValidFeatureEncoder.

        Args:
            num_features: The target number of features to transform the input into.
            nan_normalize: Indicates whether to normalize based on the number of features actually used.
            sqrt_normalize: Legacy option to normalize using the square root rather than the count of used features.
        """
        super().__init__()
        self.num_features = num_features
        self.nan_normalize = nan_normalize
        self.sqrt_normalize = sqrt_normalize
        self.in_keys = in_keys
        self.out_key = out_key
        self.valid_feature_num = None

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        x: torch.Tensor = input[self.in_keys[0]]  # type: ignore
        x_train = x[:, :input['eval_pos']]
        valid_feature = ~torch.all(x_train == x_train[:, 0:1, :], dim=1)
        self.valid_feature_num = torch.clip(valid_feature.sum(-1).unsqueeze(-1),
                                            min=1)

        # x.shape:     torch.Size([1, 1391, 2, 2])
        # valid_feature.shape:     torch.Size([1, 2, 2])
        # self.valid_feature_num.shape:     torch.Size([1, 2, 1])

        if self.nan_normalize:
            if self.sqrt_normalize:
                x = x * torch.sqrt(self.num_features / self.valid_feature_num).unsqueeze(1).expand_as(x)
            else:
                x = x * (self.num_features / self.valid_feature_num).unsqueeze(1)

        zeros = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros], -1)

        input[self.out_key] = x
        return input


class EmbYEncoderStep(nn.Module):
    """
    Use for classification task
    A simple linear input encoder step.
    """

    def __init__(
            self,
            *,
            emsize: int,
            n_classes: int = 10,
            yemb_freeze_type: Literal['freeze_train', 'freeze_all'] | None = None,
            in_keys: list[str] = ['data'],
            out_key: str = 'data',
            cls_emb_learning_scalar: bool = False
    ):
        """Initialize the EmbYEncoderStep.

        Args:
            emsize: The embedding size, i.e. the number of output features.
            n_classes: Number of classes
            yemb_weight_freeze_type: Method for freezing the weights of y embedding
        """
        super().__init__()

        self.cls_emb_learning_scalar = cls_emb_learning_scalar
        self.y_embedding = nn.Embedding(n_classes, emsize)
        self.y_mask = nn.Embedding(1, emsize)
        if yemb_freeze_type is None:
            self.y_embedding.weight.requires_grad = True  # Do not freeze the parameters of the training set
            self.y_mask.weight.requires_grad = True  # Do not freeze the parameters of the test set
        else:
            raise ValueError(f"Unknown yemb_freeze_type: {yemb_freeze_type}")
        self.in_keys = in_keys
        self.out_key = out_key
        if len(self.in_keys) > 1:
            print(
                "\033[30;43mWarning: The EmbYEncoderStepl function is only for processing Y, and in_keys must contain exactly one key.\033[0m")

        if cls_emb_learning_scalar:
            self.scalar = nn.Parameter(torch.tensor(5, dtype=self.y_embedding.weight.data.dtype))
            self.layer_norm = build_rmsnorm(emsize, eps=1e-5, elementwise_affine=True)

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        y = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        y = y.int()  # type: ignore
        y_train = y[:, :eval_pos]
        y_test = torch.zeros_like(y[:, eval_pos:], dtype=torch.int)
        y_train_emb = self.y_embedding(y_train)
        y_test_emb = self.y_mask(y_test)
        y_emb = torch.cat([y_train_emb, y_test_emb], dim=1)
        if self.cls_emb_learning_scalar:
            y_emb = y_emb * self.scalar
            y_emb = self.layer_norm(y_emb)
        input[self.out_key] = y_emb
        return input


class MulticlassTargetEncoder(nn.Module):
    """Use the target's index as the class value, with each class corresponding to an index"""

    def __init__(
            self,
            in_keys: list[str] = ['data'],
            out_key: str = 'data',
            mode: str = None
    ):
        """Initialize the ValidFeatureEncoder.

        Args:
            in_keys: the keys of the input parameter
            out_key: the key of the output result.
            mode:
                'Random_full' means mapping the origin label to all cls decoder head randomly;
                ''
        """
        super().__init__()
        self.mode = mode
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        x: torch.Tensor = input[self.in_keys[0]]  # type: ignore
        eval_pos = input['eval_pos']

        unique_xs: list[torch.Tensor] = [torch.unique(x[b, :eval_pos]) for b in range(x.shape[0])]
        for b in range(x.shape[0]):
            x[b, :eval_pos] = (x[b, :eval_pos].unsqueeze(-1) > unique_xs[b]).sum(dim=-1)

        input[self.out_key] = x
        if x.shape[0] == 1:
            input['cls_head_mapping'] = torch.arange(10).unsqueeze(dim=0).to(x.device)
        else:
            input['cls_head_mapping'] = torch.concat([torch.arange(10).unsqueeze(dim=0) for _ in range(x.shape[0])],
                                                     dim=0).to(x.device)
        input['cls_head_mask'] = torch.ones([x.shape[0], 10]).to(x.device)
        return input


class NormalizationEncoder(nn.Module):
    """normalize encoder"""

    def __init__(
            self,
            train_only: bool,
            normalize_x: bool,
            remove_outliers: bool,
            std_sigma: float = 4.0,
            in_keys: list[str] = ['data'],
            out_key: str = 'data'

    ):
        super().__init__()
        self.train_only = train_only
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.std_sigma = std_sigma
        self.in_keys = in_keys
        self.out_key = out_key
        self.mean = None
        self.std = None

    def forward(self, input: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor]:
        x = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        pos = eval_pos if self.train_only else -1
        if self.remove_outliers:
            x, lower, upper = drop_outliers(x, eval_pos=pos, std_sigma=self.std_sigma)
        if self.normalize_x:
            x, self.mean, self.std = normalize_mean0_std1(x, eval_pos=pos)

        input[self.out_key] = x
        return input


def get_x_encoder(
        *,
        num_features: int,
        embedding_size: int,
        mask_embedding_size: int,
        encoder_use_bias: bool,
        feature_embedding_type: str = 'mask_embedding',
        numeric_embed_type: str = "scino",
        RBF_config: dict | None = None,
        init_seed_dict: dict = {},
        in_keys: list = ['data'],
        **kwargs
):
    assert isinstance(in_keys, list), "The type of in_keys must be a list!"
    inputs_to_merge = {}
    for in_key in in_keys:
        inputs_to_merge[in_key] = {'dim': num_features}

    encoder_steps = []
    with SetRandomSeed(init_seed_dict.get('encoder_x_seed', None)):
        if 'mask_embedding' == feature_embedding_type:
            encoder_steps += [
                # The masked features (i.e., features with None values) are directly mapped to
                # vectors via the embedding matrix, while the numerical features obtain their
                # embedding representations through a nonlinear transformation matrix.
                MaskEmbEncoder(
                    num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
                    emsize=embedding_size,
                    mask_embedding_size=mask_embedding_size,
                    bias=encoder_use_bias,
                    RBF_config=RBF_config,
                    numeric_embed_type=numeric_embed_type,
                    **kwargs
                ),

            ]
        else:
            raise ValueError(f'Unknown feature embedding type: {feature_embedding_type}')

    return nn.Sequential(*encoder_steps, )


def get_cls_y_encoder(
        *,
        num_inputs: int,
        embedding_size: int,
        nan_handling_y_encoder: bool,
        max_num_classes: int,
        yemb_freeze_type: str | None = None,
        RBF_config: dict | None = None,
        init_seed_dict: dict = {},
        **kwargs
) -> nn.Module:
    steps = []
    if nan_handling_y_encoder:
        steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]

    if max_num_classes >= 2:
        steps += [MulticlassTargetEncoder()]

    y_encoder_type = kwargs.get('cls_y_encoder_type', 'emby_embedding')
    with SetRandomSeed(init_seed_dict.get('encoder_cls_y_seed', None)):
        if 'emby_embedding' == y_encoder_type:
            steps += [
                EmbYEncoderStep(
                    emsize=embedding_size,
                    n_classes=max_num_classes,
                    yemb_freeze_type=yemb_freeze_type,  # type: ignore
                    cls_emb_learning_scalar=kwargs.get('cls_emb_learning_scalar', False),
                )
            ]
        else:
            raise ValueError(f'Unknown cls y_encoder_type: {y_encoder_type}')
    return nn.Sequential(*steps)


def get_reg_y_encoder(
        *,
        num_inputs: int,
        num_features: int,
        embedding_size: int,
        nan_handling_y_encoder: bool,
        max_num_classes: int,
        yemb_freeze_type: str = 'yemb_open',
        RBF_config: dict | None = None,
        init_seed_dict: dict = {},
        **kwargs
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "data", "dim": num_inputs}]

    y_encoder_type = kwargs.get('y_encoder_type', 'linear')
    with SetRandomSeed(init_seed_dict.get('encoder_reg_y_seed', None)):
        if 'none_embedding' == y_encoder_type:
            steps += [
                YNoneEmbeddingEncoder(
                    num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
                    emsize=embedding_size,
                    numeric_embed_type=kwargs.get('reg_y_numeric_embed_type', 'linear'),
                    in_keys=['data', 'nan_encoding'],
                    out_key='data',
                    RBF_config=RBF_config,
                    **kwargs,
                ),
            ]
        else:
            raise ValueError(f'Unknown reg y_encoder_type: {y_encoder_type}')
    return nn.Sequential(*steps)


class PreprocessPipeline(nn.Module):
    def __init__(
            self,
            *,
            num_features: int,
            nan_handling_enabled: bool,
            normalize_on_train_only: bool,
            normalize_x: bool,
            remove_outliers: bool,
            normalize_by_used_features: bool,
    ):
        super().__init__()

        modules = nn.ModuleDict()
        step_order = []

        # Step 1: NanEncoder
        modules['nan_encoder'] = NanEncoder(in_keys=['data'], out_key='nan_encoding')
        step_order.append('nan_encoder')

        # Step 2: Optional ValidFeatureEncoder for NaN indicators
        if nan_handling_enabled:
            modules['nan_valid_encoder'] = ValidFeatureEncoder(
                num_features=num_features,
                nan_normalize=False,
                in_keys=["nan_encoding"],
                out_key="nan_encoding"
            )
            step_order.append('nan_valid_encoder')

        # Step 3: Normalization
        modules['normalization_encoder'] = NormalizationEncoder(
            train_only=normalize_on_train_only,
            normalize_x=normalize_x,
            remove_outliers=remove_outliers,
        )
        step_order.append('normalization_encoder')

        # Step 4: Final ValidFeatureEncoder
        modules['valid_feature_encoder'] = ValidFeatureEncoder(
            num_features=num_features,
            nan_normalize=normalize_by_used_features,
        )
        step_order.append('valid_feature_encoder')

        self.modules_dict = modules
        self.step_order = step_order  # preserve execution order

    def forward(self, x):
        # x is expected to be a dict-like input containing at least 'data'
        out = x
        for name in self.step_order:
            out = self.modules_dict[name](out)
        return out


def preprocesss_4_x(
        *,
        num_features: int,
        nan_handling_enabled: bool,
        normalize_on_train_only: bool,
        normalize_x: bool,
        remove_outliers: bool,
        normalize_by_used_features: bool,
):
    """Feature preprocessing pipeline implemented with nn.ModuleDict."""
    return PreprocessPipeline(
        num_features=num_features,
        nan_handling_enabled=nan_handling_enabled,
        normalize_on_train_only=normalize_on_train_only,
        normalize_x=normalize_x,
        remove_outliers=remove_outliers,
        normalize_by_used_features=normalize_by_used_features,
    )
