<div align="left">
  <img src="./doc/LimiX-Logo.png" alt="LimiX-2" width="105%">
</div>

## LimiX：面向结构化数据的大型基础模型（LDM）
[![preprint](https://img.shields.io/static/v1?label=LimiX-2-技术报告&message=2609.17488&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2609.17488)
[![preprint](https://img.shields.io/static/v1?label=LimiX-2M-论文&message=2606.04485&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2606.04485)
[![preprint](https://img.shields.io/static/v1?label=LimiX-技术报告&message=2509.03505&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2509.03505)
[![项目主页](https://img.shields.io/badge/LimiX-项目主页-green)](https://www.limix.ai/)
[![GitHub](https://img.shields.io/badge/GitHub-limiX--ldm%2FLimiX-181717)](https://github.com/limix-ldm/LimiX/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-stableai--ai-4285F4)](https://huggingface.co/stable-ai/)
[![ModelScope](https://img.shields.io/badge/ModelScope-stable--ai-f58c20)](https://modelscope.cn/organization/stable-ai/)
[![许可证](https://img.shields.io/badge/StableAI-License%20v1.0-green)](./LICENSE.txt)

<div align="left" style="line-height: 1;">
  📔 <a href="README.md" target="_blank">English</a>
  &nbsp;|
  中文
  &nbsp;
</div>

<br/>

## :🚀 更新动态

- **[2026.09.16] 🚀 LimiX-2 开源发布。** LimiX-2 权重（[`LimiX-2.ckpt`](https://huggingface.co/stable-ai/LimiX-2/tree/main)）与推理代码随本仓库一同发布。单个预训练模型在一次前向传播中即可完成分类、回归、缺失值插补，无需任务专属的参数更新。使用需遵循 [非商用许可协议](https://huggingface.co/stableai-org/LimiX-2/blob/main/LICENSE)。LimiX-2 技术报告正式发布，详见 [arxiv archive](https://arxiv.org/abs/2609.17488) 或 [LimiX_2_Technical_Report.pdf](./LimiX_2_Technical_Report.pdf)。
- **[2026.06.04] 🧠 LimiX 后续延伸研究成果被 ICML 会议录用！**
  论文：[LimiX-2M](https://arxiv.org/abs/2606.04485)，该工作是初代 [LimiX](https://arxiv.org/abs/2509.03505) 结构化数据基础模型的拓展研究。
- **[2025.11.10] 🚀 LimiX-2M 轻量化模型正式发布！**
  相较LimiX-16M版本，该轻量模型大幅降低GPU显存占用、提升推理速度；同时优化检索机制，在缩减推理耗时与内存开销的前提下进一步提升模型效果。
- **[2025.09.03] 🧠 LimiX 论文于arXiv上线发布**
  原论文：[arXiv:2509.03505](https://arxiv.org/abs/2509.03505)，LimiX是面向通用智能的首款结构化数据大模型，项目基于Apache 2.0协议开源。
- **[2025.08.29] 🚀 LimiX V1.0 正式发布**
  LimiX结构化数据基础模型首个稳定官方版本开源上线。

## ✨ LimiX-2 模型架构与预训练数据
LimiX-2 是 LimiX 家族中的新一代模型，它是基于我们此前确立的缩放定律（scaling laws），通过模型与数据规模扩展而开发的。LimiX-2 采用了上下文机制网络（CMN）范式，并利用上下文条件掩码建模（CCMM）进行预训练。CMN 将上下文学习（in-context learning）的组织原则从以目标为中心的预测，转变为以机制为导向的联合建模。该模型并非围绕传统表格型 PFN（先验拟合网络）中常见的  `p(y | x, D_context)` 目标进行构建，而是致力于学习 `p(x, y | D_context)`——即一种表征数据生成背后联合结构的、依赖于上下文的表示。预训练阶段使用了由结构因果模型（SCM）生成的合成数据集，这些数据集涵盖了多种图结构、函数机制及观测过程。在 TabArena、TALENT 和 BCCO 上的评估结果表明，LimiX-2 的表现优于现有的特定数据集模型及表格数据基础模型。除了预测性能外，CMN 范式还赋予了 LimiX-2 因果感知能力：其特征注意力机制能够编码直接因果关系，从而实现对因果骨架（causal skeleton）的精确恢复。

<div align="center">
  <img src="./doc/figures/fig2_architecture.png" alt="图 2：LimiX-2 整体结构" width="80%">
  <br>
  <sub>LimiX-2 模型架构</sub>
</div>

<div align="center">
  <img src="./doc/figures/fig3_data_generation.png" alt="图 3：预训练合成数据生成流程" width="80%">
  <br>
  <sub>预训练合成数据生成流程</sub>
</div>

### ➩ 模型下载

<div align="center">

| 模型        | 参数量 | 发布日期   | 下载链接                                                                      | 支持的任务                 |
| ----------- | ------ | ---------- | ----------------------------------------------------------------------------- | -------------------------- |
| LimiX-2     | 400M   | 2026-09-16 | [LimiX-2.ckpt](https://huggingface.co/stable-ai/LimiX-2/tree/main)             | ✅ 分类 ✅ 回归 ✅ 缺失值插补 |
| LimiX-1_2M  | 2M     | 2025-11-10 | [LimiX-1_2M.ckpt](https://huggingface.co/stable-ai/LimiX-1_2M/tree/main)       | ✅ 分类 ✅ 回归 ✅ 缺失值插补 |
| LimiX-1_16M | 16M    | 2025-08-29 | [LimiX-1_16M.ckpt](https://huggingface.co/stable-ai/LimiX-1_16M/tree/main)     | ✅ 分类 ✅ 回归 ✅ 缺失值插补 |

</div>

<br/>

## 📈 Benchmark 结果

### ➩ 总体 Elo

LimiX-2 在三大基准上均取得最高 Elo，超越所有对比的基础模型与 AutoGluon 1.6。

<div align="center">
  <img src="./doc/figures/fig1_performance_overview.png" alt="图 1：各基准性能总览" width="105%">
  <br>
  <sub>各基准性能总览</sub>
</div>

### ➩ TabArena

在完整的 TabArena 基准上，LimiX-2 在全部四项预测指标上均排名第一，取得 **1935** 的 Elo（取整前高出第二名 TabFM+ 117.4 分）、**3.3%** 的可改进度（improvability）、**5.5** 的平均排名与 **18.9** 的聚合胜场（约为 TabFM+ 的 3.6 倍）。

<div align="center">

| 模型 | Elo ↑ | 可改进↓ | 均排名↓ | 胜场↑ |
| --- | --- | --- | --- | --- |
| **LimiX-2 (D)** | **1935** | **3.3%** | **5.5** | **18.9** |
| TabFM+ | 1818 | 6.2% | 9.0 | 5.3 |
| Causilo (D) | 1790 | 8.9% | 10.1 | 1.7 |
| AutoGluon 1.6 (NC, 4h) | 1789 | 8.6% | 10.1 | 1.2 |
| TabFM (D) | 1774 | 6.5% | 10.7 | 5.9 |
| Mitra-v2 (D) | 1769 | 8.3% | 10.9 | 3.2 |
| EXAONE Tabular (D) | 1749 | 9.5% | 11.8 | 2.9 |
| AutoGluon 1.6 (EX, 4h) | 1738 | 9.2% | 12.3 | 1.2 |
| AutoGluon 1.5 (EX, 4h) | 1648 | 10.2% | 16.9 | 1.3 |
| TabPFN-3 (D) | 1632 | 11.6% | 17.9 | 0.4 |

</div>

在分类（38 个数据集，Table 3）与回归（13 个数据集，Table 4）子集上，LimiX-2 同样四项指标全面第一：

<div align="center">

| 子集 | Elo ↑ | 可改进↓ | 均排名↓ | 胜场↑ | 胜率↑ |
| --- | --- | --- | --- | --- | --- |
| 分类 | **1917** | **4.3%** | **6.0** | **10.6** | **94.5%** |
| 回归 | **2206** | **0.6%** | **3.8** | **8.3** | **96.9%** |

</div>

<div align="center">
  <img src="./doc/figures/fig4_tabarena_elo.png" alt="图 4：TabArena 基准性能" width="100%">
  <br>
  <sub>TabArena 基准性能。
  基线涵盖默认、调优与调优+集成三类配置；LimiX-2 在默认配置下即取得 1935 Elo，优于所有对比基础模型，并超过 4 小时非商业配置的 AutoGluon。</sub>
</div>

<div align="center">
  <img src="./doc/figures/fig5_tabarena_winrate.png" alt="图 5：TabArena 成对胜率" width="66%">
  <br>
  <sub>TabArena 成对胜率</sub>
</div>

### ➩ TALENT

在 TALENT 上，LimiX-2 在全部五个评测类别中均取得最高 Elo，总体 Elo 为 **1506**（高出 TabFM 35 分），可改进度为 **6.75%**，聚合胜场为 **84.3**（约为 TabFM 的 1.7 倍）。

<div align="center">

| 模型 | Elo | 分类 | 回归 | 二分类 | 多分类 | 可改进↓ | 胜场↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LimiX-2** | **1506** | **1475** | **1584** | **1455** | **1520** | **6.75%** | **84.3** |
| TabFM | 1471 | 1449 | 1529 | 1418 | 1517 | 9.17% | 50.1 |
| AutoGluon 1.6 | 1438 | 1379 | 1581 | 1340 | 1465 | 11.21% | 48.0 |
| EXAONE Tabular | 1393 | 1358 | 1477 | 1370 | 1342 | 14.74% | 12.9 |
| TabPFN-3 | 1363 | 1331 | 1441 | 1333 | 1331 | 15.98% | 15.7 |
| LimiX-16M | 1227 | 1210 | 1268 | 1217 | 1198 | 20.52% | 6.1 |
| CatBoost | 1094 | 1094 | 1094 | 1107 | 1073 | 26.84% | 4.9 |
| RandomForest | 1000 | 1000 | 1000 | 1000 | 1000 | 30.06% | 3.8 |

</div>

<div align="center">
  <img src="./doc/figures/fig6_talent_rank.png" alt="图 6：TALENT 平均排名对比" width="76%">
  <br>
  <sub>TALENT 平均排名对比。LimiX-2 在二分类、多分类与回归上均取得最低平均排名（4.62、3.91、3.88）。</sub>
</div>

<div align="center">
  <img src="./doc/figures/fig7_talent_winrate.png" alt="图 7：TALENT 成对胜率" width="66%">
  <br>
  <sub>TALENT 成对胜率</sub>
</div>

### ➩ BCCO

在 BCCO 上，LimiX-2 总体排名第一，Elo 为 **1432**（分别高出 AutoGluon 1.6 (EX, 4h)、TabFM、LimiX-16M 达 56 / 63 / 202 分），可改进度为 **6.97%**，聚合胜场为 **50.4**（约为 TabFM 的 3.0 倍）。

<div align="center">

| 模型 | Elo | 分类 | 回归 | 二分类 | 多分类 | 可改进↓ | 胜场↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LimiX-2** | **1432** | **1321** | **1859** | **1284** | 1414 | **6.97%** | **50.4** |
| AutoGluon 1.6 | 1376 | 1269 | 1782 | 1199 | **1443** | 9.77% | 24.5 |
| TabFM | 1369 | 1260 | 1785 | 1213 | 1374 | 12.24% | 16.7 |
| EXAONE Tabular | 1345 | 1255 | 1689 | 1224 | 1333 | 13.40% | 7.8 |
| TabPFN-3 | 1295 | 1188 | 1691 | 1153 | 1275 | 14.92% | 5.7 |
| LimiX-16M | 1230 | 1195 | 1385 | 1173 | 1252 | 16.94% | 6.1 |
| CatBoost | 1140 | 1101 | 1287 | 1096 | 1118 | 21.64% | 4.6 |
| RandomForest | 1000 | 1000 | 1000 | 1000 | 1000 | 26.75% | 2.3 |

</div>

<div align="center">
  <img src="./doc/figures/fig9_bcco_rank.png" alt="图 9：BCCO 平均排名对比" width="76%">
  <br>
  <sub>BCCO 平均排名对比。LimiX-2 在二分类、多分类与回归上均取得最低平均排名。</sub>
</div>

<div align="center">
  <img src="./doc/figures/fig10_bcco_winrate.png" alt="图 10：BCCO 成对胜率" width="66%">
  <br>
  <sub>BCCO 成对胜率</sub>
</div>

### ➩ 缩放定律

缩放研究评测了 LimiX-2 从 12.5M 到 406.2M 参数的配置，并将拟合的对数线性趋势外推至十亿参数量级。在全部五个评测序列上，下游性能均随模型规模呈清晰的对数线性趋势。

<div align="center">

| 评测任务 | α（Elo@100M） | β（每翻倍 Elo 增益） | R² | RMSE |
| --- | --- | --- | --- | --- |
| TabArena | 1863.88 | 34.68 | 0.9808 | 8.31 |
| TALENT 分类 | 1427.25 | 22.16 | 0.9792 | 5.53 |
| TALENT 回归 | 1545.12 | 18.26 | 0.9680 | 5.69 |
| BCCO 分类 | 1295.86 | 11.24 | 0.9617 | 3.84 |
| BCCO 回归 | 1795.89 | 30.06 | 0.9702 | 9.03 |

</div>

<div align="center">
  <img src="./doc/figures/fig12_scaling_tabarena.png" alt="图 12：TabArena 参数缩放" width="74%">
  <br>
  <sub>TabArena 参数缩放。实线连接实测的 LimiX-2 各规模点，虚线为外推至 2B 参数的对数线性拟合。</sub>
</div>

<div align="center">
  <img src="./doc/figures/fig13_scaling_talent.png" alt="图 13：TALENT 分类与回归参数缩放" width="74%">
  <br>
  <sub>TALENT 分类与回归的参数缩放曲线</sub>
</div>

<div align="center">
  <img src="./doc/figures/fig14_scaling_bcco.png" alt="图 14：BCCO 分类与回归参数缩放" width="74%">
  <br>
  <sub>BCCO 分类与回归的参数缩放曲线</sub>
</div>

<br/>

## ➤ 使用教程

### ➩ 安装

需要 Python >= 3.12，其余 Python 依赖通过 `pip install -e .` 安装（包含 `torch>=2.9.1`）。`torch` / `flash-attn` 需与本机 CUDA 版本匹配，可在步骤 1（可选）中先行安装。

#### 步骤 1（可选）：安装 PyTorch 与 flash-attn

推荐版本（见 `constraints.txt`）：`torch==2.9.1`、`torchvision==0.24.1`、`torchaudio==2.9.1`。请从 [pytorch.org](https://pytorch.org) 安装与 CUDA 匹配的构建，例如：

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

然后从 [flash-attention Releases](https://github.com/Dao-AILab/flash-attention/releases) 下载与本机 Python / CUDA / torch 版本匹配的预编译 wheel（必须与上述 torch 2.9.1 对齐；下面的文件名仅为示例）：

```bash
wget -O flash_attn.whl https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn.whl
```

#### 步骤 2：以可编辑模式安装

```bash
git clone https://github.com/limix-ldm-ai/LimiX.git
cd LimiX
python -m pip install -e .
```

该命令会安装 `LimiX-infer` 命令行入口，并使 `from inference.predictor import LimiXPredictor` / `from limix import LimiXPredictor` 在任意工作目录下均可导入。若步骤 1 中已安装 `torch==2.9.1`，此处会直接复用该构建。

<br/>

## ➤ 推理

LimiX 支持分类、回归与缺失值插补。统一的推理入口为 `inference.predictor.LimiXPredictor`，它会根据 checkpoint 的架构版本路由到 `v1_0` / `v2_0`。请使用与模型匹配的配置（LimiX-2 / V2.0 使用 `*_v2.json`）。

### ➩ 命令行：`LimiX-infer`

执行 `pip install -e .` 后，`LimiX-infer` 与 `python infer.py` 是同一入口。必填参数：`--task_type`、`--data_dir`、`--model_path`。若未指定 `--inference_config_path`，将按任务类型与 checkpoint 版本选择内置的默认配置（V2.0 使用 `*_v2.json`）。

`--data_dir` 为基准数据根目录：每个数据集一个子目录，可参考公开的 [bcco_cls](https://huggingface.co/datasets/stable-ai/bcco_cls) / [bcco_reg](https://huggingface.co/datasets/stable-ai/bcco_reg) 目录结构。缺少所需 CSV 的子目录会被跳过。

```
<data_dir>/
  <dataset_name>/
    <dataset_name>_train.csv
    <dataset_name>_test.csv
    <dataset_name>_0.05.csv    # 仅缺失值插补使用；后缀随 --mask_ratio 变化（默认 0.05）
```

每个 CSV 均含表头，最后一列为目标列（分类数据集为 `label`，回归数据集为回归目标值），其余列为特征。

```bash
LimiX-infer --help
```

分类 / 回归 / 缺失值插补：

```bash
LimiX-infer --task_type Classification --data_dir /path/to/class202512_527 --model_path /path/to/LimiX-2.ckpt --gpuid 0 --save_name limix_cls
LimiX-infer --task_type Regression --data_dir /path/to/reg202512_288 --model_path /path/to/LimiX-2.ckpt --gpuid 0 --save_name limix_reg
LimiX-infer --task_type Feature_imputation --data_dir /path/to/mvi_data --model_path /path/to/LimiX-2.ckpt --gpuid 0 --save_name limix_mvi
```

`--task_type` 也接受别名 `cls` / `reg` / `imputation`。常用可选参数：

<div align="center">

| 参数                      | 说明                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `--inference_config_path` | JSON 配置路径（默认用内置配置）                             |
| `--save_name`             | 结果目录名称                                               |
| `--device`                | `cuda`（默认）或 `cpu`                                     |
| `--gpuid`                 | GPU 编号（默认 `0`）                                       |
| `--gpu_num_per_predictor` | 每个 predictor 的 GPU 数（仅 V2.0）                        |
| `--autobatch`             | 启用自动批处理                                             |
| `--show_progress`         | 显示进度                                                   |
| `--seed`                  | 随机种子                                                   |

</div>

### ➩ 接口说明

#### 模型创建

```python
from inference.predictor import LimiXPredictor

class LimiXPredictor:
    def __init__(self,
                 device: torch.device,
                 model_path: str,
                 inference_config: dict | list | str,
                 mix_precision: bool = True,
                 outlier_remove_std: float = 12,
                 softmax_temperature: float = 0.9,
                 average_before_softmax: bool = True,
                 categorical_features_indices: List[int] | None = None,
                 inference_with_DDP: bool = False,
                 use_data_cache: bool = False,
                 seed: int = 0)
```

<div align="center">

| 参数                         | 数据类型           | 说明                                                       |
| ---------------------------- | ------------------ | ---------------------------------------------------------- |
| device                       | torch.device       | 推理设备，推荐 `cuda`                                      |
| model_path                   | str                | checkpoint 路径                                            |
| inference_config             | dict / list / str  | 推理配置（dict / list / JSON 路径）                        |
| mix_precision                | bool               | 是否混合精度                                               |
| outlier_remove_std           | float              | 异常值裁剪的标准差倍数                                     |
| softmax_temperature          | float              | Softmax 温度（须 > 0）                                     |
| average_before_softmax       | bool               | 分桶回归：softmax 前是否先平均                             |
| categorical_features_indices | list               | 类别列的索引                                               |
| inference_with_DDP           | bool               | 是否启用 DDP（V2.0 保持 False）                            |
| use_data_cache               | bool               | 预处理结果是否落盘缓存                                     |
| seed                         | int                | 随机状态种子                                               |

</div>

#### 预测

```python
def predict(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            task_type: Literal["Classification", "Regression", "Feature_imputation"] = "Classification",
            unique_dataset_name: str | None = None) -> np.ndarray:
```

<div align="center">

| 参数                | 数据类型   | 说明                                                     |
| ------------------- | ---------- | -------------------------------------------------------- |
| x_train             | np.ndarray | 训练特征，形状 `(n_train, n_features)`                   |
| y_train             | np.ndarray | 训练目标，形状 `(n_train,)`                              |
| x_test              | np.ndarray | 查询特征，列需与 `x_train` 对齐                             |
| task_type           | str        | `"Classification"`（默认）、`"Regression"` 或 `"Feature_imputation"` |
| unique_dataset_name | str        | 数据集标识（可选，用于缓存）                             |

</div>

返回值：

- 分类：类别概率，形状 `(n_query, n_classes)`，每行求和为 1
- 回归：预测值，形状 `(n_query,)`（V2.0 直接返回原始目标尺度，无需手动反归一化）
- 缺失值插补：插补后的特征矩阵

### ➩ 推理配置文件

<div align="center">

| 配置文件名                          | 适用模型              | 说明                                   |
| ----------------------------------- | --------------------- | -------------------------------------- |
| cls_default_noretrieval_v2.json     | LimiX-2 / V2.0        | 默认**分类**（不检索）               |
| reg_default_noretrieval_v2.json     | LimiX-2 / V2.0        | 默认**回归**（不检索）               |
| reg_default_noretrieval_MVI_v2.json | LimiX-2 / V2.0        | 默认**缺失值插补**                   |
| cls_default_retrieval.json          | LimiX-16M / LimiX-2M  | 分类 + 检索，精度更高                 |
| cls_default_noretrieval.json        | LimiX-16M / LimiX-2M  | 分类不检索，更快更省显存              |
| reg_default_retrieval.json          | LimiX-16M / LimiX-2M  | 回归 + 检索，精度更高                 |
| reg_default_noretrieval.json        | LimiX-16M / LimiX-2M  | 回归不检索，更快更省显存              |
| reg_default_noretrieval_MVI.json    | LimiX-16M / LimiX-2M  | 缺失值插补                             |

</div>

### ➩ 分类

```python
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
from inference.predictor import LimiXPredictor

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model_file = hf_hub_download(repo_id="stable-ai/LimiX-2", filename="LimiX-2.ckpt", local_dir="./cache")

clf = LimiXPredictor(
    device=torch.device("cuda"),
    model_path=model_file,
    inference_config=os.path.join(ROOT_DIR, "config", "cls_default_noretrieval_v2.json"),
)
prediction = clf.predict(X_train, y_train, X_test, task_type="Classification")

print("roc_auc_score:", roc_auc_score(y_test, prediction[:, 1]))
print("accuracy_score:", accuracy_score(y_test, np.argmax(prediction, axis=1)))
```

完整示例见 [examples/demo_classification.py](./examples/demo_classification.py)

### ➩ 回归

```python
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
from inference.predictor import LimiXPredictor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_path = hf_hub_download(repo_id="stable-ai/LimiX-2", filename="LimiX-2.ckpt", local_dir="./cache")

model = LimiXPredictor(
    device=torch.device("cuda"),
    model_path=model_path,
    inference_config=os.path.join(ROOT_DIR, "config", "reg_default_noretrieval_v2.json"),
)
y_pred = model.predict(X_train, y_train, X_test, task_type="Regression")

rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")
```

完整示例见 [examples/demo_regression.py](./examples/demo_regression.py)

### ➩ 缺失值插补

```python
model = LimiXPredictor(
    device=torch.device("cuda"),
    model_path=model_path,
    inference_config=os.path.join(ROOT_DIR, "config", "reg_default_noretrieval_MVI_v2.json"),
)
reconstructed_X = model.predict(X_train, y_train, x_test_with_nan, task_type="Feature_imputation")
```

完整示例见 [examples/demo_missing_value_imputation.py](./examples/demo_missing_value_imputation.py)

## ➤ 相关链接

- LimiX:Unleashing Structured-Data Modeling Capability for Generalist Intelligence: [Arxiv](https://arxiv.org/abs/2509.03505)
- LimiX 技术报告：[LimiX_Technical_Report.pdf](https://github.com/limix-ldm/LimiX/blob/main/LimiX_Technical_Report.pdf)
- LimiX-2 技术报告：[LimiX_2_Technical_Report.pdf](./LimiX_2_Technical_Report.pdf)
- LimiX 详细使用说明：[访问 Limix 官方文档](https://www.limix.ai/doc/)
- Balance Comprehensive Challenging Omni-domain 分类基准：[BCCO_cls](https://huggingface.co/datasets/stable-ai/bcco_cls)
- Balance Comprehensive Challenging Omni-domain 回归基准：[BCCO_reg](https://huggingface.co/datasets/stable-ai/bcco_reg)

## 📃 许可证

本仓库中的代码采用 Stable AI Technology Co., Ltd. License, Version 1.0 (2026年9月) 授权，该许可证衍生自 Apache License, Version 2.0：其中第 1–9 节沿用了 Apache 2.0 的条款与条件，仅对第 1 节中“License”（许可证）的定义进行了修改，以便纳入第 10 节（关于署名及模型命名的附加要求）的规定。第三方代码受其各自的许可证及署名要求约束；详情请参阅 [LICENSE.txt](./LICENSE.txt)。模型权重采用单独的许可证授权：第三方代码遵循其各自的许可与署名要求，详见 [LICENSE.txt](./LICENSE.txt)。模型权重单独授权:

- LimiX-2: [非商用许可协议](https://huggingface.co/stableai-org/LimiX-2/blob/main/LICENSE)
- LimiX-1-2M: [许可协议](https://huggingface.co/stableai-org/LimiX-1_2M/blob/main/LICENSE)
- LimiX-1-16M: [许可协议](https://huggingface.co/stableai-org/LimiX-1_16M/blob/main/LICENSE)
- LimiX-2M: [许可协议](https://huggingface.co/stableai-org/LimiX-2M/blob/main/LICENSE.txt)
- LimiX-16M: [许可协议](https://huggingface.co/stableai-org/LimiX-16M/blob/main/LICENSE.txt)


## 📝 引用
```bibtex
@misc{zhang2026limix2contextualmechanismnetwork,
      title={LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence}, 
      author={Xingxuan Zhang and Gang Ren and Hao Yuan and Hao Zou and Hongze Tan and Hui Wang and Jianhao Song and Jiansheng Li and Jiayao Zhang and Jinghan Zhang and Kaifang Li and Lang Mo and Li Mao and Mingchao Hao and Nuo Xu and Rui Ding and Ruiji Zhang and Shuyang Li and Siyu Mei and Tianyang Zhang and Weiyang Mu and Yancheng Dong and Yongxian Wei and Yuan Xue and Yuanrui Wang and Yue He and Zijia Yang and Ziyun Li and Dongzhe Li and Fuqiang Wang and Jiandong Liu and Jiawei Chen and Jiaxin Du and Kaijie Cheng and Kehan Li and Lei Sun and Linjun Zhou and Ningbo Dai and Qi Wang and Renzhe Xu and Shaoxing Du and Shumeng Yang and Wang Lu and Wenjing Chu and Xiannan Huang and Xiaoyu Lin and Xing Ai and Xinyan Han and Xuanyue Li and Xuanyue Su and Xukun Zhang and Yan Lu and Yaxin Zhang and Yi Qin and Yifei Huang and Yihan Xu and Yongle Lv and Yuanyuan Jiang and Yushan Han and Peng Cui},
      year={2026},
      eprint={2609.17488},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.17488}, 
}

@article{wang2026limix,
  title={LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models},
  author={Wang, Yuanrui and Zhang, Xingxuan and Yu, Han and Hao, Mingchao and Ren, Gang and Yuan, Hao and Mao, Li and Zhang, Yunjia and Yuan, Chun and Cui, Peng},
  journal={arXiv preprint arXiv:2606.04485},
  year={2026}
}

@article{zhang2025limix,
  title={Limix: Unleashing structured-data modeling capability for generalist intelligence},
  author={Zhang, Xingxuan and Ren, Gang and Yu, Han and Yuan, Hao and Wang, Hui and Li, Jiansheng and Wu, Jiayun and Mo, Lang and Mao, Li and Hao, Mingchao and others},
  journal={arXiv preprint arXiv:2509.03505},
  year={2025}
}
```
