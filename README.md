# 工业表面异常检测与定位 / Industrial Surface Anomaly Inspection

基于 PyTorch 的工业表面视觉检测项目：先用 MVTec bottle 验证流程，再迁移到 Severstal 钢材表面缺陷。当前完成瓶子检测与定位、钢材数据检查和固定划分；钢材模型尚未训练或建库。

A PyTorch project for industrial surface anomaly detection and localization. MVTec bottle validates the pipeline; Severstal steel is the target application. Bottle experiments and steel data preparation are complete. Steel modeling is pending.

## 进展 / Progress

| 阶段 / Stage | 状态 / Status |
|---|---|
| GPU 环境与 ResNet18 特征提取 / GPU setup and feature extraction | 已完成 / Complete |
| bottle 全局特征基线 / Global-feature baseline | 已完成 / Complete |
| bottle 局部特征与热力图 / Local features and heatmaps | 已完成 / Complete |
| 钢材 RLE 检查与固定划分 / Steel RLE audit and fixed split | 已完成 / Complete |
| 钢材滑窗与异常检测 / Steel tiling and anomaly detection | 待实现 / Planned |
| 监督分割与实际采集验证 / Supervised segmentation and field validation | 待实现 / Planned |

## 方法 / Methods

- 全局基线：冻结 ResNet18，512 维特征 L2 单位化，计算到正常特征库的最近邻距离。
  Global baseline: frozen ResNet18, unit-normalized 512-D embeddings, nearest-normal-neighbor distance.
- 局部方法：layer2/layer3 局部聚合、投影贪心 coreset（1%）、局部最近邻异常图。
  Local method: layer2/layer3 aggregation, projected greedy coreset (1%), and patch nearest-neighbor maps.
- 局部方法是 PatchCore 简化实现，不是原论文标准复现，详见阶段报告。
  This is a simplified PatchCore-style implementation, not a standard paper reproduction; see the stage report.
- 正常特征库只使用正常训练图。图像报警阈值使用正常验证图分数的第 95 百分位。
  Memory banks use normal training images only. Image thresholds use the normal-validation 95th percentile.

## 结果与限制 / Results and limitations

| bottle 指标 / Metric | 全局 / Global | 局部 / Local |
|---|---:|---:|
| Image AUROC | 0.992857 | 1.000000 |
| 缺陷检出 / Defects detected | 63/63 | 63/63 |
| 正常误报 / Normal false alarms | 2/20 | 1/20 |
| Pixel AUROC at 256×256 | — | 0.982241 |

以上为固定划分、单类别的探索性结果，不代表真实产线效果。像素指标在 256×256 下计算。热力图不是概率，也不是精确二值边界。测试结果已在开发中查看，后续研究需独立数据检验泛化。

These are exploratory single-category results, not production validation. Pixel metrics use 256×256 masks. Heatmaps are not probabilities or precise binary boundaries. Test results have been inspected during development; independent data is needed for generalization claims.

## 环境 / Environment

实测 / Tested: Windows, Python 3.12, RTX 5060 Laptop 8 GB, PyTorch 2.15.0+cu132, TorchVision 0.29.2+cu132. 实际兼容版本应结合驱动和 PyTorch 官方发布情况选择。 / Select compatible releases for your driver when recreating the environment.

```powershell
conda create -n surface-inspection python=3.12
conda activate surface-inspection
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
python -m pip install numpy==2.5.5 Pillow==13.0.0 scikit-learn==1.9.1
```

所有命令在克隆后的仓库根目录执行，不需要原作者电脑的绝对路径。
Run all commands from the cloned repository root; no author-specific absolute paths are required.

## 数据与权重 / Data and weights

数据、预训练权重、运行输出和本机环境快照不上传。请从原发布方下载并遵守条款，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
Datasets, pretrained weights, run outputs, and machine snapshots are excluded. Download separately under the original terms; see [THIRD_PARTY.md](THIRD_PARTY.md).

```text
data/
  mvtec_ad/bottle/       # contains train/, test/, ground_truth/; extra bottle/ nesting supported
  severstal/
    train.csv
    train_images/       # JPGs directly under severstal/ are also supported
weights/
  resnet18-f37072fd.pth
```

```powershell
Invoke-WebRequest -Uri 'https://download.pytorch.org/models/resnet18-f37072fd.pth' -OutFile 'weights/resnet18-f37072fd.pth'
python scripts/check_environment.py
python scripts/extract_features.py
python -m unittest discover -s tests -v
```

## bottle 复现 / Reproduce bottle

```powershell
python scripts/prepare_bottle.py
python scripts/run_baseline.py
python scripts/run_patchcore.py
```

原实验目录有额外 bottle/bottle 层。`prepare_bottle.py` 会重新生成适合实际路径的清单。保持相同文件和 seed 可复现划分；脚本修改原始数据时会拒绝后续校验。
The original experiment used an extra bottle/bottle layer. Regenerate the manifest for your actual layout. The same files and seed reproduce the split; downstream hash checks reject changed image data.

单张推理 / Single-image inference (after building the bank):

```powershell
python scripts/run_patchcore.py --image assets/synthetic_inspection_sample.png
```

上述合成图仅验证接口，不应当作有效瓶子检测实验。换成真实、存在的图片路径；任意手机照片可能不符合特征库的拍摄分布。
The synthetic fixture only checks the interface. Replace it with an existing real image path; arbitrary phone photos may be out of distribution.

结果位于 / Outputs: `runs/global_baseline/`, `runs/patchcore_lite/`. 局部方法生成 / Local method generates: `localization_preview.png`, `single_heatmap.png` (single-image mode), metrics, predictions, and raw anomaly maps.

## 钢材数据 / Steel data

```powershell
python scripts/audit_severstal.py
python scripts/split_severstal.py
```

| 集合 / Split | 全部 / Total | 正常 / Normal | 缺陷 / Defect |
|---|---:|---:|---:|
| Train | 8800 | 4132 | 4668 |
| Validation | 1886 | 885 | 1001 |
| Test | 1882 | 885 | 997 |

按原图、缺陷类别组合分层，seed=42，约 70/15/15；罕见组合受整数分配限制。每张图仅进入一个集合，后续所有切块必须继承原图集合。仅 `use_for_memory_bank=true` 的正常训练图可加入正常库。
Image-level class-combination stratification, seed 42, approximately 70/15/15; rare strata have rounding limitations. All future tiles must inherit their source split. Only normal training samples marked `use_for_memory_bank=true` may enter the memory bank.

固定清单在 `configs/`；已有清单若与新划分不同，脚本拒绝覆盖。无生产批次 ID；文件哈希不重复不代表不存在近重复或同批次相关性。
Frozen manifests are in `configs/`; changed assignments are rejected. No production batch IDs are supplied, so exact-file checks cannot establish batch independence or rule out near duplicates.

## 项目结构 / Structure

- `src/`: 特征与评分 / feature extraction and scoring.
- `scripts/`: 检查、建库、推理、评估 / audits, memory banks, inference, evaluation.
- `tests/`: 距离、掩码、划分检查 / distance, mask, and split tests.
- `configs/`: 固定样本身份、标签与校验值 / fixed sample identities, labels, hashes.
- `reports/`: 阶段报告与实验指标 / stage reports and metrics (historical detailed reports are in Chinese).
- `assets/`: 人工合成测试图 / synthetic fixture.

下一步 / Next: implement detail-preserving steel sliding windows and map stitching, then steel anomaly experiments and a supervised segmentation baseline.
