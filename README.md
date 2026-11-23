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
| 钢材滑窗与拼接 / Steel tiling and stitching | 已完成 / Complete |
| 钢材异常检测首轮试验 / Steel anomaly pilot | 已跑通，效果不合格 / Runs; inadequate detection |
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

## 钢材滑窗与拼接 / Steel tiling and stitching

```powershell
python scripts/check_steel_tiling.py
python -m unittest discover -s tests -v
```

每张 1600×256 原图按 256×256、步长 128 切为 12 块，最后一块起点为 x=1344，确保完整覆盖。直接裁切，不缩放原图。小块继承原图 split，不再随机划分；正常训练子集共有 49,584 个可读取小块。

Each 1600×256 image yields twelve 256×256 crops at stride 128, with the final start anchored at x=1344. No image resizing is applied. Tiles inherit the source split; 49,584 normal-training tiles are available on demand.

`src/severstal_tiles.py` 提供 PyTorch Dataset，返回归一化图片、掩码、原图身份、坐标和有效区域。掩码按原图坐标裁切。只缓存最近一张原图和掩码，不保存全部切块。当前顺序读取更高效；全局随机小块读取会重复解码原图，训练时应考虑按原图组织采样。

The Dataset returns normalized RGB, mask, parent identity, coordinates, and valid region. It caches only the most recent source image and mask. Sequential access avoids repeated image decoding; future training should consider image-grouped sampling.

`src/tiling.py` 的 Stitcher 对重叠区域取平均，拒绝缺失覆盖和非有限预测。小图使用边缘填充，掩码补零；拼接时丢弃填充区域。模型输出的低分辨率异常图须先上采样到小块尺寸后再拼接。

Stitcher averages overlapping scores and rejects uncovered pixels or nonfinite values. Small RGB images use edge padding, masks use zero padding, and padding is discarded on stitching. Upsample model maps to tile resolution before adding them.

已通过 16 项测试；另用 5 张正常/各类缺陷训练图检查 RGB 与掩码精确重建，验证 DataLoader 输出 `[8,3,256,256]`。详细记录见 `reports/steel_tiling_checks.json`，本机预览为 `runs/steel_tiling/tiling_preview.png`。预览不上传数据集图片。

All 16 tests pass. Five real training examples covering normal and all defect classes reconstruct RGB and masks exactly; DataLoader shape is `[8,3,256,256]`. The report is `reports/steel_tiling_checks.json`; the local preview is `runs/steel_tiling/tiling_preview.png` and is not distributed.

本阶段仅验证分块几何与数据读取，尚未运行钢材模型。切块与重叠拼接不能保证模型对接缝完全不敏感，后续应单独检查边缘缺陷和接缝响应。

This stage validates geometry and loading, not steel detection performance. Overlap averaging does not guarantee seam-free model predictions; boundary defects and seam responses require later evaluation.

下一步 / Next: steel anomaly experiments with bounded memory, followed by a supervised segmentation baseline.

## 钢材首轮实验 / Steel pilot (2026-11-23)

```powershell
python scripts/run_steel_pilot.py
python -m unittest discover -s tests -v
```

固定选择 300 张正常训练图、200 张正常验证图校准阈值，以及另外 100 张正常+100 张缺陷验证图作开发评估。选择清单在 `configs/steel_pilot_selection.json`，彼此不重叠，不读取测试图片。

Fixed selection: 300 normal training images, 200 normal validation images for threshold calibration, and a disjoint validation evaluation subset of 100 normal and 100 defective images. Identities are in `configs/steel_pilot_selection.json`. No test images are accessed.

每块随机抽取 32 个局部特征，产生 115200 个候选，再投影贪心选取 2048 个原始 384 维特征。模型保持冻结。每张图的分数是滑窗异常图平滑、拼接后的最大值；阈值取正常校准图分数的第 95 百分位。

Sample 32 local features per tile, yielding 115200 candidates, then select 2048 original 384-D descriptors by projected greedy coreset. The frozen model scores each image by the maximum stitched smoothed anomaly map. Threshold calibration uses the normal-score 95th percentile.

| 开发验证指标 / Development-validation metric | 结果 / Result |
|---|---:|
| Image AUROC | 0.5205 |
| 缺陷检出 / Defects detected | 3/100 |
| 正常误报 / Normal false alarms | 11/100 |
| 缺陷图逐图像素 AUROC 的均值 / Mean per-defective-image pixel AUROC | 0.930568 |

**检测效果不合格，不能作为可用钢材检测器。** 最后一项只衡量每张缺陷图内部的像素排序，是 100 张缺陷图各自 AUROC 的均值，不是跨图汇总像素 AUROC，也不包含正常图片。因此不能据此认为检测成功或与瓶子像素指标直接比较。

**Detection is inadequate; this is not a usable steel detector.** The last metric averages within-image pixel AUROC across 100 defective images. It is not pooled pixel AUROC and excludes normal images, so it cannot establish successful image detection or be directly compared to bottle's pixel metric.

实测峰值 PyTorch 显存约 225.5 MiB、建库约 6.1 秒、全流程约 26.4 秒（含 I/O、校准和验证，不是标准延迟基准）。17 项测试通过，保存重载和单图入口均验证。详情在 `reports/steel_pilot_metrics.json`。

Measured peak PyTorch allocation: ~225.5 MiB. Bank construction: ~6.1 s; complete pipeline: ~26.4 s including I/O, calibration, and validation (not a latency benchmark). All 17 tests pass; bank reloading and single-image inference are verified. See `reports/steel_pilot_metrics.json`.

输出 / Outputs: `runs/steel_pilot/bank.pt`, `validation_predictions.csv`, `validation_preview.png`, `metrics.json`. 单图模式 / Single-image mode (after bank construction):

```powershell
python scripts/run_steel_pilot.py --image data/severstal/034941f9d.jpg
```

该图为本轮缺陷验证图，当前模型漏检；保留它用于展示失败，不作为成功演示。生成 `single_overlay.png`、`single_map.npy`、`single_prediction.json`。不同目录布局时替换为存在的路径。

This is a defective validation example missed by the pilot, intentionally retained as a failure case. Outputs are `single_overlay.png`, `single_map.npy`, and `single_prediction.json`; adjust the path if the data layout differs.

下一步分析正常/缺陷分数重叠、边缘与接缝响应、候选抽样和纹理差异，再选择受控改动或监督分割对照。上述原因目前是待验证假设，不能作为已确认结论。钢材测试集保留用于最终配置评价。

Next: investigate score overlap, boundary/seam responses, candidate sampling, and texture variation, then run controlled changes or supervised segmentation. These are hypotheses, not confirmed causes. The steel test split remains reserved for final evaluation.
