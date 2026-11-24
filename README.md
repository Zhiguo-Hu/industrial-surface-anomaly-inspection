# 工业表面异常检测与定位 / Industrial Surface Anomaly Inspection

基于 PyTorch 的工业表面视觉检测项目。2026-11-24已断点恢复并完成全幅检测第二阶段8轮训练、BN统计校正对照、窗口/掩码/类别融合及两阶段组合。当前研究组合在986张资格验证图上召回92.87%（651/701）、正常误报7.02%（20/285），**未达到召回≥95%、正常误报≤5%的门槛，不能称为可部署成品**。62项测试、独立进程重载与全分辨率输出检查通过；原1882张最终test仍封存。详见[中英文训练验证报告](reports/strip_refinement_results.md)和[完整数值](reports/strip_refinement_summary.json)。

A PyTorch surface-inspection project. Resumed and completed eight refinement epochs on2026-11-24, plus BN controls, feature fusion and a frozen two-stage ensemble. Its qualification recall is92.87% (651/701), normal FPR7.02% (20/285): **below the recall≥95% / FPR≤5% acceptance target; not deployable**. All62 tests, independent-process reload and full-resolution output checks pass. Original test1882 remains locked. See the [bilingual report](reports/strip_refinement_results.md) and [complete metrics](reports/strip_refinement_summary.json).

## 进展 / Progress

| 阶段 / Stage | 状态 / Status |
|---|---|
| GPU 环境与 ResNet18 特征提取 / GPU setup and feature extraction | 已完成 / Complete |
| bottle 全局特征基线 / Global-feature baseline | 已完成 / Complete |
| bottle 局部特征与热力图 / Local features and heatmaps | 已完成 / Complete |
| 钢材 RLE 检查与固定划分 / Steel RLE audit and fixed split | 已完成 / Complete |
| 钢材滑窗与拼接 / Steel tiling and stitching | 已完成 / Complete |
| 钢材异常检测首轮试验 / Steel anomaly pilot | 已跑通，效果不合格 / Runs; inadequate detection |
| 轻量 U-Net 监督分割对照 / Supervised U-Net pilot | 已完成首轮，效果仍不足 / Initial run complete, inadequate |
| 预训练编码器与训练扩展 / Pretraining and training expansion | 已完成开发实验，未达部署要求 / Development runs complete, not deployable |
| 本机确定性与3种子复验 / Local determinism and three seeds | 80步训练状态及单图跨进程一致性通过 / 80-step state and single-input replay verified |
| 显式分类监督对照 / Explicit classification supervision | seed=42整图报警改善，定位与小缺陷仍不足 / Better seed-42 alarms; localization and small defects remain weak |
| 联合模型三种子复验 / Three-seed joint replication | 分类排序均改善，定位均值未改善 / Better ranking in all paired seeds; no mean localization gain |
| 稀有类别与小缺陷抽样 / Rare-class and small-tile sampling | 类别平衡改善本轮小面积检出，强小分块偏好失败 / Better small-area detection with class balance; strong small bias fails |
| 类别平衡复验 / Class-balanced replication | 类别2收益复现，整体误报与定位退步 / Class-2 gains replicate; overall alarms/localization regress |
| 完整正常特征库 / Normal steel feature memory | 已建成，独立加载/来源/检索通过 / Built; independent reload, provenance and retrieval verified |
| 实际采集验证 / Field validation | 待实现 / Planned |

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

### 失败诊断已完成 / Failure diagnosis completed

详见 [钢材失败分析 / Steel failure analysis](reports/steel_failure_analysis.md)。在固定校准和验证集上，排除边缘、改用分位数/相对分数、将特征库扩大四倍，均未解决图像级识别失败。缺陷图中 74/100 的最高分位于标注内，说明“图内定位”与“跨图异常判断”存在明显差异。测试集未使用，原模型未修改。下一步优先做监督分割对照。

Border exclusion, quantile/relative scoring, and a fourfold larger bank did not resolve image-level detection. Peaks lie inside annotations in 74/100 defective images, showing the distinction between within-image localization and cross-image detection. Test images remain untouched and the original bank is unchanged. Prioritize a supervised segmentation baseline next.

```powershell
python scripts/diagnose_steel.py
python scripts/diagnose_steel_capacity.py
```

## 轻量 U-Net 监督训练 / Supervised lightweight U-Net

```powershell
python scripts/train_steel_unet.py
python scripts/train_steel_unet.py --image data/severstal/034941f9d.jpg
python -m unittest discover -s tests -v
```

SmallUNet 从零训练，488001 参数，GroupNorm，BCE+soft Dice，AdamW，CUDA 混合精度。保持原图划分、256×256 切块与重叠平均。4 类掩码合并为二值缺陷。固定选择 250 张正常+250 张缺陷训练图，索引为 1210 个含缺陷块和 4790 个无缺陷块。每批各采样 4 个，有放回；图片与掩码同步翻转。

SmallUNet is trained from scratch: 488001 parameters, GroupNorm, BCE plus soft Dice, AdamW, CUDA mixed precision. Original-image splits, 256×256 tiles, and overlap averaging are preserved. Four classes are merged into one binary defect mask. A fixed 250-normal/250-defective parent-image pool yields 1210 positive and 4790 negative tiles. Each batch samples four of each with replacement and paired flips.

本轮训练 6 轮、每轮 80 次更新，共 480 次；“轮”不是遍历全部训练图的 epoch。80 张独立开发验证图用于按固定 0.5 阈值的 pooled Dice 选检查点，并用于选择像素阈值。第 5 轮最佳，像素阈值 0.4；图像阈值由原 200 张正常校准图确定。最后在与异常基线相同的 100 正常+100 缺陷验证图上评估，不读取测试图片。

Six rounds of 80 updates give 480 updates, not six full dataset epochs. Eighty separate development-validation images select the checkpoint by pooled Dice at threshold 0.5, then select the pixel threshold. Round 5 is best, pixel threshold 0.4. The same 200 normal calibration images determine the image threshold. Evaluation uses the original 100 normal and 100 defective validation images; no test images are read.

| 相同评估子集 / Same evaluation subset | 正常特征库 / Memory bank | U-Net |
|---|---:|---:|
| Image AUROC | 0.5205 | 0.7204 |
| 缺陷检出 / Detected | 3/100 | 25/100 |
| 正常误报 / False alarms | 11/100 | 7/100 |

U-Net 全部 200 张图的 pooled pixel Dice=0.2877、IoU=0.1680，缺陷图逐图 Dice 均值=0.4304。不能把不同定义的 Dice 混用。逐缺陷图像素 AUROC 均值=0.9143，低于之前 0.9306；监督方法没有在所有指标上同时提升。仍存在粗糙正常纹理误报、小缺陷漏检和大面积缺陷分割不足，不能用于产线。

U-Net pooled pixel Dice over all 200 images is 0.2877 and IoU is 0.1680; mean defective-image Dice is 0.4304. These are different aggregations. Mean per-defective-image pixel AUROC is 0.9143, below the earlier 0.9306, so not all metrics improve. Rough normal textures, small defects, and large defect regions remain failure modes; this is not production-ready.

两种方法使用不同监督信息与训练样本规模，不能将差异全部归因于架构。21 项测试通过，保存重载与单图推理通过。权重位于 `runs/steel_unet/best.pt`（本机保存，不上传 GitHub）；输出还包括验证预测、预览、metrics.json。训练曲线数值保存在 `reports/steel_unet_metrics.json` 的 history，固定选择在 `configs/steel_unet_selection.json`。

Supervision and training-set sizes differ, so this is not an architecture-only comparison. All 21 tests pass; checkpoint reload and single-image inference are verified. Weights remain local at `runs/steel_unet/best.pt`, alongside validation predictions, previews, and metrics. Numeric learning history is in `reports/steel_unet_metrics.json`; selected identities are in `configs/steel_unet_selection.json`.

实测训练与后续校准评估约 74 秒（不含前面的切块索引）、峰值 PyTorch 分配显存约 789 MiB。后续优先做类别/缺陷面积分组评估，检查训练样本覆盖与损失取舍，再决定扩大训练规模或使用预训练编码器。不是直接盲目增加训练轮数。

Training plus calibration/evaluation took ~74 seconds excluding initial tile indexing, with ~789 MiB peak PyTorch allocation. Next, analyze class/defect-area failures and training coverage before expanding training or using a pretrained encoder.

### U-Net 分组诊断 / U-Net grouped diagnosis

```powershell
python scripts/analyze_unet.py
```

详见 [类别与面积分析 / Class and area analysis](reports/steel_unet_failure_analysis.md)。第 2 类训练子集仅 9 张，验证 0/6 检出，类别像素召回约 2.49%。≥5% 大面积缺陷像素召回也只有 27.80%，不能仅归因于小缺陷。正常图虽然仅 7/100 报警，但 95/100 在像素阈值下有非空掩码；“图像误报率”与“分割误报”必须分别报告。

Class 2 has only 9 training images, 0/6 validation alarms and 2.49% class-pixel recall. Large defects (≥5% union area) have only 27.80% pixel recall. Although 7/100 normal images alarm, 95/100 have nonempty masks at the pixel threshold. Image alarms and segmentation false positives are distinct metrics.

本阶段未更新模型、阈值或测试集。24 项测试通过。下一步优先做类别覆盖/抽样的单变量对照，不同时改动架构、损失或训练预算。

Model and thresholds are unchanged, test data remains untouched, and all 24 tests pass. Next prioritize a single-variable class-coverage/sampling experiment without changing architecture, loss or training budget.

下一步分析正常/缺陷分数重叠、边缘与接缝响应、候选抽样和纹理差异，再选择受控改动或监督分割对照。上述原因目前是待验证假设，不能作为已确认结论。钢材测试集保留用于最终配置评价。

Next: investigate score overlap, boundary/seam responses, candidate sampling, and texture variation, then run controlled changes or supervised segmentation. These are hypotheses, not confirmed causes. The steel test split remains reserved for final evaluation.

## 类别覆盖与预训练对照 / Coverage and pretraining experiments

已完成三组新增训练，详见 [中英文对照报告 / Bilingual experiment report](reports/unet_controlled_experiments.md)。保持原数据划分、检查点/校准/评估身份、损失与 480 次优化步骤预算；不使用测试图片。

Three additional runs are complete. Original splits, validation identities, loss and 480-step budget are held fixed; test images remain untouched.

| 方法 / Method | Image AUROC | TP/100 | FP/100 | Pooled Dice |
|---|---:|---:|---:|---:|
| 原小 U-Net / Original | 0.7204 | 25 | 7 | 0.2877 |
| 类别覆盖 / Balanced coverage | 0.7204 | 15 | 8 | 0.2883 |
| ResNet18 随机 / Random encoder | 0.6541 | 5 | 2 | 0.2750 |
| ResNet18 预训练 / Pretrained encoder | 0.6121 | 10 | 3 | 0.3443 |

```powershell
python scripts/train_steel_unet.py --coverage-balanced
python scripts/train_steel_unet.py --coverage-balanced --encoder random-resnet18
python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18
python scripts/compare_unet_experiments.py
```

平衡版仍为 250 正常+250 缺陷，正常图完全不变，每个缺陷类别至少 60 张（多类图有重叠）。预训练与随机编码器结构相同、解码器初始参数相同，所有层参与训练。各实验输出独立保存为 `runs/steel_unet_balanced*`，对应清单和指标也分别保存。

Coverage-balanced runs retain the same 250 normal images and 250 defective images with at least 60 examples/class (overlapping classes). Matched random/pretrained encoders share architecture and initial decoder weights, with all layers trainable. Separate `runs/steel_unet_balanced*` outputs and matching selections/metrics preserve the original run.

单图推理 / Single-image inference:

```powershell
python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18 --image data/severstal/034941f9d.jpg
```

28 项测试通过。预训练提高本轮分割重叠，但未提升整体报警；类别覆盖改善第 2 类像素召回，但不保证总体改善。单一种子、短预算结果不能说明复杂模型或预训练必然更好/无用。下一阶段优先验证整图评分与区域级误报，测试集保持保留。

All 28 tests pass. Pretraining improves overlap but not overall alarms in this run. Better coverage helps class-2 pixel recall but not aggregate detection. Single-seed short-budget results do not establish universal superiority or futility. Next investigate image scoring and region-level false positives; keep test data reserved.

## 区域评分对照 / Region-scoring experiments

详见 [中英文报告 / Bilingual report](reports/region_scoring_experiments.md)。固定现有两模型，在原开发集选择区域过滤及评分规则，正常校准集确定阈值，评估子集不参与选择，测试集未使用。

Frozen-model experiments select cleanup/scoring on development images, thresholds on normal calibration, and evaluate the unchanged subset. No test images are used.

```powershell
python scripts/evaluate_region_scoring.py --model small
python scripts/evaluate_region_scoring.py --model pretrained
```

轻量模型开发集选定 top 0.1% 均值：评估 AUROC 0.7252，但检出降至 22/100（原为25/100），正常误报6/100（原7/100）。预训练模型仍选原最大概率，检出10/100、误报3/100。连通区域面积/概率总量未解决问题。

The small model selects top-0.1% mean: evaluation AUROC 0.7252, but recall falls to 22/100 (original 25/100), false alarms 6/100 (original 7/100). Pretrained selects original max probability: TP=10/100, FP=3/100. Component-area/mass rules do not resolve detection.

过滤小于256像素的连通区域后，正常非空掩码仍为92/100（轻量）和82/100（预训练），Dice基本不变。31项测试通过，检查点哈希保持不变。新规则单独保存在 `configs/*_region_rule.json`；原模型/默认入口不变。

Filtering regions below 256 pixels leaves nonempty masks in 92/100 small-model and 82/100 pretrained normal images; Dice barely changes. All 31 tests pass and checkpoint hashes are unchanged. New rules are separate experimental artifacts; original model/default entry points are preserved.

单图实验入口 / Single-image experimental entry:

```powershell
python scripts/evaluate_region_scoring.py --model pretrained --image data/severstal/034941f9d.jpg
```

输出在 `runs/*_regions/`，掩码不按整图报警门控。下一步优先改善训练覆盖和有效训练预算，而不是继续堆叠后处理。

Outputs are in `runs/*_regions/`; masks are not gated by image alarms. Prioritize training coverage and effective training budget over further heuristic stacking.

## 扩展训练 / Expanded training

完整方法与限制见[中英文报告 / Bilingual report](reports/training_scale_experiments.md)。原池500张扩展到4000张（2000正常、2000缺陷，均来自train且包含原500张），对照480与2400次优化步骤。模型、损失、验证身份与规则不变，测试集未使用。

The train pool grows from 500 to 4000 images (2000 normal/2000 defective, retaining the original pool). Compare 480 and 2400 steps with the same model, loss and validation rules; no test images are used.

| Pretrained ResNet18-U-Net | Image AUROC | TP/100 | FP/100 | Pooled Dice | Pooled IoU |
|---|---:|---:|---:|---:|---:|
| 500 images / 480 steps | 0.6121 | 10 | 3 | 0.3443 | 0.2079 |
| 4000 images / 480 steps | 0.5150 | 0 | 0 | 0.3657 | 0.2238 |
| 4000 images / 2400 steps | 0.7926 | 23 | 6 | 0.5218 | 0.3530 |

```powershell
python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded
python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30
python scripts/analyze_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30
python scripts/compare_training_scale.py
```

单图入口在同样参数后加 `--image` 和实际路径。扩展版权重与历史分别保存在各自 `runs/*_expanded*` 目录，不覆盖原版；每轮记录历史和最佳训练权重，不能将该简化训练快照当成完整优化器断点恢复。

Append `--image` and an existing path for inference. Expanded weights/history have separate `runs/*_expanded*` directories. Per-round best-weight snapshots do not include full optimizer state for resuming training.

长版本正常像素误报降至1.50%，但正常非空掩码仍94/100，整图召回仅23%；尚不可部署。32项测试通过。单种子且GPU训练未强制确定性，两次扩展实验的早期轨迹并非逐点一致；只能作为开发证据，不能宣称严格逐位复现或统计显著提升。

The longer run reduces normal-pixel FPR to 1.50% but still has nonempty masks in 94/100 normal images and only 23% image recall. All 32 tests pass. Single-seed GPU runs were not forced deterministic and early histories differ, so results are development evidence, not bitwise reproduction or statistical significance.

学习曲线 / Learning curves (optional matplotlib):

```powershell
python -m pip install -r requirements-plots.txt
python scripts/plot_training_scale.py
```

![Training-scale development curves](reports/steel_training_scale_curves.png)

## 确定性与多种子 / Determinism and predefined seeds

完整方法与全部结果见[中英文复验报告 / Bilingual replication report](reports/deterministic_replication.md)。使用原4000张训练图和相同开发/校准/评估身份，每个预定种子训练2400步；不选评估成绩最好的种子。

Use the same 4000 training images and all original validation identities; each predefined seed trains 2400 steps. Do not select the best evaluation seed.

| Seed | Image AUROC | TP/100 | FP/100 | Pooled Dice |
|---|---:|---:|---:|---:|
| 42 | 0.7759 | 6 | 5 | 0.5096 |
| 43 | 0.8413 | 49 | 9 | 0.5168 |
| 44 | 0.8310 | 44 | 6 | 0.5446 |

开启严格确定性算法，关闭benchmark/TF32，CUDA初始化前设置cuBLAS工作区；训练随机流与数据选择分离。相同种子的初始状态、前80步后全部参数/缓冲区及首轮记录完全一致；一次单图概率图在两个独立进程逐像素一致。不是重复了两次完整2400步，也不保证跨软硬件复现。

Strict deterministic algorithms, benchmark/TF32 disabled, cuBLAS workspace configured before CUDA, and training RNG isolated from data selection. Same-seed initial state, 80-step parameter/buffer state and first-round history match exactly; one probability map matches across separate processes. This is not two complete 2400-step repeats or a cross-runtime guarantee.

```powershell
python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 1 --deterministic --seed 42
foreach ($trainingSeed in @(42,43,44)) {
  python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30 --deterministic --seed $trainingSeed
  python scripts/analyze_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30 --deterministic --seed $trainingSeed
}
python scripts/check_deterministic_inference.py
python scripts/summarize_reproducibility.py
python scripts/plot_multiseed.py
```

35项测试通过。三次共用200张评估图，不是600个独立样本；标准差只描述训练波动。严格模式也改变了数值设置和训练随机流，不能与历史版本作纯粹“去掉随机性”的因果比较。下一步以预先指定seed=42为参考，检验显式分类监督，继续保留最终测试集。

All 35 tests pass. Runs reuse 200 images, not 600 independent examples; SD describes training variation only. Strict mode changes arithmetic and RNG handling, so historical differences are not solely attributable to removing nondeterminism. Follow-up controls use predeclared seed 42 and investigate explicit classification supervision; test data stays reserved.

![Three-seed development curves](reports/steel_multiseed_curves.png)

## 显式分类监督 / Explicit classification supervision

完整结构、对照、失败分析与单图入口见[中英文报告 / Bilingual report](reports/joint_classification_experiments.md)。同一联合结构、同一初始状态、4000张训练图、确定性seed=42、2400次更新尝试，仅改变分类损失权重0→0.5。联合组AMP跳过3次，实际更新2397次；对照为2400次。

The matched joint architecture, initial state, 4000-image pool, deterministic seed 42 and 2400 attempted updates are fixed; change classification-loss weight from 0 to 0.5. Joint AMP skips three steps (2397 successful updates); control completes 2400.

| Scoring method | Image AUROC | TP/100 | FP/100 | Raw pooled Dice |
|---|---:|---:|---:|---:|
| Segmentation-only, max pixel | 0.7759 | 6 | 5 | 0.5096 |
| Joint training, max pixel | 0.8163 | 27 | 6 | 0.5124 |
| Joint training, classification head | 0.8801 | 53 | 3 | 0.5124 |

后两行为同一模型、同一原始掩码的不同报警分数。分类分支从每个256×256分块学习是否含缺陷，整图分数取12块最大值；阈值在200张正常校准图上独立确定。默认掩码不按分类结果清空。探索性门控虽使Dice变成0.5414，却删除约14%的原本正确定位像素，缺陷像素召回由50.81%降至43.70%。

The final two rows use one model and shared raw masks. The head learns tile defect presence; image scores take the maximum across 12 tiles, independently calibrated on 200 normal images. Default masks remain ungated. Exploratory gating raises Dice to 0.5414 but removes about 14% of raw true-positive pixels and reduces defect-pixel recall from 50.81% to 43.70%.

```powershell
foreach ($clsLossWeight in @(0,0.5)) {
  python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30 --deterministic --seed 42 --classification-weight $clsLossWeight
  if ($LASTEXITCODE -ne 0) { break }
}
python scripts/analyze_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30 --deterministic --seed 42 --classification-weight 0.5
python scripts/compare_joint_training.py
python scripts/plot_joint_training.py
```

38项测试通过。权重0对照完全重现历史seed=42的30轮损失/开发Dice记录和最佳分割状态哈希；两组分类及分割分数重加载检查通过。此轮仍是单种子开发结果；类别2仅检出1/6、小面积组仅6/26。下一步先在seed=43、44复验分类收益，再做稀有类别/小缺陷分块抽样对照，测试集继续保留。

All 38 tests pass. The zero-weight control exactly reproduces historical seed-42 30-round loss/development-Dice records and best segmentation state hash; head and mask-score reload checks pass. This is single-seed development evidence: class 2 detects only 1/6, small-area group 6/26. Next replicate on seeds 43/44, then examine rare-class/small-defect tile sampling; keep test data reserved.

![Joint training curves](reports/steel_joint_training_curves.png)

## 联合模型三种子复验 / Joint three-seed replication

详见[中英文报告 / Bilingual report](reports/joint_multiseed_replication.md)和[完整数值 / Numerical audit](reports/steel_joint_multiseed.json)。保持分类损失权重0.5与原训练/验证规则，新增训练43、44，保留已有42；不挑最好种子。三次共用200张开发评估图，不是600个独立样本，最终测试集未使用。

Keep classification weight 0.5 and the original protocol; newly train seeds 43/44 and retain 42. No best-seed selection. Runs reuse 200 development images, not 600 independent samples; no final test images are used.

| Seed | Head AUROC | TP/100 | FP/100 | Raw Dice |
|---|---:|---:|---:|---:|
| 42 | 0.8801 | 53 | 3 | 0.5124 |
| 43 | 0.8868 | 41 | 3 | 0.5287 |
| 44 | 0.8590 | 58 | 4 | 0.5193 |

分类AUROC均高于各自基线，正常误报均减少，但seed=43检出由49降到41。分类AUROC均值0.8753±0.0145（原0.8161±0.0352）；分割Dice均值0.5201±0.0081（原0.5237±0.0185）。不能用排序改善宣称召回和定位均改善。

All paired AUROCs improve and false alarms decrease, but seed-43 detections fall from 49 to 41. Mean±sample SD classification AUROC is 0.8753±0.0145 (baseline 0.8161±0.0352); raw Dice is 0.5201±0.0081 (baseline 0.5237±0.0185). Better ranking does not establish uniformly better recall or localization.

43/44复用历史分割基线，仅42有同结构分类权重0对照；全部种子的分割初始状态、数据清单、实际类别抽样及确定性设置核对通过。联合实际更新分别2397/2396/2397次，均为2400次尝试。38项测试、CSV数值核对、各模型单图重加载检查通过。

Seeds 43/44 reuse historical segmentation baselines; only 42 has a same-architecture zero-weight control. Initialization, identities, actual class draws and deterministic settings pass audits. Joint runs complete 2397/2396/2397 updates from 2400 attempts each. All 38 tests, CSV checks and one-image checkpoint reload checks pass.

```powershell
# 已有本地训练结果时 / With existing local training artifacts
python scripts/summarize_joint_multiseed.py
python scripts/plot_joint_multiseed.py
```

仍有29张缺陷图所有联合种子都漏检，其中15张位于0.1%–1%小面积组。默认不做分类门控；seed=43门控会删除53.51%的原始真阳性像素。下一步固定seed=42与联合模型，优先检验稀有类别/小缺陷分块抽样，不直接改选最好种子或扩大模型。

All joint seeds miss 29 common defective images, including 15 in the 0.1%–1% area group. Default masks stay ungated: seed-43 gating removes 53.51% of raw true-positive pixels. Next keep seed 42/joint protocol and test rare-class/small-defect tile sampling instead of selecting the best seed or enlarging the model.

![Joint-model paired seeds](reports/steel_joint_multiseed.png)

## 稀有类别与小分块抽样 / Rare-class and small-tile sampling

详见[中英文报告 / Bilingual report](reports/tile_sampling_experiments.md)、[对照数值 / Comparison](reports/steel_tile_sampling_comparison.json)及[分类头诊断 / Head diagnosis](reports/steel_sampling_head_diagnosis.json)。固定seed=42联合模型、权重0.5、原4000张训练图及开发规则，对比均匀正分块、每类一个正分块，以及带训练供给保护的小分块偏好。三组负分块和翻转序列一致，2400次更新尝试。

Fix the seed-42 joint model, weight 0.5, original 4000-image pool and development rules. Compare uniform positives, one positive per numeric class, and small-tile preference with train-supply safeguards. Negative tiles/flips match across arms; each full run has 2400 attempted updates.

| Policy | Head AUROC | TP/100 | FP/100 | Raw Dice | Small-area TP/26 |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.8801 | 53 | 3 | 0.5124 | 6 |
| Class-balanced | 0.8975 | 63 | 5 | 0.5188 | 19 |
| Class + small | 0.5000 | 0 | 0 | 0.4353 | 0 |

类别平衡让类别2检出从1/6到4/6、该类像素召回约35%到73%，但正常像素误报从1.55%到2.02%，≥5%大面积组检出22/32到19/32。强小分块偏好让所有200张图分数相同；固定检查的240块分类隐层全为0，不能把其0误报当作成功。默认掩码不门控。

Class balancing raises class-2 detections from 1/6 to 4/6 and pixel recall from about 35% to 73%, but normal-pixel FPR rises from 1.55% to 2.02% and large-area detections fall from 22/32 to 19/32. Strong small preference produces constant scores on all 200 images and zero hidden activations on 240 inspected tiles. Its zero false alarms are not success. Default masks stay ungated.

43项测试通过；原均匀流程80步状态/记录复现，完整抽样回放、检查点重加载与CSV指标核对通过。训练供给检查发现类别2只有2个小块、来自2图，所以小分块偏好要求至少20块/10原图，不足时回退。所有选择仅用train；本轮单种子开发比较不代表最终泛化。

All 43 tests pass; original 80-step training replay, full sampling replay, reload and CSV checks pass. Class 2 has only two small tiles from two parents, so small preference requires 20 tiles/ten parents or falls back. Pools use train only; this single-seed development experiment is not final generalization testing.

已有本地训练结果后 / With local artifacts available:

```powershell
python scripts/check_sampling_prefix.py
python scripts/compare_tile_sampling.py
python scripts/diagnose_sampling_heads.py
python scripts/plot_tile_sampling.py
```

下一步先在预定seed=43、44复验类别平衡候选；保留完整失败结果，不直接采用当前强小分块偏好。完整训练与单图参数见报告。

Next replicate the class-balanced candidate on predefined seeds 43/44. Preserve failure artifacts and reject the current strong small-tile preference. See the report for training/inference flags.

![Tile sampling controls](reports/steel_tile_sampling.png)

## 类别平衡复验 / Class-balanced replication

新增seed=43、44后，不能复现seed=42的整体改善。三种子配对全表及控制见[中英文复验报告](reports/class_balanced_replication.md)和[完整数值](reports/steel_balanced_multiseed.json)。类别平衡平均AUROC 0.8544±0.0401、Dice 0.4736±0.0417；均匀联合参照分别0.8753±0.0145、0.5201±0.0081。三次共用200张开发图，不是600个独立样本。

New seeds 43/44 do not reproduce the seed-42 overall gain. Class-balanced mean±sample SD AUROC is 0.8544±0.0401 and Dice 0.4736±0.0417; uniform references are 0.8753±0.0145 and 0.5201±0.0081. See the linked paired report/JSON. Runs reuse 200 development images, not 600 independent samples.

```powershell
python scripts/summarize_balanced_replication.py
```

固定数据、初值、预算及负分块/翻转的核对通过；最终test未使用。正常特征库按预先指定的seed=42继续，不按复验结果另选最好种子。

Data, initial state, budget and negative/flip audits pass; no final test images are used. The normal bank keeps predeclared seed 42 rather than selecting a best replication seed.

## 正常特征库已完成 / Normal memory ready

完整方法、文件位置与限制见[中英文报告 / Bilingual report](reports/normal_feature_bank.md)。使用事先指定的类别平衡seed=42编码器提取4132张正常train图，1586688个候选经两级筛选保留4096×384向量；最终向量代表1359张正常原图。编码器曾接受缺陷掩码监督，因此是“正常向量库＋领域监督编码器”，不是完全无监督学习。

Use the predeclared class-balanced seed-42 encoder and all 4132 normal train images: 1586688 candidates become 4096×384 vectors by two-stage selection, representing 1359 normal parents. The encoder previously used defect-mask supervision, so this combines normal-only vectors with a supervised domain extractor.

库文件为本地 `runs/steel_normal_bank/bank.pt`（约16.9 MiB），自带提取器权重。47项测试通过，正常来源/校验和/8个向量原图回放/独立进程分数重载通过，PowerShell入口已从项目以外目录验证。数据、缓存、库权重及照片保留本机，GitHub保存代码、清单、数值与说明。

The local `runs/steel_normal_bank/bank.pt` (about 16.9 MiB) includes extractor weights. All 47 tests pass, along with source/checksum audits, eight source-vector replays, independent-process score reload and launcher tests outside project root. Data, caches, bank weights and photos stay local; code, selections, metrics and docs are versioned.

```powershell
# 项目根目录，已有库直接验证/检测 / At project root, verify or predict with the completed bank
.\scripts\run_normal_bank.ps1 -Mode verify
.\scripts\run_normal_bank.ps1 -Mode predict -Image 'data/severstal/034941f9d.jpg'
# 重新构建或续建 / Rebuild or resume matching intact shards
.\scripts\run_normal_bank.ps1 -Mode build
```

从其他目录运行时，将入口脚本和图片替换为实际完整路径。原Python入口为 `scripts/build_steel_normal_bank.py --verify` 或 `--image`。检测会写出 `single_overlay.png`、`single_map.npy`、`single_prediction.json` 到库目录，不需要原U-Net检查点路径。

From another working directory, use actual absolute script/image paths. Python also supports `scripts/build_steel_normal_bank.py --verify` or `--image`. Predictions write overlay, distance-map and JSON files to the bank directory without needing the original U-Net checkpoint path.

正常库在同一200张开发图上AUROC 0.8403、检出57/100、正常误报10/100；平均逐缺陷图像素AUROC 0.8657。它已可用但未达到生产精度；热力图是距离，不是缺陷概率或已校准的二值分割。最终test未使用，本轮按用户要求在库完成后停止。

Bank development AUROC is 0.8403, TP=57/100 and FP=10/100; mean per-defective-image pixel AUROC is 0.8657. Usable artifact, inadequate production accuracy. Maps are distances, not probabilities or calibrated masks. No final test images; stop this run after completing the requested bank.
