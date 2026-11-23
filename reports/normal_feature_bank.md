# 钢材正常特征库 / Normal steel feature memory

日期 / Date: 2026-11-23

## 已完成结果 / Completed artifact

正常特征库已完成并通过独立进程验证，状态为`ready`。两种子复验先完成，详见[复验报告](class_balanced_replication.md)；随后构建本库，未根据复验成绩改选编码器。

The bank is complete and independently verified with status `ready`. Complete seeds 43/44 first (linked replication report), then build this bank without changing the predeclared encoder based on their scores.

| 项目 / Item | Result |
|---|---|
| Normal training images extracted | 4132 |
| Raw sampled candidates | 1586688 |
| Shard representatives before final selection | 17408 |
| Final bank shape | 4096 × 384, float32 |
| Final represented normal parents | 1359 |
| Self-contained artifact | `runs/steel_normal_bank/bank.pt`, 17695977 bytes (about 16.9 MiB) |
| Feature extraction/selection time | 93.0 s |
| Extraction, calibration, evaluation and internal checks | 118.5 s |
| Peak PyTorch allocated GPU memory | 198.8 MiB (excludes driver/other processes) |
| Normal-calibration q95 image threshold | 3.619108 |
| Independent verification | Passed |

来源、模型与特征哈希通过；8个入库向量从8张原图重新提取核对通过；独立进程重新加载、检索及256×1600异常图检查通过。从项目以外的工作目录运行PowerShell入口，验证与单图推理均成功。47项测试通过，包括来源排除、坐标回放、独立权重保存/加载及向量/来源校验。

Source/model/vector hashes pass. Re-extract eight bank vectors from eight source images, replay a score in a separate process and check the 256×1600 anomaly map. Launch verification and prediction from outside the project directory successfully. All 47 tests pass, including source exclusion, coordinate replay, standalone weights and artifact integrity.

### 当前检测表现 / Current detection performance

| Same 200 development images | Result |
|---|---:|
| Image AUROC | 0.8403 |
| Defective images detected /100 | 57 |
| Defective images missed /100 | 43 |
| Normal false alarms /100 | 10 |
| Mean per-defective-image pixel AUROC | 0.8657 |

这证明库可加载、检索和产生结果，不代表精度已经合格。仍有43%缺陷漏报及10%正常图误报，尚不可部署。不能将此结果称为比已有监督分类器更好的新最优模型。

The artifact loads, retrieves and produces outputs, but accuracy remains inadequate: 43% missed defects and 10% false alarms. It is not deployable or established as a new best model over the supervised classifier.

固定示例`034941f9d.jpg`的实际分数5.174236，大于3.619108阈值，输出报警1与1600×256叠加图。它是固定功能演示，不替代上述全部评估图的成绩。

The fixed example scores 5.174236 versus threshold 3.619108, yielding alarm 1 and a 1600×256 overlay. It demonstrates functionality, not aggregate accuracy.

完整数值与来源信息见[metrics](steel_normal_bank_metrics.json)、[独立验证 / verification](steel_normal_bank_verification.json)、[fixed protocol](../configs/steel_normal_bank_protocol.json)和[固定图片身份 / selections](../configs/steel_normal_bank_selection.json)。本轮按用户要求完成正常库后停止，不继续启动新的模型训练。

Full metrics, independent verification, protocol and image selections are linked above. Stop this run after the requested bank is complete; do not start another training experiment.

## 方法与数据 / Method and data

正常特征库用于记录正常钢材的局部外观，检测时计算待测特征到库中正常特征的最近邻距离，产生异常分数及定位热力图。建库包括提取、筛选、保存与校准，不是再训练一个分类头。

The memory represents local normal-steel appearance. Nearest-normal-feature distances produce image anomaly scores and localization maps. Building it means extraction, selection, serialization and calibration rather than training another classifier head.

使用预先固定的类别平衡seed=42钢材模型的编码器，仅取ResNet18的layer2/layer3局部特征，3×3均值聚合后对齐并拼接为384维。编码器被冻结，库中向量只来自4132张正常train图片，原CSV中这些图片均无缺陷标注；val及test不贡献库向量。

Freeze the predeclared class-balanced seed-42 encoder. Use ResNet18 layer2/layer3 local features, 3×3 average aggregation and spatial alignment, concatenated to 384 dimensions. Memory vectors come only from all 4132 normal training images, independently checked against original CSV annotations. Validation/test images contribute no memory vectors.

编码器此前使用正常和缺陷图及掩码进行监督训练，因此整个方法不是“只用正常数据学习”的无监督方法。种子43、44复验不用于挑选建库编码器，避免按评估结果挑最好种子。

The encoder was previously trained with normal/defective images and masks, so the whole method is not unsupervised normal-only learning. Seeds 43/44 are not used to select the memory encoder.

输入保留1600×256，按256×256、步长128切成12块，不做224中心裁剪。每块特征图32×32，从中固定种子抽32个位置；每张正常图384个候选，共1586688个候选。每256张原图为一批，先保留最多1024个投影贪心coreset向量，再合并各批保留4096个最终向量。这是两级近似选择，不是标准PatchCore论文复现。

Keep 1600×256 input resolution: twelve 256×256 tiles, stride 128, no 224 center crop. Each 32×32 feature grid contributes 32 seeded random locations, 384 candidates/image and 1586688 candidates overall. Select up to 1024 projected-greedy coreset vectors per 256-parent shard, then 4096 from the merged shard sets. This is two-stage approximate selection, not standard PatchCore reproduction.

最终库是压缩表示，不保证4132张原图每张都有最终向量。文件记录每个向量的原图、分块编号及特征图坐标，并将特征提取器权重一并保存，加载与检测不依赖原U-Net权重路径。

Compression does not guarantee a final representative from every source image. Store each vector's parent, tile and feature-grid coordinates together with the extractor weights, so loading/inference does not depend on the original U-Net checkpoint path.

## 保存、续建和验证 / Persistence, resume and verification

- 分批筛选结果带模型/数据/代码指纹和张量哈希，保留本机。中断后重新执行建库可复用匹配且完整的分批结果，不一次将全部候选塞进显存。
- Cache shard selections locally with model/data/code fingerprints and tensor hashes; rerunning resumes matching intact shards without placing all candidates in GPU memory.
- 完整库原子写入 `runs/steel_normal_bank/bank.pt`，包含4096×384特征、来源坐标、提取器、阈值及校验信息。
- Atomically save `runs/steel_normal_bank/bank.pt` with 4096×384 vectors, provenance, extractor, thresholds and checksums.
- 用原200张正常val校准图确定图像距离分数q95阈值；原200张开发评估图只用于报告，1882张test保持保留。
- Calibrate image thresholds at q95 of the original 200 normal validation images. Reuse the 200 development-evaluation images for reporting; all 1882 test images stay reserved.
- 重载后检查模型/向量/来源哈希，并从原图重新提取8个最终向量核对，核对一张开发图重载前后的分数；独立进程再次验证。
- Reload and check extractor/vector/provenance hashes; re-extract eight selected vectors from original source images and replay one development score, then verify in another process.

## 使用 / Use

项目根目录已配置Python环境下 / In the configured Python environment at project root:

```powershell
# 建库或从完整分批结果继续 / Build or resume intact shard selections
python scripts/build_steel_normal_bank.py
# 已完成库的独立重载与来源检查 / Independently verify the completed artifact
python scripts/build_steel_normal_bank.py --verify
# 使用本地已有示例图片，或替换为实际路径 / Existing example or an actual image path
python scripts/build_steel_normal_bank.py --image data/severstal/034941f9d.jpg
```

PowerShell入口会自动定位项目根目录与本机py312环境，因此不需要先定义 `$projectPython`。从其他目录运行时，使用入口脚本的实际完整路径。

The PowerShell launcher resolves project root and the local py312 interpreter automatically. From another directory, use the launcher's actual absolute path.

```powershell
.\scripts\run_normal_bank.ps1 -Mode verify
.\scripts\run_normal_bank.ps1 -Mode predict -Image 'data/severstal/034941f9d.jpg'
```

输出在 `runs/steel_normal_bank/`：`bank.pt`为正常库，`single_overlay.png`为叠加图，`single_map.npy`为原分辨率距离图，`single_prediction.json`为分数/阈值/报警记录。输出完整路径会显示在PowerShell入口中。特征图是距离而非缺陷概率，不将其擅自转换成已校准的二值缺陷边界。

Outputs: `bank.pt`, `single_overlay.png`, original-resolution `single_map.npy` and `single_prediction.json`. The launcher prints the output directory. Maps are distances, not defect probabilities or calibrated binary defect boundaries.

## 结果口径与限制 / Interpretation and limits

库完成与检测精度达标是两个结论。独立加载、检索和来源验证证明文件可用，不能证明适用于任意手机照片或实际产线。原开发图已经在多个阶段被查看；建库编码器、正常图数量、筛选方式和库大小也与早期pilot不同，不能把性能差异全部归因于单一改动。

Artifact readiness and adequate detection accuracy are separate conclusions. Loading/retrieval/provenance checks establish usability, not arbitrary-phone-image or production performance. Development images were reused, and encoder, normal-image count, selection method and bank size differ from the early pilot, preventing single-factor causal attribution.

像素AUROC按100张缺陷图分别计算后取均值，不是全图像素合并AUROC，也不包括正常图。没有使用标签选择像素二值阈值，因此不报告与监督分割可直接比较的Dice。数据、权重、分批缓存和照片保留本机，版本库更新代码、固定选择、数值与中英文说明。

Pixel AUROC is the mean of separate defective-image AUROCs, not pooled pixels or normal images. No labeled binary pixel-threshold calibration is performed, so do not compare a fabricated Dice with supervised segmentation. Data, weights, shard caches and photographs stay local; code, selections, metrics and bilingual documentation are versioned.
