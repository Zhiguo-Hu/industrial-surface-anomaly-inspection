# 类别覆盖与预训练编码器对照 / Coverage and pretraining ablations

日期 / Date: 2026-11-23

## 实验设计 / Design

保留原始模型，新增三个独立实验。所有实验的检查点验证（80 张）、正常校准（200 张）、开发评估（200 张）身份相同；没有访问测试图片。每个版本按相同规则重新选择检查点、像素阈值和图像阈值，而非直接复用不同模型的数值阈值。

Three new runs preserve the original model. The 80 checkpoint-validation, 200 normal-calibration, and 200 evaluation identities are unchanged. Test images are untouched. Each version independently applies the same checkpoint/threshold selection rules; numerical thresholds are model-specific.

- 覆盖实验保留完全相同的 250 张正常图；250 张缺陷图保证每类至少出现于 60 张原图。保留原 500 张中的 406 张，替换 94 张缺陷图。
  Coverage intervention keeps the same 250 normal images and selects 250 defective images with at least 60 images/class. It retains 406/500 original identities.
- 增加结构相同的 ResNet18-U-Net 随机编码器和预训练编码器对照。解码器初始参数完全相同，编码器均参与训练。
  A matched ResNet18-U-Net pair differs only in random versus ImageNet-initialized encoder state. Initial decoder weights are identical, and encoders are trainable.
- 损失 BCE+soft Dice、AdamW lr=0.001、每批 4 正+4 负块、有放回抽样、同步翻转、AMP、480 次优化步骤预算不变。六轮不等于六个完整 epoch。
  BCE+soft Dice, AdamW lr=0.001, 4 positive/4 negative tiles, replacement sampling, paired flips, AMP and 480 optimization-step budget are held fixed. Six rounds are not six exhaustive epochs.

## 类别覆盖 / Class coverage

| Class ID | 原训练图 / Original | 新训练图 / New | 新集合含该类块 / Tiles | 实际采样次数 / Draws |
|---|---:|---:|---:|---:|
| 1 | 43 | 60 | 197 | 341 |
| 2 | 9 | 60 | 121 | 209 |
| 3 | 188 | 108 | 526 | 938 |
| 4 | 27 | 60 | 289 | 505 |

多类图/块有重复计数，父图覆盖不代表像素或采样次数完全平衡。两个 ResNet18 版本使用完全相同的训练身份与实际类别采样次数。

Multi-class images/tiles overlap. Parent-image coverage does not fully balance pixel counts or update exposure. Both encoder variants have identical training identities and actual class-tile draws.

## 开发验证结果 / Development results

| 方法 / Method | Image AUROC | TP / 100 defects | FP / 100 normal | Pooled Dice | Pooled IoU |
|---|---:|---:|---:|---:|---:|
| 原小 U-Net / Original small U-Net | 0.7204 | 25 | 7 | 0.2877 | 0.1680 |
| 类别覆盖 / Coverage-balanced small U-Net | 0.7204 | 15 | 8 | 0.2883 | 0.1684 |
| ResNet18 随机 / Random encoder | 0.6541 | 5 | 2 | 0.2750 | 0.1594 |
| ResNet18 预训练 / Pretrained encoder | 0.6121 | 10 | 3 | 0.3443 | 0.2079 |

Dice/IoU 汇总全部 200 张完整 256×1600 图片，包括正常图误报像素。图像分数仍为概率图最大值，掩码未按图像报警门控。阈值和检查点始终只从开发验证/正常校准数据选择。

Dice/IoU pool all 200 full-resolution strips, including normal false-positive pixels. Image scores are probability maxima; masks are not gated by alarms. Evaluation labels do not choose thresholds or checkpoints.

## 结论与限制 / Findings and limits

1. 类别覆盖对稀有类定位有帮助，但不是整体修复：第 2 类像素召回 2.49%→22.57%，整体 Dice 基本不变，报警检出下降。第 3 类覆盖减少是此次取舍的一部分。
   Coverage helps class-2 pixel recall (2.49%→22.57%) but does not fix overall detection. Reduced class-3 exposure is part of the tradeoff.
2. 预训练相对匹配随机编码器提高 Dice（0.2750→0.3443）、IoU（0.1594→0.2079），但图像 AUROC 下降（0.6541→0.6121）。不能说全面胜出。
   Pretraining improves Dice/IoU over the matched random encoder, but lowers image AUROC. It is not uniformly superior.
3. 更大结构并不保证更好。两个 ResNet18-U-Net 各约 1246 万参数，原小 U-Net 约 49 万；随机大模型未优于小模型。
   A larger architecture alone does not help: ~12.46M parameters versus ~0.49M do not establish superiority.
4. 预训练版正常图仍有 91/100 非空掩码、正常像素误报 6.03%（原版 4.81%）。第 4 类像素召回升至 55.16%，第 1 类则降至 21.13%。不存在单一全面最佳版本。
   The pretrained variant still produces nonempty masks in 91/100 normal images and normal-pixel FPR 6.03% (original 4.81%). Class-4 recall improves to 55.16%, class 1 falls to 21.13%.
5. 仅一个种子、短训练预算、共享学习率，不代表各模型最优。按 Dice 选检查点也不保证最佳整图报警。相同更新预算不等于相同计算量。未证明预训练无用或类别覆盖无用。
   One seed, short budget and shared learning rate are not per-model optimal settings. Dice-selected checkpoints need not optimize image alarms. Equal steps are not equal compute; these results do not show pretraining or coverage is useless.

## 验证与后续 / Verification and next steps

28 项测试通过，包括类别配额、预算、身份不泄漏、解码器初始化一致、预训练编码器加载和梯度检查。全部训练的检查点重载通过，预训练版单图入口也已运行。比较脚本核对验证身份、损失、优化器、预算、平衡版训练身份及编码器对照的类别采样次数。

All 28 tests pass, including selection quotas, budgets, identity isolation, identical decoder initialization, pretrained encoder loading and gradient checks. Checkpoint reloads and pretrained single-image inference pass. The comparison script verifies experimental controls.

保留原小 U-Net 作为整图参考、预训练版作为分割参考，均不宣称可部署。下一阶段先定义图像报警与掩码误报的联合评价，再在开发集验证区域聚合评分或分类分支；不继续盲目堆模型或调整最终测试集阈值。

Keep the original small U-Net as the image-level reference and the pretrained version as the segmentation reference; neither is deployable. Next jointly evaluate alarms and mask false positives, then investigate region-based scores or a classification branch on development data. Reserve the test split until the final configuration.

## 复现 / Reproduce

```powershell
python scripts/train_steel_unet.py --coverage-balanced
python scripts/train_steel_unet.py --coverage-balanced --encoder random-resnet18
python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18
python scripts/analyze_unet.py --coverage-balanced
python scripts/analyze_unet.py --coverage-balanced --encoder pretrained-resnet18
python scripts/compare_unet_experiments.py
python -m unittest discover -s tests -v
```

权重与预览分别保存在各自 `runs/steel_unet_balanced*` 目录，不上传。选择清单、指标、诊断和对照记录在 `configs/`、`reports/`。未覆盖原版结果。

Weights and dataset previews remain local in separate `runs/steel_unet_balanced*` folders. Selections, metrics, diagnostics and comparisons are versioned in `configs/` and `reports/`. Original results are preserved.
