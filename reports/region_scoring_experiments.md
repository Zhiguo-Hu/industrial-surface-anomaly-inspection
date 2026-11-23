# 区域评分与掩码后处理 / Region scoring and mask cleanup

日期 / Date: 2026-11-23

## 设计 / Design

固定原轻量 U-Net 和预训练 ResNet18-U-Net，模型文件 SHA256 在实验前后完全一致。每个模型使用原 80 张检查点开发图选择规则、200 张正常图校准阈值，再在原 100 正常+100 缺陷图上评估。三组不重叠，全部来自验证划分；没有读取测试图片。

Freeze the original small U-Net and pretrained ResNet18-U-Net. Checkpoint SHA256 is unchanged before/after. Each model uses the original 80 development images for rule selection, 200 normal calibration images for thresholds, and the original 100 normal/100 defective evaluation images. Groups are disjoint validation subsets; no test images are accessed.

先固定模型原像素阈值（小 U-Net 为 0.4、预训练为 0.5），按 8 邻域提取连通区域。最小区域面积仅在 {0,16,64,256} 原图像素中，根据开发图 pooled Dice 选择；两模型均选择 256。不是对概率图做膨胀、缩放或补造像素。

Keep original pixel thresholds (0.4 small, 0.5 pretrained), use 8-connected regions, and select minimum area from {0,16,64,256} original-resolution pixels by development pooled Dice. Both select 256. No dilation or fabricated pixels are applied.

比较四种整图分数：

- 原概率图最大值 / Original maximum probability.
- 原概率图最高 0.1%（本图 410 个像素）的均值 / Mean of the top 0.1% original probabilities (410 pixels).
- 过滤后的掩码面积占整图比例 / Cleaned-mask area fraction.
- 过滤后单个连通区域的最大概率总和除以整图像素数 / Largest retained component probability mass divided by image area.

前两种分数使用原概率图，后两种使用过滤后的区域。分数选取规则为开发集 AUROC 最高；每种分数的图像阈值单独用正常校准集第 95 百分位确定。规则与阈值在读取评估图片前保存。开发集已用于原检查点选择，所以属于探索性开发实验，不是独立验证。

The first two scores use raw probabilities; the others use cleaned regions. Select score by development AUROC and calibrate each score's threshold at normal q95. Save rules before reading evaluation images. Development images were previously used for checkpoint selection, so findings are exploratory, not independent validation.

## 整图结果 / Image-level results

| 模型 / Model | Score | Evaluation AUROC | TP/100 | FP/100 |
|---|---|---:|---:|---:|
| Small U-Net | Max probability | 0.7204 | 25 | 7 |
| Small U-Net | Top 0.1% mean (**开发集选定 / selected on dev**) | 0.7252 | 22 | 6 |
| Small U-Net | Mask area fraction | 0.4970 | 0 | 7 |
| Small U-Net | Largest component mass | 0.5487 | 5 | 5 |
| Pretrained U-Net | Max probability (**开发集选定 / selected on dev**) | 0.6121 | 10 | 3 |
| Pretrained U-Net | Top 0.1% mean | 0.5911 | 9 | 4 |
| Pretrained U-Net | Mask area fraction | 0.4531 | 4 | 13 |
| Pretrained U-Net | Largest component mass | 0.4956 | 8 | 7 |

轻量版 AUROC 的 0.0048 差异不能据单一种子小样本宣称显著提升；在所定阈值下召回反而下降。预训练版不支持改为这些区域评分。没有在评估集上重新挑一个表现较好的分数当作最终规则。

The small model's AUROC change of 0.0048 is not established as significant, and recall decreases at the calibrated threshold. The pretrained model provides no support for replacing max probability with these candidates. No evaluation-selected winner is deployed.

## 掩码结果 / Mask results

| 模型 / Model | 处理 / Processing | Pooled Dice | 正常图非空掩码 / Nonempty normal masks | 正常像素误报 / Normal-pixel FPR |
|---|---|---:|---:|---:|
| Small | Raw | 0.2877 | 95/100 | 4.81% |
| Small | Minimum area 256 | 0.2875 | 92/100 | 4.69% |
| Pretrained | Raw | 0.3443 | 91/100 | 6.03% |
| Pretrained | Minimum area 256 | 0.3448 | 82/100 | 5.98% |

小区域过滤只减少少量零散误报，无法去掉较大的错误预测。Dice 几乎不变，轻量版略降。小缺陷也可能被误删，因此不能无限增大面积阈值来追求干净展示。原阈值下整图报警率和非空掩码率仍是两个不同指标。

Small-region filtering removes some isolated predictions but not large false-positive regions. Dice barely changes and slightly decreases for the small model. Filtering can erase real small defects; increasing the area threshold without validation is not justified. Image-alarm rate and nonempty-mask rate remain different metrics.

## 结论与下一步 / Conclusion and next step

当前后处理不能修复模型的主要区分能力问题。较大正常纹理也会形成高概率连通区域，所以“面积大/概率总量大”并不能稳定表示缺陷。本实验不证明所有区域算法无效，但不支持在现有模型上继续堆这些简单规则。

These postprocessing rules do not fix the main discrimination problem. Normal textures can produce large high-probability regions, so area or mass is not consistently defect-specific. This does not rule out all region methods, but does not support stacking these simple heuristics on current models.

保留原评分和原模型作为参考，不覆盖它们；新规则仅作为可复现实验保存。下一阶段应转回训练数据覆盖与训练预算，增加代表性困难正常样本和各类缺陷，固定验证规则做新对照。最终方案确定后才使用保留测试集。

Keep original models/scoring as references; new rules are experimental artifacts. Next return to representative hard-normal/defect coverage and training budget, with fixed validation rules. Reserve the test split until the configuration is fixed.

## 复现 / Reproduce

```powershell
python scripts/evaluate_region_scoring.py --model small
python scripts/evaluate_region_scoring.py --model pretrained
python scripts/evaluate_region_scoring.py --model pretrained --image data/severstal/034941f9d.jpg
python -m unittest discover -s tests -v
```

规则位于 `configs/*_region_rule.json`，报告位于 `reports/*_region_metrics.json`，本机逐图分数/预览位于 `runs/*_regions/`。单图模式验证检查点哈希，输出的掩码不按报警门控，避免隐藏误报或漏检。

Rules are in `configs/*_region_rule.json`, reports in `reports/*_region_metrics.json`, and local per-image scores/previews in `runs/*_regions/`. Single-image mode checks checkpoint hashes and does not gate masks by alarms.

31 项测试通过，包括对角连通、区域移除、概率总和、空掩码和非法输入。两模型的原最大概率评分与历史 AUROC/混淆矩阵均完全复现，权重保持不变。

All 31 tests pass, covering diagonal connectivity, removal, component mass, empty masks and invalid input. Original max-score AUROC/confusion matrices are reproduced and both checkpoints remain unchanged.
