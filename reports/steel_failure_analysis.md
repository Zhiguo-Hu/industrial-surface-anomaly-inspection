# 钢材首轮失败分析 / Steel pilot failure analysis

日期 / Date: 2026-11-23

## 范围 / Scope

使用原有冻结模型、原始 2048 特征库，同样的 200 张正常校准图与 200 张开发验证图。未使用测试图片，未更改已保存模型或阈值。另做 8192 特征库的单变量容量实验，仅用于开发诊断。

The frozen model and original 2048-vector bank were evaluated on the same 200 normal calibration and 200 development-validation images. No test images were accessed; saved pilot parameters are unchanged. An 8192-vector capacity ablation is diagnostic only.

## 已确认的问题 / Established findings

### 1. 跨图片的异常分数缺乏区分力 / Cross-image scores lack discrimination

| 原始最大分数 / Original peak score | 正常 / Normal | 缺陷 / Defective |
|---|---:|---:|
| 第 5 百分位 / q05 | 2.0431 | 2.0945 |
| 中位数 / Median | 2.3624 | 2.3645 |
| 第 95 百分位 / q95 | 3.0253 | 2.7426 |

两组中位数几乎相同，正常图高分尾部甚至更高；所以正常校准阈值 2.8006 漏检绝大多数缺陷。不是只差一个合适阈值：在已查看的验证标签上反推约 90% 召回率，需要阈值约 2.1214，此时正常误报约 80%。这仅用于说明不可分离，不是推荐部署阈值。

The score medians nearly coincide and the normal high-score tail is larger. A normal-calibrated threshold of 2.8006 misses most defects. Retrospectively targeting 90% validation recall requires ~2.1214 and yields 80% normal false positives; this is diagnostic, not a deployment threshold.

### 2. 图内定位信息仍然存在 / Within-image localization signal remains

100 张缺陷图中，74 张最高分点位于官方标注内；100 张图的缺陷区域中位分数均高于本图背景中位分数。缺陷图逐图像素 AUROC 的均值为 0.930568。

For 74/100 defective images the maximum lies inside the annotation; all 100 have higher defect-region median than background median. Mean per-defective-image pixel AUROC is 0.930568.

这些数字说明局部特征并非完全无效，但不意味着绝对分数能跨产品表面比较，也不表示精确边界已恢复。该像素指标不包括正常图，不能替代整图异常识别或跨图汇总像素指标。

Local information is not absent, but its absolute magnitude does not separate normal and defective images. The pixel statistic excludes normal images and does not establish precise segmentation or pooled pixel performance.

### 3. 外圈边缘有偏高现象，但不是主要解释 / Border concentration is not the primary explanation

16 像素外圈占图像面积 14.25%，却包含 32% 正常图和 26% 缺陷图的最高分点，提示应继续检查边缘。但排除外圈后、重新用同一正常校准集定阈值，AUROC 仅从 0.5205 变为 0.5241，仍然 TP=3、FP=11。

The outer 16-pixel border is 14.25% of area but contains peaks in 32% of normal and 26% of defective images. Excluding it barely changes AUROC (0.5205 to 0.5241), retaining TP=3 and FP=11. Border removal would also discard real boundary defects and is not a justified fix.

仅 5% 的正常图及 5% 的缺陷图最高分点靠近定义的内部切块边界（±8 像素）。目前没有证据支持“接缝是全部失败的主要来源”；该统计也不能证明完全没有接缝影响。

Only 5% of peaks in either group fall within the defined internal seam bands (±8 pixels). This does not support seams as the dominant explanation, nor rule out some seam effect.

### 4. 简单改评分或扩大库均未解决 / Simple score changes and larger memory did not solve it

| 开发验证实验 / Validation experiment | Image AUROC | 检出 / TP | 误报 / FP |
|---|---:|---:|---:|
| 原始最大值 / Original peak | 0.5205 | 3 | 11 |
| 排除外圈 / Interior peak | 0.5241 | 3 | 11 |
| 99.9% 分位 / q99.9 | 0.5321 | 3 | 11 |
| 最大值减本图中位数 / Peak minus median | 0.5068 | 2 | 9 |
| 最大值除本图中位数 / Peak divided by median | 0.5003 | 4 | 3 |
| 特征库扩大四倍 / 8192-vector bank | 0.5117 | 3 | 11 |

各评分都单独使用正常校准集第 95 百分位阈值。容量实验保留相同 115200 个候选、数据、模型和评分方式；8192 库的前 2048 个向量与原库逐元素完全一致。因此，单纯扩大该候选集合的记忆库不构成有效修复。它不排除换候选抽样、扩展正常训练覆盖或更换特征可能有效。

Each score uses its own normal-calibration 95th-percentile threshold. The expanded bank uses the same candidate pool and its first 2048 vectors exactly equal the original bank. Simply increasing capacity in this candidate pool is not a fix; different training coverage, candidate sampling, or representations remain untested.

## 尚不能确认的原因 / Unconfirmed causes

- ImageNet 特征更关注外观差异而不是本任务标注的缺陷语义，这是合理假设，但尚未与领域训练特征做对照。
  ImageNet features may emphasize appearance rather than task-specific defect semantics; no domain-trained-feature comparison has established causality.
- 正常纹理变化可能影响评分。正常图最大分数与灰度标准差的 Spearman 相关系数约 0.438，但相关性不能证明纹理是主因。
  Normal texture variability may influence scores (peak vs grayscale std Spearman ~0.438), but this is not causal evidence.
- 只使用 300 张正常图、随机抽取每块 32 个特征可能没有覆盖足够多的正常模式；本轮容量实验没有改变这两点。
  Training coverage (300 normal images) and candidate sampling (32 descriptors/tile) remain plausible limitations, unchanged by the capacity experiment.
- 正常标签下也有显著外观变化。不能凭肉眼认定标注错误，更不能删除这些图来提高结果。
  Some officially normal images have visually prominent variations; appearance alone does not establish labeling error or justify deleting hard cases.

## 下一步 / Next steps

1. 保留当前失败基线，不继续盲调阈值；明确像素级排序与图像级检测是两个不同目标。
   Preserve this baseline and avoid blind threshold tuning; separate within-image localization from image detection.
2. 优先加入轻量 U-Net 监督分割对照，使用现有训练掩码学习任务定义的缺陷。相同原图划分、保留细节的分块、验证集选配置；训练损失和困难正常样本必须记录。
   Prioritize a lightweight supervised U-Net using existing training masks, the same original-image split, and detail-preserving tiles. Record losses and difficult normal cases.
3. 如果继续异常检测路线，分别检验“更多正常训练覆盖”与“不同特征/候选策略”，每次只改一个主要变量，不把目前未经验证的解释当成结论。
   For anomaly detection, separately test broader normal coverage and alternative representations/candidate strategies; change one main factor at a time.
4. 配置确定后才评价保留的钢材测试集。当前诊断与替代评分均为探索性验证结果。
   Evaluate the reserved test split only after fixing the configuration; current findings are exploratory validation results.

## 复现与证据 / Reproduction and evidence

```powershell
python scripts/diagnose_steel.py
python scripts/diagnose_steel_capacity.py
python -m unittest discover -s tests -v
```

依赖已经建立的 `runs/steel_pilot/bank.pt` 和固定数据清单。量化结果见 `steel_failure_diagnosis.json` 和 `steel_capacity_ablation.json`。本机逐图记录及案例预览位于 `runs/steel_diagnosis/`，数据集图片不上传。

Requires the saved pilot bank and fixed data. Quantitative records are `steel_failure_diagnosis.json` and `steel_capacity_ablation.json`. Local per-image records and annotated failure examples are in `runs/steel_diagnosis/`; dataset images are not uploaded.
