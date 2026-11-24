# 冻结部署模型错误复核 / Frozen deployment error review

2026-11-24 · 只读、验收后分析 / Read-only post-acceptance analysis

本轮完成已冻结v1在最终test上的全部43个错误案例复核：12张漏检、31张误报。逐图记录包含原图、原始标注、未经整图报警门控的分割响应、最后编码层分类Grad-CAM、原尺度256×256局部图、分数及中英文视觉观察。原始标签未修改。这里的视觉观察由Codex完成，尚未经过钢材工艺人员裁决。

All 43 archived errors were reviewed: 12 false negatives and 31 false positives. Each record contains the original, reference annotation, ungated segmentation response, final-encoder-layer classification Grad-CAM, native 256×256 crops, scores and bilingual visual observations. Labels are unchanged. Observations were made by Codex and have not been adjudicated by a steel-process expert.

## 冻结结果保持不变 / Frozen results remain unchanged

| 项目 / Item | 数值 / Value |
|---|---|
| 最终test / Final test | 1882 images: TP985, FN12, TN854, FP31 |
| 缺陷召回 / Defect recall | 98.7964% = 985/997 |
| 正常误报 / Normal-image FPR | 3.5028% = 31/885 |
| 图像报警阈值 / Image alarm threshold | score > -0.3139572739601135 |
| 定位阈值 / Mask threshold | probability > 0.4 |
| 模型状态SHA256 / Model state SHA256 | `df3e8bff400607a5b3b42489e36a590cf056c94ba8bd9ab32ee9b8b53aacb659` |
| 冻结指纹 / Frozen fingerprint | `eb5af201843abc092914f995afa800a5829df479dc429f9164e9656d989d4db0` |

43例重放与归档分数的最大绝对差为0.000004530，报警结论全部相同。部署指针、权重文件、数据清单、最终test报告、最终test访问记录、归档预测CSV的SHA256均与复核前一致；可视化反向求导只计算特征梯度，没有优化器、参数更新或训练模式。完整逐例数值和保护文件哈希见[机器可读报告](steel_frozen_error_review.json)。

Maximum absolute replay/archived score difference is 0.000004530; every alarm decision matches. SHA256 checks confirm that the deployment pointer, artifact, manifest, final-test report, final-test access record and archived predictions are unchanged. Attribution computes feature gradients only, without an optimizer, parameter updates or training mode. See the [machine-readable report](steel_frozen_error_review.json) for per-case evidence and hashes.

## 逐图观察 / Per-image observations

下表是互斥的主要视觉复核分组，合计43例；它们描述现象，不证明单一失败原因。逐例中英文记录见[视觉观察表](steel_error_review_annotations.json)。

These mutually exclusive primary review groups total 43 cases. They describe observations rather than proving a single failure mechanism. See the [bilingual annotations](steel_error_review_annotations.json).

| 分组 / Group | 数量 / Count | 主要证据 / Evidence |
|---|---:|---|
| 漏检：未有效定位 / FN: not localized | 5 | FN01–03、FN06–07；标注区域分割弱。Weak segmentation at annotated targets. |
| 漏检：部分定位 / FN: partial localization | 3 | FN04、08、11；只覆盖较长或较大标注的一部分。Only part of the annotation is covered. |
| 漏检：定位与报警不一致 / FN: localization/alarm disagreement | 4 | FN05、09、10、12；Dice≥0.2且像素召回≥0.25，整图仍正常。Measurable overlap without an image alarm. |
| 误报：竖向条纹 / FP: longitudinal texture | 12 | FP02–04、08、13、17、19–20、25–28；响应落在参考正常图的条纹附近。Responses near streaks on reference-normal images. |
| 误报：局部斑点 / FP: localized spots | 8 | FP01、05、09、11–12、15–16、24；点簇或局部斑块响应。Responses near spot clusters or local patches. |
| 误报：边缘附近响应 / FP: edge-adjacent response | 8 | FP06–07、10、18、21–23、31；可见钢材边缘及邻近区域响应。Responses near visible steel boundaries. |
| 误报：高亮斑块 / FP: bright patches | 3 | FP14、29–30；高亮或反光区域响应。Responses near bright/reflective patches. |

**不能把所有漏检归为“小缺陷”。** FN01的标注占图像11.953%，FN03占6.745%，分割均几乎没有覆盖标注。相反，FN09的像素召回约96.6%、Dice约0.622，整图分类仍失败。说明既存在局部表征或定位不足，也存在整图分类头与定位分支不一致。

**Small defect size cannot explain all misses.** FN01 and FN03 have annotated areas of 11.953% and 6.745% with almost no segmentation overlap. Conversely, FN09 has approximately 96.6% pixel recall and 0.622 Dice but no image alarm. The cases show both weak localization and disagreement between localization and image classification.

**边缘和纹理值得验证，但不能直接屏蔽。** 8例误报的主要响应靠近边缘；漏检中也有窄钢带和靠边目标。直接裁掉边缘、黑背景或给出强位置惩罚可能减少误报，同时增加漏检。现有错误集合不能证明黑背景是原因，也没有提供完整正常/缺陷总体的匹配对照。

**Edges and texture merit investigation, but blanket suppression is unsupported.** Eight FPs have primary edge-adjacent responses, while some FNs also contain narrow steel strips or targets near edges. Cropping/suppressing edges or black background could reduce some alarms while creating misses. This error-only collection cannot establish causality or enrichment relative to matched full-population controls.

**参考正常不等于视觉上没有纹理。** 31例按原始标注全部计误报；明显条纹、斑点或反光不能据此改成缺陷，也不能据此宣布数据标错。FP14与FP30外观相近，仅记录相似性，未证明来自同一工件。

**Reference-normal does not mean texture-free.** All 31 remain FPs under the original annotations. Visible streaks, spots or reflections do not justify relabeling or declaring annotation errors. FP14 and FP30 look similar, without evidence of shared workpiece provenance.

## 自动描述指标 / Automatic descriptive checks

这些标签允许重叠，不应与上表的互斥分组混加。阈值是用于整理观察的规则，不是新的部署阈值。

These flags overlap and must not be added to the mutually exclusive groups above. Their cutoffs organize observations; they are not new deployment thresholds.

| 指标 / Check | FN / 12 | FP / 31 | 解释 / Meaning |
|---|---:|---:|---|
| 距冻结阈值≤0.5 logit / Within 0.5 logit of frozen threshold | 1 | 1 | 仅2例接近该描述边界；不能说全部错误由临界分数引起。Only two satisfy this proximity rule. |
| 四翻转单视图跨越冻结阈值 / Individual flip views straddle threshold | 4 | 12 | 视图敏感性的描述，不意味着改成max聚合就更好。View sensitivity does not establish that max aggregation is better. |
| 标注与邻域平均灰度差/邻域标准差<0.5 / Local mean-contrast ratio <0.5 | 7 | N/A | 均值指标可能抵消纹理差异，不能作为可见性或标注质量判据。Mean differences can cancel texture variation. |
| 有定位重合但没有图像报警 / Overlap without image alarm | 4 | N/A | Dice≥0.2、像素召回≥0.25；局部掩码没有门控。Ungated mask evidence. |
| 报警但预测掩码面积<0.01% / Alarm with mask area <0.01% | N/A | 1 | 分类头和掩码分支并非同一个输出。Classification and segmentation are distinct outputs. |

漏检类别分布为类别1四张、类别3七张、类别4一张。这里只统计12例漏检，不用于推断各类别总体难度或部署泛化水平。

FN class counts are 4 in class 1, 7 in class 3 and 1 in class 4. These twelve misses alone do not estimate class difficulty or deployment generalization.

## 图册解释与复现 / Atlas interpretation and reproduction

本地入口：`runs/frozen_error_review_v1/index.html`。支持FN/FP筛选、图片ID或观察关键词搜索、展开逐图面板、打开原图、导出全43例CSV；同目录另有静态`review.csv`和8张总览。图中绿色为原始标注轮廓，红色为未经报警门控的预测轮廓，黄色仅表示轮廓交叠，像素交集以数值为准。

The local entry supports FN/FP filtering, ID/observation search, expandable panels, originals and full-43-case CSV export. The directory also contains static `review.csv` and eight contact sheets. Green outlines are references, red outlines are ungated predictions, and yellow indicates overlapping outlines; numerical overlap uses mask pixels.

分割热图表达像素分割分支的输出，不能当作分类解释。分类热图采用[Grad-CAM原论文](https://arxiv.org/abs/1610.02391)的梯度加权激活思路，只分析最后编码层，按冻结的四翻转视图对齐后汇总。该分类头另有多尺度分支，因此此图不是完整分类归因。每张CAM独立归一化；颜色不是缺陷概率，不可跨图比较强弱，也不能证明因果或精确缺陷边界。

Segmentation maps are outputs of the pixel branch, not explanations of classification. Class maps use gradient-weighted activations following the [Grad-CAM paper](https://arxiv.org/abs/1610.02391), restricted to the last encoder layer and aligned across frozen flip views. Other multiscale classifier branches are not fully represented. Per-image normalization prevents cross-image intensity comparisons; colors are neither probabilities nor causal proof or precise boundaries.

```powershell
# Requires the existing accepted local artifact, manifest, annotations and archived predictions.
python scripts/review_frozen_errors.py
# Rebuild text/HTML/CSV using saved evidence; no model replay.
python scripts/review_frozen_errors.py --render-only
python -m unittest discover -s tests -v
# Optional isolated Edge browser check of the offline atlas.
python scripts/verify_error_atlas.py
```

数据图片、派生面板、图册ZIP和权重均保留本机，不上传GitHub；公开代码、数值和视觉观察文本。图册是本地工程产物，其生成不会更改生产推理入口。

Dataset images, derived panels, atlas ZIP and weights remain local. Only code, numerical reports and observation text are published. Atlas generation does not alter the inference entry point.

验证：75项单元/回归测试通过；真实Edge浏览器已验证离线打开、43/12/31例筛选、图片ID搜索、展开原尺度面板、中英文观察和43行CSV导出回读。见[浏览器检查记录](steel_error_atlas_browser_check.json)。

Validation: all 75 unit/regression tests pass. Real Edge checks cover offline opening, 43/12/31-case filters, ID search, expanded native-size panels, bilingual observations and a 43-row CSV round trip. See the [browser check](steel_error_atlas_browser_check.json).

## 对下一阶段的约束 / Implications for the next stage

下一阶段优先收集与部署相机一致的新图片，覆盖窄钢带、边缘、亮度变化、竖向纹理以及明确缺陷，由工艺人员确认正常/缺陷边界。在新数据上验证固定v1，记录图像报警与定位效果及耗时；若继续研究，可在训练/开发数据上比较ROI感知分类、分类与分割一致性、针对正常纹理的抽样等假设。

Next, collect fresh images under the deployment camera setup, covering narrow steel regions, edges, brightness changes, streaks and confirmed defects. Obtain process-expert labels, then validate frozen v1 for alarms, localization and latency. Research candidates include ROI-aware classification, classification/segmentation consistency and normal-texture sampling, evaluated on training/development data.

本轮没有重训、改阈值、改标签或重新宣称一次独立test。最终test已用于逐例解释；基于本报告改变模型或选择方法后，后续独立验收需要新的、未参与选择的数据。

No retraining, threshold changes, label edits or new independent-test claim occurred. The final test has now been inspected case by case. Model/method selection informed by this report requires fresh untouched data for subsequent independent acceptance.
