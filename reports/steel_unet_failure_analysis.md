# U-Net 类别与面积诊断 / U-Net class and area diagnosis

日期 / Date: 2026-11-23

使用已保存的 U-Net、像素阈值 0.4、图像阈值 0.974557，重新评估同一 200 张开发验证图片；与已记录的 TP=25、FP=7、pooled Dice=0.287704 一致。没有改动权重或阈值，没有访问测试图片。

The saved U-Net and original thresholds reproduce TP=25, FP=7 and pooled Dice=0.287704 on the same 200 development-validation images. No weights or thresholds were changed; no test images were accessed.

## 类别 / Classes

| 类别 / Class ID | 训练图含该类 / Train images | 验证图含该类 / Eval images | 整图检出 / Image alarms | 该类像素召回 / Class-pixel recall |
|---|---:|---:|---:|---:|
| 1 | 43 | 9 | 2/9 | 25.27% |
| 2 | 9 | 6 | 0/6 | 2.49% |
| 3 | 188 | 75 | 22/75 | 36.27% |
| 4 | 27 | 14 | 1/14 | 16.39% |

类别来自官方编号，不擅自命名。模型只输出二值掩码，没有识别具体类别。多类图片重复出现在相关组内，因此不能把行数直接相加。整图报警也不表示该类别所在区域被正确覆盖；最后一列单独计算该类标注像素被预测覆盖的比例。

Class names are not inferred. The model is binary, not class-predicting. Multi-class images appear in multiple groups; image alarm does not establish correct localization of that class. The last column measures coverage of that class's actual annotation pixels.

当前“250 正常+250 缺陷”的抽样并没有平衡缺陷类别。第 2 类仅 9 张、22781 个标注像素；第 3 类有 188 张、4703893 个标注像素。批次中正负块各半，也不等于稀有类别获得足够更新。本轮结果与覆盖不足相容，但尚未用受控重采样实验确认因果。

Balancing normal/defective parent images and positive/negative tiles does not balance defect classes. Class 2 has only 9 images / 22781 pixels versus class 3's 188 images / 4703893 pixels. This supports investigating coverage but does not prove a causal explanation without resampling ablation.

## 面积 / Area

面积指整张原图中所有缺陷掩码的并集占比，不是单个连通区域大小。
Area is union defect area as a fraction of the original strip, not component size.

| 缺陷面积占比 / Area | 图片 / Images | 整图检出 / Alarms | 缺陷像素覆盖 / Pixel recall | 逐图 Dice 均值 / Mean Dice |
|---|---:|---:|---:|---:|
| <0.1% | 0 | 无样本 / N/A | N/A | N/A |
| 0.1–1% | 26 | 10/26 | 51.64% | 0.3817 |
| 1–5% | 42 | 10/42 | 47.75% | 0.5033 |
| ≥5% | 32 | 5/32 | 27.80% | 0.3743 |

不能归结为“只有小缺陷看不见”。大面积组也漏掉大量区域：该组预测正像素精确率约 80.81%，召回仅 27.80%，表现为覆盖不足。本组没有 <0.1% 的缺陷图，不能据此宣称已评估极小缺陷。类别和面积存在混杂，不能仅据本表认定尺度是主因。

Large-area defects also suffer undercoverage (precision 80.81%, recall 27.80%). No examples have union area below 0.1%, so very-small-defect performance is untested. Class and area are confounded; this table alone does not prove scale is causal.

## 正常误报与两级阈值 / Normal false positives and two thresholds

- 100 张正常图中仅 7 张触发图像报警，但 95 张在像素阈值 0.4 下存在非空预测掩码。
- 正常图片合计 4.81% 的像素被预测为缺陷。预测面积占比中位数约 1.84%，最大约 27.91%。
- 75 张整图漏报的缺陷图都存在非空预测掩码；其中 27 张掩码 Dice 已达到 0.5。

Only 7/100 normal images trigger alarms, but 95/100 have nonempty masks at pixel threshold 0.4. Overall normal-pixel FPR is 4.81%; median predicted area is 1.84%, maximum 27.91%. All 75 missed defective images have predicted pixels; 27 have Dice at least 0.5.

图像报警基于最大概率是否超过 0.974557，像素输出基于是否超过 0.4，当前并未用整图报警门控掩码。因此“有掩码但没报警”是不同规则的结果，不是统计矛盾。也不能直接用报警关掉掩码：这样会丢掉已有较好定位的漏报图。需独立设计图像评分和空间区域决策，而不是只选一个好看的误报率。

Image alarms use maximum probability >0.974557, masks use probability >0.4, and masks are not gated by image alarms. Nonempty masks without alarms are expected. Gating would discard some correctly localized misses; image scoring and spatial decisions need separate design and validation.

## 下一次受控实验 / Next controlled experiment

1. 优先改变训练抽样，保持模型、损失、更新次数和验证集不变。在原 train 集内增加第 2/4 类覆盖，尽可能固定同样的父图规模；记录类别重叠和每批实际采样情况。先检验覆盖假设，不同时更换编码器、损失和训练时长。
   First change training selection/sampling only, increasing class 2/4 representation within the original train split while keeping architecture, loss, update budget and validation fixed. Record overlapping classes and actual sampling.
2. 固定报告整图召回、正常图报警率、正常图非空掩码率、正常像素误报率、缺陷区域 Dice 和各类别像素召回。
   Report image recall, normal alarms, nonempty normal masks, normal-pixel FPR, defect Dice and per-class pixel recall together.
3. 抽样实验后再决定延长训练或使用预训练编码器。第 5 轮优于第 6 轮，说明本次不能仅凭训练 loss 下降断言延长训练会改善泛化。
   Consider longer training or pretrained encoding after the sampling ablation. Round 5 outperformed round 6, so decreasing training loss alone does not justify longer training.

原始逐图数据与案例图在本机 `runs/steel_unet_diagnosis/`；量化分组在 `reports/steel_unet_diagnosis.json`。24 项测试通过，包括空掩码定义、面积边界与正常误报像素计入 pooled 指标。

Per-image data and examples remain local in `runs/steel_unet_diagnosis/`; grouped records are in `reports/steel_unet_diagnosis.json`. All 24 tests pass, including empty-mask semantics, area boundaries and normal false positives in pooled metrics.
