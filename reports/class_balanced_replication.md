# 类别平衡三种子复验 / Class-balanced three-seed replication

日期 / Date: 2026-11-23

类别2的收益在三个种子中均出现，但整体精度没有稳定改善，误报增多。不能根据seed=42的一次好结果宣称类别平衡已优于原方案。以下均使用同一200张开发评估图，最终test未使用。

Class-2 gains occur in all three seeds, but overall quality does not improve consistently and false alarms increase. The favorable seed-42 run does not establish superiority. Results reuse the same 200 development images; final test remains unused.

| Seed | Uniform → balanced head AUROC | Uniform → balanced TP/100 | Uniform → balanced FP/100 | Uniform → balanced raw Dice |
|---|---:|---:|---:|---:|
| 42 | 0.8801 → 0.8975 | 53 → 63 | 3 → 5 | 0.5124 → 0.5188 |
| 43 | 0.8868 → 0.8183 | 41 → 41 | 3 → 11 | 0.5287 → 0.4367 |
| 44 | 0.8590 → 0.8473 | 58 → 47 | 4 → 9 | 0.5193 → 0.4652 |

均值 ± 样本标准差（3种子，ddof=1） / Mean ± sample SD (three seeds, ddof=1):

| Metric | Uniform | Class-balanced |
|---|---:|---:|
| Head AUROC | 0.8753 ± 0.0145 | 0.8544 ± 0.0401 |
| Image recall | 50.67% ± 8.74% | 50.33% ± 11.37% |
| Normal image FPR | 3.33% ± 0.58% | 8.33% ± 3.06% |
| Raw pooled Dice | 0.5201 ± 0.0081 | 0.4736 ± 0.0417 |
| Class-2 pixel recall | 35.33% ± 16.76% | 66.11% ± 6.28% |

| Seed | Small-area head TP/26, uniform → balanced | Class-2 head TP/6, uniform → balanced |
|---|---:|---:|
| 42 | 6 → 19 | 1 → 4 |
| 43 | 3 → 4 | 1 → 4 |
| 44 | 7 → 8 | 2 → 3 |

小面积为整图缺陷面积占比0.1%–1%。所有种子都增加了这个组的检出，但改善幅度不稳定；类别2只有6张图，不能视为充分泛化证据。均值与标准差描述训练波动，不是统计显著性或新产线置信区间，三个种子仍只有200张独立图片。

Small area means 0.1%–1% whole-image defect area. Every seed detects more of this group but the gain varies sharply. Class 2 has only six images. Mean/SD describe training variation, not significance or production confidence intervals; there are still only 200 unique evaluation images.

## 控制与核对 / Controls and audits

复用已有seed=42，新跑43、44；与各自原均匀正分块联合模型配对。模型完整初始状态、4000张训练图、80张检查点开发图、200正常校准及200评估图、损失权重0.5、优化器、严格确定性设置和2400次尝试预算相同。正常/异常仍为每批4+4，只改正分块按类别抽样。

Reuse seed 42 and newly train 43/44. Pair each with its uniform-positive joint reference. Match initial state, 4000 train images, 80 checkpoint-development, 200 normal calibration/200 evaluation images, loss weight 0.5, optimizer, strict determinism and 2400 attempted updates. Batches remain four positive/four negative; only positive sampling changes.

抽样回放重算每类实际次数，并核对与训练记录一致；负分块及翻转的完整序列哈希与各自均匀参照一致。CSV分数重算AUROC及混淆矩阵，三个新检查点重新加载的一张评估图分数与原记录一致（容差1e-5）。类别平衡三个种子均实际更新2396次，AMP各跳过4次。

Replay class exposures against training records and verify negative/flip sequence hashes match each seed's uniform reference. Recompute CSV AUROC/confusions and reload one evaluation image per candidate at tolerance 1e-5. Each balanced seed completes 2396 updates and skips four AMP steps.

下一部分按用户要求构建正常特征库，编码器仍使用事先指定的类别平衡seed=42，不因本次复验挑选最好种子。这个选择用于固定建库来源，不代表已证明它是生产最优模型。

Next build the requested normal memory using the predeclared class-balanced seed-42 encoder. It fixes provenance without selecting the best seed from this replication; it is not a production-optimality claim.

完整数值见 [steel_balanced_multiseed.json](steel_balanced_multiseed.json)。复现43/44训练：

Full numerical audits are linked above. To retrain seeds 43/44:

```powershell
foreach ($trainingSeed in @(43,44)) {
  python scripts/train_steel_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30 --deterministic --seed $trainingSeed --classification-weight 0.5 --tile-sampling class-balanced
  if ($LASTEXITCODE -ne 0) { break }
  python scripts/analyze_unet.py --coverage-balanced --encoder pretrained-resnet18 --expanded --rounds 30 --deterministic --seed $trainingSeed --classification-weight 0.5 --tile-sampling class-balanced
}
python scripts/summarize_balanced_replication.py
```
