# 全幅检测暂停记录 / Full-strip inspection pause

2026-11-23：用户因电脑电量不足要求暂停。训练已停止，没有自动重启任务。目标为图像级缺陷召回≥95%、正常误报≤5%；**尚未达标**。

Training stopped at the user's request due to low battery. No scheduled restart. Target: image-level recall≥95%, normal FPR≤5%. **Not yet achieved.**

## 已完成 / Completed

全部8800张训练图使用1600×256原始分辨率；本地缓存覆盖10686张train/val，不含test。划分development500、正常calibration400、qualification986；原1882张test仍封存。

All8800 train images are used at native resolution. Cache covers10686 train/validation images only. Development500, normal calibration400 and qualification986 are disjoint; the original test1882 remains locked.

冻结分类头预热与第一阶段12轮训练完成。最终最佳开发结果：召回268/300（89.33%）、正常误报9/200（4.50%）、AUROC0.97868。中途epoch4普通候选及epoch6融合候选在qualification上均检出628/701、误报17/285，未达标；JSON记录来源轮次，不能当成最终epoch12模型结果。

Head warmup and12 full-strip epochs completed. Best development recall89.33%, FPR4.50%, AUROC0.97868. Earlier frozen epoch4/epoch6 qualification candidates both produced TP628/701 and FP17/285, failing the target. These are not evaluations of the final epoch12 checkpoint.

第二阶段加入800像素保缺陷裁剪、四类辅助头及每步掩码监督。计划8轮，已保存第1轮完成及第2轮50批，共600次更新尝试。优化器、AMP、EMA、CPU/GPU/Python随机状态齐全，断点已验证可加载。59项测试通过；三窗口推理代码通过测试，尚未实际资格验证。

Refinement adds guided800 crops, four-class auxiliaries and denser mask supervision. Eight epochs planned; checkpoint contains epoch1 plus50 batches of epoch2, 600 attempted updates, optimizer/scaler/EMA and CPU/GPU/Python RNG. Checkpoint loading verified. All59 tests pass; new three-window inference is tested but not qualified yet.

## 恢复 / Resume

在项目根目录运行，或让Codex执行 / At project root, or ask Codex to resume:

```powershell
.\scripts\resume_strip_refinement.ps1
```

脚本按相同配置加载`runs/strip_inspector_refined/last.pt`；最后断点后未保存的更新会重做。请保留本地`runs/strip_training_cache`、`runs/strip_inspector`和`runs/strip_inspector_refined`，缓存、照片、权重不上传GitHub。

The launcher restores the matching configuration/checkpoint. Unsaved updates are replayed. Keep the local cache, warm-start and refined run folders; caches, images and weights are excluded from GitHub.

下一步：恢复训练→验证集选择融合/窗口方案→资格达标后冻结候选→一次最终test验收。图像报警门槛不等同于定位精度或现场相机验收；不得用test调参。详细断点见`strip_training_pause.json`。

Next: resume training, select fusion/window policies on validation, qualify and freeze a candidate, then run final test once. Image-alarm acceptance is separate from localization and production-camera validation. Do not tune on test. Exact pause state: `strip_training_pause.json`.
