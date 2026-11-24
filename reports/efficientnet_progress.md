# EfficientNetV2-S训练与部署入口 / Training and deployment entry

2026-11-24：**已通过最终独立图像报警验收并完成本机单张/批量部署。** test召回98.80%、正常FPR3.50%，满足95%/5%目标；完整结果见[最终验收报告](efficientnet_acceptance.md)。

**Frozen final-test image alarms passed; local single/batch deployment completed.** Test recall98.80%, normal FPR3.50%, meeting95%/5% targets. See the final report.

完成4轮后触发预声明提前验证，随后完成正常样本NP阈值校准、一次原始test、真实模型HTTP/批量CLI联调、独立打包和浏览器界面检查。71项检查通过。以下保留方法和训练过程，不表示训练仍在运行。

Four epochs trigger the predefined gate, followed by normal-only NP calibration, one original test, real HTTP/batch CLI, standalone packaging and browser checks. All71 tests pass. The method/training notes below do not mean training is still active.

## 本轮改动 / Changes

使用[PyTorch官方EfficientNetV2-S](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html)的ImageNet1K V1权重，更换特征编码器并增加轻量U-Net解码器。沿用ImageNet归一化，保留原1600×256图像，不执行官方分类用的384中心裁剪。没有使用外部Severstal训练模型。来源与完整SHA256见`efficientnet_pretrained_provenance.json`；论文见[EfficientNetV2](https://arxiv.org/abs/2104.00298)。

Use TorchVision's generic ImageNet1K V1 EfficientNetV2-S encoder with a small U-Net decoder. Keep ImageNet normalization and native1600×256 geometry, omitting the official classification center crop. No external Severstal-trained weights. Full provenance/SHA256 is recorded separately; this does not reproduce the original paper's training recipe.

已有三种候选共同漏检45张；人工复核包含弱对比边缘细缺陷与大块类别4缺陷，正常标注中也有粗糙纹理/亮点。不能据此认定标注错误；图片、标注与划分均未修改，错误图没有加入训练集。

The previous three candidates share45 misses. Reviewed failures include thin low-contrast edge defects and large class4 patches; normal-labeled surfaces can also contain rough texture/bright points. This does not establish annotation error. Labels/splits are unchanged and validation errors are not moved into train.

## 训练协议 / Protocol

8800张train，原始分辨率，二分类BCE＋四类加权辅助分类＋BCE/Dice定位监督。先冻结编码器提取特征并预热任务头，再联合训练最多12轮。测试集封存；开发500、正常校准400、qualification986保持既有划分。资格验证被用于研究迭代，不是独立最终测试。

Train on8800 native images using binary BCE, weighted four-class auxiliaries and mask BCE/Dice. Warm heads on frozen generic features, then jointly train at most12 epochs. Keep development500, normal calibration400, reused qualification986 and locked final test1882.

本机试跑micro8约5711MiB，micro4约3006MiB，均通过有限损失/梯度检查；选择micro4＋累积4次（等效16），BF16。正式训练额外保存EMA，显存峰值约3186MiB。每100次更新及轮末原子保存模型、EMA、优化器和CPU/GPU/Python RNG，支持在完整更新边界暂停与恢复。

GPU probes pass for micro8/micro4 at5711/3006MiB. Choose micro4 with accumulation4, effective16, BF16; observed training peak including EMA is about3186MiB. Save atomic complete state every100 updates/epoch; pause at complete update boundaries.

预声明一次提前资格验证：至少第3轮，且连续两轮开发召回≥97%、AUROC≥0.99。否则完成12轮。若提前资格不通过，记录尝试并继续相同计划；最终test仅在qualification满足目标、冻结权重与完整策略后开启一次。

One predefined early qualification attempt: from epoch3, two consecutive development epochs at recall≥97% and AUROC≥0.99. Otherwise complete12 epochs. If early qualification fails, record it and resume the same schedule. Unlock final test once only after qualification and freezing weights/policy.

下面仅为研究训练中断后的恢复入口；当前已验收模型保持冻结。 / Research resume entry only; the accepted model remains frozen.

```powershell
# 项目根目录；已有完整状态后恢复 / Resume matching state at project root
.\scripts\run_efficient_training.ps1 -Resume
```

## 本机服务 / Local service

已实现仅监听127.0.0.1的推理入口、多选上传、CSV导出、`GET /health`和原始PNG/JPEG字节的`POST /predict`；图片与结果留本机。拒绝错误尺寸、超过10MB图片，以及未通过最终测试的模型。当前已验收模型由`configs/strip_deployment_current.json`指向。

Implemented loopback-only multi-file upload/CSV, health and raw-image prediction APIs. Results stay local. Wrong geometry, payloads over10MB and unaccepted models are rejected. The accepted artifact has a dedicated deployment pointer.

71项测试通过。HTTP合同单元测试使用替身推理器；另已通过真实验收模型的HTTP/批量CLI及独立部署包端到端验证，详见`steel_service_smoke.json`和`steel_package_smoke.json`。

All71 tests pass. HTTP contract unit tests use a stub; separate real accepted-model HTTP/batch CLI and standalone bundle end-to-end checks pass, as recorded in their reports.
