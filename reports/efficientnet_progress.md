# EfficientNetV2-S训练与部署入口 / Training and deployment entry

2026-11-24，正在训练；**不是验收通过的部署模型**。既有组合候选召回92.87%/正常FPR7.02%，本轮目标仍为至少95%/至多5%。

Training in progress; **not an accepted deployment model**. Previous ensemble recall92.87% / normal FPR7.02% fails the unchanged95%/5% target.

当前已完成4轮、触发预声明的提前资格验证；开发召回98.67%/FPR4.5%，不代表最终验收。用户选择本机单张/批量图片部署，已补充网页多选、CSV导出与文件夹批量CLI。70项检查通过，真实模型服务与独立最终test仍待验收流程执行。

Four epochs trigger the predefined early-qualification gate: development recall98.67%/FPR4.5%, not final acceptance. User selected local single/batch images. Multi-file web upload, CSV export and folder CLI are implemented;70 checks pass. Accepted-model service/end-to-end and final-test checks remain pending.

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

```powershell
# 项目根目录；已有完整状态后恢复 / Resume matching state at project root
.\scripts\run_efficient_training.ps1 -Resume
```

## 本机服务 / Local service

已实现仅监听127.0.0.1的推理入口、上传页面、`GET /health`和原始PNG/JPEG字节的`POST /predict`；图片与结果留本机。拒绝错误尺寸、超过10MB图片，以及未通过最终测试的模型。当前尚无通过验收的模型，服务不会把研究候选作为部署模型启动。

Implemented a loopback-only upload page/API: `GET /health`, `POST /predict` with raw PNG/JPEG bytes. Images/results stay local. Reject wrong geometry, payloads over10MB and models lacking passed final-test acceptance. No accepted artifact exists yet; research models cannot start as deployment models.

67项测试已通过，含新编码器梯度/复载、输入合同、验收门槛，以及实际回环HTTP路由/文件隔离测试。此处HTTP合同测试使用替身推理器；真实验收模型的端到端服务验证须等训练和验收后执行，尚未完成。

All67 tests pass, covering encoder gradients/reload, input/acceptance contracts and real loopback HTTP routes/file isolation. HTTP contract tests use a stub inference object; accepted-model end-to-end verification remains pending.
