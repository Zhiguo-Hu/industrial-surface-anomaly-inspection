"""Bilingual final evidence, paired validation comparison and frozen-test figures."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    run=ROOT/'runs/steel_efficientnet_v2_s';history=json.loads((run/'training_history.json').read_text())
    qualification=json.loads((run/'qualification.json').read_text());final=json.loads((ROOT/'reports/strip_final_test.json').read_text())
    old=json.loads((ROOT/'reports/strip_ensemble_qualification.json').read_text());mask=json.loads((run/'mask_calibration.json').read_text())
    assert qualification['model_sha256']==final['model_sha256']
    benchmark_path=ROOT/'reports/strip_inference_benchmark.json'
    benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    if benchmark and benchmark['model_sha256']!=final['model_sha256']:benchmark=None
    q=qualification['qualification'];m=final['metrics'];loc=final['localization'];passed=final['status']=='dataset_acceptance_passed'
    summary={'status':final['status'],'target':final['target'],'model_sha256':final['model_sha256'],'trained_epochs':len(history),'planned_max_epochs':12,
             'early_stop':'Predeclared two consecutive development epochs at recall>=.97/AUROC>=.99',
             'qualification':q,'final_test':m,'final_localization':loc,'final_per_class':final['per_class'],
             'recall_wilson_95':final['recall_wilson_95'],'fpr_wilson_95':final['fpr_wilson_95'],'benchmark':benchmark,
             'test_used':True,'scope':'Original test1882 evaluated once for one frozen model/policy; comparable steel images only; no production-camera acceptance claim'}
    (ROOT/'reports/efficientnet_final_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    rows=[]
    for label,r in [('旧组合资格验证 / Old ensemble qualification',old['qualification']),('新模型资格验证 / New qualification',q),('独立最终test / Frozen final test',m)]:
        rows.append(f"| {label} | {r['tp']}/{r['tp']+r['fn']} | {r['fn']} | {r['fp']}/{r['fp']+r['tn']} | {r['recall']:.2%} | {r['fpr']:.2%} | {r['auroc']:.5f} |")
    table='\n'.join(rows)
    speed=f"中位{benchmark['latency_ms_median']:.1f}ms、P95 {benchmark['latency_ms_p95']:.1f}ms；峰值PyTorch分配{benchmark['peak_allocated_mib']:.1f}MiB。 / Median{benchmark['latency_ms_median']:.1f}ms, P95{benchmark['latency_ms_p95']:.1f}ms; PyTorch peak{benchmark['peak_allocated_mib']:.1f}MiB." if benchmark else '推理基准待完成 / Benchmark pending.'
    state='已通过指定图像报警验收 / Specified image-alarm acceptance passed' if passed else '未通过指定验收 / Specified acceptance failed'
    report=f'''# 钢材检测最终验收 / Frozen steel-inspection acceptance

2026-11-24：**{state}**。最终test1882张，权重与整套阈值在读取test前冻结；未用test选择模型或调整阈值。测试验收只针对同类1600×256钢材图的图像报警，不等于任意相机或精确像素边界的生产验收。

One frozen candidate on the original test1882, with no test-based tuning. Scope is comparable native-resolution steel-image alarms; production-camera and pixel-boundary acceptance remain separate.

## 结果 / Results

| 评估 / Evaluation | 缺陷检出 / TP | 漏检 / FN | 正常误报 / FP | 召回 / Recall | 正常FPR | AUROC |
|---|---:|---:|---:|---:|---:|---:|
{table}

前两行共用986张验证图（701缺陷/285正常）；最后一行使用从未调参的1882张test（997缺陷/885正常）。验证集已多次用于研究，不作为最终独立证据。95% Wilson区间：test召回[{final['recall_wilson_95'][0]:.2%}, {final['recall_wilson_95'][1]:.2%}]，正常FPR[{final['fpr_wilson_95'][0]:.2%}, {final['fpr_wilson_95'][1]:.2%}]。验收标准为用户指定的经验召回≥95%、正常FPR≤5%。

The first two rows are paired on the same reused qualification986. The last row is the untouched test1882. Wilson95% intervals are descriptive uncertainty, not a field guarantee. Acceptance uses the user's empirical recall≥95% and normal-FPR≤5% targets.

## 方法与冻结 / Method and freezing

[官方EfficientNetV2-S](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html) ImageNet1K V1编码器＋轻量U-Net＋图像二分类/四类辅助头。8800张train、BF16、micro4累积4次等效16。任务头先预热，联合训练实际{len(history)}轮后触发预声明提前验证；保留最多12轮计划的完整恢复状态。开发集连续两轮满足97%召回与0.99 AUROC余量，最终权重来自第{qualification['source_epoch']}轮。原分辨率与ImageNet归一化保留，未做384中心裁剪；无外部Severstal训练权重。

Use generic ImageNet EfficientNetV2-S with a small U-Net and task heads. Warm heads, then train native images jointly with BF16/effective16. Stop after{len(history)} epochs via the predefined two-epoch gate, keeping complete resume state. This is not the original paper's training recipe.

开发500选择checkpoint与推理方式；最终采用四翻转图像logit均值，没有学习额外融合器。400张正常校准图按[NP顺序统计量方法](https://jsb-lab.org/wp-content/uploads/2018/02/ScienceAdvances_eaao1659.full_.pdf)确定阈值：alpha=delta=0.05，升序第388个分数，阈值{qualification['selected_policy']['threshold']:.9f}。初始q95方案的99.14%召回/4.56%FPR已单独归档，随后仅依据正常校准样本做有限样本误报控制；未使用qualification/test标签选择这个顺序统计量。

Choose the normal threshold using rank388/400 with alpha=delta=.05. The NP design assumes a fixed score and independent iid normal calibration data; historical validation/calibration reuse and acquisition shifts limit formal coverage claims. Final acceptance relies on the separate frozen test, not a universal probability guarantee.

像素阈值0.4仅在开发500上按全局Dice选择，分类分数不受其影响。test全局Dice={loc['global_dice']:.4f}，全局IoU={loc['global_iou']:.4f}，平均缺陷图Dice={loc['mean_positive_image_dice']:.4f}，像素召回={loc['pixel_recall']:.2%}，正常像素FPR={loc['normal_pixel_fpr']:.3%}。掩码不按图像分类门控，仍作为定位辅助。

Output mask threshold0.4 is development-selected only. Test localization is measured separately; raw masks are ungated and remain auxiliary. Numeric defect classes can overlap within an image; per-class test counts are in the JSON report.

## 本机使用与验证 / Local use and verification

{speed}

基准为FP32、batch1、5次预热后20次测量，包括内存图像预处理/传输/推理/回传，不含相机、JPEG解码或写盘，不代表产线节拍。

The warmed benchmark excludes JPEG decode, camera and disk writes. User selected local single/batch images. Runtime requires the accepted frozen artifact, checks weights/policy/mask threshold, accepts native PNG/JPEG, and keeps uploads/results on this computer.

模型在本地`runs/steel_efficientnet_v2_s/deployment/model.pt`。通过验收后可运行`.\\scripts\\run_steel_deployment.ps1`，打开 http://127.0.0.1:8765 ，支持单张/多选、定位图与CSV。也支持`batch_steel_inspection.py`文件夹CLI。打包与真实HTTP/CLI验证见独立报告；照片、缓存和权重不上传GitHub。

The accepted artifact is local, self-contained and requires no original training cache/weights at inference. Code/metrics are versioned; dataset images, caches and fitted model artifacts remain local. See package/service verification reports for end-to-end evidence.

模型SHA256：`{final['model_sha256']}`。冻结指纹：`{final['frozen_fingerprint']}`。

![开发与验收 / Development and acceptance](efficientnet_acceptance.png)
'''
    (ROOT/'reports/efficientnet_acceptance.md').write_text(report,encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    ax=axes[0];ax.plot([r['epoch'] for r in history],[r['recall']*100 for r in history],'o-',color='#2466a8')
    ax.axhline(95,color='#27854b',ls='--',label='Recall target');ax.set_ylim(85,100);ax.set_xticks([r['epoch'] for r in history]);ax.set_title('Development: checkpoint recall');ax.set_xlabel('Joint-training epoch');ax.set_ylabel('Recall (%)');ax.legend(fontsize=8)
    ax=axes[1];versions=[old['qualification'],q];ax.bar([-.17,.83],[r['recall']*100 for r in versions],.34,label='Recall',color='#2466a8');ax.bar([.17,1.17],[r['fpr']*100 for r in versions],.34,label='Normal FPR',color='#dc7e24')
    for i,r in enumerate(versions):ax.text(i-.17,r['recall']*100+4,f"{r['recall']:.1%}",ha='center',fontsize=9);ax.text(i+.17,r['fpr']*100+4,f"{r['fpr']:.1%}",ha='center',fontsize=9)
    ax.axhline(95,color='#27854b',ls='--');ax.axhline(5,color='#a83434',ls=':');ax.set_xticks([0,1],['Previous ensemble','EfficientNetV2-S']);ax.set_ylim(0,108);ax.set_title('Same qualification986');ax.set_ylabel('Percent');ax.legend(fontsize=8,loc='center right')
    ax=axes[2];cm=[[m['tn'],m['fp']],[m['fn'],m['tp']]];ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(max(row) for row in cm))
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha='center',va='center',fontsize=23,color='white' if cm[i][j]>500 else '#172d3c')
    ax.set_xticks([0,1],['Normal','Defect']);ax.set_yticks([0,1],['Normal','Defect']);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f"Frozen test1882\nRecall {m['recall']:.2%}, FPR {m['fpr']:.2%}")
    fig.suptitle('Steel image-alarm acceptance — one frozen independent test',fontsize=14)
    fig.savefig(ROOT/'reports/efficientnet_acceptance.png',dpi=150);plt.close(fig)
    print(json.dumps({'status':summary['status'],'metrics':m,'localization':loc,'benchmark_available':benchmark is not None},indent=2))


if __name__=='__main__':main()
