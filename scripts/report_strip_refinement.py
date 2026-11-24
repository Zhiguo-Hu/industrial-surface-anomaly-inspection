"""Summarize completed full-strip phases and validation qualification numerically."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    first=json.loads((ROOT/'runs/strip_inspector/training_history.json').read_text())
    refined=json.loads((ROOT/'runs/strip_inspector_refined/training_history.json').read_text())
    initial=json.loads((ROOT/'reports/strip_fusion_initial_qualification.json').read_text())
    final=json.loads((ROOT/'runs/strip_inspector_refined/qualification.json').read_text())
    baseline=json.loads((ROOT/'runs/strip_inspector/qualification.json').read_text())
    ensemble=json.loads((ROOT/'runs/strip_inspector_ensemble/qualification.json').read_text())
    report={'target':{'recall_min':.95,'fpr_max':.05},'initial_full_strip_development':first,
            'refined_development':refined,'initial_qualification':initial['qualification'],
            'refined_qualification':final['qualification'],'refined_qualification_status':final['status'],
            'completed_first_phase_qualification':baseline['qualification'],'ensemble_qualification':ensemble['qualification'],
            'ensemble_status':ensemble['status'],'ensemble_model_sha256':ensemble['model_sha256'],
            'ensemble_per_class':ensemble['qualification_per_class'],
            'model_sha256':final['model_sha256'],'selected_epoch':final['source_epoch'],
            'per_class':final['qualification_per_class'],'by_area':final['qualification_by_defect_area'],
            'localization':final['qualification_localization'],'test_used':(ROOT/'reports/strip_final_test_access.json').exists(),
            'notes':['Development fixed500; recall selected at development-normal q95, FP9/200.',
                     'Initial qualification checkpoint was epoch6, not final epoch12.',
                     'Refined epoch2 training loss covers only resumed batches51..550.',
                     'Qualification986 is reused validation, not independent final test.',
                     'Numeric defect classes can overlap within an image.']}
    (ROOT/'reports/strip_refinement_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    x1=[r['epoch'] for r in first];x2=[12+r['epoch'] for r in refined]
    for ax,key,title in [(axes[0,0],'recall','Development recall at 4.5% normal FPR'),(axes[0,1],'auroc','Development AUROC')]:
        ax.plot(x1,[r[key] for r in first],'o-',label='Full-strip phase',color='#2466a8')
        ax.plot(x2,[r[key] for r in refined],'o-',label='Guided-crop + auxiliary phase',color='#dc7e24')
        ax.axvline(12.5,color='.6',ls=':',lw=1);ax.set_title(title);ax.set_xlabel('Cumulative epoch');ax.set_ylim(.65 if key=='recall' else .94,1.)
        if key=='recall':ax.axhline(.95,color='#27854b',ls='--',label='Recall target')
        ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.15)
    labels=['Earlier fusion\n(epoch6)','Phase1 final\n(linear)','Refined\n(linear)','Two-stage\n(linear)'];versions=[initial,baseline,final,ensemble]
    a=axes[1,0];a.bar([i-.18 for i in range(4)],[100*r['qualification']['recall'] for r in versions],.36,label='Defect recall',color='#2466a8')
    a.bar([i+.18 for i in range(4)],[100*r['qualification']['fpr'] for r in versions],.36,label='Normal FPR',color='#dc7e24')
    a.set_xticks(list(range(4)),labels,fontsize=8);a.set_ylim(0,105);a.set_ylabel('Percent');a.set_title('Qualification: 701 defect / 285 normal')
    a.axhline(95,color='#27854b',ls='--',lw=1);a.axhline(5,color='#a83434',ls=':',lw=1);a.legend(fontsize=8)
    a=axes[1,1];groups=ensemble['qualification_per_class'];names=list(groups)
    bars=a.bar(names,[100*groups[c]['recall'] for c in names],color='#2466a8')
    for bar,c in zip(bars,names):a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+4,f"{groups[c]['detected']}/{groups[c]['defective_images']}",ha='center',fontsize=8)
    a.axhline(95,color='#27854b',ls='--');a.set_ylim(0,108);a.set_xlabel('Numeric defect class (overlaps allowed)');a.set_ylabel('Recall (%)');a.set_title('Two-stage candidate per-class recall')
    fig.suptitle('Steel inspection refinement — validation results, not production acceptance',fontsize=13)
    fig.savefig(ROOT/'reports/strip_refinement_metrics.png',dpi=150)
    plt.close(fig);print(json.dumps({k:v for k,v in report.items() if k not in ('initial_full_strip_development','refined_development')},indent=2))


if __name__=='__main__':main()
