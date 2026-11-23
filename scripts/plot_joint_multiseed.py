"""Paired seed plots from audited development summaries, without dataset photos."""
import json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parents[1]/'work/matplotlib-cache'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((ROOT/'reports/steel_joint_multiseed.json').read_text())
    seeds=report['seeds'];x=np.arange(len(seeds));width=.34
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    specs=[('Image AUROC','joint_head','image_auroc',1),
           ('Defective images detected / 100','joint_head','recall',100),
           ('Normal false alarms / 100','joint_head','false_positive_rate',100),
           ('Raw segmentation Dice (ungated)','joint_masks','dice',1)]
    for ax,(title,group,metric,scale) in zip(axes.flat,specs):
        before=[report['per_seed'][str(s)]['baseline'][metric]*scale for s in seeds]
        after=[report['per_seed'][str(s)][group][metric]*scale for s in seeds]
        for shift,values,label,color in [(-width/2,before,'Segmentation-only baseline','tab:blue'),
                                          (width/2,after,'Joint supervision (weight 0.5)','tab:orange')]:
            bars=ax.bar(x+shift,values,width,label=label,color=color)
            ax.bar_label(bars,fmt='%.3f' if scale==1 else '%.0f',padding=3,fontsize=9)
        ax.set_title(title);ax.set_xticks(x,[f'Seed {s}' for s in seeds]);ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True);ax.set_ylim(0,1.05 if scale==1 else max(12,max(before+after)*1.2))
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Same 200 development images, three predefined seeds; no test data\nAll runs shown; no best-seed selection',fontsize=12)
    path=ROOT/'reports/steel_joint_multiseed.png';fig.savefig(path,dpi=160);plt.close(fig);print(path)


if __name__=='__main__':main()
