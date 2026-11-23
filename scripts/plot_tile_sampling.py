"""Development sampling comparison without dataset photographs."""
import json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parents[1]/'work/matplotlib-cache'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((ROOT/'reports/steel_tile_sampling_comparison.json').read_text())
    names=['uniform','class-balanced','class-small'];r=[report['results'][n] for n in names]
    x=np.arange(3);fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    specs=[('Image-head AUROC',[a['image_head']['image_auroc'] for a in r],1),
           ('Raw pooled segmentation Dice',[a['raw_masks']['dice'] for a in r],1),
           ('Small-image-group detections / 26 (GT area 0.1%-1%)',
            [a['by_image_gt_area']['0.1_to_1pct']['head_detected'] for a in r],30)]
    for ax,(title,values,top) in zip([axes[0,0],axes[0,1],axes[1,1]],specs):
        bars=ax.bar(x,values,color=['tab:blue','tab:orange','tab:green'])
        ax.bar_label(bars,fmt='%.3f' if top==1 else '%.0f',padding=3)
        ax.set_ylim(0,top*1.05);ax.set_title(title,fontsize=10)
    ax=axes[1,0]
    for shift,key,label in [(-.18,'tp','Defects detected /100'),(.18,'fp','Normal false alarms /100')]:
        bars=ax.bar(x+shift,[a['image_head']['confusion'][key] for a in r],.36,label=label)
        ax.bar_label(bars,padding=3)
    ax.set_ylim(0,110);ax.set_title('Whole-image alarms');ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(x,['Uniform','Class balance','Class + small']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Positive-tile sampling controls: seed 42, same 200 development images\nFixed model, loss, negative tiles and flips; no test data',fontsize=12)
    path=ROOT/'reports/steel_tile_sampling.png';fig.savefig(path,dpi=160);plt.close(fig);print(path)


if __name__=='__main__':main()
