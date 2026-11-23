"""Plot matched joint-training histories, without dataset photographs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s42'


def main():
    control=json.loads((ROOT/f'reports/{BASE}_cls0_metrics.json').read_text())
    joint=json.loads((ROOT/f'reports/{BASE}_cls0p5_metrics.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for record,label in [(control,'Segmentation only'),(joint,'Joint (classification weight 0.5)')]:
        history=record['history'];steps=[r['updates'] for r in history]
        axes[0].plot(steps,[r['checkpoint_validation_pooled_dice_at_05'] for r in history],label=label)
        axes[1].plot(steps,[r['segmentation_loss'] for r in history],label=label)
    history=joint['history']
    axes[2].plot([r['updates'] for r in history],[r['classification_loss'] for r in history],color='tab:orange')
    titles=['Checkpoint-development Dice at 0.5','Sampled training segmentation loss','Joint head: training classification BCE']
    for ax,title in zip(axes,titles):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Attempted optimizer updates');ax.grid(alpha=.25)
    axes[0].set_ylabel('Pooled Dice');axes[1].set_ylabel('BCE + soft Dice');axes[2].set_ylabel('BCE')
    axes[0].legend(fontsize=8)
    fig.suptitle('Seed 42, same 4000-image pool and initialization; development evidence, no test data')
    path=ROOT/'reports/steel_joint_training_curves.png';fig.savefig(path,dpi=160);plt.close(fig)
    print(path)


if __name__=='__main__':main()
