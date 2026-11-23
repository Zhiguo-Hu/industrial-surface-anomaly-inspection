"""Optional learning-curve figure (requires matplotlib); no dataset images included."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parents[1]/'work/matplotlib-cache'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base='steel_unet_balanced_pretrained_resnet18'
    runs=[(base,'500 images / 480 steps'),(base+'_expanded','4000 images / 480 steps'),
          (base+'_expanded_r30','4000 images / 2400 steps')]
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,label in runs:
        report=json.loads((ROOT/f'reports/{name}_metrics.json').read_text())
        history=report['history'];x=[h['updates'] for h in history]
        axes[0].plot(x,[h['training_loss'] for h in history],marker='.',label=label)
        axes[1].plot(x,[h['checkpoint_validation_pooled_dice_at_05'] for h in history],marker='.',label=label)
    axes[0].set_title('Training loss');axes[0].set_ylabel('BCE + soft Dice loss')
    axes[1].set_title('Checkpoint-development Dice (threshold 0.5)');axes[1].set_ylabel('Pooled Dice')
    for axis in axes:axis.set_xlabel('Attempted optimization steps');axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.suptitle('Fixed pretrained ResNet18-U-Net: data coverage and training budget\nOne seed; development curves, not final test performance',fontsize=11)
    fig.savefig(ROOT/'reports/steel_training_scale_curves.png',dpi=160)
    plt.close(fig)
    print('Saved reports/steel_training_scale_curves.png')


if __name__=='__main__':main()
