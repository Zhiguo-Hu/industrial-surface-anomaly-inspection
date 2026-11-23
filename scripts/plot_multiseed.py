"""Optional figure: all predefined deterministic seeds; no best-seed selection."""
import json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parents[1]/'work/matplotlib-cache'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s'
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for seed in (42,43,44):
        m=json.loads((ROOT/f'reports/{base}{seed}_metrics.json').read_text())
        h=m['history'];x=[v['updates'] for v in h]
        axes[0].plot(x,[v['training_loss'] for v in h],label=f'seed {seed}')
        axes[1].plot(x,[v['checkpoint_validation_pooled_dice_at_05'] for v in h],label=f'seed {seed}')
    axes[0].set_title('Training loss');axes[0].set_ylabel('BCE + soft Dice')
    axes[1].set_title('Checkpoint-development Dice (threshold 0.5)');axes[1].set_ylabel('Pooled Dice')
    for axis in axes:axis.set_xlabel('Attempted optimization steps');axis.grid(alpha=.25);axis.legend()
    fig.suptitle('Deterministic local-runtime training: predefined seeds 42, 43, 44\nSame data; development curves, not test performance',fontsize=11)
    fig.savefig(ROOT/'reports/steel_multiseed_curves.png',dpi=160);plt.close(fig)
    print('Saved reports/steel_multiseed_curves.png')


if __name__=='__main__':main()
