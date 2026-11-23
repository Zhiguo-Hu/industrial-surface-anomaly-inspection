"""Repeat seed-42 inference in separate processes and compare every probability pixel."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAME='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s42'


def main():
    image=ROOT/'data/severstal/034941f9d.jpg'
    command=[sys.executable,str(ROOT/'scripts/train_steel_unet.py'),'--coverage-balanced','--encoder',
             'pretrained-resnet18','--expanded','--rounds','30','--deterministic','--seed','42','--image',str(image)]
    snapshots=[];decisions=[]
    for _ in range(2):
        response=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8')
        snapshots.append(np.load(ROOT/f'runs/{NAME}/single_probability.npy').copy())
        decisions.append(json.loads(response.stdout))
    assert np.isfinite(snapshots[0]).all()
    np.testing.assert_array_equal(snapshots[0],snapshots[1])
    assert decisions[0]==decisions[1]
    report={'scope':'Same-runtime inference in two separate processes, same saved seed-42 model',
            'image_id':image.name,'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
            'model':NAME,'probability_shape':list(snapshots[0].shape),
            'all_probability_pixels_bitwise_equal':True,'decision_records_equal':True,
            'probability_bytes_sha256':hashlib.sha256(snapshots[0].tobytes()).hexdigest(),
            'prediction':decisions[0],'limitation':'One input on this runtime; not a universal cross-runtime guarantee'}
    text=json.dumps(report,indent=2)
    (ROOT/'reports/steel_deterministic_inference.json').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
