"""Verify real training-image tiling and stitching without fitting a model."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.severstal_tiles import SeverstalTiles
from src.tiling import crop_tile, Stitcher


def main():
    dataset=SeverstalTiles(ROOT,'train')
    normal=SeverstalTiles(ROOT,'train',normal_only=True)
    selected=[next(i for i,r in enumerate(dataset.rows) if r['label']==0)]
    selected += [next(i for i,r in enumerate(dataset.rows) if c in r['classes']) for c in (1,2,3,4)]
    output=ROOT/'runs/steel_tiling';output.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(1600,len(selected)*310),'white');draw=ImageDraw.Draw(canvas)
    checks=[]
    for row_index in selected:
        row=dataset.rows[row_index]
        first=dataset[row_index*len(dataset.layout)]
        original=dataset._image.copy(); mask=dataset._mask.copy()
        reconstructed=[Stitcher(256,1600) for _ in range(3)]
        mask_stitch=Stitcher(256,1600)
        for j,box in enumerate(dataset.layout):
            sample=dataset[row_index*len(dataset.layout)+j]
            assert sample['split']=='train' and sample['image_id']==row['image_id']
            assert tuple(sample['image'].shape)==(3,256,256)
            tile=crop_tile(original,box)
            for c in range(3):reconstructed[c].add(tile[:,:,c],box)
            mask_stitch.add(sample['mask'][0].numpy(),box)
        restored=np.stack([s.finish() for s in reconstructed],axis=-1)
        assert np.array_equal(restored,original)
        assert np.array_equal(mask_stitch.finish(),mask)
        checks.append({'image_id':row['image_id'],'classes':row['classes'],
                       'rgb_reconstruction_exact':True,'mask_reconstruction_exact':True,
                       'coverage_min':int(mask_stitch.count.min()),'coverage_max':int(mask_stitch.count.max())})
        i=len(checks)-1
        for c,array in enumerate([original,restored.astype(np.uint8)]):
            panel=Image.fromarray(array).resize((1600,128))
            painter=ImageDraw.Draw(panel)
            for box in dataset.layout:painter.rectangle((box[0],0,box[2]-1,127),outline=(255,80,0),width=1)
            canvas.paste(panel,(0,i*310+30+c*145))
        draw.text((5,i*310+5),f"{row['image_id']} classes={row['classes']} | top=original with tile boundaries; bottom=stitched RGB",fill='black')
    canvas.save(output/'tiling_preview.png')
    batch=next(iter(DataLoader(normal,batch_size=8,shuffle=False,num_workers=0)))
    assert tuple(batch['image'].shape)==(8,3,256,256)
    assert not batch['mask'].any()
    manifest=json.loads((ROOT/'configs/severstal_manifest.json').read_text())
    counts={s:sum(r['split']==s for r in manifest['samples'])*len(dataset.layout) for s in ('train','val','test')}
    report={'tile_size':256,'stride':128,'tiles_per_image':len(dataset.layout),
            'x_starts':[b[0] for b in dataset.layout], 'tiles_per_split':counts,
            'normal_training_tiles':len(normal),'dataloader_batch_shape':list(batch['image'].shape),
            'samples_checked':checks,'source_split_inherited':True,
            'storage':'On-demand; no tiles written to disk',
            'note':'RGB/mask reconstruction verifies geometry only, not model predictions. No steel model fitted.'}
    text=json.dumps(report,indent=2)
    (ROOT/'reports/steel_tiling_checks.json').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
