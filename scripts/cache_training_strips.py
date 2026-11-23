"""Cache all train/val strips without reading final-test images; resumable writes."""
import csv,hashlib,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_severstal import decode_rle
OUTPUT=ROOT/'runs/strip_training_cache'


def write_metadata(meta):
    temporary=OUTPUT/'metadata.tmp';temporary.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    temporary.replace(OUTPUT/'metadata.json')


def main():
    manifest_path=ROOT/'configs/severstal_manifest.json';manifest=json.loads(manifest_path.read_text())
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows=[r for s in ['train','val'] for r in manifest['samples'] if r['split']==s]
    assert len(rows)==10686 and all(r['split']!='test' for r in rows)
    annotation=ROOT/manifest['annotation_csv'];assert hashlib.sha256(annotation.read_bytes()).hexdigest()==manifest['source_csv_sha256']
    OUTPUT.mkdir(parents=True,exist_ok=True);existing=OUTPUT/'metadata.json'
    meta=json.loads(existing.read_text()) if existing.exists() else None
    if meta is not None and meta['manifest_sha256']!=digest:raise ValueError('Source manifest changed; use a separate cache')
    if meta is not None and meta['complete']:print('Train/val cache already ready');return
    if meta is None:
        images=np.lib.format.open_memmap(OUTPUT/'images.npy',mode='w+',dtype=np.uint8,shape=(len(rows),256,1600,3))
        masks=np.lib.format.open_memmap(OUTPUT/'masks.npy',mode='w+',dtype=np.uint8,shape=(len(rows),256,200))
        meta={'complete':False,'manifest_sha256':digest,'height':256,'width':1600,'rows':rows,'completed_rows':0}
        write_metadata(meta)
    else:
        images=np.load(OUTPUT/'images.npy',mmap_mode='r+');masks=np.load(OUTPUT/'masks.npy',mmap_mode='r+')
    annotations=defaultdict(list)
    with annotation.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['EncodedPixels'].strip():annotations[r['ImageId']].append(r['EncodedPixels'])
    for i in range(meta['completed_rows'],len(rows)):
        row=rows[i];path=ROOT/row['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with Image.open(path) as im:
            assert im.size==(1600,256);images[i]=np.asarray(im.convert('RGB'))
        mask=np.zeros((256,1600),dtype=np.uint8)
        for text in annotations[row['image_id']]:mask|=decode_rle(text)
        assert int(mask.any())==row['label'];masks[i]=np.packbits(mask,axis=-1)
        if (i+1)%256==0 or i+1==len(rows):
            images.flush();masks.flush();meta['completed_rows']=i+1;write_metadata(meta)
            print(f'Train/val cache {i+1}/{len(rows)}',flush=True)
    meta['complete']=True;write_metadata(meta);print('Train/val cache complete; no final-test images read',flush=True)


if __name__=='__main__':main()
