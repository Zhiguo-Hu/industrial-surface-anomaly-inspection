"""Read-only Severstal image and RLE integrity audit; leaves source data untouched."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def decode_rle(text, height=256, width=1600):
    mask = np.zeros(height*width, dtype=np.uint8)
    if not text.strip():
        return mask.reshape((height,width),order='F')
    tokens = text.split()
    if len(tokens)%2:
        raise ValueError('Odd RLE token count')
    numbers = np.array([int(t) for t in tokens],dtype=np.int64)
    starts, lengths = numbers[::2]-1, numbers[1::2]
    ends = starts+lengths
    if np.any(starts<0) or np.any(lengths<=0) or np.any(ends>height*width):
        raise ValueError('RLE outside image or invalid run length')
    if len(starts)>1 and np.any(starts[1:]<ends[:-1]):
        raise ValueError('Unsorted or overlapping RLE runs')
    for start,end in zip(starts,ends):
        mask[start:end]=1
    return mask.reshape((height,width),order='F')


def encode_rle(mask):
    flat=mask.ravel(order='F')
    padded=np.concatenate([[0],flat,[0]])
    edges=np.flatnonzero(padded[1:]!=padded[:-1])+1
    edges[1::2]-=edges[::2]
    return ' '.join(str(int(x)) for x in edges)


def main():
    root=ROOT/'data/severstal'
    images=root/'train_images' if (root/'train_images').is_dir() else root
    annotations=root/'train.csv'
    errors=[]
    mappings=defaultdict(dict)
    class_counts=Counter()
    example_ids={}
    row_count=0
    mask_area=defaultdict(list)
    with annotations.open(encoding='utf-8-sig',newline='') as file:
        reader=csv.DictReader(file)
        if not {'ImageId','ClassId','EncodedPixels'}.issubset(reader.fieldnames or []):
            raise ValueError('Unexpected CSV schema')
        for row in reader:
            row_count+=1
            name=row['ImageId']
            category=int(row['ClassId'])
            if Path(name).name!=name or category not in {1,2,3,4}:
                raise ValueError('Invalid image name or category')
            if category in mappings[name]:
                raise ValueError(f'Duplicate image/class row: {name}/{category}')
            text=row['EncodedPixels'] or ''
            try:
                mask=decode_rle(text)
                if not np.array_equal(mask,decode_rle(encode_rle(mask))):
                    raise ValueError('RLE roundtrip mismatch')
            except ValueError as exc:
                errors.append(f'{name}/{category}: {exc}');continue
            mappings[name][category]=text
            if mask.any():
                class_counts[category]+=1
                mask_area[category].append(int(mask.sum()))
                example_ids.setdefault(str(category),name)
    paths=sorted(p for p in images.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    names={p.name for p in paths}
    missing=sorted(set(mappings)-names)
    sizes=Counter()
    hashes=defaultdict(list)
    damaged=[]
    for i,path in enumerate(paths):
        try:
            with Image.open(path) as image:
                image.load()
                sizes[f'{image.width}x{image.height}']+=1
                if image.size!=(1600,256):
                    errors.append(f'Unexpected dimensions: {path.name} {image.size}')
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
        except Exception as exc:
            damaged.append({'image':path.name,'error':str(exc)})
        if (i+1)%2500==0: print(f'Checked {i+1}/{len(paths)} images',flush=True)
    abnormal={name for name,classes in mappings.items() if any(s.strip() for s in classes.values())}
    normal=sorted(names-abnormal)
    if normal: example_ids['normal']=normal[0]
    duplicates=[v for v in hashes.values() if len(v)>1]
    report={'image_directory':str(images),'csv_rows':row_count,'image_count':len(paths),
            'csv_referenced_images':len(mappings),'normal_images':len(normal),
            'defect_images':len(names & abnormal),
            'note':'No positive annotation in train.csv is interpreted as normal for this downloaded training set.',
            'image_sizes_wh':dict(sizes),'positive_image_count_per_class':dict(class_counts),
            'multi_class_defect_images':sum(sum(bool(s.strip()) for s in v.values())>1 for v in mappings.values()),
            'mask_area_pixels':{str(k):{'min':min(v),'median':float(np.median(v)),'max':max(v)} for k,v in mask_area.items()},
            'missing_csv_images':missing,'damaged_images':damaged,'rle_or_size_errors':errors,
            'exact_duplicate_groups':duplicates,'example_images':example_ids,
            'data_unchanged':True,'split_created':False}
    report['integrity_passed']=bool(paths) and not (missing or damaged or errors)
    out=ROOT/'reports'
    out.mkdir(exist_ok=True)
    (out/'severstal_data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['integrity_passed']:
        canvas=Image.new('RGB',(1600,160*len(example_ids)),'white')
        draw=ImageDraw.Draw(canvas)
        for i,key in enumerate(['normal','1','2','3','4']):
            if key not in example_ids:continue
            name=example_ids[key]
            with Image.open(images/name) as image:
                array=np.asarray(image.convert('RGB')).copy()
            mask=np.zeros((256,1600),dtype=np.uint8)
            for text in mappings.get(name,{}).values(): mask|=decode_rle(text)
            overlay=array.copy()
            overlay[mask>0]=(0.5*overlay[mask>0]+0.5*np.array([255,0,0])).astype(np.uint8)
            for j,panel in enumerate([array,overlay]):
                canvas.paste(Image.fromarray(panel).resize((800,128)),(j*800,i*160+30))
                draw.text((j*800+5,i*160+5),f"{key} | {name} | {'original' if j==0 else 'official-mask overlay (not prediction)'}",fill='black')
        canvas.save(out/'severstal_annotation_preview.png')
    print(json.dumps(report,indent=2),flush=True)
    if not report['integrity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
