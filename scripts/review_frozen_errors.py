"""Replay only archived mistakes; produce private atlas without altering frozen v1."""
import argparse,csv,hashlib,html,json,os,shutil,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images
from src.inspection_protocol import frozen_test_fingerprint
from src.reproducibility import configure,state_hash
from src.error_review import select_errors,boundary,patch_start,case_evidence
from src.steel import heat_rgb
from scripts.audit_severstal import decode_rle

OUT=ROOT/'runs/frozen_error_review_v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_maps(model,array,policy,device):
    """Four-view positive final-layer Grad-CAM; it covers only the deepest feature branch."""
    x=normalize_images(torch.from_numpy(array.transpose(2,0,1).copy())[None].to(device))
    views=[(),(-1,),(-2,),(-2,-1)] if policy['flip4'] else [()];logits=[];cams=[];probability=None
    for index,dims in enumerate(views):
        view=(x.flip(dims) if dims else x).detach().requires_grad_(True)
        with torch.enable_grad():
            encoded,skips=model.encode(view);pooled=model.pool_features(encoded,skips);logit=model.detector(pooled).squeeze(1)
            if index==0:
                with torch.no_grad():probability=model.decode(encoded.detach(),tuple(s.detach() for s in skips),x.shape[-2:])[0,0].float().sigmoid().cpu().numpy()
            gradient,=torch.autograd.grad(logit.sum()/len(views),encoded)
        cam=torch.relu((gradient.detach().mean((-2,-1),keepdim=True)*encoded.detach()).sum(1,keepdim=True))
        cam=F.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)
        if dims:cam=cam.flip(dims)
        cams.append(cam[0,0].cpu().numpy());logits.append(logit.detach())
    cam=np.sum(cams,axis=0);maximum=float(cam.max())
    if maximum>0:cam=cam/maximum
    score=float(torch.stack(logits).mean(0).item())
    return score,probability,cam,[float(v.item()) for v in logits]


def reference_image(array,truth,pred):
    result=array.copy();gt=boundary(truth);pd=boundary(pred)
    result[gt]=[30,235,100];result[pd]=[255,50,50];result[gt&pd]=[255,220,0];return result


def image_views(array,truth,probability,cam,pixel_threshold):
    return [array,reference_image(array,truth,probability>pixel_threshold),
            (.6*array+.4*heat_rgb(probability,1)).astype(np.uint8),(.6*array+.4*heat_rgb(cam,1)).astype(np.uint8)]


def draw_card(record,views,start,path):
    sheet=Image.new('RGB',(1600,1510),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
    e=record['evidence'];draw.text((12,7),f"{record['case_id']}  {record['image_id']}  classes={record['classes']}  score={record['score']:.4f}  margin={e['score_margin']:+.4f}",fill='black',font=font)
    captions=['Original native1600x256','Annotation reference: green=GT, red=predicted mask, yellow=overlap',
              'Segmentation response (not classification attribution)','Classification final-layer Grad-CAM (relative positive support; approximate)']
    for i,(array,caption) in enumerate(zip(views,captions)):
        top=45+i*286;draw.text((12,top),caption,fill='black',font=font);sheet.paste(Image.fromarray(array),(0,top+28))
    for i,index in enumerate((0,2,3)):
        left=20+i*525;draw.text((left,1192),f'Native256 crop x={start}: '+['original','segmentation','Grad-CAM'][i],fill='black',font=font)
        sheet.paste(Image.fromarray(views[index][:,start:start+256]),(left,1224))
    sheet.save(path)


def build_contacts(records):
    directory=OUT/'overview';directory.mkdir(exist_ok=True);links=[]
    for kind in ('FN','FP'):
        group=[r for r in records if r['kind']==kind]
        for offset in range(0,len(group),6):
            page=Image.new('RGB',(1600,1800),'white');draw=ImageDraw.Draw(page);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
            for j,r in enumerate(group[offset:offset+6]):
                left=(j%2)*800;top=(j//2)*600
                with Image.open(OUT/r['panel']) as im:card=np.asarray(im)
                draw.text((left+8,top+5),f"{r['case_id']} {r['image_id']} s={r['score']:.2f} C={r['classes']}",fill='black',font=font)
                # Exact original/reference rows from the high-resolution card.
                for row,start in enumerate((73,359)):
                    page.paste(Image.fromarray(card[start:start+256]).resize((800,128)),(left,top+35+row*145))
                for k,x in enumerate((20,545,1070)):
                    page.paste(Image.fromarray(card[1224:1480,x:x+256]),(left+8+264*k,top+330))
            name=f'overview/{kind}_{offset//6+1:02d}.png';page.save(OUT/name);links.append(name)
    return links


def render(records,metadata,notes):
    contacts=build_contacts(records)
    for r in records:r['review']=notes.get(r['image_id'],{'primary':'pending','observation_zh':'待逐图复核','observation_en':'Visual review pending'})
    data=json.dumps(records,ensure_ascii=False).replace('<',r'\u003c')
    links=' '.join(f'<a href="{name}" target="_blank">{Path(name).stem}</a>' for name in contacts)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>冻结模型错误复核 / Frozen error review</title>
<style>body{font:16px/1.6 system-ui;background:#edf2f5;color:#172d3c;margin:0}main{max-width:1250px;margin:auto;padding:24px}article,.intro{background:white;padding:20px;border-radius:12px;margin:18px 0}img{width:100%}summary{cursor:pointer;font-weight:600}button,select,input{padding:8px;margin:6px}a{color:#155a75}.muted{color:#5e7180;font-size:14px}.fn{color:#b44626}.fp{color:#855421}pre{white-space:pre-wrap}table{border-collapse:collapse}td{padding:4px 12px;border-bottom:1px solid #dde5ea}</style>
<main><h1>冻结模型错误复核 / Frozen-model error review</h1><div class="intro"><strong>12张漏检 FN · 31张误报 FP · 共43例</strong><p>本图册仅复核已验收v1的错误，不是新的独立测试；权重、阈值与既有验收记录保持不变。分割响应不是分类解释；Grad-CAM只描述最后层近似正向支持，未覆盖全部多尺度分支，不能证明因果。Grad-CAM逐图归一化，不可比较不同图片的颜色强弱。</p><p class="muted">Read-only post-acceptance review. Green=ground truth; red=ungated mask at0.4; yellow=overlapping boundaries. Segmentation response and approximate positive class attribution are distinct. No model/threshold tuning.</p><p>总览 / Contact sheets: CONTACTS</p><select id="kind"><option value="">全部43例 / All</option><option>FN</option><option>FP</option></select><input id="search" placeholder="图片ID、观察关键词 / Search"><button id="csv">导出复核表 / Export CSV</button><p id="count"></p></div><section id="cases"></section></main>
<script>const DATA=CASE_DATA;function render(){const kind=document.querySelector('#kind').value,search=document.querySelector('#search').value.toLowerCase();const list=DATA.filter(r=>(!kind||r.kind===kind)&&JSON.stringify(r).toLowerCase().includes(search));document.querySelector('#count').textContent=`显示 ${list.length} /43`;
document.querySelector('#cases').replaceChildren();for(const r of list){const article=document.createElement('article'),details=document.createElement('details'),summary=document.createElement('summary');summary.className=r.kind.toLowerCase();summary.textContent=`${r.case_id} · ${r.image_id} · score ${r.score.toFixed(3)} · margin ${r.evidence.score_margin.toFixed(3)} · class ${r.classes.join(',')||'normal'}`;details.append(summary);const text=document.createElement('p');text.textContent=r.review.observation_zh+' / '+r.review.observation_en;details.append(text);const metrics=document.createElement('p');metrics.className='muted';metrics.textContent=`GT面积 ${(r.evidence.gt_area_fraction*100).toFixed(3)}% | 预测面积 ${(r.evidence.predicted_area_fraction*100).toFixed(3)}% | 像素Dice ${r.evidence.pixel_dice===null?'N/A':r.evidence.pixel_dice.toFixed(3)} | 描述标签 ${r.evidence.descriptive_flags.join(', ')||'无'}`;details.append(metrics);const image=document.createElement('img');image.loading='lazy';image.src=r.panel;image.alt=r.image_id;details.append(image);const original=document.createElement('a');original.href=r.original;original.textContent='打开原图 / Original';original.target='_blank';details.append(original);article.append(details);document.querySelector('#cases').append(article)}}
document.querySelector('#kind').onchange=render;document.querySelector('#search').oninput=render;document.querySelector('#csv').onclick=()=>{const cols=['case_id','image_id','kind','score','primary','observation_zh','observation_en'];const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=[cols,...DATA.map(r=>[r.case_id,r.image_id,r.kind,r.score,r.review.primary,r.review.observation_zh,r.review.observation_en])];const url=URL.createObjectURL(new Blob(['\\ufeff'+rows.map(row=>row.map(q).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='frozen-error-review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();</script></html>'''
    page=page.replace('CONTACTS',links).replace('CASE_DATA',data)
    (OUT/'index.html').write_text(page,encoding='utf-8')
    (OUT/'case_records.json').write_text(json.dumps({'metadata':metadata,'cases':records},indent=2,ensure_ascii=False),encoding='utf-8')
    columns=['case_id','image_id','kind','score','score_margin','classes','primary','observation_zh','observation_en']
    with (OUT/'review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in records:
            writer.writerow({**{key:r[key] for key in ('case_id','image_id','kind','score')},
                             'score_margin':r['evidence']['score_margin'],'classes':','.join(map(str,r['classes'])),
                             **{key:r['review'][key] for key in ('primary','observation_zh','observation_en')}})
    flags={kind:dict(Counter(flag for r in records if r['kind']==kind for flag in r['evidence']['descriptive_flags'])) for kind in ('FN','FP')}
    summary={**metadata,'case_counts':dict(Counter(r['kind'] for r in records)),'descriptive_flag_counts':flags,
             'reviewed_cases':sum(r['review']['primary']!='pending' for r in records),
             'primary_review_groups':dict(Counter(r['review']['primary'] for r in records)),
             'case_statistics':[{k:r[k] for k in ('case_id','image_id','kind','classes','score','evidence','review')} for r in records],
             'scope':'Post-acceptance descriptive review of43 known mistakes; not a new independent test or causal proof. No tuning/retraining or label changes.'}
    (ROOT/'reports/steel_frozen_error_review.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'cases':len(records),'reviewed':summary['reviewed_cases'],'flags':flags,'atlas':str(OUT/'index.html')},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);notes_path=ROOT/'reports/steel_error_review_annotations.json'
    notes=json.loads(notes_path.read_text(encoding='utf-8')) if notes_path.exists() else {}
    if args.render_only:
        saved=json.loads((OUT/'case_records.json').read_text(encoding='utf-8'))
        assert all(digest(ROOT/path)==sha for path,sha in saved['metadata']['protected_file_sha256'].items()),'Saved evidence no longer matches frozen files'
        render(saved['cases'],saved['metadata'],notes);return
    configure(42,True);torch.set_num_threads(4)
    pointer=ROOT/'configs/strip_deployment_current.json';settings=json.loads(pointer.read_text());model_path=ROOT/settings['model']
    protected=[pointer,model_path,ROOT/'configs/severstal_manifest.json',ROOT/'reports/strip_final_test.json',ROOT/'reports/strip_final_test_access.json',ROOT/'runs/steel_efficientnet_v2_s/final_test_predictions.csv']
    before={str(p.relative_to(ROOT)):digest(p) for p in protected}
    frozen=torch.load(model_path,map_location='cpu',weights_only=True);policy=frozen['inspection_policy']
    assert frozen['acceptance']['status']=='dataset_acceptance_passed';assert frozen['model_sha256']==settings['model_sha256']==state_hash(frozen['model'])
    assert frozen_test_fingerprint(frozen['model_sha256'],policy)==settings['frozen_fingerprint']
    spec=policy['fusion'];assert spec.get('method','linear')=='linear' and spec['coefficient']==[1.]+[0.]*8 and spec['mean']==[0.]*9 and spec['scale']==[1.]*9 and spec['intercept']==0
    with protected[-1].open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    errors=select_errors(rows,policy['threshold']);assert Counter(r['kind'] for r in errors)=={'FN':12,'FP':31}
    manifest=json.loads(protected[2].read_text());lookup={r['image_id']:r for r in manifest['samples']};wanted={r['image_id'] for r in errors}
    annotation=ROOT/manifest['annotation_csv'];assert digest(annotation)==manifest['source_csv_sha256'];annotations={}
    with annotation.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['ImageId'] in wanted and row['EncodedPixels'].strip():annotations.setdefault(row['ImageId'],[]).append(row['EncodedPixels'])
    model=inspector_from_saved(frozen).cuda().eval().requires_grad_(False);counter=Counter();records=[]
    for directory in ('cases','originals'):(OUT/directory).mkdir(exist_ok=True)
    for number,row in enumerate(errors,1):
        source=lookup[row['image_id']];assert source['split']=='test';image_path=ROOT/source['image'];assert digest(image_path)==source['sha256']
        with Image.open(image_path) as image:array=np.asarray(image.convert('RGB')).copy()
        truth=np.zeros((256,1600),dtype=np.uint8)
        for text in annotations.get(row['image_id'],[]):truth|=decode_rle(text)
        truth=truth.astype(bool);assert int(truth.any())==row['label']
        score,probability,cam,view_logits=frozen_maps(model,array,policy,'cuda');assert abs(score-row['score'])<.001
        assert int(score>policy['threshold'])==row['prediction']
        counter[row['kind']]+=1;case_id=f"{row['kind']}{counter[row['kind']]:02d}";stem=Path(row['image_id']).stem
        record={**row,'case_id':case_id,'classes':source['classes'],'replayed_score':score,'score_replay_difference':abs(score-row['score']),
                'original':f'originals/{row["image_id"]}','panel':f'cases/{case_id}_{stem}.png',
                'evidence':case_evidence(array,truth,probability,cam,view_logits,row['score'],policy['threshold'],frozen.get('pixel_threshold',.5))}
        views=image_views(array,truth,probability,cam,frozen.get('pixel_threshold',.5));start=patch_start(truth,probability,cam)
        record['crop_x_start']=start;draw_card(record,views,start,OUT/record['panel']);shutil.copy2(image_path,OUT/record['original']);records.append(record)
        print(f'Reviewed maps {number}/{len(errors)} {case_id}',flush=True)
    assert state_hash(model.state_dict())==frozen['model_sha256']
    after={str(p.relative_to(ROOT)):digest(p) for p in protected};assert before==after,'Frozen artifacts changed during review'
    metadata={'model_sha256':frozen['model_sha256'],'frozen_fingerprint':settings['frozen_fingerprint'],'image_threshold':policy['threshold'],
              'pixel_threshold':frozen.get('pixel_threshold',.5),'readonly_integrity_verified':True,'protected_file_sha256':before,
              'gradcam_scope':'Positive final-encoder-layer attribution, aligned/averaged over frozen views; multiscale head branches are not fully represented; normalized per image, not probabilities or causal evidence',
              'attribution_source':'https://arxiv.org/abs/1610.02391','test_used_for_tuning':False,'replayed_archived_errors':43,
              'acceptance_metrics_unchanged':frozen['acceptance']['metrics']}
    render(records,metadata,notes)


if __name__=='__main__':main()
