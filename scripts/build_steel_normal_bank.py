"""Build, calibrate, save and verify a standalone bank from all normal train strips."""
import argparse,csv,hashlib,json,os,pickle,sys,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.normal_bank import normal_training_rows,domain_extractor,standalone_extractor,image_candidates,select_vectors,validate_bank
from src.reproducibility import configure,state_hash
from src.steel import infer,heat_rgb,tile_batches
from src.baseline import validation_threshold
from scripts.run_steel_pilot import read_image
from scripts.audit_severstal import decode_rle

SOURCE='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s42_cls0p5_tile_class_balanced'
OUTPUT=ROOT/'runs/steel_normal_bank'


def json_read(path):return json.loads(path.read_text(encoding='utf-8'))


def atomic_torch_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def load_ready(device,manifest,manifest_hash):
    saved=torch.load(OUTPUT/'bank.pt',map_location='cpu',weights_only=True)
    validate_bank(saved,manifest['samples'],manifest_hash)
    model=standalone_extractor(saved['feature_model']).to(device).eval()
    return saved,model,saved['bank'].to(device)


@torch.inference_mode()
def verify_vectors(saved,model,device,normal_rows):
    chosen=np.linspace(0,len(saved['bank'])-1,8,dtype=int).tolist();grouped=defaultdict(list)
    for i in chosen:grouped[int(saved['provenance'][i,0])].append(i)
    for source,indices in grouped.items():
        grids=[]
        for batch,_ in tile_batches(read_image(normal_rows[source]),device):grids.extend(model(batch).cpu())
        for i in indices:
            _,tile,y,x=saved['provenance'][i].tolist()
            torch.testing.assert_close(grids[tile][:,y,x],saved['bank'][i],rtol=1e-5,atol=1e-5)
    return {'selected_vectors_replayed':len(chosen),'source_images':len(grouped),'status':'passed at tolerance 1e-5'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--verify',action='store_true');group.add_argument('--image',type=Path)
    args=parser.parse_args();reproducibility=configure(42,True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest_path=ROOT/'configs/severstal_manifest.json';manifest=json_read(manifest_path)
    manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest();normal_rows=normal_training_rows(manifest['samples'])
    if args.image or args.verify:
        saved,model,bank=load_ready(device,manifest,manifest_hash)
        if args.verify:
            verification=verify_vectors(saved,model,device,normal_rows)
            first={r['image_id']:r for r in manifest['samples']}[saved['evaluation_ids'][0]]
            score,heat=infer(model,bank,read_image(first),device)
            np.testing.assert_allclose(score,saved['replay_score'],rtol=1e-5,atol=1e-5)
            assert heat.shape==(256,1600) and np.isfinite(heat).all()
            result={'status':'ready','bank_shape':list(bank.shape),'normal_train_images':len(normal_rows),
                    'vector_provenance_replay':verification,'independent_reload_score_check':'passed',
                    'bank_file_sha256':hashlib.sha256((OUTPUT/'bank.pt').read_bytes()).hexdigest()}
            (OUTPUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            (ROOT/'reports/steel_normal_bank_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        else:
            with Image.open(args.image) as file:array=np.asarray(file.convert('RGB')).copy()
            score,heat=infer(model,bank,array,device)
            Image.fromarray((.6*array+.4*heat_rgb(heat,saved['visual_scale'])).astype(np.uint8)).save(OUTPUT/'single_overlay.png')
            np.save(OUTPUT/'single_map.npy',heat)
            result={'image':str(args.image.resolve()),'score':score,'threshold':saved['threshold'],
                    'prediction':int(score>saved['threshold']),'map_shape':list(heat.shape),'score_is_probability':False,
                    'note':'Domain-trained encoder, normal-only memory vectors; arbitrary camera conditions remain unvalidated'}
            (OUTPUT/'single_prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));return
    OUTPUT.mkdir(parents=True,exist_ok=True);shards=OUTPUT/'shards';shards.mkdir(exist_ok=True)
    source_path=ROOT/f'runs/{SOURCE}/best.pt';source=torch.load(source_path,map_location='cpu',weights_only=True)
    assert source['manifest_sha256']==manifest_hash
    model=domain_extractor(source).to(device).eval()
    feature_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    config={'source_experiment':SOURCE,'source_model_sha256':source['model_state_sha256'],
            'feature_model_sha256':state_hash(feature_state),'manifest_sha256':manifest_hash,
            'train_normal_images':len(normal_rows),'seed':42,'runtime':reproducibility,'per_tile_candidates':32,
            'parents_per_shard':256,'per_shard_coreset':1024,'final_coreset':4096,
            'tile_size':256,'stride':128,'feature_code_sha256':hashlib.sha256((ROOT/'src/normal_bank.py').read_bytes()).hexdigest(),
            'local_feature_code_sha256':hashlib.sha256((ROOT/'src/patchcore.py').read_bytes()).hexdigest()}
    config['tile_code_sha256']=hashlib.sha256((ROOT/'src/steel.py').read_bytes()+(ROOT/'src/tiling.py').read_bytes()).hexdigest()
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    annotation=ROOT/manifest['annotation_csv']
    assert hashlib.sha256(annotation.read_bytes()).hexdigest()==manifest['source_csv_sha256']
    annotations=defaultdict(list)
    with annotation.open(encoding='utf-8-sig',newline='') as file:
        for r in csv.DictReader(file):
            if r['EncodedPixels'].strip():annotations[r['ImageId']].append(r['EncodedPixels'])
    assert all(not annotations[r['image_id']] for r in normal_rows)
    selected=json_read(ROOT/f'configs/{SOURCE}_selection.json');byid={r['image_id']:r for r in manifest['samples']}
    calibration=[byid[i] for i in selected['calibration']];evaluation=[byid[i] for i in selected['evaluation']]
    assert len(calibration)==200 and all(r['split']=='val' and r['label']==0 for r in calibration)
    assert len(evaluation)==200 and all(r['split']=='val' for r in evaluation)
    assert not set(selected['calibration'])&set(selected['evaluation'])
    selection={'normal_train':[r['image_id'] for r in normal_rows],
               'calibration':selected['calibration'],'evaluation':selected['evaluation']}
    (ROOT/'configs/steel_normal_bank_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    (ROOT/'configs/steel_normal_bank_protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    start=time.perf_counter();parts=[];refs=[];resumed=0;candidate_count=0
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for shard_number,offset in enumerate(range(0,len(normal_rows),256)):
        path=shards/f'shard_{shard_number:03d}.pt';cached=None
        if path.exists():
            try:cached=torch.load(path,map_location='cpu',weights_only=True)
            except (OSError,EOFError,RuntimeError,pickle.UnpicklingError):cached=None
            if not isinstance(cached,dict) or not {'fingerprint','features','references','sha256','candidate_count'}<=cached.keys():cached=None
            elif cached.get('fingerprint')!=fingerprint:cached=None
            elif state_hash({'features':cached['features'],'references':cached['references']})!=cached['sha256']:cached=None
        if cached is None:
            raw=[];provenance=[]
            for i in range(offset,min(offset+256,len(normal_rows))):
                values,reference=image_candidates(model,read_image(normal_rows[i]),device,i)
                raw.append(values);provenance.append(reference)
            features,references=select_vectors(torch.cat(raw),torch.cat(provenance),1024,42+shard_number,device)
            cached={'fingerprint':fingerprint,'features':features,'references':references,
                    'candidate_count':sum(len(p) for p in raw),'sha256':state_hash({'features':features,'references':references})}
            atomic_torch_save(cached,path)
        else:resumed+=1
        parts.append(cached['features']);refs.append(cached['references']);candidate_count+=cached['candidate_count']
        progress={'stage':'extract_and_select','normal_images_done':min(offset+256,len(normal_rows)),
                  'normal_images_total':len(normal_rows),'completed_shards':len(parts)}
        (OUTPUT/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8');print(json.dumps(progress),flush=True)
    bank,provenance=select_vectors(torch.cat(parts),torch.cat(refs),4096,42,device)
    bank_device=bank.to(device);build_seconds=time.perf_counter()-start
    calibration_scores=[];scales=[]
    for i,row in enumerate(calibration):
        score,heat=infer(model,bank_device,read_image(row),device)
        calibration_scores.append(score);scales.append(float(np.quantile(heat,.995)))
        if (i+1)%50==0:print(f'Normal-bank calibration {i+1}/200',flush=True)
    threshold=validation_threshold(calibration_scores);visual_scale=max(1e-6,3*float(np.quantile(scales,.995)))
    predictions=[];pixel_auc=[]
    for i,row in enumerate(evaluation):
        score,heat=infer(model,bank_device,read_image(row),device)
        if not np.isfinite(heat).all() or heat.shape!=(256,1600):raise ValueError('Invalid reconstructed anomaly map')
        mask=np.zeros((256,1600),dtype=np.uint8)
        for text in annotations[row['image_id']]:mask|=decode_rle(text)
        if mask.any() and not mask.all():pixel_auc.append(float(roc_auc_score(mask.ravel(),heat.ravel())))
        predictions.append({'image_id':row['image_id'],'label':row['label'],'score':score,'prediction':int(score>threshold)})
        if (i+1)%50==0:print(f'Normal-bank development evaluation {i+1}/200',flush=True)
    saved={'schema':1,'complete':True,'bank':bank,'provenance':provenance,'feature_model':feature_state,
           'bank_sha256':state_hash({'bank':bank}),'provenance_sha256':state_hash({'provenance':provenance}),
           'feature_model_sha256':state_hash(feature_state),'manifest_sha256':manifest_hash,
           'normal_image_ids':selection['normal_train'],'evaluation_ids':selection['evaluation'],
           'threshold':threshold,'visual_scale':visual_scale,'replay_score':predictions[0]['score'],
           'config':config,'fingerprint':fingerprint,
           'method':'Two-stage projected greedy coreset, raw layer2/layer3 local features; NOT standard PatchCore reproduction',
           'supervision':'Encoder was fine-tuned on normal and defective training masks; bank vectors only use normal train images'}
    validate_bank(saved,manifest['samples'],manifest_hash);atomic_torch_save(saved,OUTPUT/'bank.pt')
    restored,restored_model,restored_bank=load_ready(device,manifest,manifest_hash)
    repeat,_=infer(restored_model,restored_bank,read_image(evaluation[0]),device)
    np.testing.assert_allclose(repeat,predictions[0]['score'],rtol=1e-5,atol=1e-5)
    replay=verify_vectors(restored,restored_model,device,normal_rows)
    labels=[r['label'] for r in predictions];scores=[r['score'] for r in predictions]
    tn,fp,fn,tp=confusion_matrix(labels,[r['prediction'] for r in predictions],labels=[0,1]).ravel()
    report={'status':'complete','scope':'Domain-trained feature extractor; normal-train-only memory; reused development evaluation, test untouched',
            'config':config,'candidate_features_seen':candidate_count,'shard_candidates_after_first_selection':sum(len(p) for p in parts),
            'resumed_shards':resumed,'bank_shape':list(bank.shape),'represented_normal_parent_images':len(provenance[:,0].unique()),
            'bank_bytes':(OUTPUT/'bank.pt').stat().st_size,'bank_file_sha256':hashlib.sha256((OUTPUT/'bank.pt').read_bytes()).hexdigest(),
            'bank_sha256':saved['bank_sha256'],'threshold':threshold,'calibration_fpr':float(np.mean(np.array(calibration_scores)>threshold)),
            'image_auroc':float(roc_auc_score(labels,scores)),
            'confusion':dict(zip(['tn','fp','fn','tp'],map(int,[tn,fp,fn,tp]))),
            'recall':float(tp/(tp+fn)),'false_positive_rate':float(fp/(fp+tn)),
            'mean_per_defect_image_pixel_auroc':float(np.mean(pixel_auc)),
            'build_seconds':build_seconds,'total_seconds':time.perf_counter()-start,
            'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
            'normal_only_vector_sources':'Manifest and original annotation CSV checked; no defective/validation/test image contributes memory vectors',
            'vector_provenance_replay':replay,'reload_score_check':'passed',
            'limitations':['Supervised domain encoder, not an entirely normal-only learning method.',
                           'Two-stage approximate coreset and changed encoder/data budget differ from the old pilot; not a single-factor ablation.',
                           'Development results reused; successful bank construction is not production accuracy validation.',
                           'Scores are distances, not defect probabilities; no calibrated binary pixel segmentation.']}
    with (OUTPUT/'validation_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    (OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'reports/steel_normal_bank_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUTPUT/'progress.json').write_text(json.dumps({'stage':'complete','normal_images_done':len(normal_rows),'normal_images_total':len(normal_rows)}),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
