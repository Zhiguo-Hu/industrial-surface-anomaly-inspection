"""Local inference contract for a frozen, test-accepted inspection artifact."""
import hashlib,io,json,threading,time,uuid
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from src.strip_inspector import inspector_from_saved
from src.strip_inference import inspect_image
from src.inspection_protocol import frozen_test_fingerprint
from src.reproducibility import configure,state_hash
from src.steel import heat_rgb

MAX_IMAGE_BYTES=10*1024*1024


def decode_steel_image(content):
    if not content or len(content)>MAX_IMAGE_BYTES:raise ValueError('请上传不超过10MB的PNG/JPEG钢材图 / Image must be 1 byte to 10MB')
    try:
        with Image.open(io.BytesIO(content)) as im:
            if im.format not in ('PNG','JPEG') or im.mode not in ('RGB','L'):
                raise ValueError('仅支持8位RGB/灰度PNG或JPEG / 8-bit RGB or grayscale PNG/JPEG required')
            if im.size!=(1600,256):raise ValueError('需要1600×256原尺度钢材图；请勿拉伸任意照片 / Native1600×256 steel image required')
            return np.asarray(im.convert('RGB')).copy()
    except (OSError,Image.DecompressionBombError) as exc:raise ValueError('图片无法读取 / Invalid image') from exc


class SteelRuntime:
    def __init__(self,model_path,result_dir,device='cuda'):
        saved=torch.load(model_path,map_location='cpu',weights_only=True)
        acceptance=saved.get('acceptance',{})
        if acceptance.get('status')!='dataset_acceptance_passed':raise ValueError('模型尚未通过最终测试验收 / Model has not passed final-test acceptance')
        if state_hash(saved['model'])!=saved['model_sha256']:raise ValueError('Model checksum mismatch')
        fingerprint=frozen_test_fingerprint(saved['model_sha256'],saved['inspection_policy'])
        if acceptance.get('frozen_fingerprint')!=fingerprint:raise ValueError('Inference policy differs from the frozen test candidate')
        if saved.get('pixel_threshold',.5)!=saved['inspection_policy'].get('pixel_threshold',.5):raise ValueError('Mask threshold differs from frozen inference policy')
        metrics=acceptance['metrics']
        if metrics['recall']<.95 or metrics['fpr']>.05:raise ValueError('Artifact does not meet configured recall/FPR targets')
        configure(42,True);torch.set_num_threads(4)
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA GPU unavailable')
        self.model=inspector_from_saved(saved).to(device).eval();self.policy=saved['inspection_policy'];self.device=device
        self.result_dir=Path(result_dir).resolve();self.result_dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.pixel_threshold=saved.get('pixel_threshold',.5)
        self.info={'status':'ready','model_sha256':saved['model_sha256'],'device':device,'image_size_wh':[1600,256],
                   'dataset_acceptance':'dataset_acceptance_passed','test_recall':metrics['recall'],'test_normal_fpr':metrics['fpr'],
                   'scope':'Comparable native-resolution steel images; production-camera validation is separate','mask_role':'Auxiliary localization, ungated'}

    def predict(self,content):
        start=time.perf_counter();array=decode_steel_image(content)
        with self.lock:
            score,probability=inspect_image(self.model,array,self.device,self.policy['flip4'],self.policy.get('fusion'))
            if self.device=='cuda':torch.cuda.synchronize()
        identifier=uuid.uuid4().hex;folder=self.result_dir/identifier;folder.mkdir()
        Image.fromarray(array).save(folder/'input.png')
        Image.fromarray((.6*array+.4*heat_rgb(probability,1)).astype(np.uint8)).save(folder/'overlay.png')
        Image.fromarray((probability>self.pixel_threshold).astype(np.uint8)*255).save(folder/'mask.png')
        result={'request_id':identifier,'defect_detected':bool(score>self.policy['threshold']),'score':score,'threshold':self.policy['threshold'],
                'elapsed_ms':(time.perf_counter()-start)*1000,'model_sha256':self.info['model_sha256'],
                'input_sha256':hashlib.sha256(content).hexdigest(),'image_size_wh':[1600,256],
                'input_url':f'/results/{identifier}/input.png','overlay_url':f'/results/{identifier}/overlay.png',
                'mask_url':f'/results/{identifier}/mask.png','score_is_probability':False,'mask_role':'Auxiliary localization, ungated'}
        (folder/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
