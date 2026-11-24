"""Build a data-free local runtime package only from a test-accepted model."""
import argparse,hashlib,json,shutil,sys,zipfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.reproducibility import state_hash
from src.inspection_protocol import frozen_test_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/local_deployment/package');args=parser.parse_args()
    pointer=ROOT/'configs/strip_deployment_current.json'
    if not pointer.exists():parser.error('No accepted deployment artifact yet')
    settings=json.loads(pointer.read_text());source=ROOT/settings['model'];saved=torch.load(source,map_location='cpu',weights_only=True)
    acceptance=saved.get('acceptance',{});policy=saved['inspection_policy'];sha=state_hash(saved['model'])
    if acceptance.get('status')!='dataset_acceptance_passed' or acceptance['metrics']['recall']<.95 or acceptance['metrics']['fpr']>.05:
        parser.error('Model has not passed recall/FPR acceptance')
    assert sha==saved['model_sha256']==settings['model_sha256']
    assert frozen_test_fingerprint(sha,policy)==acceptance['frozen_fingerprint']==settings['frozen_fingerprint']
    output=args.output_dir.resolve()
    if output.exists() and not output.is_dir():parser.error('Output path must be a directory')
    if output.exists() and any(output.iterdir()):parser.error('Use a new/empty output directory; existing package is preserved')
    archive=output.with_suffix('.zip')
    if archive.exists():parser.error('Archive already exists; choose a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    for directory in ('src','scripts','web','configs'):(output/directory).mkdir(exist_ok=True)
    for path in (ROOT/'src').glob('*.py'):shutil.copy2(path,output/'src'/path.name)
    (output/'src/__init__.py').touch(exist_ok=True)
    for name in ('serve_steel_inspection.py','batch_steel_inspection.py','run_steel_deployment.ps1'):
        shutil.copy2(ROOT/'scripts'/name,output/'scripts'/name)
    shutil.copy2(ROOT/'web/steel_inspection.html',output/'web/steel_inspection.html')
    shutil.copy2(ROOT/'requirements.txt',output/'requirements.txt');shutil.copy2(source,output/'model.pt')
    (output/'configs/strip_deployment_current.json').write_text(json.dumps({**settings,'model':'model.pt'},indent=2),encoding='utf-8')
    (output/'acceptance.json').write_text(json.dumps(acceptance,indent=2),encoding='utf-8')
    readme=f'''# 钢材检测本机部署 / Local steel inspection

在本机已验证Python环境运行： / Use the verified local Python environment:

```powershell
.\\scripts\\run_steel_deployment.ps1
```

浏览器打开 http://127.0.0.1:8765 ，单选或多选原尺度1600×256 PNG/JPEG钢材图，逐张查看定位响应并导出CSV。模型不自动拉伸任意手机照片。图片与输出只保存在本机。

Open the loopback URL, select one/multiple native1600×256 steel images, inspect results and export CSV. Images remain local. Move to another machine only after installing matching requirements and passing `-Python` with its actual interpreter path.

独立test图像报警验收：召回 {acceptance['metrics']['recall']:.2%}、正常误报 {acceptance['metrics']['fpr']:.2%}。定位掩码另行评估；现场相机采集分布需单独验证。这里不是任意物品/任意拍摄条件的通用检测器。

Frozen independent-test image alarms: recall{acceptance['metrics']['recall']:.2%}, normal FPR{acceptance['metrics']['fpr']:.2%}. Localization and production-camera validation are separate; scope is comparable steel acquisition.

模型SHA256 / Model state SHA256: `{sha}`. This package contains no training/validation/test images or caches. Full audit: acceptance.json and bundle_manifest.json.
'''
    (output/'README.md').write_text(readme,encoding='utf-8')
    hashes={path.relative_to(output).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.rglob('*')) if path.is_file()}
    manifest={'model_sha256':sha,'files_sha256':hashes,'contains_dataset_images':False,'runtime_only':True}
    (output/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(output))
    result={'package_folder':str(output),'archive':str(archive),'archive_bytes':archive.stat().st_size,
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'model_sha256':sha,'dataset_acceptance':'passed'}
    (ROOT/'reports/steel_deployment_package.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('package_folder','archive')},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
