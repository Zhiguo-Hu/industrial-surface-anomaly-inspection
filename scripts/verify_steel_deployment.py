"""Verify real accepted-model HTTP and batch inference, including a standalone bundle."""
import argparse,csv,io,json,os,socket,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.inspection_protocol import partition


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package-folder',type=Path);args=parser.parse_args()
    runtime_root=(args.package_folder or ROOT).resolve()
    out=ROOT/'runs/local_deployment'/('package_smoke' if args.package_folder else 'service_smoke');out.mkdir(parents=True,exist_ok=True)
    pointer=json.loads((runtime_root/'configs/strip_deployment_current.json').read_text());model=runtime_root/pointer['model']
    meta=json.loads((ROOT/'runs/strip_training_cache/metadata.json').read_text());rows=meta['rows'];development=partition(rows)['development']
    sources=[rows[development[i]] for i in (0,200)];paths=[ROOT/r['image'] for r in sources]
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}';log=(out/'server.log').open('wb')
    command=[sys.executable,str(runtime_root/'scripts/serve_steel_inspection.py'),'--port',str(port),'--model',str(model),'--result-dir',str(out/'images')]
    process=subprocess.Popen(command,cwd=runtime_root,stdout=log,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    predictions=[]
    try:
        deadline=time.monotonic()+60
        while True:
            if process.poll() is not None:raise RuntimeError(f'Service exited {process.returncode}; inspect {out / "server.log"}')
            try:
                with urlopen(url+'/health',timeout=1) as response:health=json.load(response)
                break
            except (URLError,TimeoutError):
                if time.monotonic()>deadline:raise RuntimeError('Service startup timed out')
                time.sleep(.5)
        assert health['model_sha256']==pointer['model_sha256'] and health['dataset_acceptance']=='dataset_acceptance_passed'
        for source,path in zip(sources,paths):
            with urlopen(Request(url+'/predict',data=path.read_bytes(),headers={'Content-Type':'image/jpeg'}),timeout=30) as response:result=json.load(response)
            assert result['model_sha256']==pointer['model_sha256']
            for key in ('input_url','overlay_url','mask_url'):
                with urlopen(url+result[key],timeout=5) as response:
                    with Image.open(io.BytesIO(response.read())) as image:assert image.size==(1600,256)
            predictions.append({'image_id':source['image_id'],'label':source['label'],'score':result['score'],'prediction':int(result['defect_detected'])})
        buffer=io.BytesIO();Image.new('RGB',(32,32)).save(buffer,format='PNG');invalid=out/'invalid_size.png';invalid.write_bytes(buffer.getvalue())
        try:urlopen(Request(url+'/predict',data=buffer.getvalue()),timeout=5);raise AssertionError('Wrong geometry accepted')
        except HTTPError as error:assert error.code==400
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        log.close()
    batch=out/'batch'
    # Replays never overwrite an earlier batch audit.
    if batch.exists():batch=out/f'batch_{time.time_ns()}'
    completed=subprocess.run([sys.executable,str(runtime_root/'scripts/batch_steel_inspection.py'),'--model',str(model),
                              '--images',*(str(p) for p in paths),str(invalid),'--output-dir',str(batch)],cwd=runtime_root,
                             capture_output=True,text=True,encoding='utf-8',errors='replace')
    if completed.returncode:raise RuntimeError(completed.stderr)
    summary=json.loads((batch/'summary.json').read_text());assert summary['processed']==3 and summary['successful']==2 and summary['rejected']==1
    with (batch/'summary.csv').open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    for record,prediction in zip(records[:2],predictions):
        assert abs(float(record['score'])-prediction['score'])<.001
        assert (record['defect_detected']=='True')==bool(prediction['prediction'])
    report={'status':'passed','standalone_package':bool(args.package_folder),'model_sha256':pointer['model_sha256'],
            'checks':['accepted-model startup','loopback health','two real development images via HTTP','native-resolution PNG results',
                      'invalid-size HTTP rejection','real CLI batch2valid+1invalid','HTTP/CLI frozen-score agreement'],
            'images':predictions,'test_images_reused':False,'scope':'Functional deployment verification; accuracy acceptance remains the one frozen original final test'}
    (ROOT/'reports'/('steel_package_smoke.json' if args.package_folder else 'steel_service_smoke.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
