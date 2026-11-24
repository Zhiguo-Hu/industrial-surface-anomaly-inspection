"""Launch an isolated hidden Edge profile to verify the live local upload workflow."""
import csv,hashlib,io,json,os,shutil,socket,subprocess,sys,uuid
from pathlib import Path
from urllib.request import urlopen
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    service='http://127.0.0.1:8765';node=shutil.which('node');assert node,'Node22+ required for optional browser QA'
    with urlopen(service+'/health',timeout=5) as response:health=json.load(response)
    assert health['dataset_acceptance']=='dataset_acceptance_passed'
    output=ROOT/'runs/steel_browser_interaction';output.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'runs/strip_training_cache/metadata.json').read_text());lookup={r['image_id']:r for r in meta['rows']}
    fixtures=[]
    for image_id,name in [('739cf8127.jpg','=normal,样本.jpg'),('61be1f51b.jpg','positive.jpg')]:
        row=lookup[image_id];assert row['split']=='val';source=ROOT/row['image'];assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
        target=output/name;shutil.copy2(source,target);fixtures.append(target)
    invalid=output/'invalid_size.png';Image.new('RGB',(32,32)).save(invalid);fixtures.append(invalid)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    browser=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    if not browser.exists():browser=Path('C:/Program Files/Microsoft/Edge/Application/msedge.exe')
    assert browser.is_file(),'Edge browser missing'
    profile=output/('profile_'+uuid.uuid4().hex[:8]);log=(output/'browser.log').open('wb')
    command=[str(browser),'--headless','--disable-gpu','--no-first-run','--disable-background-networking','--disable-sync',
             '--no-default-browser-check','--disable-extensions','--remote-debugging-address=127.0.0.1',f'--remote-debugging-port={port}',
             '--user-data-dir='+str(profile),'--window-size=1280,1000','about:blank']
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        result=subprocess.run([node,str(ROOT/'scripts/verify_steel_browser.mjs'),f'http://127.0.0.1:{port}',service,str(output),*[str(p) for p in fixtures]],
                              cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if result.returncode:raise RuntimeError(result.stderr)
        report=json.loads(result.stdout)
        with (output/'browser_export.csv').open(encoding='utf-8-sig',newline='') as stream:records=list(csv.DictReader(stream))
        assert len(records)==3 and [r['status'] for r in records]==['ok','ok','rejected']
        assert records[0]['file']=="'=normal,样本.jpg",'Spreadsheet text escaping failed'
        assert [records[i]['defect_detected'] for i in (0,1)]==['false','true']
        report['model_sha256']=health['model_sha256'];report['csv_roundtrip_verified']=True
        (ROOT/'reports/steel_browser_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        log.close()


if __name__=='__main__':main()
