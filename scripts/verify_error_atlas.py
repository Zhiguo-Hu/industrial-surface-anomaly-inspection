"""Optional real-browser QA of the local offline error atlas; no inference or tuning."""
import csv,hashlib,io,json,os,shutil,socket,subprocess,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
JAVASCRIPT=r'''
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const [endpoint,url,out]=process.argv.slice(2);
let tab;
for(let i=0;i<100;i++){
  try{tab=(await(await fetch(endpoint+'/json/list')).json()).find(p=>p.type==='page');if(tab)break}catch{}
  await new Promise(r=>setTimeout(r,200));
}
assert(tab,'Isolated browser unavailable');
const socket=new WebSocket(tab.webSocketDebuggerUrl),pending=new Map();let next=0;
socket.addEventListener('message',e=>{const m=JSON.parse(e.data),p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)});
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true})});
function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++next,timer=setTimeout(()=>reject(Error('Timeout '+method)),10000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const r=await cdp('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
async function wait(expression){for(let i=0;i<100;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100))}throw Error('Page condition failed: '+expression)}
try{
  await cdp('Page.enable');await cdp('Page.navigate',{url});
  await wait(`document.querySelectorAll('article').length===43`);
  const count=()=>evaluate(`document.querySelectorAll('article').length`);
  const select=async kind=>{await evaluate(`document.querySelector('#kind').value=${JSON.stringify(kind)};document.querySelector('#kind').dispatchEvent(new Event('change'))`);return count()};
  assert.equal(await select('FN'),12);assert.equal(await select('FP'),31);assert.equal(await select(''),43);
  await evaluate(`document.querySelector('#search').value='b8be43898';document.querySelector('#search').dispatchEvent(new Event('input'))`);
  assert.equal(await count(),1);
  await evaluate(`document.querySelector('details').open=true;document.querySelector('article img').loading='eager'`);
  await wait(`document.querySelector('article img').complete && document.querySelector('article img').naturalWidth===1600`);
  assert((await evaluate(`document.querySelector('article p').textContent`)).includes('两块面积明显'));
  const shot=await cdp('Page.captureScreenshot',{format:'png'});
  await fs.writeFile(path.join(out,'browser_preview.png'),Buffer.from(shot.data,'base64'));
  const csv=await evaluate(`(async()=>{let blob;const original=URL.createObjectURL;URL.createObjectURL=b=>{blob=b;return original(b)};const click=HTMLAnchorElement.prototype.click;HTMLAnchorElement.prototype.click=function(){};try{document.querySelector('#csv').click();return await blob.text()}finally{URL.createObjectURL=original;HTMLAnchorElement.prototype.click=click}})()`);
  await fs.writeFile(path.join(out,'browser_export.csv'),csv,'utf8');
  console.log(JSON.stringify({status:'passed',browser:(await cdp('Browser.getVersion')).product,all_cases:43,FN:12,FP:31,search_results:1,export_scope:'all 43 records even when filtered',checks:['offline file URL','FN/FP filters','image ID search','expanded panel loads','bilingual observation text','CSV export']}));
}finally{socket.close()}
'''


def main():
    out=ROOT/'runs/frozen_error_review_v1';assert (out/'index.html').is_file()
    records=json.loads((out/'case_records.json').read_text(encoding='utf-8'))
    protected=records['metadata']['protected_file_sha256']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in protected.items())
    node=shutil.which('node');assert node,'Node22+ required for optional browser QA'
    browser=next((p for p in [Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'),Path('C:/Program Files/Microsoft/Edge/Application/msedge.exe')] if p.is_file()),None)
    assert browser,'Microsoft Edge is required for optional browser QA'
    runner=out/'browser_qa.mjs';runner.write_text(JAVASCRIPT,encoding='utf-8')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    profile=out/('qa_profile_'+uuid.uuid4().hex[:8]);flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (out/'browser.log').open('wb') as log:
        process=subprocess.Popen([str(browser),'--headless','--disable-gpu','--no-first-run','--disable-background-networking',
                                  '--disable-sync','--disable-extensions','--no-default-browser-check','--remote-debugging-address=127.0.0.1',
                                  f'--remote-debugging-port={port}','--user-data-dir='+str(profile),'--window-size=1280,1000','about:blank'],
                                 stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        try:
            result=subprocess.run([node,str(runner),f'http://127.0.0.1:{port}',(out/'index.html').as_uri(),str(out)],capture_output=True,text=True,encoding='utf-8',timeout=60,creationflags=flags)
            if result.returncode:raise RuntimeError(result.stderr)
            report=json.loads(result.stdout)
            with (out/'browser_export.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
            assert len(rows)==43 and [r['image_id'] for r in rows]==[r['image_id'] for r in records['cases']]
            assert rows[0]['observation_zh']==records['cases'][0]['review']['observation_zh']
            assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in protected.items())
            report.update(csv_roundtrip_verified=True,frozen_files_unchanged=True)
            (ROOT/'reports/steel_error_atlas_browser_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report,indent=2))
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)


if __name__=='__main__':main()
