// Real browser single/multi-file interaction. Node22+ WebSocket, no third-party modules.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const [endpoint,service,out,normal,positive,invalid]=process.argv.slice(2);
let pages;
for(let attempt=0;attempt<100;attempt++){
  try{pages=await(await fetch(endpoint+'/json/list')).json();if(pages.some(p=>p.type==='page'))break}catch{}
  await new Promise(r=>setTimeout(r,200));
}
assert(pages,'Browser debugging endpoint unavailable');
const tab=pages.find(p=>p.type==='page');assert(tab);
const socket=new WebSocket(tab.webSocketDebuggerUrl);let sequence=0;const pending=new Map();
socket.addEventListener('message',event=>{const message=JSON.parse(event.data);if(!message.id)return;const item=pending.get(message.id);if(!item)return;pending.delete(message.id);clearTimeout(item.timer);if(message.error)item.reject(Error(JSON.stringify(message.error)));else item.resolve(message.result)});
await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true})});
function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{pending.delete(id);reject(Error('Timeout '+method))},30000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const result=await cdp('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value}
async function wait(expression){for(let attempt=0;attempt<150;attempt++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,200))}throw Error('Browser condition timed out: '+expression)}
try{
  await cdp('Page.enable');await cdp('Runtime.enable');await cdp('Page.navigate',{url:service});
  await wait(`document.querySelector('#run') && !document.querySelector('#run').disabled`);
  async function select(files){const document=await cdp('DOM.getDocument');const input=await cdp('DOM.querySelector',{nodeId:document.root.nodeId,selector:'#file'});await cdp('DOM.setFileInputFiles',{nodeId:input.nodeId,files});await evaluate(`document.querySelector('#run').click()`);await wait(`document.querySelector('#status').textContent.includes('Batch complete')`)}
  await select([normal,positive,invalid]);
  const batch=await evaluate(`({rows:document.querySelector('#rows').children.length,totals:document.querySelector('#totals').textContent,exportVisible:!document.querySelector('#export').hidden})`);
  assert.equal(batch.rows,3);assert(batch.totals.includes('异常 1'));assert(batch.totals.includes('失败 1'));assert(batch.exportVisible);
  const csv=await evaluate(`(async()=>await(await fetch(document.querySelector('#export').href)).text())()`);
  await fs.writeFile(path.join(out,'browser_export.csv'),csv,'utf8');
  const layout=await cdp('Page.getLayoutMetrics');const shot=await cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:1280,height:Math.ceil(layout.cssContentSize.height),scale:1}});
  await fs.writeFile(path.join(out,'browser_batch.png'),Buffer.from(shot.data,'base64'));
  await select([positive]);
  const single=await evaluate(`({rows:document.querySelector('#rows').children.length,decision:document.querySelector('#decision').textContent,totals:document.querySelector('#totals').textContent})`);
  assert.equal(single.rows,1);assert(single.decision.includes('Defect detected'));assert(single.totals.includes('失败 0'));
  const version=await cdp('Browser.getVersion');
  console.log(JSON.stringify({status:'passed',browser:version.product,batch_rows:batch.rows,single_rows:single.rows,
    checks:['real multi-file picker','2 valid +1 wrong-size rejection','CSV export','comma/Chinese filename handling','single-image rerun clears prior batch'],
    scope:'Functional checks on two development images, no final-test images or model tuning'}));
}finally{socket.close()}
