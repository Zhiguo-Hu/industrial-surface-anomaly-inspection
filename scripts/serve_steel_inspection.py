"""Loopback-only web/API inference; requires an accepted deployment artifact."""
import argparse,json,re,sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.steel_runtime import SteelRuntime,MAX_IMAGE_BYTES


class Handler(BaseHTTPRequestHandler):
    def reply(self,status,data,kind='application/json; charset=utf-8'):
        content=json.dumps(data,ensure_ascii=False).encode('utf-8') if isinstance(data,dict) else data
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(content)))
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)

    def do_GET(self):
        if self.path=='/':return self.reply(200,(ROOT/'web/steel_inspection.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/health':return self.reply(200,self.server.runtime.info)
        match=re.fullmatch(r'/results/([0-9a-f]{32})/(input|overlay|mask)\.png',self.path)
        if match:
            path=self.server.runtime.result_dir/match[1]/(match[2]+'.png')
            if path.is_file():return self.reply(200,path.read_bytes(),'image/png')
        self.reply(404,{'error':'Not found'})

    def do_POST(self):
        if self.path!='/predict':return self.reply(404,{'error':'Not found'})
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError:return self.reply(400,{'error':'Invalid content length'})
        if not 0<length<=MAX_IMAGE_BYTES:return self.reply(413,{'error':'图片应小于10MB / Image must be at most10MB'})
        try:result=self.server.runtime.predict(self.rfile.read(length))
        except ValueError as exc:return self.reply(400,{'error':str(exc)})
        except Exception:
            import traceback
            traceback.print_exc();return self.reply(500,{'error':'检测失败，请检查本机服务日志 / Inference failed; inspect local log'})
        self.reply(200,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path)
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--result-dir',type=Path,default=ROOT/'runs/local_deployment/results');args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Port must be1024..65535')
    pointer=ROOT/'configs/strip_deployment_current.json'
    if args.model is None:
        if not pointer.exists():parser.error('当前没有通过验收的模型 / No accepted deployment artifact yet')
        args.model=ROOT/json.loads(pointer.read_text())['model']
    runtime=SteelRuntime(args.model,args.result_dir,args.device)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.runtime=runtime
    print(f'Steel inspection / 钢材检测: http://127.0.0.1:{args.port}',flush=True)
    print(json.dumps(runtime.info,ensure_ascii=False),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
