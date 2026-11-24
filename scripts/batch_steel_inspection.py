"""Single/batch image inference using an accepted artifact, with durable CSV output."""
import argparse,csv,json,sys,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.steel_runtime import SteelRuntime,MAX_IMAGE_BYTES


def inspect_files(runtime,paths,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);records=[];fatal=False
    fields=['file','status','defect_detected','score','threshold','elapsed_ms','request_id','error']
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();stream.flush()
        for path in paths:
            path=Path(path);record={'file':str(path.resolve()),'status':'ok','error':''}
            try:
                if path.stat().st_size>MAX_IMAGE_BYTES:raise ValueError('Image exceeds10MB')
                result=runtime.predict(path.read_bytes())
                record.update({k:result[k] for k in fields if k in result})
            except (ValueError,OSError) as exc:record.update(status='rejected',error=str(exc))
            except Exception as exc:record.update(status='runtime_error',error=str(exc));fatal=True
            records.append(record);writer.writerow(record);stream.flush()
            print(f'Processed {len(records)}/{len(paths)}: {record["status"]}',flush=True)
            if fatal:break
    ok=[r for r in records if r['status']=='ok']
    summary={'status':'incomplete' if fatal else 'completed','requested':len(paths),'processed':len(records),
             'successful':len(ok),'defect_detected':sum(bool(r['defect_detected']) for r in ok),
             'no_defect_detected':sum(not r['defect_detected'] for r in ok),'rejected':sum(r['status']=='rejected' for r in records),
             'runtime_errors':sum(r['status']=='runtime_error' for r in records),'model':runtime.info,
             'csv':'summary.csv','image_outputs':str(runtime.result_dir.resolve())}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8');return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-dir',type=Path);group.add_argument('--images',nargs='+',type=Path)
    parser.add_argument('--model',type=Path);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda');args=parser.parse_args()
    if args.input_dir:
        if not args.input_dir.is_dir():parser.error('Image directory not found')
        paths=sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')])
    else:paths=list(dict.fromkeys(args.images))
    if not paths:parser.error('No PNG/JPEG images selected')
    if args.model is None:
        pointer=ROOT/'configs/strip_deployment_current.json'
        if not pointer.exists():parser.error('当前没有通过最终测试验收的模型 / No accepted artifact yet')
        args.model=ROOT/json.loads(pointer.read_text())['model']
    output=args.output_dir or ROOT/'runs/local_deployment/batches'/uuid.uuid4().hex
    if output.exists() and not output.is_dir():parser.error('Output path must be a directory')
    if output.exists() and any(output.iterdir()):parser.error('Output directory must be empty to preserve existing results')
    runtime=SteelRuntime(args.model,output/'images',args.device);summary=inspect_files(runtime,paths,output)
    print(json.dumps(summary,indent=2));return int(summary['runtime_errors']>0)


if __name__=='__main__':raise SystemExit(main())
