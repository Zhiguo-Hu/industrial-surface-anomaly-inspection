import io,json,tempfile,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from PIL import Image
import torch
from src.steel_runtime import SteelRuntime,decode_steel_image,MAX_IMAGE_BYTES
from scripts.serve_steel_inspection import Handler
from scripts.batch_steel_inspection import inspect_files
from src.localization_metrics import localization_metrics
from src.np_calibration import normal_fpr_threshold


def png(size=(1600,256),mode='RGB'):
    buffer=io.BytesIO();Image.new(mode,size).save(buffer,format='PNG');return buffer.getvalue()


class RuntimeContractTests(unittest.TestCase):
    def test_np_calibration_rank_satisfies_exact_minimal_binomial_bound(self):
        import math
        result=normal_fpr_threshold(np.arange(100,dtype=float),alpha=.1,delta=.05);k=result['rank_one_based']
        def exact_tail(rank):return sum(math.comb(100,j)*(.9**j)*(.1**(100-j)) for j in range(rank,101))
        self.assertLessEqual(exact_tail(k),.05);self.assertGreater(exact_tail(k-1),.05)
        self.assertAlmostEqual(result['binomial_tail_bound'],exact_tail(k),places=12)
        self.assertEqual(result['threshold'],k-1)
        with self.assertRaises(ValueError):normal_fpr_threshold(np.arange(10,dtype=float))
        with self.assertRaises(ValueError):normal_fpr_threshold([1.,float('nan')])

    def test_localization_counts_include_normal_false_positive_pixels(self):
        metrics=localization_metrics([[4,5,8],[0,0,2]],.3,pixels_per_image=10)
        self.assertAlmostEqual(metrics['global_dice'],8/15);self.assertAlmostEqual(metrics['global_iou'],4/11)
        self.assertAlmostEqual(metrics['pixel_recall'],.8);self.assertAlmostEqual(metrics['pixel_precision'],.4)
        self.assertAlmostEqual(metrics['mean_positive_image_dice'],8/13);self.assertAlmostEqual(metrics['normal_pixel_fpr'],.2)

    def test_final_loader_rle_and_hash_contract_without_accessing_real_test(self):
        import hashlib
        from unittest.mock import patch
        from scripts import final_test_strip_inspector as module
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'synthetic.png';path.write_bytes(png())
            row={'image':'synthetic.png','image_id':'synthetic.png','label':1,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.object(module,'ROOT',root):
                dataset=module.FinalImages([row],{'synthetic.png':['409600 1']})
                image,label,index,mask=dataset[0]
                self.assertEqual(image.shape,(3,256,1600));self.assertEqual((label,index),(1,0))
                self.assertEqual(mask.sum().item(),1);self.assertEqual(mask[0,255,1599].item(),1)
                with self.assertRaises(AssertionError):module.FinalImages([row],{})[0]

    def test_batch_preserves_results_and_rejects_invalid_images_individually(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);normal=root/'正常,样本.png';normal.write_bytes(png())
            damaged=root/'invalid.png';damaged.write_bytes(png((32,32)))
            positive=root/'positive.png';buffer=io.BytesIO();Image.new('RGB',(1600,256),'white').save(buffer,format='PNG');positive.write_bytes(buffer.getvalue())
            oversized=root/'large.png'
            with oversized.open('wb') as stream:stream.truncate(MAX_IMAGE_BYTES+1)
            class FakeRuntime:
                info={'status':'test_stub'}
                result_dir=root/'outputs'
                calls=0
                def predict(self,content):
                    self.calls+=1;array=decode_steel_image(content)
                    return {'request_id':str(self.calls),'defect_detected':bool(array.mean()>127),'score':float(array.mean()),'threshold':127.,'elapsed_ms':1.}
            runtime=FakeRuntime();summary=inspect_files(runtime,[normal,damaged,positive,oversized],root/'summary')
            self.assertEqual(summary['processed'],4);self.assertEqual(summary['successful'],2);self.assertEqual(summary['rejected'],2)
            self.assertEqual(summary['defect_detected'],1);self.assertEqual(runtime.calls,3)
            with (root/'summary/summary.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(Path(rows[0]['file']).name,'正常,样本.png');self.assertEqual(rows[1]['status'],'rejected');self.assertEqual(rows[2]['status'],'ok')

    def test_native_rgb_decode_and_rejection_of_wrong_acquisition(self):
        array=decode_steel_image(png());self.assertEqual(array.shape,(256,1600,3));self.assertEqual(array.dtype,np.uint8)
        self.assertEqual(decode_steel_image(png(mode='L')).shape,(256,1600,3))
        for content in (b'',b'not an image',png((810,1440)),png(mode='RGBA'),b'x'*(MAX_IMAGE_BYTES+1)):
            with self.assertRaises(ValueError):decode_steel_image(content)

    def test_research_model_is_rejected_before_cuda_or_model_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'research.pt';torch.save({'acceptance':{'status':'dataset_acceptance_failed'}},path)
            with self.assertRaisesRegex(ValueError,'final-test'):SteelRuntime(path,Path(folder)/'out')

    def test_loopback_api_contract_media_and_path_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);identifier='a'*32;(directory/identifier).mkdir();(directory/identifier/'mask.png').write_bytes(png())
            class FakeRuntime:
                info={'status':'ready'}
                result_dir=directory
                def predict(self,content):
                    decode_steel_image(content);return {'request_id':identifier,'defect_detected':False}
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.runtime=FakeRuntime()
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(url+'/health',timeout=5) as response:self.assertEqual(json.load(response)['status'],'ready')
                request=Request(url+'/predict',data=png(),headers={'Content-Type':'image/png'})
                with urlopen(request,timeout=5) as response:self.assertFalse(json.load(response)['defect_detected'])
                with urlopen(url+f'/results/{identifier}/mask.png',timeout=5) as response:self.assertEqual(response.headers['Content-Type'],'image/png')
                for path in ('/results/../requirements.txt',f'/results/{identifier}/prediction.json'):
                    with self.assertRaises(HTTPError) as error:urlopen(url+path,timeout=5)
                    self.assertEqual(error.exception.code,404)
                with self.assertRaises(HTTPError) as error:urlopen(Request(url+'/predict',data=png((32,32))),timeout=5)
                self.assertEqual(error.exception.code,400)
            finally:server.shutdown();server.server_close();thread.join(timeout=5)
