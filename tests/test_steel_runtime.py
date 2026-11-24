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


def png(size=(1600,256),mode='RGB'):
    buffer=io.BytesIO();Image.new(mode,size).save(buffer,format='PNG');return buffer.getvalue()


class RuntimeContractTests(unittest.TestCase):
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
