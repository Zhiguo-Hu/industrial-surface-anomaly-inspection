import copy,io,unittest
import torch
from torchvision.models import resnet18
from src.normal_bank import normal_training_rows,sample_grid,standalone_extractor,validate_bank
from src.patchcore import LocalFeatures
from src.reproducibility import state_hash


class NormalBankTests(unittest.TestCase):
    def row(self,name,split='train',label=0):
        return {'image_id':name,'sha256':name,'split':split,'label':label,'classes':[1] if label else []}

    def test_only_normal_train_and_unique_content(self):
        rows=[self.row('a'),self.row('b','val'),self.row('c','test'),self.row('d',label=1)]
        self.assertEqual([r['image_id'] for r in normal_training_rows(rows)],['a'])
        with self.assertRaises(ValueError):normal_training_rows(rows+[self.row('a')])
        bad=self.row('e');bad['classes']=[1]
        with self.assertRaises(ValueError):normal_training_rows(rows+[bad])

    def test_grid_provenance_reconstructs_selected_features(self):
        f=torch.arange(2*3*4*4).reshape(2,3,4,4).float()
        values,refs=sample_grid(f,7,5,torch.Generator().manual_seed(42),3)
        self.assertEqual(tuple(values.shape),(6,3))
        for v,(source,tile,y,x) in zip(values,refs.tolist()):
            self.assertEqual(source,7);torch.testing.assert_close(v,f[tile-5,:,y,x])
        self.assertEqual(len(set(map(tuple,refs.tolist()))),6)

    def test_self_contained_weights_only_roundtrip(self):
        torch.set_num_threads(2);model=LocalFeatures(resnet18(weights=None)).eval()
        file=io.BytesIO();torch.save(model.state_dict(),file);file.seek(0)
        restored=standalone_extractor(torch.load(file,weights_only=True))
        with torch.inference_mode():
            x=torch.randn(1,3,64,64);torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)

    def test_bank_tampering_sources_and_provenance_are_rejected(self):
        bank=torch.ones(3,384);state={'weight':torch.ones(1)};rows=[self.row('a')]
        saved={'schema':1,'complete':True,'normal_image_ids':['a'],'manifest_sha256':'fixed',
               'bank':bank,'provenance':torch.zeros(3,4,dtype=torch.int64),'bank_sha256':state_hash({'bank':bank}),
               'provenance_sha256':state_hash({'provenance':torch.zeros(3,4,dtype=torch.int64)}),
               'feature_model':state,'feature_model_sha256':state_hash(state),'threshold':1.,'visual_scale':2.}
        self.assertTrue(validate_bank(saved,rows,'fixed'))
        moved=torch.zeros(3,4,dtype=torch.int64);moved[0,1]=1
        for field,value in [('complete',False),('normal_image_ids',['b']),('bank',bank+1),
                            ('provenance',torch.ones(3,4,dtype=torch.int64)),('provenance',moved),('feature_model',{'weight':torch.zeros(1)})]:
            changed=copy.deepcopy(saved);changed[field]=value
            with self.assertRaises(ValueError):validate_bank(changed,rows,'fixed')
