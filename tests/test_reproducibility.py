import unittest
import io
import numpy as np
import random
import torch
from src.reproducibility import configure,state_hash


class ReproducibilityTests(unittest.TestCase):
    def test_seed_replays_python_numpy_and_torch(self):
        configure(42);first=(random.random(),np.random.rand(),torch.rand(3))
        configure(42);second=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(first[:2],second[:2]);torch.testing.assert_close(first[2],second[2],rtol=0,atol=0)
        configure(43);self.assertFalse(torch.equal(first[2],torch.rand(3)))

    def test_state_hash_is_order_independent_and_value_sensitive(self):
        a={'weight':torch.tensor([1.,2.]),'counter':torch.tensor(5)}
        self.assertEqual(state_hash(a),state_hash(dict(reversed(list(a.items())))))
        b={k:v.clone() for k,v in a.items()};b['weight'][0]=3
        self.assertNotEqual(state_hash(a),state_hash(b))

    def test_settings_metadata_supports_weights_only_loading(self):
        stream=io.BytesIO()
        settings=configure(42)
        torch.save({'settings':settings,'model':{'weight':torch.ones(2)}},stream)
        stream.seek(0)
        result=torch.load(stream,weights_only=True)
        self.assertEqual(result['settings'],settings)
        self.assertIs(type(result['settings']['torch']),str)
