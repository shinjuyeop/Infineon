from __future__ import annotations
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terrain_fast_reflex_v2_detector import Normalizer, make_windows

def fixture(split="train"):
    rows=[{"split":split,"run_id":"r1","surface_family":"multisine","mode":"sink_dominant","valid":"1"}]
    sensors=np.arange(1500,dtype=np.float32).reshape(1,150,10)
    labels={name:np.zeros((1,150),bool) for name in ("confirmed_slip","incipient_risk","sustained_sink","sustained_tilt")}
    return rows,sensors,labels

class V2DetectorDatasetTest(unittest.TestCase):
    def test_causal_confirmed_endpoint_and_pre_onset_negative(self):
        rows,x,l=fixture("validation");l["confirmed_slip"][0,50+7:]=True
        d=make_windows(rows,x,l,"slip",5,"validation")
        self.assertEqual(d["y"][:7].tolist(),[0]*7);self.assertEqual(int(d["y"][7]),1)
        np.testing.assert_array_equal(d["x"][7,:,0],np.arange(530,580,10))

    def test_sink_target_ignores_tilt_and_allows_dual_positive(self):
        rows,x,l=fixture("validation");l["sustained_tilt"][0,50:]=True
        self.assertFalse(make_windows(rows,x,l,"sink",10,"validation")["y"].any())
        l["sustained_sink"][0,50:]=True;l["confirmed_slip"][0,50:]=True
        self.assertTrue(make_windows(rows,x,l,"sink",10,"validation")["y"].all())
        self.assertTrue(make_windows(rows,x,l,"slip",5,"validation")["y"].all())

    def test_train_normalizer_does_not_depend_on_validation(self):
        train=np.ones((2,5,10),np.float32);normalizer=Normalizer.fit(train)
        self.assertAlmostEqual(float(normalizer.mean[0]),1.0)
        changed_validation=np.full((2,5,10),999.,np.float32)
        self.assertTrue(np.isfinite(normalizer.transform(changed_validation)).all())

if __name__=="__main__":unittest.main()
