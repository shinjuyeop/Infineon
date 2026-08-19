from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terrain_fast_reflex_system_v1 import Decision,case_for
class TestSystem(unittest.TestCase):
 def test_cases(self):
  self.assertEqual([case_for(*x) for x in [('marble','ice'),('marble','sand'),('ice','marble'),('sand','marble')]],list('ABCD'))
 def test_decision(self):
  d=Decision('marble');self.assertFalse(d.update('ice',False,False)['case_reflex_required']);self.assertTrue(d.update('ice',True,False)['case_reflex_required'])
  d=Decision('ice');self.assertTrue(d.update('marble',False,False)['recovery_required'])
if __name__=='__main__':unittest.main()
