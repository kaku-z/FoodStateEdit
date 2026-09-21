import sys,unittest
from pathlib import Path
import numpy as np
import tempfile
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_spatial_stress_v1 import fit_similarity
from review_spatial_stress_v1 import masked_mae, sheet
class SimilarityTests(unittest.TestCase):
    def test_known_rotation_scale(self):
        x=np.array([[-2.,-1.],[2.,-1.],[2.,1.],[-2.,1.]])
        a=.4;m=1.7*np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        np.testing.assert_allclose(fit_similarity(x,x@m),m,atol=1e-12)
    def test_no_reflection(self):
        x=np.array([[-2.,-1.],[2.,-1.],[2.,1.],[-2.,1.]])
        self.assertGreaterEqual(np.linalg.det(fit_similarity(x,x*[-1,1])),0)
    def test_masked_mae_does_not_wrap_uint8(self):
        x=np.zeros((2,2,3),dtype=np.uint8)
        y=x.copy();y[0,0]=255
        mask=np.array([[True,False],[False,False]])
        self.assertEqual(masked_mae(x,y,mask),255.)
        self.assertEqual(masked_mae(x,y,~mask),0.)
    def test_sheet_contains_all21_frames(self):
        frames=[np.full((24,32,3),i*10,dtype=np.uint8) for i in range(21)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sheet.png'
            sheet(frames,[str(i) for i in range(21)],p,columns=7,cell_width=32)
            im=np.array(Image.open(p))
            self.assertEqual(im.shape,(144,224,3))
            for i in range(21):
                self.assertTrue(np.all(im[i//7*48+30,i%7*32+16]==i*10))
if __name__=='__main__':unittest.main()
