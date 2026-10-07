"""Fixed source-native pixel sampling, applied to every cell without selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--gate',required=True);a.add_argument('--geometry',required=True);args=a.parse_args()
 g=ROOT/args.gate;cfg=json.loads((g/'config.json').read_text());cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};out=g/'source_sampling_v1';out.mkdir(exist_ok=False);rows=[]
 for j in cfg['jobs']:
  ps=list(g.glob('worker_*/'+j['id']+'/composited.png'));assert len(ps)==1,j['id']
  p=ps[0];c=cases[j['case_id']];im=Image.open(p).convert('RGB');original=np.asarray(im,float)
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
  native=im.crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS);sampled=native.resize((w,h),Image.Resampling.LANCZOS)
  canvas=original.copy();canvas[t:t+h,l:l+w]=np.asarray(sampled,float)
  mask=np.asarray(Image.open(j['files']['edit_mask']['path']).convert('L'))>127;alpha=np.clip(distance_transform_edt(mask)/5,0,1)[...,None]
  result=np.uint8(np.clip(np.rint(original*(1-alpha)+canvas*alpha),0,255));source=np.asarray(Image.open(j['files']['source']['path']).convert('RGB'))
  assert np.array_equal(result[~mask],source[~mask]);d=out/j['id'];d.mkdir();Image.fromarray(result).save(d/'composited.png');Image.fromarray(result).crop(rect).save(d/'view.png')
  row={'id':j['id'],'parent_composite':str(p),'parent_sha256':sha(p),'original_source_native_size':c['source_size'],'all_cells_same_rule':True,'outside_edit_source_exact':True,'raw_generation':False,'scope':'Resample the candidate through the existing photograph native pixel grid; no measured PSF or recovered optical calibration is claimed.'};(d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'script_sha256':sha(Path(__file__))},indent=2))
 print('DERIVED',len(rows))
if __name__=='__main__':main()
