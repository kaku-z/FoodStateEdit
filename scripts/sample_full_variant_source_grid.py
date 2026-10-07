"""Fixed photograph-native sampling for full-image derived variants."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gate',required=True);ap.add_argument('--parent-variant',required=True);args=ap.parse_args();g=ROOT/args.gate;parent=g/args.parent_variant;manifest=json.loads((parent/'manifest.json').read_text());out=g/(args.parent_variant+'_source_grid_v1');out.mkdir(exist_ok=False);cases={c['case_id']:c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']};rows=[]
 for row in manifest['rows']:
  ident=row['id'];cid=ident.split('__')[0];c=cases[cid];p=parent/ident/'composited.png';im=Image.open(p).convert('RGB');image=np.asarray(im,float);l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
  native=im.crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS);resampled=native.resize((w,h),Image.Resampling.LANCZOS);canvas=image.copy();canvas[t:t+h,l:l+w]=np.asarray(resampled,float)
  geometry=ROOT/'geometry_spoon_open_corner_v1'/cid;mask=np.asarray(Image.open(geometry/'edit_mask.png'))>0;source=np.asarray(Image.open(geometry/'source.png').convert('RGB'));alpha=np.clip(distance_transform_edt(mask)/5,0,1)[...,None]
  final=np.uint8(np.clip(np.rint(image*(1-alpha)+canvas*alpha),0,255));assert np.array_equal(final[~mask],source[~mask]);d=out/ident;d.mkdir();Image.fromarray(final).save(d/'composited.png');Image.fromarray(final).crop(rect).save(d/'view.png');Image.fromarray(final).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).save(d/'native_resolution.png')
  report={'id':ident,'raw_generation':False,'parent_sha256':sha(p),'fixed_source_native_size':c['source_size'],'outside_action_original_source_exact':True,'scope':'Fixed source-native pixel-grid resampling, no measured PSF recovery or camera calibration claim.'};(d/'result.json').write_text(json.dumps(report,indent=2));rows.append(report)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'all_cells_same_rule':True,'additional_raw_calls':0,'script_sha256':sha(Path(__file__))},indent=2));print('SAMPLED',len(rows))
if __name__=='__main__':main()
