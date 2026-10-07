"""Verify material evaluation on the same raster labels as visibility."""
from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[];total=0
 for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']:
  cid=c['case_id'];geo=ROOT/'geometry_spoon_source_fit_ellipsoid_v3'/cid;f=np.load(ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4'/cid/'field.npz');a=np.load(ROOT/'spoon_source_fit_ellipsoid_channels_v3'/cid/'geometry_channels.npz');food=a['labels']==2;known=np.zeros(food.shape,bool);known[f['yy'][f['valid']],f['xx'][f['valid']]]=True;unknown=food&~known;coef=np.asarray(json.loads((geo/'appearance_calibration.json').read_text())['coefficient_rgb']);pred=np.zeros((*food.shape,3),np.uint8);pred[food]=np.uint8(np.clip(np.rint(np.maximum(np.c_[np.ones(food.sum()),a['scene_normals'][food]]@coef,.05)*255),0,255));source=np.asarray(Image.open(geo/'source.png').convert('RGB'));edit=np.asarray(Image.open(geo/'edit_mask.png'))>0;source_samples=np.uint8(np.clip(np.rint(f['sampled'][f['valid']]),0,255));previous=np.asarray(Image.open(geo/'rgb_control.png').convert('RGB'));row={'case_id':cid,'unobserved_food_pixels':int(unknown.sum()),'prior_gpu_proxy_dark_food_pixels':int(np.sum(food&(previous.mean(2)<100))),'cells_verified':0}
  for arm in ['source_photo_cut','source_locked_cut','band_photo_cut','band_locked_cut']:
   for seed in [41,163,907]:
    dest=ROOT/'gate_v40'/arm/(cid+'__'+str(seed));pre=np.asarray(Image.open(dest/'pre_sampling.png').convert('RGB'));final=np.asarray(Image.open(dest/'composited.png').convert('RGB'));assert np.array_equal(pre[unknown],pred[unknown]),(cid,arm,seed,'GPU material contamination');assert np.array_equal(final[~edit],source[~edit]);request=json.loads((dest/'result.json').read_text());assert request['gpu_proxy_rgb_used_inside_food'] is False
    if arm.startswith('source_'):assert np.array_equal(pre[f['yy'][f['valid']],f['xx'][f['valid']]],source_samples)
    row['cells_verified']+=1;total+=1
  rows.append(row)
 out={'status':'verified','compositions_verified':total,'cases':rows,'known_source_rgb_exact_before_sampling_in_source_arms':True,'unknown_food_color_evaluated_on_same_cpu_label_normals':True,'scope':'Code and image invariants only. Source-camera approximation, unknown texture and perceived realism are not validated by construction.','script_sha256':sha(Path(__file__))};(ROOT/'label_consistent_food_audit.json').write_text(json.dumps(out,indent=2));print('LABEL_MATERIAL_AUDIT',total,flush=True)
if __name__=='__main__':main()
