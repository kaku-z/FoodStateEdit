"""Reconstruct the complete source cut: remaining material and revealed plate.

The cut is a monocular prior. Background continuation and hidden material are
inferred, never claimed to be observed before/after truth.
"""
import argparse,json,hashlib,time,zipfile
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,distance_transform_edt,gaussian_filter
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arms',nargs='+',default=['consumptive_surface_neural_light','no_food_only_crop_ablation']);ap.add_argument('--gate',default='gate_v35');ap.add_argument('--geometry',default='geometry_spoon_observed_surface_v1');ap.add_argument('--channels',default='spoon_observed_surface_channels_v1');ap.add_argument('--transport',default='all_observed_surface_consumptive_v2');ap.add_argument('--ablation',default='source_geometry_food_ablation_v1');ap.add_argument('--method',default='observed_surface_food_intrinsic');ap.add_argument('--cavity-field',default='cavity_source_material_v2');ap.add_argument('--output',default='coupled_source_cut_reconstruction_v3');a=ap.parse_args();g=ROOT/a.gate;out=g/a.output;out.mkdir(exist_ok=False);(out/'executed_script.py').write_bytes(Path(__file__).read_bytes());cases=json.loads((ROOT/'inputs/manifest.json').read_text())['cases'];rows=[]
 plan={'created_unix':time.time(),'frozen_before_projection':True,'raw_calls':0,'cases':8,'seeds':[41,163,907],'args':vars(a),'rule':'Whole source-cutter footprint is reconstructed from remaining geometry: observed inherited material retains source, new cut material uses fixed source-color prior, and disappeared silhouette reveals an inferred plate continuation. Never choose an output seed.'};(out/'frozen_plan.json').write_text(json.dumps(plan,indent=2))
 for c in cases:
  cid=c['case_id'];geo=ROOT/a.geometry/cid;data=np.load(ROOT/a.channels/cid/'geometry_channels.npz');rep=json.loads((geo/'geometry_report.json').read_text());L=max(np.asarray(rep['fit']['high'])-np.asarray(rep['fit']['low']));z=data['remaining_depth'];full=data['full_depth'];finite=np.isfinite(z)&np.isfinite(full);delta=np.zeros(z.shape);np.subtract(z,full,out=delta,where=finite);cut=np.asarray(Image.open(geo/'source_bite_mask.png'))>0;food=np.asarray(Image.open(geo/'food_mask.png'))>0;source=np.asarray(Image.open(geo/'source.png').convert('RGB'));hole=np.asarray(Image.open(geo/'hole_mask.png'))>0
  fresh=finite&cut&(delta>L*1e-5)&(data['labels']==1);background=cut&np.isfinite(full)&~np.isfinite(z);reconstruct=hole&food;reconstruct|=fresh|background;assert fresh.sum()>50
  cf=np.load(g/a.cavity_field/(cid+'_field')/'field.npz');color=np.zeros(source.shape,float);color[cf['mask']]=cf['pixels_rgb'];assert np.all(cf['mask'][fresh])
  plate=cv2.inpaint(source,np.uint8(binary_dilation(food,iterations=5))*255,3,cv2.INPAINT_TELEA)
  material=source.astype(float).copy();material[fresh]=color[fresh];material[background]=plate[background]
  # Restore the source outside the Boolean-changed surface within the cut
  # region. This removes generated ledges which are absent from the proxy.
  l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h);edit=np.asarray(Image.open(geo/'edit_mask.png'))>0
  fd=out/(cid+'_field');fd.mkdir();np.savez_compressed(fd/'field.npz',fresh=fresh,background=background,reconstruct=reconstruct,rgb=material);Image.fromarray(np.uint8(background)*255).save(fd/'background_reveal_mask.png');Image.fromarray(np.uint8(reconstruct)*255).save(fd/'source_cut_mask.png')
  for seed in [41,163,907]:
   for arm in a.arms:
    if arm=='consumptive_surface_neural_light':jid=cid+'__'+a.method+'__'+str(seed);parent=g/a.transport/'source_rgb_scalar_neural_light'/jid/'transported.png'
    else:jid=cid+'__source_geometry_food__'+str(seed);parent=g/a.ablation/jid/'pre_sampling.png'
    base=np.asarray(Image.open(parent).convert('RGB'),float);candidate=base.copy();candidate[reconstruct]=material[reconstruct];alpha=np.clip(distance_transform_edt(reconstruct)/1,0,1)[...,None];pre=np.uint8(np.clip(np.rint(base*(1-alpha)+candidate*alpha),0,255));assert np.array_equal(pre[~reconstruct],base.astype(np.uint8)[~reconstruct]);native=Image.fromarray(pre).crop(rect).resize(tuple(c['source_size']),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.LANCZOS);grid=source.copy();grid[t:t+h,l:l+w]=np.asarray(native);blend=np.clip(distance_transform_edt(edit)/5,0,1)[...,None];final=np.uint8(np.clip(np.rint(source*(1-blend)+grid*blend),0,255));assert np.array_equal(final[~edit],source[~edit]);dest=out/arm/(cid+'__'+str(seed));dest.mkdir(parents=True);Image.fromarray(pre).save(dest/'pre_sampling.png');Image.fromarray(final).save(dest/'composited.png');Image.fromarray(final).crop(rect).save(dest/'view.png');row={'id':cid+'__'+str(seed),'case_id':cid,'seed':seed,'arm':arm,'parent':str(parent),'parent_sha256':sha(parent),'raw_calls':0,'fresh_pixels':int(fresh.sum()),'revealed_background_pixels':int(background.sum()),'reconstructed_source_cut_pixels':int(reconstruct.sum()),'outside_cut_parent_exact_before_sampling':True,'outside_edit_source_exact':True,'scope':'Shared proxy geometry with inferred cavity material and plate continuation. No actual hidden-scene measurement or perceptual pass.'};(dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
  for arm in a.arms:
   paths=[('source',geo/'source.png')]+[('complete source cut seed'+str(s),out/arm/(cid+'__'+str(s))/'composited.png') for s in [41,163,907]];board=Image.new('RGB',(2560,h+28),'white');dr=ImageDraw.Draw(board)
   for i,(title,p) in enumerate(paths):board.paste(Image.open(p).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));dr.text((i*640+5,6),title,fill='black')
   board.save(out/(cid+'_'+arm+'_three_seed.jpg'),quality=95)
  print('CUT_RECONSTRUCTION',cid,int(background.sum()),flush=True)
 (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','derived_composites':len(rows),'all_cells_same_rule':True,'additional_raw_calls':0,'rows':rows,'script_sha256':sha(Path(__file__))},indent=2))
 with zipfile.ZipFile(ROOT/(a.gate+'_'+a.output+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(ROOT))
if __name__=='__main__':main()
