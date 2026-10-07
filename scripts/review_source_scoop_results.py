import json,zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageDraw
r=Path('outputs/first_bite_structure_20260930');o=r/'r37';o.mkdir(exist_ok=True)
z=zipfile.ZipFile(r/'gate_v37_complete.zip');q=zipfile.ZipFile(r/'gate_v38_complete.zip')
cases=json.loads((r/'geometry_v3_controls/inputs/manifest.json').read_text())['cases'];rows=[]
def im(z,n):return Image.open(BytesIO(z.read(n))).convert('RGB')
for c in cases:
 cid=c['case_id'];l,t=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(l,t,l+w,t+h)
 src=im(z,'geometry_spoon_source_fit_ellipsoid_v3/'+cid+'/source.png')
 groups=[('parent',None),('body','source_geometry_food_ablation_v1'),('cut','coupled_source_cut_reconstruction_v3/no_food_only_crop_ablation'),('locked_photo','geometry_locked_action_v1/photographic_cut'),('locked_color','geometry_locked_action_v1/source_color_cut')]
 for part,gg in [('a',groups[:3]),('b',groups[3:])]:
  board=Image.new('RGB',(2048,390*len(gg)),'white');draw=ImageDraw.Draw(board)
  for ri,(method,folder) in enumerate(gg):
   imgs=[('source',src)]
   for s in [41,163,907]:
    if method=='parent':n=next(n for n in z.namelist() if '/parent_worker_' in n and '/'+cid+'__source_fit_ellipsoid_spoon_photo_canny__'+str(s)+'/composited.png' in n)
    elif method=='body':n='gate_v37/'+folder+'/'+cid+'__source_geometry_food__'+str(s)+'/composited.png'
    else:n='gate_v37/'+folder+'/'+cid+'__'+str(s)+'/composited.png'
    imgs.append((method+' '+str(s),im(z,n)));rows.append({'case':cid,'method':method,'seed':s,'member':n})
   for j,(label,img) in enumerate(imgs):
    img=img.crop(rect);img.thumbnail((504,354));board.paste(img,(512*j+(504-img.width)//2,390*ri+28));draw.text((512*j+5,390*ri+6),label,fill='black')
  board.save(o/(cid+'_'+part+'.jpg'),quality=95)
 board=Image.new('RGB',(2048,780),'white');draw=ImageDraw.Draw(board)
 for ri,gate in enumerate(['gate_v36','gate_v37']):
  imgs=[('source',src)]+[(gate+' gradient '+str(s),im(q,'gate_v38/'+gate+'/'+cid+'__'+str(s)+'/composited.png')) for s in [41,163,907]]
  for j,(label,img) in enumerate(imgs):
   img=img.crop(rect);img.thumbnail((504,354));board.paste(img,(512*j+(504-img.width)//2,390*ri+28));draw.text((512*j+5,390*ri+6),label,fill='black')
 board.save(o/(cid+'_grad.jpg'),quality=95)
(o/'review_index.json').write_text(json.dumps(rows,indent=2));print('review',len(rows))
print('geometry images',[n for n in z.namelist() if n.startswith('geometry_spoon_source_fit_ellipsoid_v3/new_01_7442/') and n.endswith('.png')])
