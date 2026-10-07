"""Collect every fixed cell, with fixed source-content views and separate raw/composited artifacts."""
import argparse,json,hashlib,time,zipfile
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def read(p):
 for i in range(10):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):time.sleep(.3)
 raise RuntimeError(str(p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--tag',required=True);a.add_argument('--require-complete',action='store_true');args=a.parse_args()
 g=ROOT/'gate_v18';cfg=read(g/'config.json');cases=read(ROOT/'inputs/manifest.json')['cases'];found={};states=[]
 for p in sorted(g.glob('worker_*/manifest.json')):
  m=read(p);states.append(dict(worker=p.parent.name,status=m['status'],completed=len(m['completed']),error=m.get('error')))
  for x in m['completed']:found[x['id']]=p.parent/x['id']
 complete=set(found)=={j['id'] for j in cfg['jobs']}
 if args.require_complete:assert complete,states
 rows=[];out=g/('collection_'+args.tag);out.mkdir(exist_ok=False)
 for c in cases:
  cid=c['case_id'];geometry=ROOT/'geometry_spoon_open_corner_v1'/cid
  left,top=c['preprocessing']['pad_left_top'];w,h=c['preprocessing']['resized'];rect=(left,top,left+w,top+h)
  tiles=[('source',Image.open(geometry/'source.png')),('geometry',Image.open(geometry/'rgb_control.png'))]
  for j in cfg['jobs']:
   if j['case_id']!=cid or j['id'] not in found:continue
   d=found[j['id']];raw=d/'raw_rgba.png';final=d/'composited.png';view=Image.open(final).convert('RGB').crop(rect)
   view.save(out/(j['id']+'__view.png'))
   tiles.append((j['method']+' seed '+str(j['seed']),Image.open(final)))
   rows.append({'id':j['id'],'raw_sha256':sha(raw),'composite_sha256':sha(final),'view_crop':rect,'outside_source_exact':read(d/'result.json')['outside_source_exact']})
  board=Image.new('RGB',(640*len(tiles),h+28),'white');draw=ImageDraw.Draw(board)
  for i,(label,im) in enumerate(tiles):
   crop=im.convert('RGB').crop(rect);crop=crop.resize((640,h),Image.Resampling.LANCZOS);board.paste(crop,(640*i,28));draw.text((640*i+5,6),label,fill='black')
  board.save(out/(cid+'_comparison.jpg'),quality=95)
 manifest={'status':'complete_unreviewed' if complete else 'partial_snapshot','expected_raw':len(cfg['jobs']),'completed_raw':len(found),'states':states,'rows':rows,'review':'None assigned by collector','script_sha256':sha(Path(__file__))}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 if complete:(g/'completion.json').write_text(json.dumps(manifest,indent=2))
 archive=ROOT/('gate_v18_'+args.tag+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in g.glob('*.json'):z.write(p,p.relative_to(ROOT))
  for d in found.values():
   for p in d.iterdir():
    if p.is_file():z.write(p,p.relative_to(ROOT))
  for p in out.iterdir():z.write(p,p.relative_to(ROOT))
  audit=ROOT/'visibility_pose_independent_audit.json'
  if audit.exists():z.write(audit,audit.relative_to(ROOT))
 print(manifest['status'],len(found),str(archive),flush=True)
if __name__=='__main__':main()
