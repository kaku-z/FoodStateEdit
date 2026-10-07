"""Freeze approximate source-image surface landmarks for UV correspondence QA.

Annotations were made from original images only, after detecting source-plane
misalignment in development. They are neither measured 3-D nor heldout labels.
Clockwise order matches the existing full top: (hi,lo),(hi,hi),(lo,hi),(lo,lo).
"""
import json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
LANDMARKS={
 'new_01_7442':[[130,223],[463,292],[563,187],[302,127]],
 'new_02_7496':[[165,221],[364,337],[474,191],[306,92]],
 'new_03_7443':[[175,164],[215,301],[513,253],[432,132]],
 'new_04_7459':[[222,169],[308,323],[490,273],[369,126]],
 'prospective_01_7473':[[178,241],[349,297],[425,157],[258,108]],
 'prospective_02_7441':[[237.7143,274.2857],[466.2857,278.8571],[416,187.4286],[272,171.4286]],
 'prospective_03_11160':[[83,281],[287,365],[402,97],[229,70]],
 'prospective_04_7498':[[165,316],[402,357],[445,270],[247,236]],
}
REAR_INFERRED=['new_01_7442','new_03_7443','prospective_01_7473','prospective_02_7441','prospective_04_7498']
def main():
 out=ROOT/'source_top_landmarks_v1';out.mkdir(exist_ok=False);rows=[]
 projected=json.loads((ROOT/'source_landmark_diagnostics_v1/projected_corners.json').read_text())
 for item in projected:
  cid=item['case_id'];old=np.asarray(item['projected_top_corners'],np.float32);new=np.asarray(LANDMARKS[cid],np.float32)
  H=cv2.getPerspectiveTransform(old,new);assert np.linalg.det(H)!=0
  source=ROOT/'geometry_spoon_open_corner_v1'/cid/'source.png'
  row={'case_id':cid,'canvas_top_corners':new.tolist(),'old_projected_top_corners':old.tolist(),
       'old_to_source_top_homography':H.tolist(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'rear_corner_partly_occluded_inferred':cid in REAR_INFERRED,
       'manual_approximate_source_landmarks':True,
       'in_sample_old_plane_landmark_discrepancy_px':np.linalg.norm(new-old,axis=1).tolist(),
       'scope':'A source-only projective appearance correspondence correction, not a source 3D refit, independent geometry validation or measured garnish height.'}
  (out/(cid+'.json')).write_text(json.dumps(row,indent=2));rows.append(row)
  im=Image.open(source).convert('RGB');d=ImageDraw.Draw(im);d.line([tuple(x) for x in np.r_[new,new[:1]]],fill='lime',width=2)
  for i,(x,y) in enumerate(new):d.text((x+4,y-12),str(i),fill='lime')
  im.save(out/(cid+'.png'))
 (out/'manifest.json').write_text(json.dumps({'rows':rows,'scope':'Approximate manual source-only development landmarks. Original geometry and raw generations unchanged.'},indent=2));print('PREPARED',len(rows))
if __name__=='__main__':main()
