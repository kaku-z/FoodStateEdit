"""Complete all G37 source/geometry derivatives after every raw cell finishes."""
import json,time,subprocess,zipfile
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');PY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def run(s,args=[]):subprocess.run([PY,'-u',str(ROOT/s)]+args,cwd=str(ROOT),check=True)
def main():
 g=ROOT/'gate_v37';p=g/'postprocessing_execution.json';p.write_text(json.dumps({'status':'waiting_for_generation'}))
 while json.loads((g/'execution.json').read_text())['status']!='all_24_raw_complete_unreviewed':time.sleep(20)
 p.write_text(json.dumps({'status':'derivatives_running'}));run('compose_source_fit_ellipsoid_food.py');run('prepare_shared_cut_material_field.py',['--gate','gate_v37','--geometry','geometry_spoon_source_fit_ellipsoid_v3','--channels','spoon_source_fit_ellipsoid_channels_v3']);run('coupled_source_cut_reconstruction.py',['--gate','gate_v37','--geometry','geometry_spoon_source_fit_ellipsoid_v3','--channels','spoon_source_fit_ellipsoid_channels_v3','--cavity-field','shared_cut_material_field_v3','--arms','no_food_only_crop_ablation']);run('compose_geometry_locked_action.py');p.write_text(json.dumps({'status':'complete_unreviewed','finished_unix':time.time()}))
 with zipfile.ZipFile(ROOT/'gate_v37_complete.zip','w',zipfile.ZIP_DEFLATED) as z:
  for d in [g,ROOT/'geometry_spoon_source_fit_ellipsoid_v3',ROOT/'spoon_source_fit_ellipsoid_channels_v3',ROOT/'source_fit_ellipsoid_food_fields_consumptive_v4']:
   for f in d.rglob('*'):
    if f.is_file():z.write(f,f.relative_to(ROOT))
 print('G37_COMPLETE',flush=True)
if __name__=='__main__':main()
