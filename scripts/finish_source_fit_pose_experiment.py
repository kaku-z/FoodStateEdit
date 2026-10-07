"""Complete deterministic derivatives after all G36 neural cells finish."""
import json,time,subprocess,traceback
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930');PY='/host/space0/guo-z/tf-ufi/food3d_pilot_20260928/venv/bin/python'
def run(s,args=[]):subprocess.run([PY,'-u',str(ROOT/s)]+args,cwd=str(ROOT),check=True)
def main():
 p=ROOT/'gate_v36/postprocessing_execution.json';p.write_text(json.dumps({'status':'waiting_for_generation'}))
 while not (ROOT/'gate_v36/completion.json').exists():time.sleep(20)
 p.write_text(json.dumps({'status':'derivatives_running'}));run('transport_source_fit_surfaces.py');run('compose_source_fit_food_ablation.py');run('prepare_shared_cut_material_field.py',['--gate','gate_v36','--geometry','geometry_spoon_source_fit_v2','--channels','spoon_source_fit_channels_v2']);run('coupled_source_cut_reconstruction.py',['--gate','gate_v36','--geometry','geometry_spoon_source_fit_v2','--channels','spoon_source_fit_channels_v2','--transport','source_fit_surface_consumptive_v3','--method','source_fit_food_intrinsic','--cavity-field','shared_cut_material_field_v3']);p.write_text(json.dumps({'status':'complete_unreviewed','finished_unix':time.time()}));print('G36_DERIVATIVES_COMPLETE',flush=True)
if __name__=='__main__':main()
