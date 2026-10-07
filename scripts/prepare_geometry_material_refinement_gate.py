"""Freeze geometry-aligned source appearance before local partial-noise refinement.

This prior is an inferred composite, not a photographed cut pair. Every cell is
retained; no output-driven crop, seed retry or case-specific prompt is allowed.
"""
import copy, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p): return {'path': str(p), 'sha256': sha(p)}

def main():
    gate = ROOT/'gate_v23'; gate.mkdir(exist_ok=False)
    cfg = json.loads((ROOT/'gate_v9/config.json').read_text())
    cfg['stage'] = 'development_geometry_aligned_observed_surface_partial_noise'
    cfg['inference']['negative_prompt'] += ', caramel, toasted brown spot, pudding, custard, hollow cup, printed sticker, large cracks'
    jobs = []
    for prototype in json.loads((ROOT/'gate_v20/config.json').read_text())['jobs']:
        if prototype['method'] != 'silken_food_only_canny': continue
        cid = prototype['case_id']; folder = gate/cid; folder.mkdir()
        context = Path(prototype['transform']).parent
        old_transform = json.loads((context/'transform.json').read_text())
        parent = ROOT/'gate_v20/collection_complete'/prototype['id']/'composited.png'
        image = np.asarray(Image.open(parent).convert('RGB'), float)
        geometry = ROOT/'geometry_spoon_open_corner_v1'/cid
        proxy = np.asarray(Image.open(geometry/'rgb_control.png').convert('RGB'), float)
        channels = np.load(ROOT/'spoon_open_corner_channels_v1'/cid/'geometry_channels.npz')
        food = channels['labels'] == 2
        alpha = np.clip(distance_transform_edt(food),0,1)[...,None]
        aligned = np.uint8(np.clip(np.rint(image*(1-alpha)+proxy*alpha),0,255))
        Image.fromarray(aligned).save(folder/'full_prior.png')
        box = tuple(old_transform['box'])
        Image.fromarray(aligned).crop(box).resize((640,640),Image.Resampling.LANCZOS).save(folder/'layout.png')
        full_mask = np.asarray(Image.open(context/'full_mask.png')) > 0
        assert np.all(~food | full_mask)
        Image.fromarray(np.uint8(full_mask)*255).save(folder/'full_mask.png')
        Image.fromarray(np.uint8(full_mask)*255).crop(box).resize((640,640),Image.Resampling.NEAREST).save(folder/'edit_mask.png')
        selection = json.loads((ROOT/'gate_v13_run03'/cid/'reference_selection.json').read_text())
        source = Image.open(geometry/'source.png').convert('RGB')
        source.crop(tuple(selection['material_box_canvas'])).resize((512,512),Image.Resampling.LANCZOS).save(folder/'material_reference.png')
        transform = dict(old_transform, full_composite_parent=str(parent),parent_sha256=sha(parent),
                         prior_full_path=str(folder/'full_prior.png'),prior_sha256=sha(folder/'full_prior.png'),
                         food_prior='Source UV on inherited surfaces, inferred source-calibrated fresh surfaces at exact CPU visible geometry; not a real cut photograph.')
        (folder/'transform.json').write_text(json.dumps(transform,indent=2))
        for sigma in [.08,.16,.28]:
            method='aligned_material_sdedit_'+str(round(sigma*100))
            jobs.append({'id':cid+'__'+method+'__41','case_id':cid,'method':method,'seed':41,
                         'input_order':['layout','reference'],'lock_context':True,'start_raw_sigma':sigma,
                         'transform':str(folder/'transform.json'),
                         'files':{'layout':entry(folder/'layout.png'),'reference':entry(folder/'material_reference.png'),'edit_mask':entry(folder/'edit_mask.png')},
                         'prompt':'Image 1 shows an already lifted small portion of cold silken tofu resting on a stainless steel spoon. Image 2 is the actual plain tofu surface photographed in this meal. Refine only the existing food in Image 1 into matching natural photographic texture: dense moist soybean gel, delicate compact grain, subtle wet translucency and soft cut surfaces. Preserve the exact solid silhouette and every source-photographed sauce or garnish already on its top. The top is a filled solid surface. Keep the existing pose, spoon contact, small scale, highlights and lighting. Preserve the original spoon and background exactly. Add no food, topping, person or hand.'})
    cfg['jobs']=jobs; cfg['frozen_raw_cells']=len(jobs)
    cfg['selection_rule']='All eight development images, all three predeclared noise levels; no case-specific output selection.'
    (gate/'config.json').write_text(json.dumps(cfg,indent=2)); print('PREPARED',len(jobs),flush=True)

if __name__=='__main__': main()
