"""Freeze a high-noise refinement with/without observed-top latent projection.

All eight images are development data. The observed top is transported through
an inferred 3-D pose; garnish height and calibrated relighting are unavailable.
The paired arms use identical initialization, noise, prompt and references.
"""
import copy, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p): return {'path':str(p), 'sha256':sha(p)}

def main():
    gate=ROOT/'gate_v30'; gate.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'gate_v23/config.json').read_text())
    cfg['stage']='development_observed_top_projection_controlled_ablation'
    cfg['not_formal']=True
    jobs=[]
    for prototype in json.loads((ROOT/'gate_v29/config.json').read_text())['jobs']:
        cid=prototype['case_id']; d=gate/cid; d.mkdir()
        context=Path(prototype['transform']).parent
        transform=json.loads((context/'transform.json').read_text())
        parent=ROOT/'gate_v29/collection_complete'/prototype['id']/'composited.png'
        appearance=ROOT/'gate_v29/observed_material_neural_light_v1'/prototype['id']
        initial=appearance/'composited.png'
        box=tuple(transform['box'])
        Image.open(initial).convert('RGB').crop(box).resize((640,640),Image.Resampling.LANCZOS).save(d/'layout.png')
        (d/'full_mask.png').write_bytes((context/'full_mask.png').read_bytes())
        top=np.asarray(Image.open(appearance/'transport_mask.png'))>0
        weight=binary_erosion(top,iterations=1)
        Image.fromarray(np.uint8(weight)*255).crop(box).resize((640,640),Image.Resampling.NEAREST).save(d/'observed_top_weight.png')
        transform=dict(transform,full_composite_parent=str(parent),parent_sha256=sha(parent),
                       appearance_initialization=str(initial),appearance_initialization_sha256=sha(initial))
        (d/'transform.json').write_text(json.dumps(transform,indent=2))
        material=ROOT/'gate_v23'/cid/'material_reference.png'
        for anchored in [False,True]:
            method='observed_top_projected' if anchored else 'observed_top_unanchored'
            j={'id':cid+'__'+method+'__41','case_id':cid,'method':method,'seed':41,
               'start_raw_sigma':.8,'lock_context':True,'input_order':['layout','reference'],
               'transform':str(d/'transform.json'),
               'files':{'layout':entry(d/'layout.png'),'reference':entry(material),
                        'edit_mask':copy.deepcopy(prototype['files']['edit_mask'])},
               'prompt':'Image 1 shows a small solid first mouthful of cold silken tofu already supported by the existing stainless steel spoon. Image 2 shows the actual plain tofu material in this meal. Make the existing piece a coherent natural photograph: dense compact moist soybean gel, continuous delicate fresh cut surfaces, subtle wet translucency, soft food highlights and plausible contact against the spoon. The photographed colors, sauce and garnish already on its upper surface belong to this exact mouthful and must stay in their existing positions. The food has a filled solid top and continuous solid sides. Preserve the original silhouette, scale, pose, spoon and background. Add no topping, colored spot, leaf, condiment, person or hand.'}
            if anchored:
                j['files']['structure_weight']=entry(d/'observed_top_weight.png')
                j['structure']={'begin_taper_fraction':.25,'end_fraction':.75,'tail_weight':.65,'lowpass':False,
                                'scope':'Observed source-top VAE projection only; max-pool expands the latent weight. Not a pixel-identity or 3-D guarantee.'}
            jobs.append(j)
    cfg['jobs']=jobs;cfg['frozen_raw_cells']=len(jobs)
    cfg['selection_rule']='All eight development images, paired unanchored/projected arms, same seed41. One global recipe; every output retained.'
    cfg['limitations']='Source top geometry and relighting are inferred. A flat photographed garnish does not supply its height. VAE spatial coupling is audited, not treated as physical supervision.'
    (gate/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',len(jobs),flush=True)

if __name__=='__main__':main()
