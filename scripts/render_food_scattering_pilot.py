"""Scattering material pilot on the existing inferred shared-cut geometry.

Optical constants and lighting are priors, not measured tofu properties. The
photographed inherited top is kept by the compositor; new surfaces are tested
with volumetric transport instead of foundation-model material hallucination.
"""
import os, sys, json, argparse, hashlib, time
from pathlib import Path
ROOT=Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=6);a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    sys.path.insert(0,'/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
    from run_qwen_image_edit_direct_baseline import gpu_snapshot
    snap=gpu_snapshot(a.gpu);assert not snap['compute_processes'] and snap['memory_free_mib']>20000
    import mitsuba as mi
    import numpy as np
    from PIL import Image,ImageDraw
    from scipy.ndimage import distance_transform_edt
    mi.set_variant('cuda_ad_rgb')
    gate=ROOT/'gate_v24';gate.mkdir(exist_ok=False);rows=[]
    cfg=json.loads((ROOT/'gate_v20/config.json').read_text())
    cases=[j for j in cfg['jobs'] if j['method']=='silken_food_only_canny']
    freeze={'status':'frozen_before_render','cases':[j['case_id'] for j in cases],'optical_depths_over_food_extent':[30,100,300],
            'roughness':.22,'interior_ior':1.34,'spp':256,'integrator':'volpath','geometry_inferred':True,'light_and_optics_measured':False,
            'script_sha256':sha(Path(__file__))}
    (gate/'recipe.json').write_text(json.dumps(freeze,indent=2))
    for job in cases:
        cid=job['case_id'];g=ROOT/'geometry_spoon_open_corner_v1'/cid;folder=gate/cid;folder.mkdir()
        report=json.loads((g/'geometry_report.json').read_text());fit=report['fit'];R=np.asarray(fit['axes_camera_columns']);L=max(np.asarray(fit['high'])-np.asarray(fit['low']))
        center=np.asarray(report['cut_and_support']['destination_center'])@R.T
        mp=Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz' if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz'
        K=np.load(mp)['intrinsics'].copy();K[0]*=640;K[1]*=480
        assert abs(K[0,2]-320)<1e-3 and abs(K[1,2]-240)<1e-3
        assert abs(K[0,0]-K[1,1])<1e-3
        calibration=json.loads((g/'appearance_calibration.json').read_text());coef=np.asarray(calibration['coefficient_rgb'])
        direction=coef[1:].mean(1);direction/=max(np.linalg.norm(direction),1e-9)
        if np.linalg.norm(direction)<.5:direction=np.array([-.4,-.6,-.7]);direction/=np.linalg.norm(direction)
        light_center=center+direction*3*L
        up=[0,1,0] if abs(direction[1])<.95 else [1,0,0]
        light_pose=mi.ScalarTransform4f().look_at(origin=light_center.tolist(),target=center.tolist(),up=up)@mi.ScalarTransform4f().scale([1.8*L,1.8*L,1])
        sensor={'type':'perspective','fov':float(np.degrees(2*np.arctan(320/K[0,0]))),'fov_axis':'x',
                'to_world':mi.ScalarTransform4f().scale([-1,-1,1]),'near_clip':.001*L,'far_clip':100*L,
                'sampler':{'type':'independent','sample_count':256},'film':{'type':'hdrfilm','width':640,'height':480,'pixel_format':'rgba','rfilter':{'type':'box'}}}
        channels=np.load(ROOT/'spoon_open_corner_channels_v1'/cid/'geometry_channels.npz');food=channels['labels']==2
        parent=ROOT/'gate_v20/collection_complete'/job['id']/'composited.png';base=np.asarray(Image.open(parent).convert('RGB'),float)
        prior=np.asarray(Image.open(g/'rgb_control.png').convert('RGB'),float)
        action=report['cut_and_support'];Q=np.asarray(action['rotation_food_frame']);dest=np.asarray(action['destination_center']);src=np.asarray(action['source_center']);hi=np.asarray(fit['high'])
        yy,xx=np.where(food);rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(K).T;cam=rays*channels['scene_depth'][food][:,None];back=(cam@R-dest)@Q+src;norm=channels['scene_normals'][food]@R@Q
        top=(np.abs(back[:,2]-hi[2])<L*1e-4)&(norm[:,2]>.99);inherited=np.zeros(food.shape,bool);inherited[yy[top],xx[top]]=True
        case=next(c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases'] if c['case_id']==cid);l,t=case['preprocessing']['pad_left_top'];w,h=case['preprocessing']['resized'];rect=(l,t,l+w,t+h)
        tiles=[('neural baseline',Image.fromarray(np.uint8(base)))]
        for optical in [30,100,300]:
            start=time.time();d=folder/('optical_'+str(optical));d.mkdir()
            scene_dict={'type':'scene','integrator':{'type':'volpath','max_depth':32,'rr_depth':8},'sensor':sensor,
                        'environment':{'type':'constant','radiance':{'type':'rgb','value':[.25,.25,.25]}},
                        'key_light':{'type':'rectangle','to_world':light_pose,'emitter':{'type':'area','radiance':{'type':'rgb','value':[5,5,5]}}}}
            for name in ['bite_lifted','remaining']:
                scene_dict[name]={'type':'ply','filename':str(g/(name+'.ply')),
                                  'bsdf':{'type':'roughdielectric','distribution':'ggx','alpha':.22,'int_ior':1.34,'ext_ior':1.000277},
                                  'interior':{'type':'homogeneous','sigma_t':float(optical/L),'albedo':{'type':'rgb','value':[.997,.996,.990]}}}
            scene_dict['spoon']={'type':'ply','filename':str(g/'fork.ply'),'bsdf':{'type':'roughconductor','material':'Al','alpha':.15}}
            scene=mi.load_dict(scene_dict);render=np.asarray(mi.render(scene,seed=41,spp=256));mi.util.write_bitmap(str(d/'render_rgba.exr'),render)
            rgb=render[:,:,:3];rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-.055)*255
            rgb=np.clip(rgb,0,255)
            # Light amplitude is not measured. Match one global RGB tone to the
            # source-calibrated food prior, preserving the renderer's shading.
            gains=np.clip(np.median(prior[food],axis=0)/np.maximum(np.median(rgb[food],axis=0),1),.5,2)
            material=np.clip(rgb*gains,0,255);material[inherited]=prior[inherited]
            alpha=np.clip(distance_transform_edt(food),0,1)[...,None]
            result=np.uint8(np.clip(np.rint(base*(1-alpha)+material*alpha),0,255));assert np.array_equal(result[~food],base[~food].astype(np.uint8))
            Image.fromarray(result).save(d/'composited.png');Image.fromarray(result).crop(rect).save(d/'view.png');Image.fromarray(np.uint8(np.clip(rgb,0,255))).save(d/'render_rgb.png')
            tiles.append(('scattering optical depth '+str(optical),Image.fromarray(result)))
            row={'case_id':cid,'optical_depth_over_extent':optical,'sigma_t':float(optical/L),'source_tone_gain_rgb':gains.tolist(),'seconds':time.time()-start,
                 'parent_sha256':sha(parent),'outside_food_parent_exact':True,'photographed_inherited_top_preserved':True,'scope':'Volumetric material prior on inferred geometry; RGB tone normalization is an explicit derived stage. Optical constants, illumination and hidden geometry are not measured.'}
            (d/'result.json').write_text(json.dumps(row,indent=2));rows.append(row);(gate/'manifest.json').write_text(json.dumps({'status':'running','rows':rows},indent=2));print('RENDERED',cid,optical,flush=True)
        board=Image.new('RGB',(640*len(tiles),h+28),'white');draw=ImageDraw.Draw(board)
        for i,(title,im) in enumerate(tiles):board.paste(im.crop(rect).resize((640,h),Image.Resampling.LANCZOS),(i*640,28));draw.text((i*640+4,6),title,fill='black')
        board.save(folder/'comparison.jpg',quality=95)
    (gate/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'neural_raw_calls':0,'physical_render_cells':len(rows),'optics_measured':False},indent=2))

if __name__=='__main__':main()
