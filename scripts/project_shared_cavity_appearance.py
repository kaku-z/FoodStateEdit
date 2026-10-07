"""Project a shared Boolean cavity with source-calibrated material and 3-D visibility.

The generated food and spoon are retained. This is a deterministic geometry/
appearance prior, not new neural output or a measurement of hidden surfaces.
"""
import os, sys, json, hashlib, time, zipfile
from pathlib import Path

ROOT = Path('/host/space0/guo-z/tf-ufi/first_bite_structure_20260930')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    sys.path.insert(0, '/mnt/tmp/guo-z_first_bite_structure_20260930_runtime/mitsuba_vendor')
    import mitsuba as mi
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation
    from scipy.stats import qmc
    mi.set_variant('cuda_ad_rgb')
    gate = ROOT/'gate_v32'
    out = gate/'cavity_geometry_projection_v1'
    out.mkdir(exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    cases = {c['case_id']: c for c in json.loads((ROOT/'inputs/manifest.json').read_text())['cases']}
    plan = {'status': 'frozen_before_projection', 'created_unix': time.time(),
            'neural_calls': 0, 'cells': 24, 'seeds': [41, 163, 907],
            'rule': 'Shared remaining depth determines cavity; source RGB directional fit plus bounded neural microdetail; 256 cosine-hemisphere visibility rays, ambient scalar shade 0.40; global source-native resample inside geometry edit mask.',
            'limits': 'Monocular geometry, hidden material, ambient light and flat garnish remain priors; no paired truth, human realism ratings or heldout data.'}
    (out/'frozen_plan.json').write_text(json.dumps(plan, indent=2))
    rows = []
    for cid, case in cases.items():
        geo = ROOT/'geometry_spoon_box_cap_v1'/cid
        channels = np.load(ROOT/'spoon_box_cap_channels_v1'/cid/'geometry_channels.npz')
        report = json.loads((geo/'geometry_report.json').read_text())
        fit = report['fit']; R = np.asarray(fit['axes_camera_columns'])
        L = max(np.asarray(fit['high'])-np.asarray(fit['low']))
        maps = (Path('/host/space0/guo-z/tf-ufi/first_bite_coupled_20260929/geometry')/cid/'maps.npz') if cid.startswith('new_') else ROOT/'geometry_v3'/cid/'maps.npz'
        K = np.load(maps)['intrinsics'].copy(); K[0] *= 640; K[1] *= 480
        z = channels['remaining_depth']; full = channels['full_depth']
        cut = binary_dilation(np.asarray(Image.open(geo/'source_bite_mask.png'))>0, iterations=2)
        finite = np.isfinite(z)&np.isfinite(full)
        delta = np.zeros(z.shape); np.subtract(z, full, out=delta, where=finite)
        mask = finite&cut&(delta>L*1e-5)&(channels['labels']==1)
        assert mask.sum()>100, (cid, mask.sum())
        yy, xx = np.where(mask)
        p = np.c_[xx+.5, yy+.5, np.ones(len(xx))]@np.linalg.inv(K).T
        p *= z[mask][:, None]
        n = channels['scene_normals'][mask].copy()
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
        scene = mi.load_dict({'type':'scene', 'remaining': {'type':'ply', 'filename':str(geo/'remaining.ply'), 'bsdf':{'type':'diffuse'}}})
        samples = qmc.Sobol(d=2, scramble=True, seed=41).random_base2(8)
        frame = mi.Frame3f(mi.Normal3f(n.T))
        origin = mi.Point3f((p+n*L*1e-5).T)
        occluded = np.zeros(len(p))
        for sample in samples:
            local = mi.warp.square_to_cosine_hemisphere(mi.Point2f(sample.tolist()))
            ray = mi.Ray3f(origin, frame.to_world(local)); ray.maxt = mi.Float(.75*L)
            occluded += np.asarray(scene.ray_test(ray), float)
        ao = np.zeros(mask.shape); ao[mask] = occluded/len(samples)
        # Normalize smoothing by membership so the cavity edge is not washed out.
        ao = gaussian_filter(ao, .6)/np.maximum(gaussian_filter(mask.astype(float), .6), 1e-8)
        ao *= mask
        coefficient = np.asarray(json.loads((geo/'appearance_calibration.json').read_text())['coefficient_rgb'])
        material = np.clip(np.c_[np.ones(len(n)), n]@coefficient, .15, 1)*255
        material *= (1-.40*ao[mask])[:, None]
        folder = out/(cid+'_field'); folder.mkdir()
        Image.fromarray(np.uint8(mask)*255).save(folder/'cavity_mask.png')
        Image.fromarray(np.uint8(np.clip(ao,0,1)*255)).save(folder/'ambient_occlusion.png')
        np.savez_compressed(folder/'projection_field.npz',mask=mask,occlusion=ao,pixels_rgb=material)
        source = np.asarray(Image.open(geo/'source.png').convert('RGB'))
        edit = np.asarray(Image.open(geo/'edit_mask.png'))>0
        l,t = case['preprocessing']['pad_left_top']; w,h = case['preprocessing']['resized']; rect=(l,t,l+w,t+h)
        for seed in [41,163,907]:
            jid = cid+'__box_cap_food_intrinsic__'+str(seed)
            parent = gate/'observed_material_box_cap_uv_v1'/jid/'composited.png'
            candidate = np.asarray(Image.open(parent).convert('RGB'), float)
            # The network supplies only small spatial detail, not the concavity's
            # low-frequency lighting or shape interpretation.
            gray = candidate@np.array([.2126,.7152,.0722])
            detail = np.clip(gray-gaussian_filter(gray, 2), -3, 3)
            projected = candidate.copy(); projected[mask] = material+detail[mask][:,None]
            alpha = (mask.astype(float)*np.clip(distance_transform_edt(mask)/.75,0,1))[...,None]
            projected = np.uint8(np.clip(np.rint(candidate*(1-alpha)+projected*alpha),0,255))
            assert np.array_equal(projected[~mask], candidate.astype(np.uint8)[~mask])
            native = Image.fromarray(projected).crop(rect).resize(tuple(case['source_size']), Image.Resampling.LANCZOS).resize((w,h), Image.Resampling.LANCZOS)
            sampled = source.copy(); sampled[t:t+h,l:l+w] = np.asarray(native)
            blend = np.clip(distance_transform_edt(edit)/5,0,1)[...,None]
            final = np.uint8(np.clip(np.rint(source*(1-blend)+sampled*blend),0,255))
            assert np.array_equal(final[~edit],source[~edit])
            dest=out/jid;dest.mkdir()
            Image.fromarray(projected).save(dest/'projected.png')
            Image.fromarray(final).save(dest/'composited.png')
            Image.fromarray(final).crop(rect).save(dest/'view.png')
            row={'id':jid,'case_id':cid,'seed':seed,'parent_sha256':sha(parent),
                 'final_sha256':sha(dest/'composited.png'),'raw_generation':False,
                 'outside_source_geometry_edit_mask_exact':True,'cavity_pixels':int(mask.sum()),
                 'occlusion_mean':float(ao[mask].mean()),'occlusion_max':float(ao.max()),
                 'source_geometry_report_sha256':sha(geo/'geometry_report.json'),
                 'scope':'New hidden cavity appearance is source-calibrated ambient visibility prior; no measured lighting or recovered hidden texture.'}
            (dest/'result.json').write_text(json.dumps(row,indent=2));rows.append(row)
        board=Image.new('RGB',(640*4,h+28),'white');draw=ImageDraw.Draw(board)
        for index,(title,path) in enumerate([('source',geo/'source.png')]+[('shared cavity seed'+str(s),out/(cid+'__box_cap_food_intrinsic__'+str(s))/'composited.png') for s in [41,163,907]]):
            board.paste(Image.open(path).convert('RGB').crop(rect).resize((640,h),Image.Resampling.LANCZOS),(640*index,28));draw.text((640*index+5,6),title,fill='black')
        board.save(out/(cid+'_three_seed.jpg'),quality=95)
        print('PROJECTED',cid,mask.sum(),float(ao[mask].mean()),flush=True)
    (out/'manifest.json').write_text(json.dumps({'status':'complete_unreviewed','rows':rows,'additional_raw_calls':0,'all_cells_same_rule':True,'script_sha256':sha(Path(__file__))},indent=2))
    with zipfile.ZipFile(ROOT/'shared_cavity_projection_v1.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in out.rglob('*'):
            if p.is_file(): archive.write(p,p.relative_to(ROOT))
if __name__=='__main__':main()
