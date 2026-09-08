"""Build four matched pilot inputs; raster proxies are NOT target photographs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodstateedit.projection3d import PinholeCamera, smoothstep
from build_flexible_completion_dataset import draw_segment, draw_polyline, render_strand, write_video

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('planar', 'relative3d')
INDICES = [0, 3, 6, 10, 15, 20]
SOURCES = {
    'soup': ('artifacts/day2_anchor_sources_x4_v1/soup_002.jpg', '6f633e7416d069c149fecfb37f5e67d7f4b568487788ee73cba75b4702d188e3'),
    'rice': ('artifacts/day2_anchor_sources_x4_v1/rice_007.jpg', '60c6e0870bf35caf686db9bcdf636f2cdd302762d5902a4b5937214f5c5b88cc'),
    'cake': ('artifacts/day19_multimaterial_sources_v1/cake_imagegen_input.png', 'ab287a7b20fe428e907214509ca36e0fc9f9999f1a85588044f4d9edb31f5e21'),
    'noodle': ('artifacts/day18_high_lift_swept_support_v1/vace_reference_image/udon_chopsticks_imagegen_pseudo_v1.png', '4d1b59dc94598ee25e8bf824509f297dbc199283d7cf240da023e49be2c42ace'),
}
SPECS = {
    'soup': dict(contact=[.33,.60], final=[.27,.31], approach=[-.05,-.26], handle=[-.24,-.29], radius=[.078,.043], yaw=.16,
                 prompt='A realistic close-up photograph of the same potage. One stainless-steel spoon scoops and lifts a visible spoonful of thick soup above the surface. Liquid stays inside the concave spoon, with a natural meniscus. Preserve the soup crock, garnish, table and lighting. No hand, no spill, no crater.'),
    'rice': dict(contact=[.64,.56], final=[.68,.28], approach=[.27,-.24], handle=[.32,-.23], radius=[.090,.065], yaw=-.20,
                 prompt='A realistic close-up photograph of the same fried rice. One stainless-steel serving spatula lifts a coherent small scoop of distinct rice grains, egg and vegetables. Food is supported on the blade, with a corresponding shallow reduction in the source mound. Preserve the plate pattern, surrounding ingredients and lighting. No hand and no duplicated payload.'),
    'cake': dict(contact=[.525,.545], final=[.63,.30], approach=[.25,-.28], handle=[.29,-.22], radius=[.06,.06], yaw=-.12,
                 prompt='A realistic close-up photograph of the same cream-topped sponge cake. Exactly one four-tined metal fork lifts the already pre-cut front-right bite above the plate. The intact cake bite stays attached to the fork, revealing a matching gap and natural crumb cut surfaces in the remaining cake. Preserve the plate, tabletop and lighting. No hand, no second bite, no duplication.'),
}
NEGATIVE = 'hand, person, arm, extra utensil, duplicate utensil, malformed utensil, floating unsupported food, duplicate payload, altered plate, altered bowl, background change, text, watermark, severe blur, ghosting'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')


def file_record(path, root):
    return dict(path=path.relative_to(root).as_posix(), size_bytes=path.stat().st_size, sha256=sha(path))


def mask_poly(points, w, h):
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def ellipse(center, radii, n=64):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.asarray(center) + np.stack([radii[0]*np.cos(t), radii[1]*np.sin(t)], axis=1)


def motion(i, spec, camera):
    w,h = camera.width,camera.height
    c = np.asarray(spec['contact'])*[w,h]
    amount = smoothstep((i-8)/7)
    if i < 6:
        uv = c + np.asarray(spec['approach'])*[w,h]*(1-smoothstep((i-1)/4))
        target = camera.backproject(uv[None], 1.1)[0]
    else:
        start = camera.backproject(c[None], 1.1)[0]
        end = camera.backproject((np.asarray(spec['final'])*[w,h])[None], .95)[0]
        target = start*(1-amount)+end*amount
    return target, amount


def transform(uv, z, target, amount, spec, camera, arm):
    center = np.asarray(spec['contact'])*[camera.width,camera.height]
    target_uv = camera.project(target[None])[0][0]
    if arm == 'planar':
        return np.asarray(uv) + target_uv-center
    xyz = camera.backproject(np.asarray(uv), z)
    origin = camera.backproject(center[None], 1.1)[0]
    a = spec['yaw']*amount
    rotation = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    return camera.project((xyz-origin)@rotation.T+target)[0]


def warp_patch(canvas, source, source_quad, destination, source_mask):
    matrix = cv2.getPerspectiveTransform(np.float32(source_quad), np.float32(destination))
    h,w = canvas.shape[:2]
    texture = cv2.warpPerspective(source, matrix, (w,h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(source_mask, matrix, (w,h), flags=cv2.INTER_NEAREST)>127
    canvas[mask] = texture[mask]


def render_solid_case(source, name, arm):
    h,w = source.shape[:2]
    spec = SPECS[name]
    camera = PinholeCamera.normalized_relative(w,h)
    contact = np.asarray(spec['contact'])*[w,h]
    rx,ry = np.asarray(spec['radius'])*[w,h]
    if name == 'cake':
        vertices = np.array([[.502,.454],[.580,.490],[.541,.582],[.459,.546],
                             [.502,.606],[.579,.636],[.535,.739],[.455,.698]])*[w,h]
        depths = np.array([1.15,1.17,1.10,1.08,1.15,1.17,1.10,1.08])
        faces = [[0,1,2,3],[3,2,6,7],[2,1,5,6]]
        body = np.maximum.reduce([mask_poly(vertices[f],w,h) for f in faces])
        repaired = source.copy()
        plate_color = source[int(.80*h):int(.825*h),int(.46*w):int(.485*w)].mean(axis=(0,1))
        repaired[body>0] = np.rint(plate_color).astype(np.uint8)
        # Explicit unobserved cut faces: source-textured procedural hypotheses.
        front = vertices[[3,2,6,7]]
        front_mask = mask_poly(front,w,h)
        for cut_face in [vertices[[0,1,5,4]],vertices[[0,3,7,4]]]:
            warp_patch(repaired, source, front, cut_face, front_mask)
    else:
        body = mask_poly(ellipse(contact,[rx,ry]),w,h)
        repaired = source.copy()
        if name == 'rice':
            # Nearby grain texture avoids a radial inpaint smear in the control.
            donor = np.asarray([.47*w,.60*h])
            donor_quad = np.array([donor+[-rx,-ry],donor+[rx,-ry],donor+[rx,ry],donor+[-rx,ry]])
            destination = np.array([contact+[-rx,-ry],contact+[rx,-ry],contact+[rx,ry],contact+[-rx,ry]])
            warp_patch(repaired,source,donor_quad,destination,mask_poly(ellipse(donor,[rx,ry]),w,h))
            # A shallow source depression, not a claim of measured grain count.
            repaired[body>0] = np.rint(repaired[body>0]*.92).astype(np.uint8)
        vertices = np.array([contact+[-rx,-ry],contact+[rx,-ry],contact+[rx,ry],contact+[-rx,ry]])
        depths = np.array([1.13,1.13,1.07,1.07])
        faces = [[0,1,2,3]]
    frames, trace = [], []
    for i in range(21):
        frame = source.copy()
        target, amount = motion(i,spec,camera)
        if i == 0:
            frames.append(frame)
            continue
        contact_uv = camera.project(target[None])[0][0]
        if i >= 9 and amount>0:
            frame = repaired.copy()
        move = lambda points,z=1.1: transform(points,z,target,amount,spec,camera,arm)
        handle_uv = np.stack([contact,contact+np.asarray(spec['handle'])*[w,h]])
        hp = move(handle_uv,np.array([1.10,1.20]))
        cv2.line(frame,tuple(np.rint(hp[0]).astype(int)),tuple(np.rint(hp[1]).astype(int)),(68,72,77),12,cv2.LINE_AA)
        cv2.line(frame,tuple(np.rint(hp[0]).astype(int)),tuple(np.rint(hp[1]).astype(int)),(196,202,209),8,cv2.LINE_AA)
        cv2.line(frame,tuple(np.rint(hp[0]-[1,1]).astype(int)),tuple(np.rint(hp[1]-[1,1]).astype(int)),(244,245,245),2,cv2.LINE_AA)
        if name == 'soup':
            rim = move(ellipse(contact,[rx*1.20,ry*1.30]),1.1)
            cv2.fillPoly(frame,[np.rint(rim).astype(int)],(190,196,202))
            cv2.polylines(frame,[np.rint(rim).astype(int)],True,(75,82,88),3,cv2.LINE_AA)
        elif name == 'rice':
            blade = move(np.array([contact+[-rx*1.16,-ry*.55],contact+[rx*1.08,-ry*.55],contact+[rx*1.20,ry*1.18],contact+[-rx*1.14,ry*1.18]]),[1.14,1.14,1.07,1.07])
            cv2.fillPoly(frame,[np.rint(blade).astype(int)],(173,183,195))
            cv2.polylines(frame,[np.rint(blade).astype(int)],True,(77,84,90),2,cv2.LINE_AA)
        else:
            # Four visible tines terminate inside the cut bite's top face.
            axis = np.asarray(spec['handle'])*[w,h]
            axis = axis/np.linalg.norm(axis)
            across = np.array([-axis[1],axis[0]])
            for offset in (-12,-4,4,12):
                line = np.stack([contact+across*offset-axis*12,contact+across*offset+axis*31])
                line = move(line,1.065)
                cv2.line(frame,tuple(np.rint(line[0]).astype(int)),tuple(np.rint(line[1]).astype(int)),(185,193,203),4,cv2.LINE_AA)
        if i >= 6:
            # Face/payload texture is sampled from the source, never imagegen-ed output.
            order = sorted(faces,key=lambda f:float(depths[f].mean()),reverse=True)
            for face in order:
                original = vertices[face]
                dest = move(original,depths[face])
                face_mask = body if name != 'cake' else mask_poly(original,w,h)
                warp_patch(frame,source,original,dest,face_mask)
            if name == 'cake':
                # Re-show distal tine tips on top of the bite, not a floating fork.
                for offset in (-12,-4,4,12):
                    line=move(np.stack([contact+across*offset-axis*9,contact+across*offset+axis*14]),1.065)
                    cv2.line(frame,tuple(np.rint(line[0]).astype(int)),tuple(np.rint(line[1]).astype(int)),(200,207,213),3,cv2.LINE_AA)
        trace.append(dict(frame=i,phase='approach' if i<6 else 'contact' if i<9 else 'lift' if i<16 else 'hold',
                          anchor_xyz=target.tolist(),anchor_uv=contact_uv.tolist(),lift=amount))
        frames.append(frame)
    return frames,dict(camera_intrinsic=camera.intrinsic.tolist(),vertices_uv=vertices.tolist(),
                       vertex_relative_depths=depths.tolist(),trace=trace,
                       geometry_source='manually_parameterized_relative_3d_not_reconstruction',
                       source_update='self_leveled_unchanged_surface' if name=='soup' else 'procedural_source_reduction',
                       mass_conservation_in_rgb_not_established=True)


def render_noodle(source,arm):
    old_root=ROOT/'artifacts/day17_high_lift_relative3d_udon_v1'
    geom=json.loads((ROOT/'configs/flexible_completion_udon_relative3d_geometry_high_lift_v1.json').read_text())
    manifest=json.loads((old_root/'dataset_manifest.json').read_text())
    xyz=np.load(old_root/'relative3d_geometry/udon_relative3d_geometry.npz')
    K=xyz['intrinsic']; h,w=source.shape[:2]; support=np.ones((h,w),bool)
    frames=[]
    for i in range(21):
        frame=source.copy()
        if i==0:
            frames.append(frame);continue
        pinch=np.asarray(manifest['frame_records'][i]['pinch_uv'])
        s=geom['chopsticks']; strand=xyz['strand_xyz'][i].astype(float)
        if i>=6:
            uvh=strand@K.T; uv=uvh[:,:2]/uvh[:,2:]
        if arm=='relative3d' and i>=6:
            render_strand(frame,uv,strand[:,2],(1.,1.06),support,geom['strand'],front=False)
        for side,color in [(1.,'far_color_rgb'),(-1.,'near_color_rgb')]:
            if arm=='planar' and side<0 and i>=6:
                for offset,col,width in [([2,3],'shadow_rgb',18),([0,0],'color_rgb',13),([-1,-1],'highlight_rgb',3)]:
                    draw_polyline(frame,uv+offset,tuple(geom['strand'][col]),width,support)
            start=pinch+[0,side*s['tip_separation_normalized']*h/2]
            end=pinch+np.asarray(s['handle_offset_normalized'])*[w,h]+[0,side*s['handle_separation_normalized']*h/2]
            draw_segment(frame,start+[2,3],end+[2,3],(40,24,17),s['shadow_width_px'],support)
            draw_segment(frame,start,end,tuple(s[color]),s['line_width_px'],support)
            draw_segment(frame,start-[1,1],end-[1,1],tuple(s['highlight_rgb']),2,support)
        if arm=='relative3d' and i>=6:
            render_strand(frame,uv,strand[:,2],(1.,1.06),support,geom['strand'],front=True)
        frames.append(frame)
    return frames,dict(geometry_config_sha256=sha(ROOT/'configs/flexible_completion_udon_relative3d_geometry_high_lift_v1.json'),
                       treatment='fixed far-stick/strand/near-stick paint order vs depth-varying strand visibility',
                       control_amplitude_not_a_generated_measurement=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args(); out=args.output_root.resolve()
    if out.exists():raise FileExistsError(out)
    for rel,digest in SOURCES.values():
        if sha(ROOT/rel)!=digest:raise ValueError('Source hash mismatch: '+rel)
    out.mkdir(parents=True)
    manifest=dict(schema_version='foodstateedit.multimaterial_dataset.v1',builder_sha256=sha(Path(__file__)),
                  status='controls_require_visual_review_before_execution',arms=list(ARMS),cases=[],
                  claim_limit='Four single-image pilots, mixed real/synthetic sources, no generalization, learned efficacy or physical correctness claim.')
    for name in ('soup','rice','cake','noodle'):
        folder=out/name;folder.mkdir()
        image=Image.open(ROOT/SOURCES[name][0]).convert('RGB')
        if name!='noodle':
            w,h=image.size;scale=688/max(w,h)
            image=image.resize((round(w*scale/16)*16,round(h*scale/16)*16),Image.Resampling.LANCZOS)
        source=np.asarray(image);h,w=source.shape[:2];image.save(folder/'reference.png')
        arms={}; geoms={}; union=np.zeros((h,w),bool)
        for arm in ARMS:
            frames,geometry=render_noodle(source,arm) if name=='noodle' else render_solid_case(source,name,arm)
            assert np.array_equal(frames[0],source)
            union |= np.logical_or.reduce([np.any(f!=source,axis=2) for f in frames])
            arms[arm]=frames;geoms[arm]=geometry
        full=np.asarray(Image.fromarray(np.uint8(union)*255).filter(ImageFilter.MaxFilter(17)))
        alpha=full.copy()
        for r in range(9,17):
            mask=np.asarray(Image.fromarray(np.uint8(union)*255).filter(ImageFilter.MaxFilter(2*r+1)))>0
            alpha=np.maximum(alpha,np.uint8(mask)*round(255*(17-r)/9))
        assert np.all(alpha[union]==255)
        Image.fromarray(alpha).save(folder/'edit_alpha.png')
        save_json(folder/'geometry.json',geoms)
        for arm,frames in arms.items():
            for frame in frames:assert not np.any(frame[alpha==0]!=source[alpha==0])
            write_video(folder/f'{arm}.mp4',frames,8)
            sheet=Image.new('RGB',(w*3,(h+28)*2),'white');draw=ImageDraw.Draw(sheet)
            for k,i in enumerate(INDICES):
                x,y=(k%3)*w,(k//3)*(h+28);sheet.paste(Image.fromarray(frames[i]),(x,y+28));draw.text((x+5,y+5),f'{name} / {arm} / f{i} / CONTROL',fill='black')
            sheet.save(folder/f'{arm}_review.png')
            Image.fromarray(frames[-1]).save(folder/f'{arm}_final.png')
        prompt = SPECS[name]['prompt'] if name!='noodle' else json.loads((ROOT/'configs/day18_high_lift_swept_support_gp40_v1.json').read_text())['inference']['prompt']
        rec=dict(case_id=name,source_original_path=SOURCES[name][0],source_original_sha256=SOURCES[name][1],
                 source_kind='real_previously_used_upscaled_pilot' if name in ('soup','rice') else 'synthetic_imagegen_input',
                 license='UECFOOD256_noncommercial_research_only_no_public_redistribution' if name in ('soup','rice') else 'generated_input_not_experimental_output',
                 width=w,height=h,prompt=prompt,negative_prompt=NEGATIVE,
                 support_fraction=float((alpha>0).mean()),uncovered_changed_pixels=int((union&(alpha<255)).sum()),
                 files={p.name:file_record(p,out) for p in folder.iterdir() if p.is_file()})
        manifest['cases'].append(rec)
    save_json(out/'dataset_manifest.json',manifest)
    print(json.dumps({'output':str(out),'manifest_sha256':sha(out/'dataset_manifest.json'),
                      'cases':[{k:c[k] for k in ('case_id','support_fraction','uncovered_changed_pixels')} for c in manifest['cases']]},indent=2))


if __name__=='__main__':main()
