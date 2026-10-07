"""Programmatic multi-topology scenes for MLD2; no photograph/after-image truth."""
from __future__ import annotations

import hashlib
import math
import numpy as np

SEED = 20261004
CAMERA_SCALE = 1.45
FAMILIES = ("rough_ellipsoid", "layered_block", "ring", "cluster", "capsules",
            "perforated_block", "bowl", "noodles")
PALETTES = np.asarray([[.75,.58,.27],[.67,.23,.08],[.32,.50,.13],[.88,.81,.63],
                       [.43,.17,.07],[.78,.69,.42],[.58,.31,.34],[.22,.43,.22]])


def rng_for(identity, stream):
    raw = hashlib.sha256((identity + ":" + stream).encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:8], "little"))


def scene_identity(index, seed=SEED):
    return hashlib.sha256(f"mld2:{seed}:{index}".encode()).hexdigest()


def axis_angle_matrix(a):
    a = np.asarray(a, dtype=np.float64)
    theta = np.linalg.norm(a)
    if theta < 1e-12:
        return np.eye(3)
    x,y,z = a / theta
    k = np.asarray([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + np.sin(theta)*k + (1-np.cos(theta))*(k@k)


def camera_basis():
    yaw, elev = np.deg2rad([30.,35.])
    return np.asarray([[-np.sin(yaw),np.cos(yaw),0],
        [-np.sin(elev)*np.cos(yaw),-np.sin(elev)*np.sin(yaw),np.cos(elev)],
        [np.cos(elev)*np.cos(yaw),np.cos(elev)*np.sin(yaw),np.sin(elev)]])


def canonical_grid(grid_size=16):
    c = (np.arange(grid_size)+.5)*(2/grid_size)-1
    return np.stack(np.meshgrid(c,c,c,indexing="ij"),-1).reshape(-1,3).astype(np.float32)


def subcell_offsets(grid_size=16):
    v = [-.5/grid_size,.5/grid_size]
    return np.stack(np.meshgrid(v,v,v,indexing="ij"),-1).reshape(8,3).astype(np.float32)


def known_camera_projection(xyz):
    uv = np.asarray(xyz) @ camera_basis()[:2].T / CAMERA_SCALE
    uv[...,1] *= -1
    return uv.astype(np.float32)


def scene_parameters(identity, index):
    r = rng_for(identity,"geometry-material")
    family = FAMILIES[index % len(FAMILIES)]
    extent = r.uniform(.38,.66,3)
    extent[2] = r.uniform(.27,.51)
    segments, centers, radii = [], [], []
    if family == "cluster":
        for j in range(int(r.integers(3,8))):
            centers.append(r.uniform(-.45,.45,3).tolist())
            radii.append(r.uniform(.13,.28,3).tolist())
    if family in ("capsules","noodles"):
        count = int(r.integers(3,6)) if family == "capsules" else int(r.integers(6,11))
        for j in range(count):
            radius = float(r.uniform(.065,.14) if family == "capsules" else r.uniform(.028,.057))
            phase = r.uniform(-math.pi,math.pi)
            amp = r.uniform(.10,.26)
            offset = r.uniform(-.32,.32,3)
            t = np.linspace(-.61,.61,5)
            curve = np.stack([t,amp*np.sin(t*r.uniform(2.0,4.0)+phase),
                              amp*.55*np.cos(t*r.uniform(2.0,4.0)-phase)],-1)
            theta = r.uniform(-math.pi,math.pi)
            curve = curve @ axis_angle_matrix([0,0,theta]).T + offset
            for a,b in zip(curve[:-1],curve[1:]):
                segments.append([a.tolist(),b.tolist(),radius])
    base = np.clip(PALETTES[int(r.integers(len(PALETTES)))]+r.normal(0,.025,3),.02,.94)
    layer_axis = r.normal(size=3); layer_axis /= np.linalg.norm(layer_axis)
    angle = r.uniform(-.17,.17,3); angle[2]=r.uniform(-math.pi,math.pi)
    # Keep the complete solids in the canonical query box, including thin tips.
    rot=axis_angle_matrix(angle)
    if segments:
        bound=max(float(np.max(np.abs(np.asarray([a,b])@rot.T)))+rad for a,b,rad in segments)
        scale=min(1.,.90/bound)
        segments=[[(np.asarray(a)*scale).tolist(),(np.asarray(b)*scale).tolist(),rad*scale] for a,b,rad in segments]
    if family in ("layered_block","perforated_block"):
        bound=float((np.abs(rot)@extent).max())
        extent*=min(1.,.88/bound)
    return {"family":family,"center":r.uniform(-.06,.06,3).tolist(),
        "axis_angle":angle.tolist(),"extent":extent.tolist(),
        "roughness":float(r.uniform(.012,.045)),"rough_frequency":r.uniform(5,10,3).tolist(),
        "ring_radius":float(r.uniform(.36,.53)),"tube_radius":float(r.uniform(.12,.20)),
        "centers":centers,"radii":radii,"segments":segments,
        "holes":r.uniform(-.28,.28,(int(r.integers(3,7)),2)).tolist(),
        "hole_radius":float(r.uniform(.075,.135)),"shell_thickness":float(r.uniform(.065,.12)),
        "base_rgb":base.tolist(),"layer_axis":layer_axis.tolist(),
        "layer_frequency":float(r.uniform(2,7)),"layer_phase":float(r.uniform(-math.pi,math.pi)),
        "layer_contrast":r.uniform(-.13,.13,3).tolist(),
        "speckle_vectors":r.uniform(-4,4,(5,3)).tolist(),
        "speckle_phases":r.uniform(-math.pi,math.pi,5).tolist(),
        "speckle_weights":r.uniform(-.02,.02,(5,3)).tolist()}


def _box(q, extent):
    d = np.abs(q)-extent
    return np.linalg.norm(np.maximum(d,0),axis=-1) + np.minimum(d.max(-1),0)


def shape_sdf(xyz, p):
    """Signed distance proxies: exact primitive composition except ellipsoid/roughness."""
    x = np.asarray(xyz,dtype=np.float64)
    q = (x-np.asarray(p["center"])) @ axis_angle_matrix(p["axis_angle"])
    ext = np.asarray(p["extent"]); f = p["family"]
    if f == "rough_ellipsoid":
        d = (np.linalg.norm(q/ext,axis=-1)-1)*ext.min()
        wave = np.sin(q*np.asarray(p["rough_frequency"])).prod(-1)
        return d + p["roughness"]*wave
    if f == "layered_block":
        return _box(q,ext)-.04
    if f == "ring":
        return np.sqrt((np.linalg.norm(q[...,:2],axis=-1)-p["ring_radius"])**2+q[...,2]**2)-p["tube_radius"]
    if f == "cluster":
        out = np.full(q.shape[:-1],10.)
        for c,e in zip(p["centers"],p["radii"]):
            e = np.asarray(e)
            out = np.minimum(out,(np.linalg.norm((q-c)/e,axis=-1)-1)*e.min())
        return out
    if f in ("capsules","noodles"):
        out = np.full(q.shape[:-1],10.)
        for a,b,rad in p["segments"]:
            a,b = np.asarray(a),np.asarray(b); ba=b-a
            t=np.clip(np.sum((q-a)*ba,axis=-1)/np.sum(ba*ba),0,1)
            out=np.minimum(out,np.linalg.norm(q-a-t[...,None]*ba,axis=-1)-rad)
        return out
    if f == "perforated_block":
        solid = _box(q,ext)-.025
        hole = np.full(q.shape[:-1],10.)
        for center in p["holes"]:
            hole=np.minimum(hole,np.linalg.norm(q[...,:2]-center,axis=-1)-p["hole_radius"])
        return np.maximum(solid,-hole)
    if f == "bowl":
        outer=(np.linalg.norm(q/ext,axis=-1)-1)*ext.min()
        inner_ext=ext-p["shell_thickness"]
        inner=(np.linalg.norm(q/inner_ext,axis=-1)-1)*inner_ext.min()
        return np.maximum(np.maximum(outer,-inner),q[...,2]-.09)
    raise KeyError(f)


def material_rgb(xyz,p):
    x=np.asarray(xyz,dtype=np.float64); shape=x.shape
    flat=x.reshape(-1,3)
    phase=flat@np.asarray(p["layer_axis"])*p["layer_frequency"]*math.pi+p["layer_phase"]
    layer=np.tanh(3*np.sin(phase))
    color=np.tile(p["base_rgb"],(len(flat),1))+layer[:,None]*p["layer_contrast"]
    modes=np.sin(flat@np.asarray(p["speckle_vectors"]).T*math.pi+p["speckle_phases"])
    color+=modes@p["speckle_weights"]
    return np.clip(color,.005,.995).reshape(shape)


def linear_to_srgb(x):
    x=np.clip(x,0,1)
    return np.rint(255*np.clip(np.where(x<=.0031308,x*12.92,1.055*x**(1/2.4)-.055),0,1)).astype(np.uint8)


def render_source(p,identity,image_size=64,surface_count=512):
    basis=camera_basis()
    pixels=(np.arange(image_size)+.5)/image_size*2-1
    u,v=np.meshgrid(pixels,-pixels,indexing="xy")
    origins=(u[...,None]*CAMERA_SCALE*basis[0]+v[...,None]*CAMERA_SCALE*basis[1]+2.5*basis[2]).reshape(-1,3)
    direction=-basis[2]
    b=origins@direction
    disc=b*b-(np.sum(origins*origins,axis=1)-1.42**2)
    active=disc>=0; t=np.zeros(len(origins)); end=np.zeros(len(origins))
    t[active]=-b[active]-np.sqrt(disc[active]); end[active]=-b[active]+np.sqrt(disc[active])
    hit=np.zeros(len(origins),bool)
    for _ in range(128):
        ids=np.flatnonzero(active)
        if not len(ids): break
        point=origins[ids]+t[ids,None]*direction
        sd=shape_sdf(point,p)
        arrived=sd<.001
        hit[ids[arrived]]=True; active[ids[arrived]]=False
        march=ids[~arrived]
        t[march]+=np.maximum(sd[~arrived]*.50,.0005)
        active[march]&=t[march]<=end[march]
    ids=np.flatnonzero(hit)
    point=origins[ids]+t[ids,None]*direction
    r=rng_for(identity,"lighting-background")
    background=r.uniform(.55,.9)+r.uniform(-.015,.015,3)
    gradient=r.uniform(-.035,.035)+r.uniform(-.005,.005,3)
    out=np.clip(background+v.reshape(-1,1)*gradient,.01,.97)
    depth=np.zeros(len(origins),np.float32)
    if len(ids):
        eps=.001
        normal=np.stack([shape_sdf(point+np.eye(3)[j]*eps,p)-shape_sdf(point-np.eye(3)[j]*eps,p) for j in range(3)],-1)
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-9)
        light=basis[2]+r.uniform(-.4,.4,3); light/=np.linalg.norm(light)
        shading=.55+.43*np.maximum(normal@light,0)
        out[ids]=material_rgb(point,p)*shading[:,None]
        depth[ids]=point@basis[2]
    sample_rng=rng_for(identity,"visible-surface-samples")
    chosen=sample_rng.choice(len(ids),surface_count,replace=len(ids)<surface_count) if len(ids) else np.zeros(surface_count,dtype=int)
    surface_xyz=point[chosen] if len(ids) else np.zeros((surface_count,3))
    surface_rgb=material_rgb(surface_xyz,p) if len(ids) else np.zeros((surface_count,3))
    return {"source_rgb":linear_to_srgb(out.reshape(image_size,image_size,3)),
        "source_mask":hit.reshape(image_size,image_size),
        "source_depth":depth.reshape(image_size,image_size).astype(np.float16),
        "surface_xyz":surface_xyz.astype(np.float16),"surface_rgb":surface_rgb.astype(np.float16),
        "surface_uv":known_camera_projection(surface_xyz).astype(np.float16),
        "surface_valid":np.full(surface_count,bool(len(ids)),dtype=bool)}


def generate_scene(index,seed=SEED,grid_size=16,image_size=64):
    identity=scene_identity(index,seed); p=scene_parameters(identity,index)
    xyz=canonical_grid(grid_size)
    sdf=shape_sdf(xyz,p); rgb=material_rgb(xyz,p)
    sub=xyz[:,None,:]+subcell_offsets(grid_size)[None,:,:]
    corners=shape_sdf(sub,p)
    record=render_source(p,identity,image_size)
    near_rng=rng_for(identity,"near-visible-surface")
    near=np.clip(record["surface_xyz"].astype(np.float64)+near_rng.normal(0,.035,(512,3)),-1,1)
    near_sdf=shape_sdf(near,p)
    near_rgb=material_rgb(near,p)
    near_corner=shape_sdf(near[:,None,:]+subcell_offsets(grid_size)[None,:,:],p)
    record.update(near_xyz=near.astype(np.float16),
        near_joint=np.concatenate([np.clip(near_sdf[:,None],-.5,.5)/.5,near_rgb*2-1],-1).astype(np.float16),
        near_corner_sdf=(np.clip(near_corner,-.5,.5)/.5).astype(np.float16))
    corner_target=(np.clip(corners,-.5,.5)/.5).astype(np.float16)
    record.update(joint=np.concatenate([np.clip(sdf[:,None],-.5,.5)/.5,rgb*2-1],-1).astype(np.float16),
        corner_sdf=corner_target,occupancy=sdf<=0,
        cell_occupancy=(corner_target<=0).mean(-1).astype(np.float16))
    return record,{"index":index,"scene_id":identity,"shape_family":p["family"],"target_generator_parameters":p}
