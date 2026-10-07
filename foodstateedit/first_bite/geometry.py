"""Source-annotated relative geometry and a matched planar layout control.

World units are container diameters. The orthographic camera, support plane,
food shape and lighting are explicit hypotheses, not reconstructed ground truth.
Both guides use the same colors, camera frame and semantic source annotations.
"""
import math
import numpy as np
from PIL import Image, ImageDraw


class Camera:
    def __init__(self, width, height, annotation):
        self.width, self.height = width, height
        self.center = np.array(annotation['container_center']) * [width, height]
        self.scale = annotation['container_diameter'] * width
        self.angle = math.radians(annotation['elevation_degrees'])
        if not 10 < annotation['elevation_degrees'] < 85:
            raise ValueError('Degenerate or unsupported view')

    def project(self, points):
        p = np.asarray(points, float)
        return np.stack([self.center[0]+self.scale*p[..., 0],
                         self.center[1]+self.scale*(math.sin(self.angle)*p[..., 1]-math.cos(self.angle)*p[..., 2])], -1)

    def unproject(self, uv, z=0):
        u, v = np.asarray(uv) * [self.width, self.height]
        return np.array([(u-self.center[0])/self.scale,
                         ((v-self.center[1])/self.scale+math.cos(self.angle)*z)/math.sin(self.angle), z])


def ellipse(center, rx, ry, n=48):
    t=np.arange(n)*2*np.pi/n
    p=np.tile(center,(n,1))
    p[:,0]+=rx*np.cos(t);p[:,1]+=ry*np.sin(t)
    return p


def build_scene(width, height, ann, family, lift=.16):
    if not 0 <= lift <= .3:
        raise ValueError('Lift outside predefined diagnostic range')
    camera=Camera(width,height,ann)
    center=camera.unproject(ann['target_ground_xy'])
    center[2]=lift
    hand=camera.unproject(ann['hand_entry_xy'])
    hand[2]=lift+.05
    axis=hand-center;axis[2]=0;axis/=np.linalg.norm(axis)
    across=np.array([-axis[1],axis[0],0.])
    faces=[];lines=[]
    def face(points, color, semantic):
        faces.append({'points':np.asarray(points).tolist(),'color':color,'semantic':semantic})
    def line(points, color, thickness, semantic):
        lines.append({'points':np.asarray(points).tolist(),'color':color,'thickness':thickness,'semantic':semantic})
    def oriented_ring(c, along, transverse, n=48):
        t=np.arange(n)*2*np.pi/n
        return c+np.cos(t)[:,None]*along*axis+np.sin(t)[:,None]*transverse*across
    if family in ['liquid','granular']:
        outer=oriented_ring(center,.095,.057)
        bottom=oriented_ring(center-[0,0,.026],.062,.033)
        face(bottom,'#778792','spoon_bottom')
        for i in range(len(outer)):
            j=(i+1)%len(outer)
            face([outer[i],outer[j],bottom[j],bottom[i]],'#9baab4','spoon_wall')
        face(outer,'#dce4e7','spoon_rim')
        payload=oriented_ring(center+[0,0,.003],.076,.041)
        face(payload,'#dfa940','payload')
        if family=='granular':
            for i in range(16):
                t=i*2.39996;r=.011*np.sqrt(i)
                c=center+[r*np.cos(t),r*np.sin(t),.012+.01*(1-r/.05)]
                face(ellipse(c,.009,.005,10),'#f2c65d','grain')
        neck=center+axis*.075
        face([neck-across*.013,hand-across*.018,hand+across*.018,neck+across*.013],'#bac7cf','handle')
        contact=center+[0,0,.003]
        work_bottom=lift-.026
    elif family=='cohesive':
        heel=center+axis*.06
        face([heel-across*.045,heel+across*.045,center+across*.045,center-across*.045],'#bac7cf','fork_head')
        for offset in [-.036,-.012,.012,.036]:
            a=center+across*offset;b=a-axis*.095
            face([a-across*.006,a+across*.006,b+across*.006,b-across*.006],'#bac7cf','fork_tine')
        face([heel-across*.013,hand-across*.018,hand+across*.018,heel+across*.013],'#bac7cf','handle')
        pc=center-axis*.035+[0,0,.024]
        lo=np.array([[-.045,-.04,-.024],[.045,-.04,-.024],[.045,.04,-.024],[-.045,.04,-.024]])+pc
        hi=lo+[0,0,.057]
        for i in range(4):
            j=(i+1)%4;face([lo[i],lo[j],hi[j],hi[i]],['#c59641','#e3b15b','#bd9144','#dca44a'][i],'payload_side')
        face(hi,'#f1cc77','payload_top')
        contact=pc-[0,0,.024];work_bottom=lift
    else:
        for sign in [-1,1]:
            tip=center+across*sign*.008
            grip=hand+across*sign*.028
            line([tip,grip],'#9b6a3e',7,'chopstick')
        source=camera.unproject(ann['source_center'])
        for i in range(7):
            start=center+across*(i-3)*.009
            end=source+across*(i-3)*.012
            # Each strand starts at a pinched contact and droops to source food.
            points=[]
            for t in np.linspace(0,1,35):
                p=(1-t)*start+t*end
                p[0]+=.014*np.sin(np.pi*t)*np.sin(i*1.5)
                p[1]+=.04*np.sin(np.pi*t)
                points.append(p)
            line(points,'#d7ad61',5,'noodle')
        contact=center.copy();work_bottom=lift
    target_uv=camera.project(center)
    ground_uv=camera.project([center[0],center[1],0])
    return {'camera':camera,'faces':faces,'lines':lines,'center':center,'hand':hand,
            'contact':contact,'work_bottom_z':work_bottom,'lift':lift,
            'target_uv':target_uv,'ground_uv':ground_uv,'hand_uv':camera.project(hand)}


def render_control(width,height,ann,family,mode,lift=.16):
    if mode not in ['planar','geometry']:
        raise ValueError(mode)
    scene=build_scene(width,height,ann,family,lift)
    camera=scene['camera'];out=Image.new('RGB',(width,height),'#f8fafb');d=ImageDraw.Draw(out)
    scale=width/640
    def poly(points,fill,outline='#52616b'):
        uv=camera.project(points)
        d.polygon([tuple(x) for x in uv],fill=fill,outline=outline)
    # The same source support outline and source ROI appear in both controls.
    plate=ellipse(np.array([0.,0.,-.025]),.5,.5)
    poly(plate,'#eef1f3','#95a0a7')
    x0,y0,x1,y1=ann['source_box']
    box=[x0*width,y0*height,x1*width,y1*height]
    d.rounded_rectangle(box,radius=max(2,int(4*scale)),outline='#45917b',width=max(2,int(3*scale)))
    d.text((box[0],max(3,box[1]-14)), 'SOURCE BITE',fill='#2c6b59')
    center=scene['target_uv'];hand=scene['hand_uv']
    if mode=='geometry':
        # Simple directional-light ground projection: an assumption, not estimated illumination.
        shadow=[]
        for a in np.linspace(0,2*np.pi,50):
            p=scene['center']+np.array([.09*np.cos(a),.055*np.sin(a),0])
            p[:2]+=p[2]*np.array([.3,.45]);p[2]=-.025
            shadow.append(p)
        poly(shadow,'#d4d9dd','#d4d9dd')
        # Faces are ordered by approximate camera depth, retaining a real z coordinate.
        def depth(f):
            p=np.mean(f['points'],axis=0)
            return p[1]*math.cos(camera.angle)+p[2]*math.sin(camera.angle)
        for f in sorted(scene['faces'],key=depth):poly(f['points'],f['color'])
        for line in scene['lines']:
            d.line([tuple(x) for x in camera.project(line['points'])],fill=line['color'],width=max(2,int(line['thickness']*scale)),joint='curve')
        ground=scene['ground_uv']
        for t in np.arange(0,1,.12):
            a=(1-t)*ground+t*center;b=(1-min(t+.06,1))*ground+min(t+.06,1)*center
            d.line([tuple(a),tuple(b)],fill='#708a9b',width=max(1,int(scale)))
    else:
        # Same requested screen-space center/hand entry, without depth, side faces or cast shadow.
        thickness=max(3,int(12*scale))
        if family=='strand':
            for offset in [-6,6]:
                d.line([(center[0],center[1]+offset*scale),(hand[0],hand[1]+offset*scale)],fill='#9b6a3e',width=max(2,int(5*scale)))
            source=np.array(ann['source_center'])*[width,height]
            for offset in range(-3,4):d.line([(center[0]+offset*5*scale,center[1]),(source[0]+offset*5*scale,source[1])],fill='#d7ad61',width=max(2,int(3*scale)))
        else:
            d.line([tuple(center),tuple(hand)],fill='#bac7cf',width=thickness)
            rx=.095*camera.scale;ry=.055*camera.scale
            if family=='cohesive':
                d.rectangle([center[0]-rx*.5,center[1]-ry,center[0]+rx*.5,center[1]+ry],fill='#f1cc77',outline='#52616b')
            else:
                d.ellipse([center[0]-rx,center[1]-ry,center[0]+rx,center[1]+ry],fill='#bac7cf',outline='#52616b')
                d.ellipse([center[0]-rx*.76,center[1]-ry*.7,center[0]+rx*.76,center[1]+ry*.7],fill='#dfa940')
    d.ellipse([hand[0]-10*scale,hand[1]-10*scale,hand[0]+10*scale,hand[1]+10*scale],outline='#ad7e60',width=max(2,int(3*scale)))
    d.text((max(0,hand[0]-55*scale),hand[1]+14*scale),'HAND ENTRY',fill='#805b44')
    d.text((10,10),'SPATIAL GUIDE ONLY - USE PHOTO APPEARANCE',fill='#3c5261')
    audit={'coordinate_system':'relative orthographic; units=annotated container diameter',
           'geometry_measured':False,'camera_elevation_degrees':ann['elevation_degrees'],
           'lift_above_source_support_plane':lift,'work_bottom_z':scene['work_bottom_z'],
           'target_uv':scene['target_uv'].tolist(),'ground_uv':scene['ground_uv'].tolist(),
           'projected_lift_pixels':float(np.linalg.norm(scene['target_uv']-scene['ground_uv'])),
           'hand_uv':scene['hand_uv'].tolist(),'mode':mode,'family':family,
           'contact_world':scene['contact'].tolist(), 'faces':scene['faces'],'lines':scene['lines']}
    return out,audit
