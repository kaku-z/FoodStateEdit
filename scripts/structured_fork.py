"""Rounded one-piece fork proxy with flat support patches and a shaped handle."""
import numpy as np
import manifold3d as manifold
import trimesh

def loft(xs,widths,zs,thickness,center,offset=0.):
    # Beveled rectangular cross-section: flat top contact, rounded side highlights.
    verts=[]
    for x,w,z in zip(xs,widths,zs):
        b=min(w*.40,thickness*.35)
        ring=[(-w+b,z), (w-b,z),(w,z-b),(w,z-thickness+b),
              (w-b,z-thickness),(-w+b,z-thickness),(-w,z-thickness+b),(-w,z-b)]
        verts.extend([[x,y+offset,zz] for y,zz in ring])
    faces=[];n=8
    for i in range(len(xs)-1):
        for j in range(n):
            a=i*n+j;b=i*n+(j+1)%n;c=(i+1)*n+(j+1)%n;d=(i+1)*n+j
            faces.extend([[a,b,c],[a,c,d]])
    for j in range(1,n-1):faces.extend([[0,j+1,j],[(len(xs)-1)*n,(len(xs)-1)*n+j,(len(xs)-1)*n+j+1]])
    m=trimesh.Trimesh(np.asarray(verts)+center,np.asarray(faces),process=True)
    trimesh.repair.fix_normals(m)
    assert m.is_watertight and m.volume>0
    return manifold.Manifold(manifold.Mesh64(np.asarray(m.vertices),np.asarray(m.faces,dtype=np.uint64)))

def make_fork(center,edge):
    c=np.asarray(center)/edge;pieces=[];contacts=[]
    for offset in [-.36,-.12,.12,.36]:
        pieces.append(loft([-.82,-.79,-.72,-.63,.20,.48],[.005,.030,.050,.055,.058,.065],[-.045,-.018,-.003,0,0,0],.055,c,offset))
        # Flat upper support patch excludes the bevel and tapered tip.
        contacts.append((np.asarray(center[:2])+np.array([-.50,offset-.035])*edge,
                         np.asarray(center[:2])+np.array([.47,offset+.035])*edge))
    pieces.append(loft([.40,.55,.72,.92,1.15,1.35,2.0,3.0,4.2,4.5],
                       [.435,.435,.36,.24,.13,.115,.145,.19,.22,.10],
                       [0,0,-.015,-.05,-.10,-.15,-.18,-.18,-.15,-.14],.075,c))
    solid=pieces[0]
    for p in pieces[1:]:solid=solid+p
    return solid.scale((edge,edge,edge)),contacts
