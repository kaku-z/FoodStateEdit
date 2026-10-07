import unittest
import copy
import numpy as np
from foodstateedit.material_lineage.strands import StrandBundle, deform_strands, flatten_bundle, reconstruct_strands, tube_geometry
from foodstateedit.material_lineage.strand_selection import select_visible_strand_bite


def synthetic_bundle():
    t = np.linspace(-.20,.20,65)
    xyz = np.c_[t,.045*np.sin(t*18),.08+.035*np.cos(t*12)]
    curve = dict(curve_id=0,graph_id=0,xyz=xyz,rest_xyz=xyz.copy(),source_surface_xyz=xyz.copy(),
        source_uv=np.c_[np.linspace(5,60,65),np.full(65,30.)],source_normal_uv=np.tile([0.,1.],(65,1)),
        source_rgb=np.tile([190,145,55],(65,1)),material_ids=np.arange(65),radius=np.full(65,.004),radius_pixels=np.full(65,1.8))
    return StrandBundle([curve],np.empty((0,0)),np.empty((0,0)),np.empty((0,0)))


class MaterialStrandTests(unittest.TestCase):
    def test_contact_and_lineage(self):
        bundle = synthetic_bundle()
        spoon = dict(axes=np.eye(3),center_local=[0,0,0],radius_ab=[.21,.09],floor_local=.025,bowl_curvature_height=.01)
        moved,metrics = deform_strands(bundle,[0,0,0],spoon,np.eye(3),iterations=240,self_collision=False)
        self.assertGreater(metrics['nonrigid_displacement_mean'],.01)
        self.assertLess(metrics['edge_strain_p95'],.04)
        self.assertGreaterEqual(metrics['support_min_gap'],-1e-10)
        self.assertGreaterEqual(metrics['tube_surface_min_support_gap'],-1e-10)
        self.assertLess(metrics['capsule_volume_proxy_relative_change'],.015)
        self.assertEqual(metrics['source_uv_max_change'],0.)
        np.testing.assert_array_equal(flatten_bundle(bundle)['material_ids'],flatten_bundle(moved)['material_ids'])

    def test_ablation_keeps_gravity_scale(self):
        bundle = synthetic_bundle()
        spoon = dict(axes=np.eye(3),center_local=[0,0,0],radius_ab=[.21,.09],floor_local=.025,bowl_curvature_height=.01)
        _,full=deform_strands(bundle,[0,0,0],spoon,np.eye(3),iterations=60,self_collision=False)
        _,no_contact=deform_strands(bundle,[0,0,0],spoon,np.eye(3),iterations=60,contact=False,self_collision=False)
        np.testing.assert_array_equal(full['gravity_acceleration_camera'],no_contact['gravity_acceleration_camera'])

    def test_source_overlap_is_reported_not_invented_motion(self):
        bundle = synthetic_bundle()
        second = copy.deepcopy(bundle.curves[0])
        second['curve_id'] = 1
        second['xyz'] += [0,.002,0]
        second['rest_xyz'] = second['xyz'].copy()
        second['material_ids'] += 1000
        bundle.curves.append(second)
        moved,metrics = deform_strands(bundle,[0,0,0],None,np.eye(3),gravity_strength=0,self_collision=True)
        self.assertGreater(metrics['source_ambiguous_overlap_pair_count'],0)
        np.testing.assert_allclose(flatten_bundle(bundle)['xyz'],flatten_bundle(moved)['xyz'],atol=1e-10)

    def test_actual_bowl_neck_tube_contact(self):
        bundle=synthetic_bundle()
        bundle.curves[0]['xyz'][:,0]=np.linspace(.16,.25,65)
        bundle.curves[0]['xyz'][:,1]=.002*np.sin(np.linspace(0,6,65))
        bundle.curves[0]['xyz'][:,2]=.031
        bundle.curves[0]['rest_xyz']=bundle.curves[0]['xyz'].copy()
        bundle.curves[0]['source_surface_xyz']=bundle.curves[0]['xyz'].copy()
        spoon=dict(axes=np.eye(3),spoon_axes_camera=np.eye(3),center_local=[0,0],radius_ab=[.21,.09],
            floor_local=.025,bowl_curvature_height=.01,bowl_handle_blend=True)
        moved,metrics=deform_strands(bundle,[0,0,0],spoon,np.eye(3),iterations=120,self_collision=False)
        self.assertGreater(metrics['tube_surface_vertices_inside_bowl'],0)
        self.assertGreaterEqual(metrics['tube_surface_min_support_gap'],-1e-9)
        self.assertLess(metrics['edge_strain_p95'],.04)

    def test_no_gravity_preserves_rigid_action(self):
        bundle = synthetic_bundle()
        moved,metrics = deform_strands(bundle,[.5,.2,.4],None,np.eye(3),gravity_strength=0,self_collision=False)
        np.testing.assert_allclose(flatten_bundle(moved)['xyz'],flatten_bundle(bundle)['xyz']+[.5,.2,.4],atol=1e-10)
        self.assertLess(metrics['capsule_volume_proxy_relative_change'],1e-10)

    def test_observed_ridge_patch_partition(self):
        yy,xx=np.indices((80,100));stripe=180*np.exp(-((yy-40)/2.2)**2)
        image=np.repeat((45+stripe).astype(np.uint8)[...,None],3,axis=2)
        points=np.stack([xx*.001,yy*.001,np.ones_like(xx)],axis=-1)
        food=(yy>30)&(yy<50);patch=food&(xx>35)&(xx<75)
        bundle=reconstruct_strands(image,food,points,patch,min_length_pixels=6)
        self.assertGreater(len(bundle.curves),0)
        self.assertFalse(np.any(bundle.removal_mask&~food))
        self.assertGreater(bundle.metrics['ownership_expanded_beyond_seed_pixels'],0)
        self.assertEqual(bundle.metrics['tube_source_uv_owned_fraction'],1.)
        for c in bundle.curves:
            self.assertTrue(np.all((c['source_uv'][:,0]>34)&(c['source_uv'][:,0]<76)))
        mesh=tube_geometry(bundle,image)
        self.assertTrue(np.isfinite(mesh['vertices']).all())
        self.assertEqual(len(mesh['vertices']),len(mesh['material_ids']))

    def test_ingredient_selection_excludes_distractor(self):
        yy,xx=np.indices((100,140));signal=120*np.exp(-((yy-64)/1.7)**2)+120*np.exp(-((yy-76)/1.7)**2)+180*np.exp(-((yy-25)/2.)**2)
        image=np.repeat((35+signal).clip(0,255).astype(np.uint8)[...,None],3,axis=2)
        points=np.stack([xx*.001,yy*.001,np.ones_like(xx)],axis=-1)
        ingredient=(yy>58)&(yy<82)
        patch,center,radius,metrics=select_visible_strand_bite(image,ingredient,points,minimum_span_pixels=24)
        self.assertTrue(metrics['sufficient_visible_bite'])
        self.assertGreater(center[1],58)
        self.assertFalse(np.any(patch&~ingredient))
        self.assertAlmostEqual(radius,14.4)

    def test_ingredient_hole_terminates_texture_cross_section(self):
        yy,xx=np.indices((80,100));stripe=180*np.exp(-((yy-40)/4.)**2)
        image=np.repeat((45+stripe).astype(np.uint8)[...,None],3,axis=2)
        points=np.stack([xx*.001,yy*.001,np.ones_like(xx)],axis=-1)
        food=(yy>30)&(yy<50);food[43,:]=False
        patch=food&(xx>35)&(xx<75)
        bundle=reconstruct_strands(image,food,points,patch,min_length_pixels=6)
        self.assertGreater(len(bundle.curves),0)
        mesh=tube_geometry(bundle,image)
        uv=mesh['uv_pixels']
        # A strip of excluded garnish cannot be crossed and sampled on its
        # far side merely because the farther pixels belong to the ingredient.
        above=np.concatenate([np.repeat(np.median(c['source_uv'][:,1])<43,len(c['xyz'])*10) for c in bundle.curves])
        self.assertTrue(np.all(uv[above,1]<43))
        self.assertEqual(bundle.metrics['tube_source_uv_owned_fraction'],1.)
        # A high-contrast excluded ingredient must contribute zero RGB through
        # any nonzero bilinear weight, not merely less than half the sample.
        audit_texture=np.where(food[...,None],np.array([15,180,55]),np.array([240,10,200])).astype(np.uint8)
        actual=tube_geometry(bundle,audit_texture)['source_rgb']
        np.testing.assert_allclose(actual,np.tile([15,180,55],(len(actual),1)),atol=1e-10)


if __name__=='__main__':unittest.main()
