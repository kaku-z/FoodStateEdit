import numpy as np
import unittest
from foodstateedit.first_bite.geometry import Camera,build_scene,render_control


ANN={'container_center':[.5,.6],'container_diameter':.85,'elevation_degrees':40,
     'target_ground_xy':[.4,.6],'hand_entry_xy':[.98,.6],
     'source_center':[.5,.7],'source_box':[.45,.65,.55,.75]}


class FirstBiteGeometryTests(unittest.TestCase):
    def test_camera_roundtrip_with_nonzero_height(self):
        camera=Camera(800,600,ANN)
        for z in [0,.08,.16]:
            point=camera.unproject([.32,.54],z)
            np.testing.assert_allclose(camera.project(point),[256,324])


    def test_height_changes_projected_clearance_and_keeps_support(self):
        for family in ['cohesive','granular','strand','liquid']:
            with self.subTest(family=family):
                scenes=[build_scene(800,600,ANN,family,h) for h in [0,.08,.16]]
                self.assertGreater(scenes[2]['work_bottom_z'],0)
                self.assertGreater(scenes[0]['target_uv'][1],scenes[1]['target_uv'][1])
                self.assertGreater(scenes[1]['target_uv'][1],scenes[2]['target_uv'][1])
                self.assertEqual(scenes[0]['target_uv'][0],scenes[2]['target_uv'][0])
                np.testing.assert_allclose(scenes[2]['contact'][:2],scenes[0]['contact'][:2])
                self.assertAlmostEqual(scenes[2]['contact'][2]-scenes[0]['contact'][2],.16)


    def test_layout_and_geometry_share_contact_position_but_have_distinct_evidence(self):
        planar,a=render_control(640,480,ANN,'cohesive','planar')
        spatial,b=render_control(640,480,ANN,'cohesive','geometry')
        np.testing.assert_allclose(a['target_uv'],b['target_uv'])
        self.assertFalse(np.array_equal(np.asarray(planar),np.asarray(spatial)))
        scene=build_scene(640,480,ANN,'cohesive')
        self.assertEqual(sum(f['semantic']=='fork_tine' for f in scene['faces']),4)


    def test_invalid_height_cannot_silently_become_translation(self):
        with self.assertRaises(ValueError):build_scene(640,480,ANN,'liquid',-.1)


if __name__=='__main__':unittest.main()
