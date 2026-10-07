"""Regression: missing MoGe support must not poison an observed bite's SVD."""
import importlib.util
from pathlib import Path
import unittest
import sys

import numpy as np
from scipy.ndimage import gaussian_filter


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('finite_geometry',ROOT/'scripts/mld2_real_geometry_v2.py')
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)


class FiniteObservedSmoothingTest(unittest.TestCase):
    def test_missing_neighbors_do_not_poison_valid_patch(self):
        yy,xx=np.mgrid[:15,:15]
        points=np.stack([xx*.03,yy*.03,np.ones_like(xx)],axis=-1).astype(np.float32)
        valid=np.ones((15,15),bool);valid[:,0]=False;points[:,0]=np.inf
        patch=np.zeros_like(valid);patch[3:12,2:10]=True
        legacy=np.stack([gaussian_filter(points[...,k],.7) for k in range(3)],axis=-1)
        self.assertFalse(np.isfinite(legacy[patch]).all())
        result=geometry.smooth_observed_points(points,valid)
        self.assertTrue(np.isfinite(result[patch]).all())
        self.assertTrue(np.isnan(result[~valid]).all())
        self.assertEqual(int(np.isfinite(result).all(-1).sum()),int(valid.sum()))
        flat=result[patch,:2]/result[patch,2:]
        singular=np.linalg.svd(flat-np.median(flat,axis=0),full_matrices=False)[1]
        self.assertGreater(float(singular.min()),0)

    def test_constant_surface_not_pulled_toward_missing_zero(self):
        points=np.broadcast_to(np.array([1.,2.,5.],np.float32),(15,15,3)).copy()
        valid=np.ones((15,15),bool);valid[4:9,4:9]=False;points[~valid]=np.nan
        result=geometry.smooth_observed_points(points,valid)
        np.testing.assert_allclose(result[valid],np.tile([1.,2.,5.],(int(valid.sum()),1)),rtol=2e-7,atol=1e-6)
        self.assertTrue(np.isnan(result[~valid]).all())

    def test_fully_observed_input_matches_existing_smoothing(self):
        points=np.random.default_rng(17).normal(size=(19,23,3)).astype(np.float32)
        legacy=np.stack([gaussian_filter(points[...,k],.7) for k in range(3)],axis=-1)
        result=geometry.smooth_observed_points(points,np.ones(points.shape[:2],bool))
        np.testing.assert_array_equal(result,legacy)

    def test_finite_source_support_produces_finite_initial_translation(self):
        spec=importlib.util.spec_from_file_location('initial_geometry',ROOT/'scripts/mld2_real_geometry.py')
        initializer=importlib.util.module_from_spec(spec);spec.loader.exec_module(initializer)
        self.assertIs(initializer.smooth_observed_points,geometry.smooth_observed_points)
        yy,xx=np.mgrid[:15,:15]
        points=np.stack([xx*.03,yy*.03,4+yy*.01],axis=-1).astype(np.float32)
        valid=np.ones((15,15),bool);valid[:,0]=False;points[:,0]=np.inf
        selected=initializer.smooth_observed_points(points,valid)[3:12,2:10].reshape(-1,3)
        axes=np.diag([1.,-1.,-1.]);K=np.array([[300.,0.,200.],[0.,300.,150.],[0.,0.,1.]])
        result=initializer.lift_plan(selected,axes,1.,K,400,300)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.isfinite(initializer.project(selected+result,K)).all())


if __name__=='__main__':unittest.main()
