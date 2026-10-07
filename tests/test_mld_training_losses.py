"""Regression checks for fractional-material calibration and learned null flow."""
import unittest
import numpy as np
import torch
from scripts.train_mld_shared import rigid_delta, supervised_losses
from scripts.prepare_mld_pretraining_data import action_delta

def fixture(action=None):
    frac=torch.tensor([[[.25],[.5],[.75]]])
    source={'sdf':torch.zeros(1,3,1),'rgb':torch.zeros(1,3,3),
            'occupancy_logits':torch.zeros(1,3,1)}
    pred=frac.clone().requires_grad_()
    transition={'carried_fraction':pred,'carried_delta':torch.zeros(1,3,3,requires_grad=True),
                'remaining_delta':torch.zeros(1,3,3,requires_grad=True)}
    b={'occ':torch.ones(1,3,1),'joint':torch.zeros(1,3,4),'valid':torch.ones(1,3,1),
       'mass':torch.ones(1,3,1),'fraction':frac,'action':torch.zeros(1,10) if action is None else action,
       'carried_delta':torch.zeros(1,3,3)}
    return source,transition,b

class TrainingLossTests(unittest.TestCase):
    def test_partial_material_fraction_is_calibrated_at_target(self):
        s,t,b=fixture();supervised_losses(s,t,b)['fraction'].backward()
        self.assertLess(float(t['carried_fraction'].grad.abs().max()),1e-6)

    def test_fraction_gradient_points_toward_true_partial_budget(self):
        s,t,b=fixture();t['carried_fraction']=torch.full((1,3,1),.9,requires_grad=True)
        supervised_losses(s,t,b)['fraction'].backward()
        self.assertTrue(bool((t['carried_fraction'].grad>0).all()))

    def test_null_command_flow_is_learned_even_without_carried_material(self):
        s,t,b=fixture();b['fraction'].zero_()
        t['carried_delta']=torch.full((1,3,3),.2,requires_grad=True)
        supervised_losses(s,t,b)['carried_flow'].backward()
        self.assertTrue(bool((t['carried_delta'].grad>0).all()))

    def test_rigid_target_matches_independent_numpy_and_null(self):
        rng=np.random.default_rng(123);points=rng.normal(size=(2,10,3)).astype(np.float32)
        act=rng.normal(size=(2,10)).astype(np.float32);act[0]=0
        actual=rigid_delta(torch.from_numpy(points),torch.from_numpy(act)).numpy()
        expected=np.stack([action_delta(points[i],act[i]) for i in range(2)])
        np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=1e-6)
        np.testing.assert_array_equal(actual[0],np.zeros_like(actual[0]))

if __name__=='__main__':unittest.main()
