import unittest
from types import SimpleNamespace
import torch
from foodstateedit.observed_edit.interior import InteriorProjection,highpass_grid,release_strength


class FakePipe:
    vae_scale_factor=1
    scheduler=SimpleNamespace(sigmas=torch.tensor([1.,.8,.6,.4,0.]))
    @staticmethod
    def _pack_latents(x,b,c,h,w):return x.reshape(b,c,h//2,2,w//2,2).permute(0,2,4,1,3,5).reshape(b,h*w//4,c*4)
    @staticmethod
    def _unpack_latents(x,h,w,f):
        b,n,c=x.shape
        return x.reshape(b,h//2,w//2,c//4,2,2).permute(0,3,1,4,2,5).reshape(b,c//4,1,h,w)


class InteriorTests(unittest.TestCase):
    def test_constant_brightness_change_is_not_anchored_by_detail(self):
        x=torch.full((1,2,1,12,12),7.)
        self.assertEqual(float(highpass_grid(x).abs().max()),0.)

    def test_texture_change_is_retained_by_detail_operator(self):
        x=torch.zeros(1,2,1,12,12);x[:,:,:,::2,::2]=1
        self.assertGreater(float(highpass_grid(x).abs().max()),.5)

    def test_last_step_is_fully_released(self):
        self.assertEqual(release_strength(39,40),0.)
        self.assertAlmostEqual(release_strength(29,40),.8)
        self.assertLess(release_strength(35,40),.8)

    def test_callback_never_restores_background_and_releases_endpoint(self):
        pipe=FakePipe();shape=(1,36,8)
        clean=torch.zeros(shape);noise=torch.ones(shape);core=torch.zeros(shape);core[:,10:20]=1
        before=torch.full(shape,4.)
        callback=InteriorProjection(clean,noise,core,12,12,4,'interior')
        first=callback(pipe,0,None,{'latents':before})['latents']
        torch.testing.assert_close(first[core==0],before[core==0])
        self.assertLess(float(first[core>0].mean()),4)
        last=callback(pipe,3,None,{'latents':before})['latents']
        torch.testing.assert_close(last,before)

    def test_detail_callback_leaves_uniform_illumination_difference_free(self):
        pipe=FakePipe();clean=torch.zeros(1,36,8);noise=torch.zeros_like(clean)
        core=torch.zeros_like(clean);core[:,10:20]=1
        before=torch.full_like(clean,4.)
        callback=InteriorProjection(clean,noise,core,12,12,4,'detail')
        out=callback(pipe,0,None,{'latents':before})['latents']
        torch.testing.assert_close(out,before)


if __name__=='__main__':unittest.main()
