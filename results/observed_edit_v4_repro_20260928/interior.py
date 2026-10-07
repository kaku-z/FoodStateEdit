"""Soft interior constraints with a fully generated background and boundary.

The distance-based weights are a heuristic confidence proxy. Latent high-pass
is not physical albedo and is not a guarantee of semantic identity.
"""
import torch
import torch.nn.functional as F


def release_strength(step, total_steps, maximum=.8, release_start=.75):
    progress=(step+1)/total_steps
    return maximum*min(1.,max(0.,(1-progress)/(1-release_start)))


def highpass_grid(delta, kernel_size=5):
    b,c,t,h,w=delta.shape
    if t!=1:raise ValueError('Only single-frame latents are supported')
    x=delta[:,:,0]
    low=F.avg_pool2d(F.pad(x,(kernel_size//2,)*4,mode='reflect'),kernel_size,stride=1)
    return (x-low).unsqueeze(2)


class InteriorProjection:
    def __init__(self,clean,noise,core_weight,height,width,total_steps,mode,
                 maximum=.8,release_start=.75,kernel_size=5):
        if clean.shape!=noise.shape or clean.shape!=core_weight.shape:
            raise ValueError('Aligned packed tensors required')
        if mode not in ['interior','detail']:raise ValueError('Unknown constraint mode')
        if total_steps<1 or not 0<=maximum<=1 or not 0<=release_start<1:
            raise ValueError('Invalid constraint schedule')
        if kernel_size<3 or kernel_size%2!=1:raise ValueError('Odd detail kernel required')
        if core_weight.min()<0 or core_weight.max()>1:raise ValueError('Core weights out of range')
        self.clean,self.noise,self.core=clean,noise,core_weight
        self.height,self.width,self.steps=height,width,total_steps
        self.mode,self.maximum,self.release_start=mode,maximum,release_start
        self.kernel_size=kernel_size
        self.audit=[]

    def __call__(self,pipe,step,timestep,kwargs):
        before=kwargs['latents']
        sigma=pipe.scheduler.sigmas[step+1].to(before.device,before.dtype)
        known=(1-sigma)*self.clean+sigma*self.noise
        delta=known-before
        if self.mode=='detail':
            grid=pipe._unpack_latents(delta,self.height,self.width,pipe.vae_scale_factor)
            detail=highpass_grid(grid,self.kernel_size)
            delta=pipe._pack_latents(detail,detail.shape[0],detail.shape[1],detail.shape[-2],detail.shape[-1])
        strength=release_strength(step,self.steps,self.maximum,self.release_start)
        out=before+strength*self.core*delta
        if not torch.isfinite(out).all():raise ValueError('Non-finite interior projection')
        self.audit.append({'step':step,'next_sigma':float(sigma),'strength':strength,
                           'update_rms':float((out.float()-before.float()).square().mean().sqrt()),
                           'outside_core_max_update':float((out[self.core==0]-before[self.core==0]).abs().max())})
        return {'latents':out}
