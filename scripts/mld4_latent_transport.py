"""Soft proximal material transport along the actual flow-matching schedule.

The prior is the source-corresponded visible surface only. Unobserved surfaces,
utensil and source cavity get no transport weight. This does not establish real
3D accuracy; it tests whether location-specific material survives generation.
"""
import numpy as np
from PIL import Image


def prepare_transport(pipe, guide, texture, confidence, food, canvas, seed):
    import torch
    import torch.nn.functional as F
    device = torch.device('cuda'); dtype = torch.bfloat16
    rgb = np.asarray(guide).copy()
    visible = (confidence >= .75) & food
    rgb[visible] = texture[visible]
    tensor = pipe.image_processor.preprocess(Image.fromarray(rgb), height=canvas[1], width=canvas[0]).to(device, dtype)
    tensor = torch.cat([tensor, torch.ones_like(tensor[:, :1])], dim=1).unsqueeze(2)
    with torch.inference_mode():
        prior = pipe._encode_vae_image(tensor, torch.Generator(device).manual_seed(seed))
    h, w = prior.shape[-2:]
    prior = pipe._pack_latents(prior, 1, pipe.latent_channels, h, w)
    weight = torch.from_numpy((confidence * visible).astype(np.float32))[None, None].to(device)
    weight = F.interpolate(weight, size=(h, w), mode='area').flatten(2).transpose(1, 2).to(dtype)
    noise5 = torch.randn((1, 1, pipe.latent_channels, h, w), generator=torch.Generator(device).manual_seed(seed), device=device, dtype=dtype)
    noise = pipe._pack_latents(noise5, 1, pipe.latent_channels, h, w)
    records = []

    def callback(pipeline, index, timestep, values):
        latent = values['latents']
        sigma = float(pipeline.scheduler.sigmas[index + 1])
        # Late material correction, with two final denoising steps for integration.
        strength = .28 if .04 < sigma < .65 else 0.
        reference = (1 - sigma) * prior + sigma * noise
        correction = strength * weight * (reference - latent)
        records.append(dict(step=index, sigma_next=sigma, strength=strength,
                            correction_rms=float(correction.float().square().mean().sqrt())))
        values['latents'] = latent + correction
        return values

    audit = dict(method='soft flow-schedule proximal source-material transport',
                 visible_pixels=int(visible.sum()), latent_grid=[w, h],
                 weighted_latent_tokens=int((weight > 0).sum()),
                 effective_latent_tokens=float(weight.float().sum()),
                 strength=.28, sigma_interval=[.04, .65], steps=records)
    return noise, callback, audit
