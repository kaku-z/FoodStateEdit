"""Flow-matching latent projection, called after the scheduler's update."""


class RegionProjection:
    def __init__(self, clean, noise, free_weight):
        if clean.shape != noise.shape or clean.shape != free_weight.shape:
            raise ValueError('Packed latent/reference/mask shapes must match')
        if free_weight.min() < 0 or free_weight.max() > 1:
            raise ValueError('Projection weights must be in [0,1]')
        self.clean, self.noise, self.free = clean, noise, free_weight
        self.audit = []

    def __call__(self, pipe, step, timestep, kwargs):
        # step() has already advanced the scheduler. Use NEXT sigma, not t/1000.
        sigma = pipe.scheduler.sigmas[step+1].to(self.clean.device, self.clean.dtype)
        known = (1-sigma)*self.clean+sigma*self.noise
        before = kwargs['latents']
        projected = self.free*before+(1-self.free)*known
        pinned = self.free == 0
        error = float((projected[pinned]-known[pinned]).abs().max()) if pinned.any() else 0.
        self.audit.append({'step': int(step), 'next_sigma': float(sigma),
                           'pinned_latent_max_error': error})
        return {'latents': projected}
