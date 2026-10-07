"""Ablation that disables Fourier bands above a declared frequency."""
import torch


def attach_lowpass(model, maximum_frequency=4):
    bands = model.config.coordinate_bands
    high = [band for band in range(bands) if 2**band > maximum_frequency]
    columns = [3+axis*bands+band for axis in range(3) for band in high]
    columns += [3+3*bands+axis*bands+band for axis in range(3) for band in high]

    def mask_position(module, arguments):
        value = arguments[0].clone()
        value[..., columns] = 0
        return (value,)

    model.coordinate_projection.register_forward_pre_hook(mask_position)
    return model
