"""Source-conditioned material-coordinate DDPM/DDIM pilot.

This trains a small epsilon-prediction MLP on *observed bare source food RGB*.
It is not the full proposed material-lineage diffusion network, a 3-D shape
model, or a reconstruction of hidden material. Lineage, geometry, and persistent
material-ID noise are supplied by the caller. The sampling function never draws
noise or reads world-space coordinates.

Input NPZ: coords (N, 3), normalized canonical material coordinates in [0, 1];
colors (N, 3), linear RGB in [0, 1]; base_color (3,), source-only linear RGB.
Example:
    train_field(samples_path, output_dir, seed=41, steps=6000, device="cuda")
    field = load_field(output_dir / "material_denoiser.pt", device="cuda")
    rgb = field.sample_material(coords, material_id_noise, steps=24)

The train/validation split contains pixels from the same source photograph. It
is not held-out food, measured 3-D truth, or an independent realism evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

# Required by CUDA deterministic GEMM; configure before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


FORMAT_VERSION = 1
SCOPE = (
    "Small source-conditioned epsilon MLP trained only on observed source color "
    "noise pairs. Unobserved material is inferred; no learned shape, complete "
    "lineage model, calibrated optics, or measured hidden material is claimed."
)


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(array):
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(json.dumps(list(value.shape)).encode("ascii"))
    h.update(value.tobytes())
    return h.hexdigest()


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def _check_rows(array, name, bounded=False):
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("%s must have shape (N, 3)" % name)
    if not np.isfinite(result).all():
        raise ValueError("%s contains a nonfinite value" % name)
    if bounded and result.size and (result.min() < 0 or result.max() > 1):
        raise ValueError("%s must be in [0, 1]" % name)
    return np.ascontiguousarray(result)


def _schedule(count=1000, beta_start=0.0001, beta_end=0.02):
    if count < 2 or not 0 < beta_start <= beta_end < 1:
        raise ValueError("Invalid DDPM beta schedule")
    # Construct in float64, then store float32 for training and sampling.
    betas = torch.linspace(beta_start, beta_end, count, dtype=torch.float64)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0).float()
    return betas.float(), alpha_bar


class MaterialEpsilonMLP(nn.Module):
    """Pointwise epsilon network; no BatchNorm, dropout, or batch aggregation."""

    def __init__(self, width=128, hidden_layers=4, coord_bands=6, time_bands=8):
        super().__init__()
        if width < 8 or hidden_layers < 1 or coord_bands < 1 or time_bands < 1:
            raise ValueError("Invalid MLP configuration")
        self.register_buffer("coord_freq", 2.0 ** torch.arange(coord_bands, dtype=torch.float32))
        self.register_buffer("time_freq", 2.0 ** torch.arange(time_bands, dtype=torch.float32))
        dimension = 3 + 6 * coord_bands + 3 + 1 + 2 * time_bands + 3
        layers = []
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(dimension, width), nn.SiLU()])
            dimension = width
        layers.append(nn.Linear(width, 3))
        self.network = nn.Sequential(*layers)
        nn.init.normal_(self.network[-1].weight, std=0.001)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, coords, noisy_rgb, normalized_t, base_color):
        phase = 2.0 * torch.pi * coords.unsqueeze(-1) * self.coord_freq
        coords_features = torch.cat(
            [coords, phase.sin().flatten(1), phase.cos().flatten(1)], dim=1
        )
        t = normalized_t.reshape(-1, 1)
        time_phase = 2.0 * torch.pi * t * self.time_freq
        time_features = torch.cat([t, time_phase.sin(), time_phase.cos()], dim=1)
        base = base_color.reshape(1, 3).expand(coords.shape[0], -1)
        return self.network(torch.cat([coords_features, noisy_rgb, time_features, base], dim=1))


class MaterialField:
    """Loaded learned color field with persistent-noise deterministic sampling."""

    def __init__(self, checkpoint, device="cpu"):
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported material denoiser checkpoint")
        self.device = torch.device(device)
        self.model = MaterialEpsilonMLP(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.base_color = torch.tensor(checkpoint["base_color"], dtype=torch.float32, device=self.device)
        self.alpha_bar = checkpoint["alpha_bar"].float().to(self.device)
        self.diffusion_steps = len(self.alpha_bar)
        self.compute_block_size = int(checkpoint["compute_block_size"])
        self.actual_optimizer_steps = int(checkpoint["actual_optimizer_steps"])
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def _sample_block(self, coords, material_noise, steps):
        """Fixed-size pointwise block keeps GEMM shape stable across callers."""
        length = len(coords)
        block = self.compute_block_size
        padded_coords = np.zeros((block, 3), dtype=np.float32)
        padded_noise = np.zeros((block, 3), dtype=np.float32)
        padded_coords[:length] = coords
        padded_noise[:length] = material_noise
        c = torch.from_numpy(padded_coords).to(self.device)
        x = torch.from_numpy(padded_noise).to(self.device)
        # A subsampled DDIM trajectory, starting at the final DDPM noise level.
        ts = np.rint(np.linspace(self.diffusion_steps - 1, 0, steps)).astype(np.int64)
        if len(np.unique(ts)) != len(ts):
            raise AssertionError("Repeated DDIM timesteps")
        for index, timestep in enumerate(ts):
            a = self.alpha_bar[int(timestep)]
            next_t = int(ts[index + 1]) if index + 1 < len(ts) else -1
            next_a = self.alpha_bar[next_t] if next_t >= 0 else torch.ones_like(a)
            t = torch.full((block,), float(timestep) / (self.diffusion_steps - 1), device=self.device)
            epsilon = self.model(c, x, t, self.base_color)
            x0 = ((x - (1.0 - a).sqrt() * epsilon) / a.sqrt()).clamp(0.0, 1.0)
            # Recompute epsilon after clipping x0. This is standard clipped
            # deterministic DDIM: eta=0, hence no extra random term.
            epsilon = (x - a.sqrt() * x0) / (1.0 - a).sqrt()
            x = next_a.sqrt() * x0 + (1.0 - next_a).sqrt() * epsilon
        return x[:length].clamp(0.0, 1.0).cpu().numpy()

    def sample_material(self, coords_np, material_noise_np, steps=24, device=None, chunk_size=None):
        """Return (N,3) linear RGB; coords and persistent-ID noise are required.

        No RNG is called. The same coordinates/noise/checkpoint/device produce
        the same values. Internally every MLP evaluation is padded to the fixed
        checkpoint block size, so point batches have no semantic dependence.
        ``chunk_size`` bounds caller staging only and never changes GEMM shape.
        Cross-device bitwise identity is not claimed.
        """
        if device is not None and torch.device(device) != self.device:
            raise ValueError("Load the field on the requested device before sampling")
        coords = _check_rows(coords_np, "coords", bounded=True)
        noise = _check_rows(material_noise_np, "material_noise")
        if len(coords) != len(noise):
            raise ValueError("coords and material_noise lengths differ")
        steps = int(steps)
        if not 1 <= steps <= self.diffusion_steps:
            raise ValueError("DDIM steps must be between 1 and diffusion_steps")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive")
        output = np.empty_like(coords)
        # Stage requests in multiples of the immutable compute block. A caller
        # can also call this method independently on any subset of points.
        stage = self.compute_block_size if chunk_size is None else max(
            self.compute_block_size,
            (int(chunk_size) // self.compute_block_size) * self.compute_block_size,
        )
        for outer in range(0, len(coords), stage):
            stop = min(len(coords), outer + stage)
            for start in range(outer, stop, self.compute_block_size):
                end = min(stop, start + self.compute_block_size)
                output[start:end] = self._sample_block(coords[start:end], noise[start:end], steps)
        if not np.isfinite(output).all() or (output.size and (output.min() < 0 or output.max() > 1)):
            raise RuntimeError("Material sampler produced invalid linear RGB")
        return output


def load_field(checkpoint_path, device="cpu"):
    """Load a local checkpoint; returned object is callable via sample_material."""
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if int(checkpoint.get("actual_optimizer_steps", 0)) < 1:
        raise ValueError("Checkpoint has no actual training steps")
    return MaterialField(checkpoint, device=device)


@torch.inference_mode()
def _validation(model, alpha_bar, coords, colors, base, seed, block_size):
    """Fixed held-out noise/t pairs plus their clean-RGB reconstruction errors."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(coords), 3)).astype(np.float32)
    device = base.device
    schedule = sorted(set(np.rint(np.linspace(0, len(alpha_bar) - 1, 7)).astype(int).tolist()))
    rows = []
    model.eval()
    for timestep in schedule:
        eps_sse = clean_sse = 0.0
        count = 0
        for start in range(0, len(coords), block_size):
            end = min(len(coords), start + block_size)
            c = torch.from_numpy(coords[start:end]).to(device)
            y = torch.from_numpy(colors[start:end]).to(device)
            e = torch.from_numpy(noise[start:end]).to(device)
            a = alpha_bar[timestep]
            noisy = a.sqrt() * y + (1 - a).sqrt() * e
            t = torch.full((len(c),), timestep / (len(alpha_bar) - 1), device=device)
            prediction = model(c, noisy, t, base)
            clean = ((noisy - (1 - a).sqrt() * prediction) / a.sqrt()).clamp(0, 1)
            eps_sse += float((prediction - e).square().sum())
            clean_sse += float((clean - y).square().sum())
            count += e.numel()
        rows.append(dict(timestep=timestep, epsilon_mse=eps_sse / count, clipped_clean_rgb_mse=clean_sse / count))
    return dict(
        epsilon_mse=float(np.mean([x["epsilon_mse"] for x in rows])),
        clipped_clean_rgb_mse=float(np.mean([x["clipped_clean_rgb_mse"] for x in rows])),
        timesteps=rows,
        source_pixel_count=len(coords),
        noise_sha256=_array_sha(noise),
        interpretation="Same-photo held-out observed pixels; spatially correlated, not object-level generalization.",
    )


def train_field(samples_path, output_dir, seed=41, steps=6000, device=None, batch_size=1024,
                learning_rate=0.001, validation_fraction=0.20, width=128,
                hidden_layers=4, diffusion_steps=1000, compute_block_size=256,
                validation_max_pixels=4096, log_interval=100):
    """Train epsilon MSE on source-only colors and save checkpoint + receipt.

    Each update samples t and epsilon, constructs
    x_t=sqrt(alpha_bar_t)*color+sqrt(1-alpha_bar_t)*epsilon,
    and minimizes mean squared epsilon prediction error. No generated image or
    hidden-material target is used. The caller is responsible for selecting bare
    observed food pixels and supplying linear (not sRGB) colors.
    """
    steps, seed, batch_size = int(steps), int(seed), int(batch_size)
    compute_block_size = int(compute_block_size)
    if steps < 1 or batch_size < 1 or compute_block_size < 1:
        raise ValueError("steps, batch_size, and compute_block_size must be positive")
    if not 0 < validation_fraction < 1 or learning_rate <= 0:
        raise ValueError("Invalid validation_fraction or learning_rate")
    if validation_max_pixels < 1 or log_interval < 1:
        raise ValueError("validation_max_pixels and log_interval must be positive")
    samples_path, output_dir = Path(samples_path), Path(output_dir)
    with np.load(samples_path, allow_pickle=False) as data:
        coords = _check_rows(data["coords"], "coords", bounded=True)
        colors = _check_rows(data["colors"], "colors", bounded=True)
        base_np = np.asarray(data["base_color"], dtype=np.float32)
    if len(coords) != len(colors) or len(coords) < 32:
        raise ValueError("Need at least 32 matching observed source color/coordinate rows")
    if base_np.shape != (3,) or not np.isfinite(base_np).all() or base_np.min() < 0 or base_np.max() > 1:
        raise ValueError("base_color must have shape (3,) and linear RGB in [0,1]")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    _deterministic(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split_rng = np.random.default_rng(seed)
    permutation = split_rng.permutation(len(coords))
    nval = min(len(coords) - 16, max(8, int(round(len(coords) * validation_fraction))))
    val_all = permutation[:nval]
    train_ids = permutation[nval:]
    val_ids = val_all[:int(validation_max_pixels)]
    model_config = dict(width=int(width), hidden_layers=int(hidden_layers), coord_bands=6, time_bands=8)
    model = MaterialEpsilonMLP(**model_config).to(target_device)
    betas_cpu, alpha_cpu = _schedule(int(diffusion_steps))
    alpha = alpha_cpu.to(target_device)
    base = torch.from_numpy(base_np).to(target_device)
    train_coords = torch.from_numpy(coords[train_ids]).to(target_device)
    train_colors = torch.from_numpy(colors[train_ids]).to(target_device)
    val_coords, val_colors = coords[val_ids], colors[val_ids]
    parameter_count = sum(p.numel() for p in model.parameters())
    receipt = dict(
        status="training", format_version=FORMAT_VERSION, scope=SCOPE,
        created_unix=started, samples_path=str(samples_path.resolve()),
        samples_sha256=_file_sha(samples_path),
        samples_arrays_sha256=dict(coords=_array_sha(coords), colors=_array_sha(colors), base_color=_array_sha(base_np)),
        script_sha256=_file_sha(Path(__file__)), seed=seed,
        requested_optimizer_steps=steps, actual_optimizer_steps=0,
        parameter_count=parameter_count, model_config=model_config,
        device=str(target_device), torch_version=str(torch.__version__), numpy_version=str(np.__version__),
        source_observed_sample_count=len(coords), train_source_pixels=len(train_ids),
        heldout_source_pixels=len(val_all), evaluated_heldout_source_pixels=len(val_ids),
        split_hashes=dict(train_indices=_array_sha(train_ids), heldout_indices=_array_sha(val_all), evaluated_heldout_indices=_array_sha(val_ids)),
        base_color_linear_rgb=base_np.tolist(),
        schedule=dict(type="linear_beta_ddpm", steps=int(diffusion_steps), beta_start=float(betas_cpu[0]), beta_end=float(betas_cpu[-1]), final_alpha_bar=float(alpha_cpu[-1])),
        training=dict(objective="DDPM epsilon prediction MSE", batch_size=batch_size, learning_rate=learning_rate, validation_fraction=validation_fraction),
        sampling=dict(method="DDIM", eta=0.0, clipping="x0 in [0,1], epsilon recomputed after clipping", compute_block_size=compute_block_size,
                      randomness="Only caller-supplied persistent material-ID noise; sampler makes no RNG calls", output="Linear RGB in [0,1]"),
        limitations=["Caller must select bare visible source pixels; this script cannot verify their provenance.",
                     "Same-photo random pixel validation is not held-out food or real hidden-material accuracy.",
                     "Identity and conservative geometry are external; this MLP predicts color only.",
                     "Cross-device bitwise identity and physically calibrated scattering are not claimed."],
        losses=[],
    )
    receipt_path = output_dir / "training_receipt.json"
    _write_json(receipt_path, receipt)
    before = _validation(model, alpha, val_coords, val_colors, base, seed + 1907, compute_block_size)
    receipt["heldout_before_training"] = before
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device=target_device).manual_seed(seed + 811)
    running_loss = 0.0
    model.train()
    for step in range(1, steps + 1):
        indices = torch.randint(len(train_coords), (batch_size,), generator=generator, device=target_device)
        c, y = train_coords[indices], train_colors[indices]
        timestep = torch.randint(len(alpha), (batch_size,), generator=generator, device=target_device)
        noise = torch.randn((batch_size, 3), generator=generator, device=target_device)
        a = alpha[timestep, None]
        noisy = a.sqrt() * y + (1.0 - a).sqrt() * noise
        prediction = model(c, noisy, timestep.float() / (len(alpha) - 1), base)
        loss = (prediction - noise).square().mean()
        if not torch.isfinite(loss):
            receipt.update(status="failed_nonfinite_loss", actual_optimizer_steps=step - 1)
            _write_json(receipt_path, receipt)
            raise RuntimeError("Nonfinite epsilon training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        optimizer.step()
        running_loss += float(loss.detach())
        receipt["actual_optimizer_steps"] = step
        if step % log_interval == 0 or step == steps:
            window = log_interval if step % log_interval == 0 else step % log_interval
            row = dict(step=step, epsilon_mse=running_loss / window, seconds=time.time() - started)
            receipt["losses"].append(row)
            running_loss = 0.0
            _write_json(receipt_path, receipt)
            print("MATERIAL_DENOISER_STEP", json.dumps(row), flush=True)
    model.eval()
    receipt["heldout_after_training"] = _validation(model, alpha, val_coords, val_colors, base, seed + 1907, compute_block_size)
    checkpoint = dict(
        format_version=FORMAT_VERSION, model_config=model_config,
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        base_color=base_np.tolist(), betas=betas_cpu, alpha_bar=alpha_cpu,
        compute_block_size=compute_block_size, actual_optimizer_steps=steps,
        input_samples_sha256=receipt["samples_sha256"], seed=seed, scope=SCOPE,
    )
    checkpoint_path = output_dir / "material_denoiser.pt"
    torch.save(checkpoint, checkpoint_path)
    field = MaterialField(checkpoint, device=target_device)
    sample_rng = np.random.default_rng(seed + 2503)
    check_count = min(len(val_coords), 256)
    check_coords = val_coords[:check_count]
    check_noise = sample_rng.standard_normal((check_count, 3)).astype(np.float32)
    sample_steps = min(24, int(diffusion_steps))
    sampled = field.sample_material(check_coords, check_noise, steps=sample_steps)
    repeated = field.sample_material(check_coords, check_noise, steps=sample_steps, chunk_size=compute_block_size * 3)
    split = max(1, check_count // 3)
    independent_batches = np.concatenate([
        field.sample_material(check_coords[:split], check_noise[:split], steps=sample_steps),
        field.sample_material(check_coords[split:], check_noise[split:], steps=sample_steps),
    ], axis=0)
    chunk_error = float(np.max(np.abs(sampled - independent_batches)))
    receipt["sampling_validation"] = dict(
        points=check_count, ddim_steps=sample_steps,
        persistent_noise_sha256=_array_sha(check_noise), output_sha256=_array_sha(sampled),
        repeat_bitwise_equal=bool(np.array_equal(sampled, repeated)),
        independent_batch_bitwise_equal=bool(np.array_equal(sampled, independent_batches)),
        independent_batch_max_absolute_error=chunk_error,
        final_linear_rgb_range=[float(sampled.min()), float(sampled.max())],
        heldout_ddim_rgb_mse=float(np.mean((sampled - val_colors[:check_count]) ** 2)),
        heldout_base_color_rgb_mse=float(np.mean((base_np - val_colors[:check_count]) ** 2)),
    )
    # Strict point-batch identity is an implementation requirement. Fail visibly
    # instead of declaring deterministic identity on a backend that changes it.
    if not np.array_equal(sampled, repeated) or not np.array_equal(sampled, independent_batches):
        receipt.update(status="failed_batch_identity", checkpoint_sha256=_file_sha(checkpoint_path), seconds=time.time() - started)
        _write_json(receipt_path, receipt)
        raise RuntimeError("Sampling batch identity failed; inspect receipt and backend")
    receipt.update(status="complete", checkpoint_path=str(checkpoint_path.resolve()),
                   checkpoint_sha256=_file_sha(checkpoint_path), seconds=time.time() - started)
    _write_json(receipt_path, receipt)
    print("MATERIAL_DENOISER_COMPLETE", json.dumps(dict(
        actual_optimizer_steps=steps, parameter_count=parameter_count,
        heldout_epsilon_mse=receipt["heldout_after_training"]["epsilon_mse"],
        checkpoint_sha256=receipt["checkpoint_sha256"])), flush=True)
    return dict(checkpoint_path=str(checkpoint_path), receipt_path=str(receipt_path), receipt=receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    args = parser.parse_args()
    train_field(args.samples_path, args.output_dir, seed=args.seed, steps=args.steps,
                device=args.device, batch_size=args.batch_size, learning_rate=args.learning_rate,
                width=args.width, hidden_layers=args.hidden_layers, diffusion_steps=args.diffusion_steps)


if __name__ == "__main__":
    main()
