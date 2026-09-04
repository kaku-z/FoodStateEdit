#!/usr/bin/env python3
"""Seeded Wan/VACE LoRA trainer with a matched uniform or topology loss.

This file is executed inside the frozen DiffSynth-Studio runtime.  It leaves
that runtime untouched and records the exact timestep/noise sequence used by
each optimizer step so the two relative-3D arms can be compared causally.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from diffsynth.core import OffloadTrainingManager, UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadAudio,
    LoadVideo,
    ToAbsolutePath,
)
from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    save_training_args,
)
from examples.wanvideo.model_training.train import WanTrainingModule, wan_parser


MASK_KEYS = (
    "flexible_strand_mask_video",
    "pinch_contact_mask_video",
    "source_connection_mask_video",
)
LOSS_MODES = ("uniform", "topology_weighted")


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().to("cpu").contiguous().view(torch.uint8)
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def trainable_state_manifest(model: WanTrainingModule) -> dict[str, object]:
    state = model.export_trainable_state_dict(model.state_dict())
    aggregate = hashlib.sha256()
    tensors: list[dict[str, object]] = []
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        aggregate.update(name.encode("utf-8"))
        aggregate.update(str(tensor.dtype).encode("ascii"))
        aggregate.update(json.dumps(list(tensor.shape)).encode("ascii"))
        aggregate.update(raw)
        tensors.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": digest,
            }
        )
    return {
        "schema_version": "foodstateedit.vace_lora_initialization.v1",
        "trainable_tensor_count": len(tensors),
        "aggregate_sha256": aggregate.hexdigest(),
        "tensors": tensors,
    }


def set_matched_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def mask_frames_to_tensor(frames: list[object]) -> torch.Tensor:
    values = []
    for frame in frames:
        array = np.asarray(frame.convert("L"), dtype=np.uint8)
        values.append(array > 127)
    stacked = np.stack(values, axis=0).astype(np.float32)
    return torch.from_numpy(stacked)[None, None, ...]


def pool_topology_masks(
    masks: dict[str, torch.Tensor], output_shape: tuple[int, int, int]
) -> dict[str, torch.Tensor]:
    return {
        key: functional.adaptive_max_pool3d(value.float(), output_shape)
        for key, value in masks.items()
    }


class MatchedWanTrainingModule(WanTrainingModule):
    def __init__(
        self,
        *args,
        loss_mode: str,
        training_seed: int,
        randomness_trace_path: str,
        topology_weights: tuple[float, float, float],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_mode not in LOSS_MODES:
            raise ValueError(f"Unsupported loss mode: {loss_mode}")
        self.loss_mode = loss_mode
        self.training_seed = int(training_seed)
        self.randomness_trace_path = Path(randomness_trace_path)
        self.topology_weights = topology_weights
        self.forward_step = 0

    def deterministic_flowmatch_loss(
        self,
        inputs: dict[str, object],
        topology_masks: dict[str, torch.Tensor],
        trace_context: dict[str, object],
    ) -> torch.Tensor:
        pipe = self.pipe
        max_boundary = int(
            inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
        )
        min_boundary = int(
            inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
        )
        if max_boundary <= min_boundary:
            raise ValueError("Empty frozen timestep interval")
        step_index = self.forward_step
        timestep_seed = self.training_seed + step_index * 2
        noise_seed = timestep_seed + 1
        timestep_generator = torch.Generator(device="cpu")
        timestep_generator.manual_seed(timestep_seed)
        timestep_id = int(
            torch.randint(
                min_boundary,
                max_boundary,
                (1,),
                generator=timestep_generator,
            ).item()
        )
        timestep = pipe.scheduler.timesteps[timestep_id].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        input_latents = inputs["input_latents"]
        noise_generator = torch.Generator(device=input_latents.device)
        noise_generator.manual_seed(noise_seed)
        noise = torch.randn(
            input_latents.shape,
            dtype=input_latents.dtype,
            device=input_latents.device,
            generator=noise_generator,
        ) * inputs.get("noise_scale", 1.0)
        noise_digest = tensor_sha256(noise)
        inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
        training_target = pipe.scheduler.training_target(input_latents, noise, timestep)
        has_conditioned_first_frame = "first_frame_latents" in inputs
        if has_conditioned_first_frame:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
        pooled: dict[str, torch.Tensor] = {}
        if self.loss_mode == "topology_weighted":
            if set(topology_masks) != set(MASK_KEYS):
                raise ValueError("Topology-weighted loss requires all three frozen masks")
            pooled = pool_topology_masks(
                topology_masks,
                tuple(int(value) for value in noise_prediction.shape[-3:]),
            )

        if has_conditioned_first_frame:
            noise_prediction = noise_prediction[:, :, 1:]
            training_target = training_target[:, :, 1:]
            pooled = {key: value[:, :, 1:] for key, value in pooled.items()}

        squared_error = (
            noise_prediction.float() - training_target.float()
        ).square()
        raw_weight_mean = 1.0
        normalized_weight_mean = 1.0
        if self.loss_mode == "uniform":
            loss = squared_error.mean()
        else:
            strand_weight, contact_weight, connection_weight = self.topology_weights
            weight = (
                1.0
                + strand_weight * pooled[MASK_KEYS[0]]
                + contact_weight * pooled[MASK_KEYS[1]]
                + connection_weight * pooled[MASK_KEYS[2]]
            )
            raw_weight_mean = float(weight.mean().detach().cpu())
            weight = weight / weight.mean().clamp_min(1e-12)
            normalized_weight_mean = float(weight.mean().detach().cpu())
            expanded_weight = weight.expand_as(squared_error)
            loss = (expanded_weight * squared_error).sum() / expanded_weight.sum()
        loss = loss * pipe.scheduler.training_weight(timestep)

        trace_record = {
            "optimizer_step": step_index + 1,
            "training_row_id": int(trace_context["training_row_id"]),
            "training_sample_id": str(trace_context["training_sample_id"]),
            "loss_mode": self.loss_mode,
            "timestep_seed": timestep_seed,
            "noise_seed": noise_seed,
            "timestep_id": timestep_id,
            "timestep": float(timestep.detach().float().cpu().reshape(-1)[0]),
            "noise_sha256": noise_digest,
            "input_latent_shape": list(input_latents.shape),
            "prediction_shape_after_first_frame_policy": list(noise_prediction.shape),
            "conditioned_first_frame_excluded": has_conditioned_first_frame,
            "raw_topology_weight_mean": raw_weight_mean,
            "normalized_topology_weight_mean": normalized_weight_mean,
        }
        self.randomness_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.randomness_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace_record, sort_keys=True) + "\n")
        self.forward_step += 1
        return loss

    def forward(self, data, inputs=None):
        if inputs is not None:
            raise ValueError("Cached-input mode is outside the frozen Day 13 protocol")
        row = dict(data)
        trace_context = {
            "training_sample_id": row.pop("training_sample_id"),
            "training_row_id": row.pop("training_row_id"),
        }
        topology_masks: dict[str, torch.Tensor] = {}
        for key in MASK_KEYS:
            if key in row:
                topology_masks[key] = mask_frames_to_tensor(row.pop(key))
        if self.loss_mode == "uniform" and topology_masks:
            raise ValueError("Uniform arm must not receive topology masks")
        pipeline_inputs = self.get_pipeline_inputs(row)
        pipeline_inputs = self.transfer_data_to_device(
            pipeline_inputs, self.pipe.device, self.pipe.torch_dtype
        )
        topology_masks = {
            key: value.to(device=self.pipe.device, dtype=torch.float32)
            for key, value in topology_masks.items()
        }
        for unit in self.pipe.units:
            pipeline_inputs = self.pipe.unit_runner(unit, self.pipe, *pipeline_inputs)
        inputs_shared, inputs_posi, _ = pipeline_inputs
        loss_inputs = {**inputs_shared, **inputs_posi}
        return self.deterministic_flowmatch_loss(
            loss_inputs, topology_masks, trace_context
        )


def launch_matched_training(
    accelerator: accelerate.Accelerator,
    dataset: UnifiedDataset,
    model: MatchedWanTrainingModule,
    model_logger: ModelLogger,
    args,
) -> None:
    if args.dataset_num_workers != 0:
        raise ValueError("Frozen Day 13 protocol requires dataset_num_workers=0")
    if accelerator.num_processes != 1:
        raise ValueError("Frozen Day 13 protocol requires exactly one process")
    if accelerator.is_main_process:
        save_training_args(args)
        initialization = trainable_state_manifest(model)
        initialization.update(
            {
                "training_seed": args.training_seed,
                "loss_mode": args.loss_mode,
            }
        )
        initialization_path = Path(args.initialization_manifest_path)
        initialization_path.parent.mkdir(parents=True, exist_ok=True)
        initialization_path.write_text(
            json.dumps(initialization, indent=2) + "\n", encoding="utf-8"
        )

    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(args.training_seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
        num_workers=0,
        generator=shuffle_generator,
    )
    if args.enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(
            optimizer, dataloader, scheduler
        )
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(
            model,
            accelerator.device,
            args.enable_optimizer_cpu_offload,
            args.cpu_offload_split_threshold,
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        offload_manager = None

    initialize_deepspeed_gradient_checkpointing(accelerator)
    for _ in range(args.num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if offload_manager is not None:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(
                    accelerator, model, args.save_steps, loss=loss
                )
    model_logger.on_training_end(accelerator, model, args.save_steps)


def main() -> None:
    parser = wan_parser()
    parser.add_argument("--loss_mode", choices=LOSS_MODES, required=True)
    parser.add_argument("--training_seed", type=int, required=True)
    parser.add_argument("--randomness_trace_path", required=True)
    parser.add_argument("--initialization_manifest_path", required=True)
    parser.add_argument("--strand_weight", type=float, default=3.0)
    parser.add_argument("--contact_weight", type=float, default=7.0)
    parser.add_argument("--source_connection_weight", type=float, default=5.0)
    args = parser.parse_args()
    set_matched_seed(args.training_seed)
    trace_path = Path(args.randomness_trace_path)
    initialization_path = Path(args.initialization_manifest_path)
    if trace_path.exists() or initialization_path.exists():
        raise FileExistsError("Refusing to overwrite Day 13 reproducibility evidence")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path)
            >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )
    model = MatchedWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device=(
            "cpu"
            if args.initialize_model_on_cpu or args.enable_model_cpu_offload
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        loss_mode=args.loss_mode,
        training_seed=args.training_seed,
        randomness_trace_path=args.randomness_trace_path,
        topology_weights=(
            args.strand_weight,
            args.contact_weight,
            args.source_connection_weight,
        ),
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launch_matched_training(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    main()
