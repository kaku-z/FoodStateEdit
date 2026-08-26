# Environment snapshot

Snapshot date: 2026-08-26

## Local orchestration machine

- OS: Microsoft Windows NT 10.0.26200.0
- Installed system Python: 3.13.3 at
  `C:\Users\kaku\AppData\Local\Programs\Python\Python313\python.exe`
- Day 1 validation interpreter: Python 3.11.9 at
  `C:\Users\kaku\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`
- Git: 2.53.0.windows.2
- Historical workspace: local files under the experiment root

## Remote inference host

- SSH alias: `gp40`
- Python: 3.12.12 (Anaconda build)
- PyTorch: 2.4.0+cu121
- NumPy: 1.26.4
- Pillow: 12.0.0
- OpenCV: 4.12.0
- Transformers: 4.45.0
- GPUs: 8 × NVIDIA RTX A6000, 49,140 MiB reported per device
- NVIDIA driver: 580.119.02
- System `nvcc`: CUDA 11.2; PyTorch runtime reports CUDA 12.1
- GeoEdit Python:
  `/host/space0/guo-z/envs/geoedit/bin/python`

The Hugging Face library emits a warning for the default user cache. Formal
launchers override `HF_HOME` and `TRANSFORMERS_CACHE` with the existing writable
model cache and set `DIFFSYNTH_SKIP_DOWNLOAD=True`.

## Reproduction rules

- never rely on the user's default Python;
- use explicit interpreter and model paths;
- record `CUDA_VISIBLE_DEVICES`, seed, step count, frame count, schedule, and hashes;
- do not overwrite completed outputs;
- distinguish technical reruns from visual-result selection;
- freeze a complete package lock after the Day 3 baseline audit.
