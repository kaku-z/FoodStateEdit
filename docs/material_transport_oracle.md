# Complementary material transport: executable stage 0

This is the geometry oracle stage of the 2026-09-27 redesign, not a trained
single-image food editor. Its purpose is to establish an auditable representation
before paying for reconstruction or diffusion training.

Each occupied unit cube owns one persistent material ID. A source-only pre-cut
label partitions IDs into moving and remaining sets. The moving set is removed
and inserted once at an integer translation. Remaining geometry never changes.
The ray renderer finds the first occupied cube, then queries the same canonical
material field for source and edited images, including newly exposed faces.

The source fixture is synthetic layered cake with exact geometry. No source or
target photograph, manual RGB proxy, diffusion refinement or target-conditioned
source selection is used. The background is an analytic plane.

## Run

Python >=3.11 with NumPy and Pillow; no Torch or GPU is needed at this stage.

```sh
python -m unittest discover -s tests -p test_material_transfer_oracle.py
python scripts/run_material_transport_oracle.py --output outputs/material_oracle_NEW --cases 16
python scripts/deploy_material_transport_oracle.py --host gp40 --bundle-dir outputs/server_bundle_NEW
```

Use a fresh output directory each time. `--package-only` creates the deployment
archive without connecting. Remote defaults come from historical project
configuration and must pass a fresh SSH/runtime preflight before writing files.
The deployment script verifies archive and file hashes, runs tests, executes the
CPU oracle in a fresh remote directory, and retrieves results. It does not install
packages or occupy GPUs. Authentication failure stops before remote writes.

If im00 has password access but a broken home directory, a user can keep a local
terminal running `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:22240:gp40:22 im00`.
Then add `--local-tunnel-port 22240` to deployment. This preserves gp40 host-key
checking and requires gp40 to accept the existing local public key. No password
is stored in scripts, and no SSH configuration file is changed.

## Measurements

`report.json` records host, runtime, code hashes, per-case source and edited
volume, duplicate/missing IDs, source residual occupancy, destination material
correspondence, and shuffled-material visible RGB error. NPZ files retain grids,
canonical hit coordinates, IDs and ray depths; PNGs show source, transport and
shuffled-material renders. Counted cube volume belongs to this representation,
not to measured real mass.

Copy and shuffle are deliberately broken negative controls. Copy must be caught
by duplicate/residual/volume metrics; shuffle preserves volume and must be
caught by correspondence. They are not competitive method baselines, and passing
these invariants is expected by construction rather than evidence of novelty.

## Limits and next gates

Supported: integer translation, fixed orthographic camera, hard canonical IDs,
exact complementary cube partition, first-hit geometry visibility.

Not implemented: source-image geometry/partition estimation, posterior inference,
continuous rigid rotation, certified continuous collision, spoon geometry/contact,
shadows, learned appearance, real multiview training, diffusion comparison.
Rounded lattice path collision checks are discrete checks only.

Before training, inspect oracle images, add finite-thickness tool and plate
collision/contact with reachable/unreachable tests, then identify actual
calibrated multiview episodes on the server. Old single-case pseudo-target data
cannot establish learned single-image 3D material transfer. Real-image claims
require held-out evaluation against matched-data baselines.
