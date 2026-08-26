# Dataset provenance

## UECFOOD256

- Dataset: UEC FOOD 256, release 1.1 on the laboratory filesystem.
- Canonical project page: <https://foodcam.mobi/dataset256.html>
- Use restriction verified on 2026-08-26: non-commercial research only.
- Redistribution policy for this repository: do not commit or redistribute the
  source images; publish case identifiers, category-relative paths, hashes, and
  evaluation metadata only.
- Resolved remote root used by the inventory:
  `/host/data/dataset/UECFOOD/UECFOOD256`
- Accessible compatibility symlink:
  `/host/space0/ma-y/SR/UECFOOD/UECFOOD256`
- Stale path found in the historical manifest and not used:
  `/host/space0/ma-y/SR/UECFOOD256`
- Dataset `README.txt` SHA-256:
  `7e3c400d1e10416d12601d54d131c6fb7b061c74caf947619fa63f98900cd0b1`
- Dataset `category.txt` SHA-256:
  `3df3a9e834bf87112bd752d4308367954db1b886de4992c6570474051517a2e8`

Required citation:

```bibtex
@inproceedings{kawano2014automatic,
  title={Automatic Expansion of a Food Image Dataset Leveraging Existing Categories with Domain Adaptation},
  author={Kawano, Yoshiyuki and Yanai, Keiji},
  booktitle={ECCV Workshop on Transferring and Adapting Source Knowledge in Computer Vision},
  year={2014}
}
```

The dataset page, rather than the local README alone, is the authority for the
non-commercial research restriction.

## Canonical editing-input layer

The frozen identity/provenance layer remains the 60 original UECFOOD256 images
in `data_manifest_v1.csv`. Ten originals have a minimum side below 256 pixels,
so formal editing does not consume the originals directly. Every method instead
receives the same existing precomputed OSEDiff input recorded in
`canonical_input_manifest_v1.csv`.

- Existing remote root: `/host/space0/ma-y/SR/UECFOOD256x4_osediff`
- Method: OSEDiff x4 real-world image super-resolution; no model was downloaded,
  retrained, or rerun for this freeze.
- Dimension rule observed and validated for all 60 cases:
  `floor((source_dimension * 4) / 8) * 8`.
- Minimum canonical dimensions: 688 x 600.
- Traceability: all 60 original hashes match the frozen manifest and all 60
  canonical hashes are unique.
- Coarse structural audit: source/canonical dHash distance median 1, maximum 5,
  below the predeclared acceptance limit of 8.
- Pixel audit against bicubic x4: mean RGB MAE 7.5211 on the 0--255 scale. This
  is descriptive only and is not treated as a quality or truth metric.

OSEDiff is a generative real-image super-resolution method and can synthesize
details. Its output is therefore not high-resolution ground truth. Locking the
same exact input for every method removes an input-resolution confound, while
the original image and hash remain the authority for case identity. Source and
canonical images remain outside this repository.

References: [OSEDiff NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf),
[official implementation](https://github.com/cswry/OSEDiff).
