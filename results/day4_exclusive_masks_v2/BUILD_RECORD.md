# Exclusive projection-mask build record

The ownership rule and inference controls were frozen at commit
`f1170db8481cd4bcb56067816c52c4ff77a7d204` before any v2 stochastic output.

The first deterministic remote build at
`/tmp/foodstateedit_day4_exclusive_masks_v1` stopped while saving the first PNG
because `save_mask` lacked a local NumPy import. No model was loaded and no
stochastic output was generated. That failed root and its launcher directory
remain untouched.

The import-only fix was committed as
`b996c44afa40d5a2ed84235e7debea8bf2f12e1c`. The successful build used:

- launcher root: `/tmp/foodstateedit_day4_exclusive_launchers_v2`;
- output root: `/tmp/foodstateedit_day4_exclusive_masks_v2`;
- source proxy root:
  `/host/space0/guo-z/tf-ufi/outputs/paper_sprint_day4_20260827_common_proxy_v1`;
- builder SHA-256:
  `8375fe85efc19271146adcadd24c28abb992a877b53f03958d73cb4071aca58c`;
- resident worker v2 SHA-256:
  `b59f4abb30f55d29a9f180b195b3412940eebe4e65e403eac109ab99a5a2fede`;
- shared anchor helper SHA-256:
  `4fcc4baca473dd5b7a3b116af7536cdfbd0f0472b0ca2e3c53dfd00f50e4ccf2`.

All four source-proxy manifest hashes were verified. For every case, the four
output masks are nonempty and pairwise disjoint, and their union is exactly the
original four-mask union. The PNGs and remote manifests are copied in this
directory so inference inputs remain recoverable if `/tmp` is cleared.
