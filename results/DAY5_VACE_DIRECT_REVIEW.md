# Day 5 VACE-direct static-proxy review

Review date: 2026-08-29 (Asia/Tokyo)

## Outcome

The frozen VACE-direct pilot completed all four anchors at seed `1`. It reused
the same source, common proxy, prompts, Wan2.2-VACE model, 21 frames, 20 steps,
and VACE scale `1.0`, but disabled GeoEdit TTM and all projection schedules.
The static `motion_signal.png` and `edit_alpha.png` were passed directly to
native VACE for every frame.

Technical completion is `4/4`, but editing success is not. The single internal,
non-blind diagnostic review gives `0/4` action successes and `0/4` photo
successes. Exact outside-mask preservation remains `4/4`.

| Anchor | Action | Photo | Exact protection | Main diagnostic |
| --- | --- | --- | --- | --- |
| Soup + spoon | fail | fail | pass | Translucent outline; no convincing liquid-bearing spoon |
| Fried rice + spatula | fail | fail | pass | Spatula disappears into a blurred food patch |
| Udon + chopsticks | fail | fail | pass | Ghosted sticks/strand; not two solid sticks with one continuous noodle |
| Pasta + fork | fail | fail | pass | Four tines and twirl relation are absent |

## Runtime and reproducibility

Both two-case workers loaded the audited pipeline once. Shared-filesystem cold
loading made the first cases take 1,399.496 and 1,408.383 seconds; resident
second cases took 247.412 and 246.731 seconds. All twelve generated artifacts
(MP4, raw last frame, and protected 2D image for four cases) matched their run
manifest hashes after transfer. Every pixel outside `edit_alpha` has maximum
channel difference `0`. There were no downloads, inference failures, output
overwrites, or seed replacements.

## Scientific conclusion

Disabling TTM does not recover utensil identity or food-contact topology. The
failure therefore cannot be attributed to the GeoEdit projection schedule
alone. Native VACE conditioning and its latent/video prior also fail to retain
thin exact-count structures and deformable-food relations from the current
static 2D proxy.

Close this candidate and do not expand it. The evidence now supports a method
direction with explicit high-resolution topology and z-order constraints,
while a learned model is restricted to local material/texture synthesis rather
than allowed to reinterpret utensil count, silhouettes, or connectivity.
