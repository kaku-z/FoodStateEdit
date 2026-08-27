# Quota-aborted launch record

At approximately 2026-08-27 12:30 JST, the first Day 4 launch attempted to use
`/host/space0/guo-z/tf-ufi/outputs/paper_sprint_day4_20260827_staged_v1_gpu0`.
The second worker's log redirection failed with `Disk quota exceeded`. GPU 0 had
started model initialization but had not produced a stochastic image or video.

The GPU 0 process (PID 1620440) was terminated deliberately to avoid an
inevitable output-write failure. The partial command, running manifest, and
`RUNNING` timestamp are retained here; the remote partial directory was not
deleted or reused.

The formal pilot used two previously absent `/tmp` roots, the same code commit
`eb0c0b01174f3d873db295f1168c5bfc14e45489`, the same seed `1`, and the same
frozen schedule. It completed all four cases. This record is an infrastructure
preflight incident and is not an additional candidate or seed.
