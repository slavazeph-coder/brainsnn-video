# Local separation record

Source project: CRUMB format, native video experiment at
`experiments/wavefield_video`, HEAD `8c1a63bc0e69af10a3e678714e719839f12e97ba`.
The frozen observation experiment was untracked in the incumbent checkout when
copied. Every migrated Python source is byte-identical and independently hashed;
the wrapper/package files are new local code.

The standalone checkout keeps a complete dependency closure for its included
observation adapter/protocol/train tests and selected native stateful, half-life,
resume, health-monitor and signed-rollout tests. Historical legacy checkpoint
reconstruction includes the small `v1_backup/wfvideo.py` model source. The local
runner is retained for its failure/status contract; operational queues are not.

Migration is additive: source files and historical artifact paths were not moved
or deleted. No upstream branch was reset or committed, and no public repository,
remote or push was created. Full upstream Git objects are not duplicated; the
original repository retains the full history and the bounded index identifies
the snapshot's relevant recent commits.

Finite receipts were copied or summarized explicitly. The numerics receipt's
absolute source paths were replaced with filenames. The one-update training
receipt's per-example rows and binary payloads were omitted while aggregate
metrics, resource counts, gate failures and original binary hashes were retained.
These reductions are listed in the migration manifest. Historical actual-video
rejections have a compact numeric record with original report hashes; this
checkout did not rerun those GPU experiments.

Private correspondence, Hermes conversations, keys, large weights and media are
not part of the project. Current independent validation utilities being developed
by another agent were excluded pending their declared source freeze; this cannot
be read as evidence that those further checks have passed.
