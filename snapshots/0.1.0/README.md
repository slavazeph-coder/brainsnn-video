# BrainSNN Video

A separate local workspace for video memory research, with a small package and
CPU commands. This snapshot separates the video experiment from CRUMB's
handoff-format/parser project. The existing CRUMB checkout, source paths and
full upstream Git history remain available. This repository has no remote and
has not been published.

## What the evidence supports

| Layer | Current evidence | Limit |
|---|---|---|
| Observed appearance reference and RGB latch | Registered synthetic 256-frame cue retention passes on three fixture seeds, with actual erase and opposite-donor state interventions | A hand-built observed-reference control; no learned memory or natural-video claim |
| Native wave, generic SSM and bounded attention adapters | Real native classes, causal frame API, actual finite backward/update and prediction replay pass CPU contract checks | A one-update CPU smoke has joint recovery zero for all three learned arms; not a recall-training result |
| Actual-video latent rollout training | Historical 512-step K2 and six 4,096-step K0/K1 GPU runs completed and were rejected against frozen persistence | Two source names, limited independent windows, unequal training cost and no decoded quality win |
| FFT and streaming kernels | CPU operation scaling and bounded recurrent state are implemented and tested | Not token savings, end-to-end GPU performance or coherent generated video |
| Product | Native-clip evaluation requirements are specified | No paid clip rescue, independent human acceptance or product-quality validation here |

The synthetic fixture observes raw frames 0–287, shows its last cue at frame31,
hides it during `[32,288)`, and scores a target at frame288 that inference never
receives. Camera paths, target pixels and scorer masks stay outside inference.
Metrics separate full RGB reconstruction, known/unknown regions and the actual
intact-minus-own-reference memory contribution. Privileged oracle outputs are
explicit controls. Rejected/ambiguous and unobservable cases remain reported.

The RGB latch can also retain a small transient occluder. The registered
small-occluder control has false presence in six of six cases; object-versus-cover
semantics cannot always be recovered from RGB observations alone. Sparse near-zero
FFT outputs can amplify roundoff through RMSNorm, so the additive wave adapter
uses the actual causal frame API for public prefix outputs and restricts chunk
execution to one layer. These limits are retained in the evidence files.

## Run locally

The package root imports and `info` command do not load torch or download models.
The verified existing interpreter is
`/Users/slavaz/crumb-format/.venv-standalone/bin/python`.

```sh
cd /Users/slavaz/Documents/Codex/2026-09-08/go/brainsnn-video
python3 -B -m brainsnn_video info
python3 -B verify_snapshot.py
/Users/slavaz/crumb-format/.venv-standalone/bin/python -B -m brainsnn_video test --native
```

To use a Python environment that already has its appropriate torch build,
install this package with `python -m pip install -e . --no-deps`. The `test`
extra supplies pytest and image-writing dependencies for the native suite;
`video` supplies optional frozen-VAE/media dependencies. We did not install
dependencies or fetch pretrained weights during separation.

```sh
# Each runner requires a fresh output directory. These commands are CPU-only.
python -m brainsnn_video benchmark --out runs/reference
python -m brainsnn_video train-smoke --out runs/training-smoke
python -m brainsnn_video native long_horizon memory --pole-param softplus halflife
```

```python
from brainsnn_video import ObservationMemory, ObservationConfig, VideoPredictor
from brainsnn_video.engine.observation_memory_v2.protocol import make_case, score

case = make_case(0, "clean", "chromatic")
memory = ObservationMemory(ObservationConfig())
for frame in case["frames"]:
    memory.step(frame)
prediction = memory.forecast()["rgb"]
metrics = score(prediction, case)  # privileged scorer truth, never frontend input
```

## Snapshot and provenance

- `MIGRATION_MANIFEST.json` records source and destination SHA256s, source HEAD,
  original relative paths, byte counts and receipt transformations.
- `brainsnn_video/engine/` preserves the native code and test dependency closure.
  `observation_memory_v2/` is the frozen new reference/adapter/protocol layer.
- `docs/history/` preserves the earlier technical analyses and a bounded upstream
  commit index. These documents are historical and may describe superseded plans.
- `evidence/registered_reference/` contains finite registered CPU aggregates,
  protocol/verdict and independent saved-metric arithmetic replay.
- `evidence/native_numerics.json`, `native_training_smoke_summary.json` and
  `historical_rejections.json` retain failure evidence and its scope.
- `verification/` records this checkout's import, dependency, test and package
  checks. `SHA256SUMS` binds the delivered source/docs/evidence snapshot.

Weights, raw tensor predictions, large metric dumps, media, credentials, private
letters, transcripts and operational queue/controller infrastructure are excluded.
The original larger artifacts remain at their incumbent locations. The copied
GPU plan is staged documentation; no GPU run, controller change, upload or model
promotion accompanied this separation. Native video decoding and human review
remain separate gates before any quality claim.
