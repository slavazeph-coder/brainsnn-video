# BrainSNN Video

Video postprocessing and memory research separated from CRUMB's handoff-format
project. Version 0.1.1 is a research snapshot with known failures. The incumbent
CRUMB sources and history remain available. Source and provenance are packaged
separately; verify repository links before citing publication.
The original 0.1.0 snapshot is preserved byte-for-byte in `snapshots/0.1.0/`.

## Evidence and limits

| Layer | Evidence | Limit |
|---|---|---|
| Analytic ENG-1 clip postprocessor | Actual unmodified spectral engine; two synthetic 1280×720 CPU frames process with finite outputs and unchanged inputs | The owned 121-frame 720p clip **fails** motion preservation: ratio 0.5163 before encoding; both unlabeled encoded ratio orders fail [0.9, 1.1]. Human review is pending; no stranger clips or paid orders |
| Observed RGB reference/latch | Registered 256-frame synthetic cue retention passes on three fixture seeds, including actual erase and opposite-donor state interventions | Hand-built reference control; no learned-memory or natural-video claim. Small transient occluders can be retained as false events |
| Native wave, generic SSM and bounded attention adapters | Actual finite backward/update and prediction replay pass CPU contract checks | One-update smoke has joint recovery zero for all three learned arms; this is executable training, not trained recall |
| Cached pretrained LTX encoder | Official pinned existing weights, complete state keys, 297 raw frames at 32×32, prefix cuts 33/129/289 and nine future-poison/context comparisons pass; maximum past delta 1.37e-6 | CPU encoder contract only. Decoder/RF, frozen DINO, complete preflight, CUDA, native-video quality and three-seed learned comparison remain unrun |
| Historical actual-video latent training | 512-step K2 and six 4,096-step K0/K1 GPU runs completed | Rejected against frozen persistence; limited source windows, unequal training cost and no decoded quality win |
| FFT/streaming kernels | CPU operation scaling and bounded recurrent state are implemented | No token-saving, end-to-end GPU or generated-video coherence claim |

The inherited analytic suite has **28 passes and three failures**, reproduced
with byte-identical canonical source/tests and the same CPU interpreter:
`test_smoothstep_bounds_and_monotone` compares 0.5000000000000001 to exact 0.5;
`test_gate_preserves_coherent_translation` and `test_gate_still_corrects_wander`
fail optional motion-gate expectations. These tests remain enabled. The frozen
ENG-1 shipping configuration is unchanged and uses the default `mc_gate=False`.
This is not a product-quality release.

## Run

Package import and `info` do not load torch or download models. Use an existing
Python environment with torch and numpy; install this source with
`python -m pip install -e . --no-deps` if needed. Tests require pytest and the
optional image dependencies. Native assessment also requires existing `ffmpeg`
and `ffprobe` on PATH. The `video` extra supplies optional codec/feature packages;
model weights are never bundled or automatically fetched by the validator.

```sh
python -B -m brainsnn_video info
python -B verify_snapshot.py
python -B -m brainsnn_video test --native --mary
# Preserves the three known analytic failures and exits nonzero:
python -B -m brainsnn_video test --analytic

# Fresh output directories are required; these commands are CPU-only.
python -B -m brainsnn_video benchmark --out runs/reference
python -B -m brainsnn_video train-smoke --out runs/training-smoke
python -B -m brainsnn_video preflight-smoke --out runs/validator-fixture
python -B -m brainsnn_video assess-native --input owned-720p.mp4 \
  --out runs/native-clip --source-role owned-generated
```

The assessor retains native pixels and full frame cadence, identical A/B
encoding, a private random role key and the frozen numerical gates. It refuses
unsupported resolutions or silent retiming. Permission to process a client clip
does not establish an independent stranger or a human-quality result. Do not
share the private role key with the reviewer. A failed native gate blocks a
product-success claim.

```python
from brainsnn_video import SpectralCoherenceEngine
import torch

engine = SpectralCoherenceEngine(alpha=.95, rho=.995, cutoff_frac=.14,
                               anchor_mode="complex_mc", mc_strength=.5,
                               mc_band=.03, phase_anchor=0.)
frame = torch.rand(3, 720, 1280)
corrected, state = engine.process_frame(frame, engine.init_state(720, 1280))
```

```python
from brainsnn_video import ObservationMemory, ObservationConfig
from brainsnn_video.engine.observation_memory_v2.protocol import make_case, score

case = make_case(0, "clean", "chromatic")
memory = ObservationMemory(ObservationConfig())
for frame in case["frames"]:
    memory.step(frame)
metrics = score(memory.forecast()["rgb"], case)
```

The RGB fixture observes frames 0–287, last cue 31, hidden interval `[32,288)`,
and an unobserved scoring target at 288. Targets, paths and scorer masks stay
outside inference; intact-minus-own-reference metrics isolate the memory
contribution from shared background errors. Unobservable and ambiguous cases
remain reported. The separate causal-codec protocol observes 33 context plus
256 hidden raw frames (289 total) and targets the next unseen VAE stride block.
It checks fair completed-run rows but does not implement a learned training
campaign. Production preflight requires independently pinned existing local
VAE and DINO snapshots; the fixture smoke cannot certify pretrained behavior.

## Provenance

`MIGRATION_MANIFEST.json` records source/destination hashes and every explicit
transformation. `brainsnn_video/engine/` preserves the numerical model, analytic
engine, validators and their tests. The portable assessor changes only the
incumbent engine-path literal; its source hash is distinguished from the
historical driver actually used for the owned clip. `docs/history/` contains
historical analyses, which may describe superseded plans. `evidence/` retains
finite CPU receipts, the cached LTX encoder result and rejected results.
`verification/` and `SHA256SUMS` bind the delivered package and known failures.

The LTX receipt records 11.017 seconds of worker time and a **sampled** peak
worker RSS of 2.66 GiB, observed by outer `ps` polls every 0.25 seconds. This is
not a continuous memory maximum or GPU evidence.

Credentials, private correspondence/transcripts, blind keys, weights, media and
queue/controller infrastructure are excluded. No source files were moved or
deleted. This separation did not start a GPU job, download weights or tune the
frozen engine. Independent native-pixel review and paid fulfillment remain
unverified.
