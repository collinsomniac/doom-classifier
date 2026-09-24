# doom-classifier

Browser-native research harness for ultra-low-latency semantic decision policies.

The project asks a narrow question:

> How far can a structured-state -> typed-decision controller go if semantic priors, temporal memory, online experience, calibration, and optional escalation are kept modular and cheap?

DOOM is the first target environment, not the architecture.

## Current experiment

The first runnable lab is deliberately dependency-free and designed for GitHub Pages. It includes:

- a generic `EnvironmentAdapter` boundary;
- a deterministic synthetic arena for repeatable experiments;
- a policy interface that returns a distribution over typed actions;
- recurrent temporal state;
- a tiny online residual Q learner;
- entropy / margin / novelty uncertainty signals;
- a policy-blind orchestration state machine;
- latency and reward instrumentation;
- reset / learn / memory toggles;
- hooks for a future Laya / ONNX / Transformers.js semantic backbone;
- hooks for a future DOOM WASM adapter.

The default semantic adapter is **not a pretrained model**. It is an intentionally weak hashed-text structural baseline so the complete control loop can be measured before model downloads or WebGPU are involved.

## Run locally

Serve the repository over HTTP (ES modules do not reliably run from `file://`):

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/>.

## Research invariants

1. The state machine may understand program lifecycle, never strategy.
2. Environment adapters expose observations and mechanics, never policy.
3. The fast path remains one bounded decision pass.
4. Online adaptation is separable, tiny, and can be disabled.
5. Teacher/escalation performance is reported separately from teacher-off evaluation.
6. DOOM-specific signals must not leak into the core policy API.
7. Every component should be ablatable.

## Planned sequence

1. Validate the browser lab and latency instrumentation.
2. Add replay/export and deterministic benchmark traces.
3. Add a real semantic backbone adapter and compare it with the structural baseline.
4. Add a DOOM WASM environment adapter and telemetry bridge.
5. Add uncertainty-triggered teacher collection.
6. Add counterfactual rollout / consequence-learning experiments.
7. Benchmark transfer across renamed, shuffled, masked, and unseen schemas.

See `docs/architecture.md` for the working design.
