# Project-owned DOOM engine build

The live lab currently boots a known-good browser runtime from a pinned
`lukaske/jev-doom-agent` commit. This directory is the migration path to an
owned telemetry ABI built reproducibly by `doom-classifier`.

## Why this exists

Post-hoc hostile HP deltas and vanilla single-player kill counts are not safe
combat rewards: monster infighting can change both without the player causing
the event. The policy should learn from causal consequences, not convenient
correlations.

The owned ABI therefore records factual cumulative counters at the native
engine hook where attribution is already known:

- player-attributed damage to `MF_COUNTKILL` hostiles;
- player-attributed hostile kills;
- successful player pickups;
- level completions;
- secret exits.

No tactical behavior is added. The engine still receives only literal player
controls and exposes factual state/events.

## Reproducibility

`SOURCE.lock` pins the upstream repository, commit, Chocolate Doom version,
Emscripten version, and default IWAD lineage.

`scripts/instrument_engine.py` uses exact source anchors and fails closed if
the pinned source no longer matches. `scripts/build_owned_wasm.sh` fetches
that exact source, applies the instrumentation, invokes its Emscripten build,
and writes the generated browser runtime under `engine/dist/`.

The GitHub Actions workflow uploads the generated engine as a build artifact.
The live site must not switch to the owned bundle until the artifact passes
browser ABI tests.

## License

Chocolate Doom is GPL-2.0-or-later. Source provenance and the instrumentation
needed to reproduce the generated engine are kept alongside this build path.
Freedoom has its own included license in the pinned upstream repository.
