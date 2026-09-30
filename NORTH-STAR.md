# North star: a typed-choice decision model

This is the thesis for the successor project. It states what we are building,
the decisions we have already committed to, and what would make our own design
obsolete — so the bet is falsifiable rather than aspirational.

---

## 1. The interface (fixed)

```text
decide(
  information,          # observed state: text and/or typed fields
  choices,              # typed candidate objects, described by a schema
  teacher?              # optional natural-language guidance: training only
) -> [
  { choice, arguments, confidence, provenance }
]
```

Properties that must hold:

- **Choices are inputs, not a vocabulary.** Nothing in the weights corresponds
  to a permanent output neuron named `turn_left`. Reordering, renaming, adding
  or removing candidates is an input change, not a retraining event.
- **Runtime never requires the teacher.** Teacher text is training evidence.
  If the production path can read the rationale, training has leaked the answer.
- **Outputs are constructed, not merely selected.** An option may be a template
  with parameters: enum values, references to supplied objects, bounded numbers,
  dependent fields. A deterministic serializer turns the prediction into a
  structurally valid object.
- **Validity is structural, not tactical.** The schema may forbid a field value
  that cannot exist. It must never encode "fire only when aligned" — that is
  the behavior the model must learn.
- **Confidence is a measured quantity**, revalidated after any weight change or
  explicitly marked stale.
- **Abstention is a first-class outcome.**

## 2. Architecture

Four responsibilities. One shared representation. **Exactly one runtime
scoring path.**

```
information ─┐
choices ─────┼─→ [ typed tokenizer + shared encoder ] ─┬─→ A. choice scoring   (fast path)
schema ──────┘                                          ├─→ B. argument construction
                                                        ├─→ C. consequence prediction (training/aux)
teacher ────────(training only)──────────────────────────┴─→ D. rationale alignment (training only)
```

| Part | Responsibility | Runs at inference |
| --- | --- | --- |
| Encoder | context, entities, candidates as tokens | yes |
| A. Scoring | one score per candidate, candidate-conditioned attention | yes |
| B. Construction | field values, references, dependent fields | yes |
| C. Consequence | action-conditioned future measurements at several horizons | no (aux) |
| D. Rationale | align student representation with teacher text | no (training only) |

Removed relative to the DOOM experiment: the semantic prior, the KL-budgeted
prior/value arbitration, the ensemble critic gates, the typed value projection,
and the linear control head. Each had a story; none earned its place.

### Why candidate-conditioned scoring
Learning `(context, candidate) -> score` means the same weights handle options
the model has never seen, because the candidate is an input. This is what makes
the Jev/Laya capability reachable without a fixed 15-action vocabulary.

### Why a separate consequence branch
The DOOM experiment's central failure was credit assignment, not capacity: a
4-tic label window made firing invisible, and a fixed 24-tic window gave
orientation no credit. Predicting action-conditioned future measurements at
multiple horizons attacks that directly and is **measurable in isolation**.

Causal honesty is required here. Predicting that "damage follows firing" is an
association, not an intervention effect. Only the exact-state, RNG-restored
branching makes the controlled comparison; and even that establishes the effect
*under a chosen continuation and horizon*. The consequence branch is evidence,
never proof.

## 3. Adapters (environments are plug-ins)

The core must not know DOOM exists. An environment provides:

- a **schema**: fields, collections, option fields, enums, ranges;
- `observe()` -> typed observation;
- `step(action)` / `stepTics(action, n)` -> reward plus a factual outcome record;
- `save()` / `restore()` -> exact state including RNG;
- a **fast reward-only path** for counterfactual probing (the DOOM build's
  65 KB JSON-per-branch telemetry was pure overhead).

DOOM remains the primary training environment because it is exact, inspectable
and cheap. A second environment — ideally a *typed-choice* task (tool selection,
form filling, routing) rather than another game — is the transfer test. Without
it we cannot distinguish a decision model from a DOOM policy.

## 4. Training

Signals into the same representation, in ascending order of cost:

1. **Teacher distributions** — an LLM returns a distribution over the options,
   optionally with a rationale used for part D. DAgger-style relabeling of the
   states the student actually visits.
2. **Exact interventions** — snapshot, vary the action, measure. The only
   source of causal contrast.
3. **Outcome learning** — part C, at several horizons.
4. **Advantage-weighted behavior learning** — weight the teacher/behavior loss
   by measured advantage. This is the classification-style RL surface.

Automatic freezing: no teacher and no fresh label stream above a trust threshold
-> gradients stop, calibration is revalidated on verifiable outcomes, and the
model can be exported.

## 5. Scale and the phone

Apple silicon is **unified memory**: there is no VRAM/system-RAM boundary to
offload across. Browser-accessible memory is the binding constraint, not the
device's total. Working figures to treat as budgets, not measurements:

| Backbone | FP16 weights | LoRA training (weights + grads + Adam) |
| --- | --- | --- |
| 100M | ~200 MB | feasible in-browser |
| 500M | ~1 GB | feasible, small rank, careful activations |
| 1B | ~2 GB | borderline in a browser tab |
| 3B | ~6 GB (1.5 GB at 4-bit) | not a browser tab; native/MLX only |

Consequences:

- A **500M backbone with a small trainable decision module is the working
  target**, with a distillation path to a ~30–150M frozen student for runtime.
- Parameter count and active computation are separate levers. Lookup-heavy and
  depth-laddered designs (per Cactus Needle) trade arithmetic for storage.
- Quantized *adapter training* in a browser needs real training kernels;
  inference support alone is not enough. Verify before committing.

## 6. What we refuse to do

- Tune DOOM-specific heuristics to make a benchmark look better.
- Report a label-side statistic as if it measured learning.
- Accept a single-run comparison as evidence (see §4 of the experiment record:
  identical code produced turn margin 0.104 and 0.028).
- Lower a grounding gate to obtain green CI.
- Add an auxiliary mechanism whose removal we have not measured.

## 7. What would make *this* design obsolete

Stated as falsifiable predictions, so we notice if the bet is wrong:

1. **Representation is the whole game.** If a frozen pretrained encoder plus a
   linear candidate scorer matches our contender under equal data, then the
   contribution is (data quality + causal labels), not the architecture. We
   should then shrink the system rather than grow it, and say so.
2. **Search beats a learned policy.** If exact-world-model search over a
   simulator plus a learned value function outperforms the direct scorer at
   comparable latency, the direct scorer is the wrong runtime and the model
   should become a value/measurement predictor only.
3. **One general model beats adapters.** If a single model trained across many
   environments transfers zero-shot to a new typed-choice task, then
   environment-specific adapters and schemas are the bottleneck, not a feature.
4. **The teacher is the product.** If teacher-quality labeled data dominates
   every architectural choice, the contribution is a data/labeling pipeline.
5. **Causal identification turns out to be unnecessary.** If purely
   observational next-token or contrastive training reaches the same grounding
   on held-out interventions, then our intervention machinery is cost without
   benefit.
6. **Latency is dominated by token count, not parameters.** If a decoder that
   emits the object token-by-token beats our scorer once both are optimized,
   then constrained decoding was the right answer and typed classification was
   premature.

Each of these is testable with the same harness. The project's value is partly
in discovering which one holds.

## 8. Evaluation protocol (applies to every stage)

- **Unit grounding gate** before any expensive run: a synthetic relational task
  the production pathway must solve, with correct signs, mirrored symmetry, and
  checkpoint round-trip equivalence. Existing artefact:
  `tests/factor-grounding.mjs` in the archived experiment.
- **Held-out interventions**, not held-out timesteps: same observations with
  actions varied.
- **Invariance tests**: choice order, choice IDs, irrelevant field renaming.
- **Composition tests**: unseen parameter combinations; goals changed at runtime.
- **Calibration**: reliability before and after adaptation, or a stale flag.
- **Multiple seeds**, variance reported. No single-run claims.
- **Latency and memory** measured on the target device, not extrapolated.
