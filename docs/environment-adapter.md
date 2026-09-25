# Environment adapter contract

The research target is a reusable typed decision learner. DOOM is the first environment adapter, not part of the policy architecture.

An environment adapter supplies four things:

1. **Schema** — named scalar fields, categorical meanings, variable record collections, and an objective description.
2. **Observation** — current factual state expressed through that schema.
3. **Typed actions** — request-time candidate actions with labels/descriptions and optional numeric/categorical parameters.
4. **Outcome** — factual transition consequences and reward. Reward may encode task success/failure, but the adapter must not encode a behavioral policy such as `enemy visible -> fire`.

The core policy receives only these interfaces. It does not import DOOM engine types, map logic, enemy rules, or weapon heuristics.

## Reusable core path

`environment -> structured state -> set/temporal encoder -> semantic prior + consequence value -> KL-constrained fusion -> typed action distribution -> actuator`

The semantic branch can be prepared from a language/decision teacher and then run without the teacher. The consequence branch learns from transitions through n-step returns, replay, delayed target values, and bootstrap disagreement. KL-constrained fusion limits how far learned value can move the final distribution away from the semantic prior as value trust grows.

## Teacher contract

A semantic teacher maps the same structured observation and candidate actions to action scores. It is supervision, not a hidden policy API. The browser exposes each teacher call with:

- trigger reason;
- state text supplied to the teacher;
- ranked action scores/probabilities;
- latency;
- distillation fit;
- temperature calibration.

A different teacher can replace MobileBERT without changing the environment or value learner.

## Transfer test

A new task should be implementable by adding a new adapter while leaving the policy modules untouched. Useful transfer environments should vary:

- field names and categorical vocabularies;
- record cardinality/order;
- action count;
- numeric action parameters;
- reward timing;
- whether semantics alone are sufficient or consequence learning is required.

A valid transfer result is competence that survives those changes without adding task-specific rules to the core policy.
