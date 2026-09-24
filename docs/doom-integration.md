# Real DOOM integration

## Purpose

The real-engine experiment is deliberately stricter than a tactical-macro demo. The policy receives factual structured telemetry and may choose only player-level primitives: forward, back, turn, strafe, fire, use, or wait.

There is no MOVE_TO_ENEMY, RETREAT, FACE_ENEMY, pathfinder policy, distance-triggered firing rule, or scripted state-to-action mapping in doom-classifier.

## Runtime provenance

The initial browser runtime is fetched at runtime from a pinned commit of lukaske/jev-doom-agent:

- pinned commit: 318c32a24851444c1170bf083671c38723f3a35a
- engine: Chocolate Doom 3.1.1, GPL-2.0-or-later
- content: Freedoom 0.13.0

This neighboring project is valuable because it already solved the difficult Emscripten/browser telemetry plumbing. doom-classifier does not reuse its Jev policy, deterministic fallback policy, tactical macros, or pathfinding controller.

The runtime is fetched remotely rather than copied into this repository while the engine ABI is still experimental. Long term we should build and ship our own pinned engine artifact from corresponding source.

## Telemetry boundary

The engine bridge exposes native player state and iterates Doom's thinker list. Raw entity records include position, velocity, health, distance, relative angle and engine flags for kill-count actors and collectible/special actors, visibility through the engine line-of-sight check, and whether an actor currently targets the player.

The first browser experiment intentionally reduces this variable entity set to a small scalar observation because the current semantic adapter expects fixed numeric fields. The projection includes health, armor, ammo, recent damage, visible hostile count, nearest visible hostile geometry/health/targeting, visible pickup count and kills.

This projection belongs in the environment adapter, not the engine. It can later be replaced by a set encoder that consumes the raw entity table directly.

## Reward

The online residual currently sees a deliberately sparse consequence signal: positive reward for a kill-count increase, negative reward for health loss, smaller positive reward for health recovery, a terminal penalty at zero health, and a tiny per-decision cost.

This defines the objective but does not prescribe an action.

The controlled runtime's PromptFPS_SetStart entry point creates a repeatable combat scene. This is useful for early learning curves, but it must not become the only benchmark. Natural-map episodes and schema perturbation remain required.

## Why not use the comparator's tactical schema?

The neighboring Jev experiment gives the decision layer actions such as MOVE_TO_ENEMY, RETREAT_FROM_ENEMY, FACE_ENEMY, COLLECT_NEAREST_PICKUP and USE_NEAREST_LINE. Its local controller then performs turning, path checks, movement and firing.

That is a reasonable Jev demonstration, but it moves substantial competence into deterministic software.

Our experiment asks a harder question: can semantic priors plus tiny learned experience discover useful primitive control without those tactical helpers?

## Next engine step

The remote runtime is a bootstrap. The preferred final engine ABI is our own small instrumentation layer:

1. packed player snapshot;
2. generic entity table;
3. optional world-line / interaction table;
4. primitive control bitmask;
5. reset/scenario entry point.

The engine should expose facts and mechanics, never labels such as threat=high. A set/attention encoder can then consume a bounded number of entity records directly, avoiding the current hand-reduced nearest_hostile fields and making schema-transfer tests much stronger.
