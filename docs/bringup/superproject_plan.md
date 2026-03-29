# PTO 910B Superproject Bring-Up Plan

This repository is the **bring-up control plane** for migrating `ops-transformer`
AI Core kernels onto the `PTO-DSL -> PTOAS -> pto-isa` stack on the current
`910B1 / ascend910b / a3 / dav-2201` environment.

The goal is not only to make kernels runnable. The end goal is:

1. migrate the targeted operator surface completely,
2. keep the migration state auditable and reproducible,
3. drive cross-stack fixes in `pto-dsl`, `PTOAS`, `pto-isa`, and runtime
   integrations when kernel-local changes are insufficient,
4. converge correctness first, then regression stability, then performance.

## Roles of this superproject

This repo owns the top-level bring-up workflow:

- **inventory truth**: which operators are in scope, excluded, seeded, or backlogged
- **gate truth**: which gates a kernel must pass before status changes
- **gap truth**: which cross-stack blockers exist and which kernels they block
- **regression truth**: which kernels belong in the active regression matrix
- **evidence routing**: where traces, reports, and benchmark results live

Sibling repos remain the implementation homes for lower-stack fixes:

- `pto-dsl`: source-language and frontend surface
- `PTOAS`: lowering, legality, codegen, Python bindings, capability manifests
- `pto-isa`: ISA contract, tile legality, type/runtime semantics, templates
- `ops-transformer`: baseline contract replication and runtime compatibility work

## Bring-up principles

1. **Superproject first**
   - New work starts from operator inventory + gate status + blocker mapping.
   - Do not treat isolated kernel hacks as completion.

2. **One constrained slice at a time**
   - For each operator, land a bounded executable slice first.
   - Expand semantics only after the bounded slice is stable.

3. **Correctness before performance, but performance is mandatory**
   - A kernel is not “done” just because it compiles or matches reference once.
   - Performance evidence must be recorded and tracked to closure.

4. **Cross-stack blockers are first-class**
   - If an operator is blocked by PTODSL/PTOAS/PTO-ISA/runtime, record and fix it there.
   - Do not hide cross-stack issues inside local kernel workarounds unless explicitly intended.

5. **Checklist + gate + gap must agree**
   - Status changes must be explainable from the gate results and blocker board.

## Execution model

### Phase 0 — Workspace + governance

Deliverables:

- environment detection and pinning (`external/manifest.lock`)
- workspace bootstrap and local sibling repo discovery
- canonical inventory/checklist/gap files
- baseline gate definitions

### Phase 1 — Seed kernel closure

Initial seed scope:

- `apply_rotary_pos_emb`
- `grouped_matmul`
- `ffn`
- `moe_token_permute`
- `flash_attention_score`
- `matmul_reduce_scatter`

For each seed kernel, close the following loop:

1. bounded PTO slice exists
2. baseline or reference contract exists
3. compile trace is captured
4. correctness gate passes
5. benchmark gate passes
6. regression inclusion is wired
7. remaining performance or semantic blockers are recorded

### Phase 2 — Family expansion

Expand by operator family once the seed path is stable:

- wave 1: posembedding / gmm / ffn
- wave 2: moe
- wave 3: attention
- wave 4: advanced attention
- wave 5: mc2 distributed kernels

### Phase 3 — Cross-stack closure

Systematically reduce open blockers in:

- `pto-dsl`
- `PTOAS`
- `pto-isa`
- runtime / baseline integration layers

### Phase 4 — Performance convergence

For operators that are correctness-green and regression-green:

- compare PTO latency to baseline latency
- identify performance gaps that are due to source surface, lowering, runtime, or ISA limitations
- close gaps or record justified performance debt explicitly

## Status semantics

This repo currently uses a mix of checklist markers and inventory states. The
working interpretation is:

- `planned`: no stable bounded executable PTO migration slice yet
- `prototype`: bounded slice exists and at least part of the gate path is wired
- `blocked`: a current blocker prevents advancing to the next gate

Checklist markers in `checklists/910b_ai_core_migration.md` are interpreted as:

- `[ ]`: not yet closed for the current migration phase
- `[~]`: partially closed; some gates pass but closure is incomplete
- `[x]`: fully closed for the intended migration target

Gap board status is interpreted as:

- `open`: active blocker or unresolved capability gap
- `mitigated`: a bounded slice is no longer blocked, but the general capability gap still exists
- `resolved`: no longer an active blocker for the tracked scope

## Cross-repo fix routing

When a kernel cannot advance, classify the blocker first:

- **pto-kernels**
  - benchmark harness, runtime wrapper, seed contract, staging choice, trace capture
- **pto-dsl**
  - missing source-level primitive, missing frontend expression surface, missing reusable family helper
- **PTOAS**
  - legality failure, pass pipeline bug, verifier gap, codegen issue, Python binding/runtime ABI issue
- **pto-isa**
  - missing tile/type/runtime semantic contract, instruction legality gap, backend template limitation
- **ops-transformer/runtime**
  - baseline entrypoint unavailable, runtime contract unreproducible, environment/package issue

## Evidence expectations for each operator step

Each operator step should leave behind:

- updated inventory/checklist/gap context
- a benchmark spec
- baseline adapter state
- PTO adapter state
- compile artifacts or blocked compile evidence
- correctness evidence
- benchmark evidence
- explicit blocker mapping when incomplete

## Immediate next-step policy

Near-term work should prioritize:

1. making the superproject governance explicit,
2. closing the seed kernel loop consistently,
3. reducing repeated frontend/runtime bring-up work,
4. only then expanding breadth.
