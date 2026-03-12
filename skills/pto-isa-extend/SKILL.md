---
name: pto-isa-extend
description: Use when a PTO migration is blocked by missing ptodsl surface area, PTOAS legality or lowering, or pto-isa A2/A3 backend coverage. Extend the stack in the correct order and land reusable functionality with tests.
---

# PTO ISA Extend

Use this skill only after a blocker has been reproduced through the normal flow.

Order of work:

1. Extend `ptodsl` if the missing piece is a reusable frontend primitive.
2. Extend `PTOAS` if the IR exists but legality, lowering, auto-sync, memory planning, or diagnostics fail.
3. Extend `pto-isa` only if the A2/A3 backend lacks the required template or is materially too slow.

Rules:

- Avoid one-off kernel hacks.
- Add tests in the repo you touch.
- Link the blocker in `bench/gap_board.yaml`.
- Keep the target backend fixed to A2/A3 for this machine.

Important files:

- `bench/gap_board.yaml`
- `external/manifest.lock`
- sibling repos `pto-dsl`, `PTOAS`, `pto-isa`
