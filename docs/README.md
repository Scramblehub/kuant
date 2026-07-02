# kuant docs

Design decisions, per-kernel API references, and usage examples.

## Structure

- `kernels/` — one page per implemented kernel; auto-populated from docstrings
- `design/` — architectural decisions, edge case handling rationale, GPU vs CPU
  strategy, testing policy
- `examples/` — end-to-end usage patterns (composing kernels to solve a task)

## Reference

The master planning index lives in the project's research memory (external —
not committed to this repo). Each kernel's file docstring links back to the
plan entry it implements.
