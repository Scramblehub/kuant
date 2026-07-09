# Plan: kuant.auto (automation root) + backend layer + queueing expansion

**Status**: DRAFT / speculative. Not implementation-blocking. Tentative
throughout: everything below is a starting position to iterate against,
not a spec. Sections marked "open" are unresolved and worth revisiting
before code lands.

## Motivation

kuant kernels turn out to be well-shaped for LLM tool use, mostly by
accident of existing design choices:

- Every Result is a dataclass with a `.summary()` method: perfect
  context injection.
- Every error carries a namespace, a stable `KE-*` code, and a
  `-> Fix:` line: an agent that hits an error can self-remediate
  without docs.
- Every warning carries a `KW-*` code with the same structure: the
  agent can decide whether to react or ignore.
- Naming is uniform (lowercase-concatenated), so schema generation
  from the module tree is mechanical.

The proposal is to expose this shape deliberately as an automation
root, so a local or remote LLM can drive kernels against a dataset
under the same discipline the user applies by hand (pre-registered
criteria, walk-forward + Monte Carlo gates, test-partition secrecy
until ship decision).

## Design principles

1. **The researcher core is provider-blind.** It sees a `Backend`
   interface; whether that is Ollama, Claude, or a fallback chain is a
   config concern.
2. **Discipline is enforced by the harness, not the model.** The
   researcher cannot see the test partition. It cannot ship a signal
   without a pre-registered criteria block. It cannot bypass WF+MC on
   borderline candidates.
3. **Every result artifact is reproducible.** Backend, model, seed
   (where supported), prompts, tool schemas, and data slice are all
   logged.
4. **Fail loud on cold start.** If the model is misconfigured, VRAM
   is short, or the API key is invalid, the researcher raises before
   the research loop starts, not mid-run.
5. **Compose with what already exists.** `kuant.queueing` is the
   natural home for rate limits, fallback, and cost budgets; each is
   a `Backend` decorator.

## Rough shape (subject to change)

```text
kuant/auto/
    __init__.py              # public entrypoint
    researcher.py            # the outer loop
    warming.py               # cold-start priming
    tools.py                 # tool-card generator (walks a subpackage)
    prompts/                 # system prompts, versioned
    backends/
        base.py              # Backend ABC + GenerationResult + ToolCall
        ollama.py            # local via HTTP
        llamacpp.py          # local via bindings (optional)
        anthropic.py         # Claude
        openai.py            # OpenAI
        google.py            # Gemini

kuant/queueing/              # expanded
    throttle.py              # existing hardware throttle
    coordination.py          # existing request coordination
    ratelimit.py             # token bucket, per-provider
    fallback.py              # FallbackChain
    circuitbreaker.py
    budget.py                # cost/token ceiling
    retry.py                 # backoff on rate-limit / provider-down
    routelogger.py           # records which backend served each turn
```

Every queueing primitive implements `Backend` too, so the stack is a
decorator chain:

```python
researcher = Researcher(
    model=CostBudget(
        FallbackChain(
            primary=CircuitBreaker(RateLimiter(AnthropicBackend(...))),
            fallback=[OllamaBackend(...)],
        ),
        max_usd=5.0,
    ),
    tools=kuant.stats.auto.tools() + kuant.signals.auto.tools(),
)
```

## Per-subpackage automation root

Each subpackage gets an `auto/` module that exposes its kernels as
tool cards, not the full function surface. Tool cards carry:

- Name (kernel function name)
- Purpose (one-line, LLM-facing)
- Input JSONSchema (derived from the signature + type hints)
- Output JSONSchema (derived from the Result dataclass)
- `KE-*` codes the kernel can raise
- `KW-*` codes it can emit
- Cost hint (S/M/L for compute cost; used by budget-aware planners)

Open: whether tool cards live as decorators on each kernel, a
manifest file per subpackage, or generated at import. Preference:
decorators. Least drift.

## Discipline gates (maps to durable rules)

The researcher is under three hard gates:

1. **Commit criteria first.** The researcher must produce a
   pre-registered criteria block (metric, threshold, WF partition
   spec, MC config) before it can invoke any evaluation kernel.
   Anything less returns `KE-CRITERIA-MISSING`.
2. **Test partition sealed.** Data ingestion splits train/test at
   researcher-init. The researcher's toolset receives only the train
   view. The test view is invocable exactly once, by a
   `researcher.finalize(candidate)` call at end of run.
3. **WF+MC required for borderline.** If a candidate's train-partition
   result falls in a config-defined borderline band, the harness
   requires walk-forward + Monte Carlo before the finalize call is
   allowed.

Open: where the criteria block schema lives, and whether it should be
enforced as JSONSchema or a Python dataclass.

## Backend abstraction

`Backend` interface:

- `generate(messages, tools=None, **params) -> GenerationResult`
- `warm() -> None`
- `health_check() -> bool`

`GenerationResult` carries:

- `text`, `tool_calls` (normalized `ToolCall(name, args, id)` shape)
- `input_tokens`, `output_tokens`, `cost_usd` (None for local)
- `latency_ms`, `finish_reason`, `backend_id` (for the route log)

Config-driven via TOML at `~/.kuant/backends.toml` or a project-local
`kuant.toml`:

```toml
[backends.local-big]
provider = "ollama"
host = "http://localhost:11434"
model = "qwen2.5-coder:32b"
keep_alive = -1

[backends.claude-sonnet]
provider = "anthropic"
model = "claude-sonnet-4-5"
api_key_env = "ANTHROPIC_API_KEY"    # env-ref only, never inline
max_tokens = 8192
```

Loaded via `Backend.load("claude-sonnet")` -> instantiates the adapter.

## Warming suite

`WarmingSuite(model, tools, data=None, embed_model=None).prime()`
brings the following to hot state before the research loop starts:

- Model weights (VRAM or process memory)
- KV cache primed with the system prompt
- Tool schemas parsed and cached
- Optional embedding model loaded and warmed
- Working data slice materialized (dask etc)
- JIT'd kernels (if we ship numba on hot loops) pre-compiled

Open: heartbeat cadence, or `keep_alive=-1` and rely on the process
lifecycle. Depends on whether the session is interactive or batch.

## New error codes (queueing / backend surface)

Each with the standard `-> Fix:` line.

- `KE-CTX-OVERFLOW`
- `KE-RATE-LIMIT` (retry-after included)
- `KE-AUTH-INVALID`
- `KE-PROVIDER-DOWN`
- `KE-TOOL-INVALID` (model returned malformed args)
- `KE-BUDGET-EXHAUSTED`
- `KE-CIRCUIT-OPEN`
- `KE-CRITERIA-MISSING` (discipline gate 1)
- `KE-TEST-PARTITION-LEAK` (discipline gate 2, safety net for bugs)

Warnings:

- `KW-BACKEND-FALLBACK` (route logger emits; researcher can react)
- `KW-WF-BORDERLINE` (candidate near the borderline band; discipline
  gate 3 will require WF+MC)

Also worth adding: a `kuant.errors.CODES` machine-readable registry
mapping every `KE-*/KW-*` code to `{description, typical_cause,
remediation_class}`. Lets the researcher enumerate possible failure
modes before running anything.

## Reproducibility and logging

Every research run writes:

- `run_id`, timestamp, kuant version, config hash
- Backend adapter used per turn (route log)
- Model + seed where supported
- Data slice descriptor (train partition only)
- Full turn transcript (prompts + tool calls + tool results)
- Criteria block(s) and their disposition
- Final finalize result and its verdict

Open: format. JSONL streaming plus a parquet summary is the strong
candidate. Storage location: `~/.kuant/runs/<run_id>/`.

## Security callouts (probably worth durable rules)

1. API keys only via env var reference, never inline in config.
2. Redact API keys from tracebacks and error messages.
3. Remote backends require a `max_cost_usd` on any run.
4. Every run artifact must record the provider and model.
5. Optional deps via extras: `pip install kuant[anthropic]` /
   `[openai]` / `[ollama]`; no hard imports at package top-level.
6. Consider a `local_only=True` mode that refuses to instantiate any
   remote backend. Useful when the data slice contains information the
   user does not want leaving the machine.

## Decisions from initial review

The following open questions were resolved in an initial pass. Kept
here (not deleted from the doc) so the reasoning survives.

### Prior findings memory: warn for now, defer full solution

The researcher does NOT read `MEMORY.md` or prior run logs in v1.
Instead, on any run against a data slice that has been researched
before (detected by a slice hash in `~/.kuant/runs/`), the harness
emits `KW-PRIOR-RUN-EXISTS` pointing at the prior run artifact and
lets the user disposition manually.

Reason: full prior-findings integration is a memory-sleeve problem,
not an automation-root problem. It is scheduled as a separate product
line coordinated with the user's long-memory sleeve work. This doc
does not commit to a specific integration shape for it.

### Failure disposition policy: four-mode enum

Config-driven per run:

```python
class FailureDisposition(Enum):
    LOG_AND_DROP = "log-and-drop"           # signal fails, log, move on
    ONE_TWEAK = "one-tweak"                 # one retry with tweaked params
    THREE_STRIKES = "three-strikes"         # up to three tweaked retries
    UNTIL_IT_WORKS = "until-it-works"       # unlimited; SEE GUARDRAIL
```

Guardrail: `UNTIL_IT_WORKS` is intended for single-factor / single-
test fitting, NOT for open-ended signal search. When the researcher
is invoked in signal-search mode with `UNTIL_IT_WORKS`, the harness
emits `KW-NARRATIVE-CHASE-RISK` at run start and requires the user
to acknowledge (interactive) or explicitly pass
`acknowledge_narrative_risk=True` (batch). Rationale: repeated
parameter tweaks against noise is exactly the failure mode the
"shapes not narratives" rule was written to prevent.

Open sub-question: what counts as a "tweak" is not fully specified.
Working definition: any hyperparameter change that keeps the kernel
identity + input columns the same. Changing the kernel or the input
columns is a new candidate, not a tweak.

### Human-in-the-loop: binary config

Config-driven per run:

```python
class HitlMode(Enum):
    NOTIFY_AND_CONTINUE = "notify-and-continue"
    NOTIFY_AND_STOP = "notify-and-stop"
```

Notification hooks fire on: failed criteria block, WF+MC borderline
result, finalize decision, budget threshold reached (any of cost,
tokens, wall time). Notifications delivered via a `NotificationSink`
that the user can wire to stdout, a file, a webhook, or later a
push-notification adapter.

### Data privacy layer: warn, defer

No `PrivacyFilter` decorator in v1. On any run that instantiates a
remote backend (Anthropic, OpenAI, Google), the harness emits
`KW-DATA-LEAVES-MACHINE` at run start listing the columns and row
count that will be sent over the wire.

Reason: cybersecurity is a real concern but not the current
bottleneck for a small user base. The warning creates awareness
without blocking the flow. Deferred until either (a) the user base
grows enough that a real leak becomes actionable, or (b) the
researcher is used against genuinely sensitive datasets.

### Kernel exposure control: out of scope

Not implemented. Rationale: models operate through the tool-use
interface only; they cannot escape the tool boundary and reach
arbitrary Python. Untrusted-model risk is bounded by "which tools
did you register" at researcher construction, which is already user-
controlled. No per-model whitelist needed.

Reopen this decision if we ever attach a code-execution tool (which
would break the boundary) or if we discover a way for a model to
exfiltrate data via tool-argument abuse.

## Decisions from second review

### Session state: atomic v1, checkpointing later

Each run is atomic. Crash / kill / OOM discards the session. State
schema and checkpoint plumbing are deferred until runs get long
enough that atomic actually hurts.

### Prompt versioning: package defaults + user override

System prompts ship inside the package at `kuant/auto/prompts/*.md`
and are versioned with releases. Config can point at
`~/.kuant/prompts/` to override, so users can iterate without a
package rebuild. Fresh installs just work; power users can diverge.

### Determinism contract: loud and proud

Every run artifact records: backend id, model id, seed (or null if
provider does not support one), determinism class (`deterministic`,
`temperature-0-approximate`, `non-deterministic`). Researcher emits
`KW-NON-DETERMINISTIC-BACKEND` at run start if reproducibility is
requested but the backend cannot guarantee it. Silent
non-determinism is a foot-gun the discipline gates cannot catch,
so we make it visible.

### Compute budget: predictive, rolling, not a flat ceiling

Each kernel's tool card carries a cost hint (`O(n)`, `O(n log n)`,
`O(n^2)`, or a measured constant for JIT'd hot kernels).

Session start:

1. Researcher composes an ETA for the planned experiment set from
   kernel hints, input shape, and the user's calibrated baseline
   (`~/.kuant/perf.json`, populated by a `kuant.auto.benchmark`
   command that runs once).
2. Presents ETA to the user: "Estimated compute: ~22 min on this
   machine. Continue?"

During run:

- Silent recomputation after each experiment (pre-run estimates
  drift; rolling estimates stay honest).
- `KW-COMPUTE-OVER-BUDGET` at 1.5x initial estimate.
- `KE-COMPUTE-EXHAUSTED` at 3x initial estimate (hard-stop safety
  net for runaway loops).

Under the same queueing decorator pattern:
`ComputeBudget(researcher, hard_multiple=3.0)`.

### Multi-agent orchestration: single v1, hybrid v2

One researcher, single system prompt, unified toolset. Router +
specialist worker pattern is queued as a v2 refinement once the
single-researcher shape is proven end-to-end.

### Output artifact shape: process-type manifests, no charts

Instead of freeform markdown reports, each **process type** declares
a fixed output schema. The researcher fills structured slots.
Markdown becomes a rendered view of structured artifacts, not a
first-class artifact.

Example catalog (starting point, extendable):

- `SignalSearchProcess`: `signals.parquet` (one row per candidate),
  `criteria.jsonl`, `turn_transcript.jsonl`
- `SingleFactorFitProcess`: `fit_result.parquet`, `mc_samples.parquet`,
  `verdict.json`
- `WalkForwardValidateProcess`: `wf_splits.parquet`, `wf_metrics.parquet`
- `TailFitProcess`: `gpd_params.json`, `tail_diagnostics.parquet`

Markdown rendering is a separate deterministic step (template +
structured data), triggered on-demand:
`kuant.auto.render(run_id, format="markdown")`. Cheap because the
LLM never wrote it: a template did.

**Charts / visualization are NOT part of the automation-root output
pipeline.** Process outputs stay purely structured (parquet, json).
Each Result gains a `.to_dataframe()` method returning plot-ready
pandas data; users bring their own viz library (matplotlib, plotly,
altair, hvplot, seaborn). Markdown references parquet by relative
path; users who want inline images generate them themselves and drop
into the run dir.

Bonus alignment: this reinforces "shapes not narratives" all the way
down to the artifact layer. Nothing narrative-shaped or aesthetic-
shaped is persisted by the researcher.

Separately, feature #13 from the QOL list (opt-in `Result.plot()`
matplotlib helper) stays on the roadmap as a **user-facing
convenience** for at-the-desk exploration. NOT called from the
automation root. Two different contexts, two different responsibilities.

### Kernel discovery: hybrid with stability flags

Every session start walks `kuant.*` and generates tool cards from
`.__kuant_tool__` decorated kernels. Zero-effort discovery for kernels
that land after the system prompt was written.

The system prompt does not enumerate every kernel. It tells the
researcher: "you have stats, risk, signals, portfolio, and causal
kernels available; call `list_tools(category='stats')` for the full
menu."

Kernel promotion is a metadata flag, not a manifest bump:

```python
@kuant_tool(
    stability="experimental",   # or "stable"
    purpose="...",
    cost_hint="O(n log n)",
)
def newkernel(...): ...
```

System prompt tells the researcher to **prefer** stable kernels
but experimental ones are usable if it explicitly picks them. New
kernels are technically available the moment they land; graduation
to "stable" is one line and can happen in the same PR that adds
tests.

## Collection-Result convenience hooks

A distinction that emerged during the second review: Result types
are not uniform in shape. Two broad categories:

- **Scalar Results** carry one answer per invocation.
  `CornishFisherVarResult`, `EvtVarResult`, `IvResult`,
  `SynthControlResult` (one ATT), `MesResult`, `CoVarResult`.
  Filter hooks do not apply.
- **Collection Results** carry N candidates / N splits / N draws with
  a natural tabular dimension. Future examples:
  `SignalSearchResult` (many candidates), `WalkForwardResult`
  (many splits), `EnsembleResult`, `SweepResult`. Some existing
  Results also have collection-shaped internal fields
  (`MfdfaResult.h_q`, `SindyLassoResult.coefficients`,
  `HmmResult.states`, `PermutationTestResult.null_distribution`).

Collection Results get a small delegating shim over pandas:

```python
result.top(n, by=None)         # n largest by metric
result.bottom(n, by=None)      # n smallest
result.where(query_or_fn)      # pandas .query() syntax OR callable
result.sort(by, ascending=True)
result.take(mask_or_indices)
result.to_dataframe()          # escape hatch for full pandas
```

Six methods. Chainable, immutable, each returns a new instance of the
same Result type. Anything more exotic (`groupby`, `rolling`, `pivot`)
goes through `.to_dataframe()`. kuant does not reinvent pandas; it
gives the top-95% queries a typed, tool-card-advertisable surface.

Query language for `.where()`:

- **String form** uses pandas `df.query()` syntax verbatim
  (`"sharpe > 1 and mdd > -0.2"`). Users who know pandas learn
  nothing new; users who don't learn a transferable skill.
- **Callable form** (`.where(lambda r: r.mdd > -0.15)`) supported as
  escape hatch.

`__repr__` for collection Results shows a top-N summary (pandas
convention with `df.head()`), not a full dataclass dump. Same
principle as the `__repr__ = summary` decision for scalar Results,
extended to a top-N view. Default N = 10, sort key = first metric
declared on the Result.

### Audit prerequisite

Rolling this out requires classifying every existing Result:
scalar, pure-collection, or hybrid (has tabular internal fields but
also scalar fields). The classification is not obvious for every
kernel:

- `PcAlgoResult` has an adjacency matrix but is a graph, not a
  tabular collection: probably no hooks.
- `MfdfaResult` has scalar `h_2` alongside `h_q[]`: hybrid, hooks
  apply to the `h_q` view.
- `SindyLassoResult` has a coefficient matrix: collection over
  (feature, term) pairs, hooks apply.
- `HmmResult` has a state trajectory: collection over time, hooks
  apply.
- `PermutationTestResult` has a null distribution: collection over
  draws, hooks apply.

Full pass across `kuant.stats`, `kuant.risk`, `kuant.signals`,
`kuant.portfolio`, `kuant.causal`, `kuant.sindy`, `kuant.qm`,
`kuant.topology` is a real audit. Best done as a dedicated agent
sweep once the base convention is defined and one reference
implementation exists (probably `SignalSearchResult`, since that
kernel is landing new and can define the shape). Queued but not
scoped in this doc.

Also worth pulling `.to_dataframe()` out as a separate concern.
Even scalar Results with tabular internal fields benefit from a
consistent `.to_dataframe()` convention (single-row DF for scalars,
multi-row for collections). That is a lighter audit and a
prerequisite for the QOL feature 5 pandas accessor
(`df.kuant.<kernel>()`) making sense.

## Open questions (still unresolved)

None from the initial batch. New questions will accumulate here as
implementation exposes them.

## Explicit non-decisions

The following are intentionally not locked here:

- Which backend ships first (Anthropic vs Ollama as the tier-1
  reference implementation).
- Whether the researcher is one-shot or interactive-turn.
- Exact schema for tool cards.
- Storage format for session state.
- Whether the researcher is invoked programmatically, from a CLI,
  from a Jupyter widget, or all of the above.
- The full set of `KE-*/KW-*` codes for the automation layer;
  above list is starting position only.

## Risks worth naming

1. **Narrative drift.** LLMs are narrative machines. Even with
   discipline gates, a persistent researcher will backfit stories
   to train-partition noise. Guardrails matter more than tools.
2. **Cost blowout.** A researcher looping on a remote backend can
   burn a lot of money quickly. Cost budget is not optional.
3. **Reproducibility rot.** External APIs change, model versions
   are deprecated. Run artifacts need to be interpretable a year
   later without the model in question being live.
4. **Data leakage to third parties.** Sending strategy internals
   over the wire to a hosted API is a category the user may not
   want to allow. `local_only=True` and a `PrivacyFilter` are the
   mitigation.
5. **Over-abstraction.** Building six layers of Backend decorators
   before we have one working researcher is a real risk. Start with
   one concrete backend and one researcher, then generalize.

## Suggested first slice (when we do build)

1. `Backend` ABC + `OllamaBackend` only.
2. `WarmingSuite` for that one backend.
3. Tool card generator for one subpackage (say, `kuant.stats`).
4. Minimal `Researcher` that can call one tool, get a Result, and
   summarize. No discipline gates yet.
5. Then, in order: RateLimiter, FallbackChain, CostBudget,
   CircuitBreaker, RouteLogger.
6. Then, in order: criteria block enforcement, test-partition
   sealing, WF+MC gate.
7. Then: Anthropic backend, config-driven loading, multi-provider.

Deliberately backwards from the finished architecture: the queueing
layer is only needed once we have more than one backend to route
between.
