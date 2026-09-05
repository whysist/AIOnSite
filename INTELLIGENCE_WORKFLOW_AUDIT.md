# AIOnSite — Intelligence Workflow Audit

Scope: `src/orchestrator.py`, `src/agents/*`, `src/pipeline/*`, `src/tools/*`,
`src/verification/*`, `src/state/*`, `src/audit/*`, `src/llm/*`, `src/api/*`,
`src/core/config.py`, and the relevant tests. Read in full; conclusions below
are traced to line-level evidence, not inferred from naming or docstrings.

---

## 1. Executive Summary

**Classification: Working but architecturally weak, with one confirmed
implementation bug in the completeness/verification path.**

The system is not "multiple generic LLM calls arranged in a DAG." It has
real, deliberate machinery that a naive multi-agent demo would not have:
deterministic plan validation, a `Criticality`/`NodeOutcome` model that
separates "ran" from "succeeded," a `TaskRequirement` checklist independent
of the DAG, a layered (rule-based + LLM) verifier that an optimistic LLM
critique cannot override, explicit `Evidence` records, and sovereignty
enforcement that raises rather than silently downgrades. These are
genuine intelligence-orchestration primitives, and they are the parts of
the system doing the real work in the P-101 case (the verifier failed
*correctly*, for a fixable reason — see §10).

Its weaknesses are architectural, not superficial:

- There is **no intent/complexity classification** anywhere. Every request,
  from "tell me about P-101" to "should P-101 be taken out of service,"
  goes through the identical Planner → DAG → multi-agent → Verifier path.
- **Role prompts and role tool-grants do not create behavioural
  boundaries.** `researcher`, `analyst`, and `executor` are the same
  `LLMAgent` class with different system-prompt strings, and the default
  tool grants (`agents/roles.py::_DEFAULT_TOOLS`) give `equipment_lookup` to
  all three. Nothing stops (and the `analyst`/`executor` prompts actively
  *tell* the model) to re-retrieve data a prior step already fetched.
- **Evidence flow between DAG nodes is text-only and lossy.** A node
  receives only the prior node's final natural-language `output` string
  (`pipeline/state.py::outputs_for`), never the structured `ToolResult` or
  `Evidence` record. Structured evidence exists (`state/evidence.py`) but
  is visible only to the verifier, not to downstream agents — which is the
  direct mechanical cause of duplicate `equipment_lookup` calls.
- **A confirmed bug**: `extract_requirements` (`pipeline/requirements.py`)
  never sets `source_step_id` on the common path (planner-supplied
  `requirements`), so the verifier's deterministic per-requirement check
  (`node_outcomes.get(req.source_step_id)`) is unreachable for exactly the
  requirements a normal (non-degraded) plan produces, and every one of them
  falls back to a coarse keyword-overlap heuristic
  (`verifier.py::_requirement_mentioned`). This is very likely the actual
  cause of the P-101 verification failure — not a hallucination or a
  factual error, but a text-matching heuristic failing to recognize a
  paraphrased but correct answer.

None of this means "rewrite the system." The correct fixes are additive and
fit the existing seams (a shared evidence/state layer already exists in
skeleton form; it is under-used, not absent). See §8–§9.

---

## 2. Actual Current Architecture

```
HTTP POST /tasks (api/app.py)
  -> Orchestrator.run_task(task, confidential)          [orchestrator.py]
       -> ExecutionContext created (task, sovereign_mode, confidential)
       -> ModelRouter constructed (policy: sovereign/confidential -> local only)
       -> Planner.plan(task)                             [agents/planner.py]
            - ONE LLM call, JSON-only, Pydantic-validated
            - invalid JSON / bad deps -> 1 bounded retry -> deterministic
              3-step fallback plan (researcher -> analyst -> summarizer)
       -> extract_requirements(plan)                     [pipeline/requirements.py]
            - prefers plan.requirements (free text) -> TaskRequirement list
              with NO source_step_id  <-- see Finding V-1
       -> build_pipeline(plan)                            [pipeline/builder.py]
            - plan steps -> PipelineNode/PipelineEdge (1:1, ids preserved)
            - always appends a terminal "verify_final" VERIFIER node
              depending on every leaf node
            - PipelineGraph(pipeline).validate() (cycle check)
       -> build_agent_library(llm)                        [agents/roles.py]
            - 5 LLMAgent instances (researcher/analyst/executor/summarizer/
              custom), differing ONLY in system_prompt + allowed tool list
       -> PipelineExecutor.run(ctx, pipeline)              [pipeline/executor.py]
            - PipelineGraph.execution_layers(): topological layering,
              nodes in the same layer run concurrently via asyncio.gather
            - per node: dependency gate -> ModelRouter.get_llm(node)
              -> agent.execute(description, context=outputs_for(depends_on))
              -> tool-use loop (LLMAgent.execute, agents/base_agent.py)
              -> successful tool calls -> Evidence recorded in ctx.state
              -> VERIFIER node -> ResultVerifier.verify(...)
       -> _finalise(): last SUMMARIZER (or last completed) node's .output
              becomes ctx.final_answer; last VERIFIER's verdict becomes
              ctx.final_verification
  -> task_status = compute_task_status(requirements, verification, exec_status)
  -> if task_status != COMPLETED: prepend a deterministic
     "[INSPECTION STATUS: ...]" banner to final_answer (not another LLM call)
  -> return ExecutionContext (final_answer, verification, audit trail, ...)
```

Key structural facts, verified in code:

- **`agents/router.py::AgentRouter`** (name → agent lookup) is dead code in
  this path — the orchestrator never constructs or calls it. `ModelRouter`
  in the same file is a *provider/model* router (which LLM backend serves a
  node), not an *intent* router. There is no component anywhere that reads
  the user's task text and classifies it into a request type or a
  complexity tier before planning.
- **`InspectionAgent`** (`agents/inspection_agent.py`) is a second,
  independent entry point using the legacy `BaseAgent.run(state)` contract.
  It is not wired into `Orchestrator` or the API at all (grep confirms no
  caller). It is inert legacy code, not part of the live request path.
- The **Planner is the only routing/classification decision-maker in the
  system**, and it is a single unconstrained LLM call: it does not know
  what tools exist, what their schemas are, what data already exists, or
  which agents can use which tools. It only knows the fixed prompt in
  `planner.py::_SYSTEM` (role names + a step-count target).

---

## 3. End-to-End Trace: "Tell me about P-101. What is the equipment?"

1. **Entry**: `POST /tasks {"task": "Tell me about P-101. What is the equipment?"}`
   → `Orchestrator.run_task`. The raw string is stored verbatim as
   `ExecutionContext.task` — no transformation, no intent tag.
2. **Planning**: `Planner.plan(task)` sends the raw task string, unmodified,
   as the user turn to the planning LLM, with `planner.py::_SYSTEM` as the
   system prompt. The planner LLM decides *everything*: how many steps,
   which agent per step, and the `requirements` checklist. For this input
   the model apparently produced something like a 4-step plan
   (researcher → analyst → executor → summarizer) with requirements
   spanning equipment identity **and** limits/inspection/maintenance —
   even though the user only asked "what is the equipment." Nothing in the
   planner constrains requirement scope to what was literally asked; the
   prompt only asks for "one requirement per concrete thing the final
   answer must contain," which a compliant model reasonably interprets
   expansively for an "tell me about X" phrasing in an inspection domain.
3. **Pipeline build**: 1:1 step→node mapping, terminal `verify_final`
   VERIFIER node appended, depending on all leaves.
4. **Execution, layer by layer** (`PipelineExecutor._run_node`):
   - **Researcher** node: `context = outputs_for([])` = `{}` (no
     dependencies). System prompt (`roles.py::_RESEARCHER`) explicitly
     lists `equipment_lookup` as the primary retrieval tool. Model calls
     `equipment_lookup(query_type="info")`. Final `output` text is whatever
     the model wrote — a short prose description of P-101's identity, not
     necessarily a full restatement of the raw JSON tool payload.
   - **Analyst** node: `context = outputs_for(["step_1"])` =
     `{"step_1": <researcher's final text only>}`. Critically, **the
     analyst never sees the researcher's `ToolResult` or `Evidence`
     record** — only whatever prose the researcher chose to write. The
     analyst's system prompt (`roles.py::_ANALYST`) says: *"If a specific
     fact ... is missing from context, look it up with `equipment_lookup`
     ... rather than assuming a value."* Because the researcher's prose
     summary does not literally contain every field (limits, inspection,
     maintenance), and the analyst is told to fetch rather than assume,
     the analyst calls `equipment_lookup(query_type="all")` itself. This
     is not a bug in tool-parsing or context-passing — it is the
     documented, designed behavior of the prompt plus the context
     mechanism.
   - **Executor** node: same mechanism, same tool grant
     (`_DEFAULT_TOOLS["executor"]` includes `equipment_lookup`), same
     result: calls `equipment_lookup(query_type="all")` again, seeing only
     the analyst's prose (not its tool output) as context.
   - **Summarizer** node: sees `outputs_for` of whichever nodes are its
     `depends_on` (its prose-only context is the union of upstream
     agents' text answers). It synthesizes all of it — including limits,
     inspection, and maintenance detail nobody explicitly asked for —
     because the *plan's requirements* told every step, including
     summarizer, that this information must appear in the final answer.
5. **Verification**: `_run_verifier_node` builds `combined` = concatenation
   of the leaf nodes' text outputs, gathers `ctx.state.evidence.values()`
   (the `Evidence` records — three, one per successful `equipment_lookup`
   call, redundant but real), and calls `ResultVerifier.verify(...)` with
   `requirements=ctx.requirements`. Because `extract_requirements` used the
   planner's free-text `requirements` list, **every `TaskRequirement` here
   has `source_step_id=None`** (`requirements.py:59-64`). In
   `verifier.py::verify`, the per-requirement loop computes
   `outcome = node_outcomes.get(req.source_step_id or "")` →
   `node_outcomes.get("")` → `None` → falls through to
   `_requirement_mentioned(req.description, text)`, a keyword-overlap
   heuristic requiring at least half of up to 8 four-plus-letter words from
   the requirement description to literally appear in the combined text.
   If even one planner-authored requirement is phrased in a way that
   doesn't share enough vocabulary with the summarizer's paraphrase (very
   plausible — LLM-authored requirement text and LLM-authored summary text
   are two independent paraphrases of the same idea), that requirement is
   marked `unsatisfied`, `missing_requirements` becomes non-empty, and
   `verifier.py:182-186` forces `passed = False, score <= 0.4,
   recommendation = REPLAN` — a **hard, LLM-critique-proof failure**
   (`verifier.py:205-210`: the LLM critique can never override a
   completeness gap back to a pass; confirmed by
   `test_llm_critique_cannot_override_a_completeness_gap_back_to_pass`).
6. **Result**: `ctx.task_status = compute_task_status(...)` becomes
   `INCOMPLETE` (verification failed →
   `task_status.py:53-54`), and the orchestrator prepends a deterministic
   `[INSPECTION STATUS: INCOMPLETE]` banner to the final answer
   (`orchestrator.py:192-204`).

**This is the mechanical explanation of the reported P-101 failure.** It is
not evidence of hallucination, and the factual content of the answer was
very likely correct — the failure is a **verification-heuristic gap**
triggered by a **planner scope-creep** problem (requirements broader than
the literal ask), compounded by duplicate retrieval that, while wasteful,
did not itself cause the verification failure.

---

## 4. Findings Table

| ID | Severity | Component | Finding | Evidence | Impact |
|----|----------|-----------|---------|----------|--------|
| F-01 | **HIGH** | `pipeline/requirements.py` + `verification/verifier.py` | `extract_requirements` never sets `source_step_id` when the planner supplied explicit `requirements` (the common, non-degraded path) — only the never-taken fallback path sets it. The verifier's deterministic per-requirement check (`node_outcomes.get(source_step_id)`) is therefore unreachable in normal operation; every requirement falls back to a coarse keyword-overlap heuristic. | `requirements.py:59-64` vs `:66-79`; `verifier.py:151`; confirmed by `tests/test_agents/test_planner.py:156` only exercising the fallback path and every verifier test that uses `source_step_id` constructing it by hand rather than through `extract_requirements`. | Verification pass/fail for the majority of real runs hinges on lexical overlap between two independently LLM-generated texts (the requirement description and the final summary), not on whether the producing step actually succeeded. Root-cause of the P-101 failure. |
| F-02 | **HIGH** | `agents/roles.py`, `agents/base_agent.py`, `pipeline/state.py` | Downstream agents receive only the prior node's final free-text `output` (`outputs_for`), never its structured `ToolResult`/`Evidence`. Combined with `_ANALYST`/`_EXECUTOR` prompts explicitly instructing "look it up ... rather than assuming," and identical tool grants for `equipment_lookup` across researcher/analyst/executor, this reliably reproduces duplicate retrieval. | `state.py:57-62` (`outputs_for`); `base_agent.py:289-296` (`_render_task`, text-only); `roles.py:58-72` (`_DEFAULT_TOOLS`); `roles.py:25-32` (`_ANALYST` prompt). | Every multi-step plan touching retrieval tends to re-retrieve the same data once per agent that has the tool — 2-3x redundant tool calls and LLM turns observed in the P-101 case, at proportional latency/cost. |
| F-03 | **MEDIUM** | Whole request path (`orchestrator.py`, `agents/planner.py`) | No intent/complexity classification exists anywhere. Every task, from a single-fact lookup to a multi-step investigation, is planned by the same LLM call and always produces a plan of "2-6 steps... end with a summarizer" plus a mandatory terminal verifier. | `planner.py:31-56` (`_SYSTEM`, no branching logic); `orchestrator.py:112-162` (`_run`, unconditional Planner → build_pipeline → Executor → Verifier for every task). | Simple factual lookups pay for 3-5 LLM calls + a verifier call when 1 tool call + formatting would suffice; conversely nothing guarantees a genuinely complex investigative task gets *more* rigor than a simple one — the ceiling and floor are the same. |
| F-04 | **MEDIUM** | `agents/roles.py` | `researcher`, `analyst`, `executor` are the same `LLMAgent` class, differing only by system-prompt string and a `tools` list that is 60-80% overlapping (`equipment_lookup`, `json_parse`/`calculator` etc. all appear in 2-3 roles). There is no enforced notion of "this role may only retrieve" vs "this role may only compute." | `roles.py:58-72` (`_DEFAULT_TOOLS`). | Role names describe intent, not capability. A prompt that ignores its instructions (or a model too weak to reliably follow "don't re-fetch") degrades directly into duplicate work with no structural backstop. |
| F-05 | **MEDIUM** | `pipeline/executor.py::_finalise` | The final answer is *whichever* `SUMMARIZER`-typed node completed last (or, if none, the last completed node of any type). If a plan has multiple summarizer nodes or the planner mislabels a node's `agent`, node selection for the final answer is positional/type-based, not semantically chosen. | `executor.py:315-323`. | Usually correct in practice (plans are asked to "end with a summarizer"), but nothing enforces exactly one summarizer or that it is the last node topologically; a plan with parallel summarizer branches would silently pick one over the other with no signal to the user. |
| F-06 | **MEDIUM** | `verification/verifier.py::_requirement_mentioned` | The completeness fallback heuristic is literally `>= half of up to 8 words from the requirement description appear in the final text`. This has no semantic understanding and is gameable in both directions: a summarizer that pads its text with the same nouns as the requirement (without actually answering it) passes; a correct answer phrased with different vocabulary fails. | `verifier.py:250-264`. | Directly causes both false negatives (P-101-style) and, in principle, false positives (keyword stuffing). This is the weakest link in an otherwise well-designed verification layer. |
| F-07 | **LOW** | `agents/router.py::AgentRouter` | Dead code: constructed nowhere, called nowhere in the live path. | grep across `src/` shows no `AgentRouter(` call site outside its own file/tests. | Not a bug, but a maintenance/clarity risk — a reader can mistake it for "the" router and miss that intent routing doesn't exist. |
| F-08 | **LOW** | `agents/inspection_agent.py` | `InspectionAgent`/`BaseAgent.run(state)` is a second, unused entry point (legacy `AgentState`-based contract), not wired into `Orchestrator` or the API. | No import of `InspectionAgent` in `orchestrator.py` or `api/app.py`; only used in its own test. | Same as F-07: dead code that could confuse future maintainers about where request handling begins. |
| F-09 | **LOW** | `pipeline/models.py::PipelineNode` / `builder.py` | A dependency edge means "wait for `dep` to finish" and (separately, via `outputs_for`) "receive `dep`'s text output" — but does *not* mean "receive everything `dep` retrieved." The two concepts (ordering vs. data-passing) are coupled only for the single hop directly upstream; transitively-upstream evidence (e.g. what the researcher retrieved, two hops before the summarizer if there's an intermediate analyst) is not passed down at all unless every intermediate node re-states it in its own prose. | `state.py:57-62`; `executor.py:109` (`context = ctx.state.outputs_for(node.depends_on)` — only *direct* deps). | Long plans lose information geometrically with depth: each hop is a lossy LLM paraphrase of the previous hop's paraphrase. |
| F-10 | **OBSERVATION** | `tools/registry.py`, `tools/base_tool.py` | Tool calls are *not* deduplicated or cached within an execution — identical `equipment_lookup(equipment_id="P-101", query_type="all")` calls from different nodes each hit the underlying DB independently. | `registry.py:59-60` (`call` has no memoization); confirmed no cache anywhere in `ToolRegistry`/`BaseTool`. | Low cost for a local SQLite lookup; would matter more for a network/API tool or an expensive OCR/vision call. |
| F-11 | **OBSERVATION (strength)** | `verification/verifier.py`, `pipeline/models.py` (Criticality/NodeOutcome), `pipeline/task_status.py` | `NodeStatus` (did it run) and `NodeOutcome` (did it succeed) are explicitly separated, and a `TaskStatus` computed independently of `ExecutionStatus` prevents "the DAG finished" from being reported as "the task was done." Verified end-to-end by `tests/test_integration/test_end_to_end.py::test_v101_style_scenario_cannot_produce_false_completion`. | `models.py:33-80`; `task_status.py:29-64`. | This is the single most sophisticated and correctly-implemented part of the system and should be preserved and extended, not replaced. |
| F-12 | **OBSERVATION (strength)** | `core/config.py`, `llm/factory.py`, `agents/router.py::ModelRouter` | Sovereignty is enforced at three independent layers (Settings validator, factory, and router) and *refuses* rather than silently downgrades a disallowed cloud route (`SovereigntyError`). | `config.py:136-145`; `factory.py:46-50`; `router.py:99-107`. | Correctly implemented defense in depth; no path found that would let a "sovereign" run silently reach a cloud provider. |
| F-13 | **OBSERVATION** | `tools/builtin/*` | Tools are not formally classified (retrieval vs computation vs transformation vs document-processing vs action vs verification) anywhere in code — the classification exists only informally in prose (tool descriptions, role prompt text). | `roles.py` prompts reference categories in English, not in a structured field on `BaseTool`. | A structured classification would let the planner/executor reason about "has this category of evidence already been gathered" without needing free-text inference — see §8. |

---

## 5. Intelligence Workflow Problems, Grouped

**Intent/routing** — No classification stage exists (F-03); `AgentRouter`
is unused scaffolding (F-07); `ModelRouter` only routes *provider/model*,
never *agent selection or plan shape*. The planner is simultaneously the
intent classifier, the capability-requirements deriver, and the plan
author — three concerns collapsed into one unconstrained LLM call with no
intermediate, checkable representation.

**Planning** — The planner cannot see: what tools exist, their schemas, or
which agents can use which tools (it only knows 5 hardcoded role names from
its own system prompt). It cannot know "this fact was already gathered"
because there is no execution history to consult before planning (planning
happens once, up front, from the raw task string only). It *can* produce
plans with heavily overlapping steps (nothing checks for semantic
duplication, only structural cycle/dangling-dependency validity in
`Plan._validate_steps`). Requirements are free text with no link back to
the step meant to satisfy them (F-01).

**Agent roles** — Prompt-only differentiation (F-04); tool grants overlap
substantially; nothing prevents (and prompts actively encourage) redundant
independent retrieval when upstream context is incomplete prose (F-02).

**Evidence flow** — There is a real `Evidence` model
(`state/evidence.py`), but it is populated and consumed only by the
executor (recording) and the verifier (reading via
`ctx.state.evidence.values()`); it never reaches an agent's prompt context.
Agents get prose, not evidence — so the system has no way to let a
downstream agent distinguish "this is what the tool returned" from "this is
what a previous LLM said about what the tool returned" (F-02, F-09). This
is the single biggest engineering gap: the exact primitive needed already
exists in the codebase, it's just wired to the wrong consumer.

**Tool use** — Tool schemas *are* rendered accurately and specifically to
the LLM (`base_agent.py::_render_tool_schema` renders real Pydantic JSON
schema, not just a description string — this is well done). Argument
validation is real (Pydantic) with a genuine repair loop
(`_repair_feedback`, bounded by `max_repair_attempts`). Tool errors are
distinguished by kind (`validation` / `execution` / `not_found`) and fed
back meaningfully. What's missing: role-based tool restriction, call
deduplication/caching, and a formal tool taxonomy (F-10, F-13).

**Context management** — Each node's LLM sees: its own system prompt, the
node's task description + overall goal string, and a dict of
`{dep_node_id: dep_node's_final_text}` — nothing else. No working
memory / evidence / task-context separation exists; it's one flat prompt
string built by `_render_task`. Session-level `memory/` package
(`memory_manager.py`, `in_memory.py`) exists but is not wired into the
orchestrator or executor at all (no import from either) — another
unused-but-present primitive.

**DAG semantics** — `depends_on` means both "wait for" and "receive
(lossy) text from," but only for the immediate parent, not transitively
(F-09). Parallel nodes in the same layer cannot corrupt shared state (each
writes only to its own node's `ctx.state.record`/`record_evidence`, both
keyed and additive, not overwritten) — this part is safe. Failure
propagation (a `FAILED`/`SKIPPED`/`CANCELLED` dependency skips the
dependent, cascading correctly) is correctly implemented
(`executor.py:89-100`).

**Verification** — Genuinely layered (rule-based deterministic checks →
requirement/evidence completeness gate → optional LLM critique, merged
conservatively, never overridable back to a pass by the LLM layer). The
one real defect is F-01/F-06: the completeness gate's *deterministic* path
is unreachable in normal operation and silently substitutes a weak
heuristic.

**Hallucination risk** — Structurally mitigated better than a "generic
LLM opinion" system would be: every role prompt has an explicit
"never fabricate" guardrail (`GUARDRAILS` constant, reused in
`INSPECTION_SYSTEM_PROMPT`, and echoed in `_RESEARCHER`/`_SUMMARIZER`), and
the verifier's evidence/requirement checks are deterministic, not another
LLM's opinion. What is *not* checked: whether individual sentences in the
final answer are traceable to a specific `Evidence` id (no per-claim
citation enforcement exists) — so a summarizer *can* still produce
broader/softer language than the evidence supports (e.g., turning "one
inspection record exists" into "regularly inspected") without the verifier
catching it, since the verifier checks requirement-coverage and
evidence-presence, not per-sentence grounding.

**Failure handling** — Distinct failure kinds are well modeled at the node
level (`NodeStatus`, `NodeOutcome`, `error_kind` on `ToolResult`,
`Criticality`) and at the task level (`TaskStatus`,
`ExecutionStatus`) — this is one of the strongest parts of the system (see
F-11). "Execution completed" cannot mask an incomplete task: the
orchestrator explicitly separates these two axes and annotates a degraded
answer rather than presenting it silently (`orchestrator.py:176-204`).

**Sovereignty** — Enforced at three layers, refuses rather than downgrades
(F-12). No leak path found in the reviewed code. Audit metadata is
redacted for common secret-key names (`audit/trail.py::_redact`,
`tools/base_tool.py::_sanitise`) though this is a keyword denylist, not a
schema-driven allowlist — acceptable for a local demo, worth hardening
before handling real PII/regulatory data.

**Testing** — Existing tests are strong on unit-level and one very good
targeted regression test
(`test_v101_style_scenario_cannot_produce_false_completion`) that
specifically defends the "can't fake completion" property end-to-end. What
is *not* tested: duplicate-retrieval prevention (there is no mechanism to
test), role tool-restriction (there is no restriction to test), or the
`extract_requirements` explicit-requirements path feeding into the
verifier's `source_step_id` check end-to-end (which is exactly the gap that
produced F-01 — the unit tests for `extract_requirements` and for the
verifier are both correct in isolation, but nothing tests them wired
together, which is precisely how F-01 stayed invisible).

---

## 6. Root Cause Analysis

### Symptom: P-101 verification failed despite a plausible-looking answer
- **Immediate cause**: one or more planner-authored `requirements` entries
  did not share enough vocabulary with the summarizer's final text to pass
  `_requirement_mentioned`.
- **Architectural cause**: `extract_requirements` treats "planner supplied
  explicit requirements" and "requirement derived from a step" as mutually
  exclusive (an `if explicit: return [...] ` early-return at
  `requirements.py:60-64`), so the common path never attaches
  `source_step_id`, permanently disabling the verifier's intended
  deterministic check for that path.
- **Recommended direction**: make the two mechanisms compose instead of
  branch — keep the planner's free-text requirements (they're better
  aligned with user intent than one-requirement-per-step), but have the
  planner (or a deterministic post-process step) associate each
  requirement with the step(s) expected to satisfy it, OR fall back to
  evidence-based grounding (does *any* recorded `Evidence` plausibly
  satisfy this requirement) rather than pure text keyword overlap when no
  step mapping exists.

### Symptom: equipment_lookup called 3 times for one entity
- **Immediate cause**: analyst and executor both have `equipment_lookup`
  in their tool grant, and their system prompts instruct them to look up
  facts rather than assume them when facts are "missing from context" —
  and facts *are* missing from context, because context is prose, not
  data.
- **Architectural cause**: no shared evidence store is exposed to agent
  prompts; `Evidence` records exist but are wired only to the verifier.
  There is no capability-based contract that says "retrieval for this
  entity is this step's job; downstream steps consume, they don't
  re-fetch."
- **Recommended direction**: inject upstream `Evidence` (not just text)
  into downstream agent context, keyed by tool+arguments, and/or let the
  planner mark a step's retrieved-evidence as authoritative so downstream
  prompts are told explicitly "equipment_lookup(P-101, all) was already
  called; its result is below — do not call it again unless you need a
  different equipment_id or query_type." This is deterministic prompt
  injection, not a new LLM decision.

### Symptom: simple lookups and complex investigations get identical treatment
- **Immediate cause**: `Orchestrator._run` has no branch; it always calls
  `Planner.plan` then `build_pipeline` then the full executor + verifier.
- **Architectural cause**: intent/complexity was never modeled as a
  first-class concept; the planner conflates "decide how much process this
  task needs" with "write the process."
- **Recommended direction**: see §8 (a cheap, deterministic or
  small-model pre-classification step that can *choose* a lighter path
  without removing the multi-agent path for tasks that need it).

---

## 7. What Is Actually Working Well (preserve these)

1. **Plan validation is genuinely defensive**: unknown agent → coerced to
   `custom` (not rejected, not silently dropped); dangling/self/cyclic
   dependencies rejected; duplicate ids rejected; bounded retry with a
   deterministic, always-available fallback plan (`planner.py`). This is
   real engineering, not just an LLM call.
2. **Status/outcome separation** (`NodeStatus` vs `NodeOutcome` vs
   `ExecutionStatus` vs `TaskStatus`) is a mature failure-semantics model
   most agent frameworks don't have. It correctly prevents "the code ran"
   from being reported as "the task succeeded," end-to-end, with a
   regression test proving it (F-11).
3. **The verifier's completeness gate cannot be overridden by an
   optimistic LLM critique** (`verifier.py:205-210`, tested explicitly). A
   verifier that were "just another LLM opinion" would not have this
   property; this one is architected specifically to avoid it.
4. **Tool schemas are rendered precisely** (`_render_tool_schema`) from
   real Pydantic models, with a structured repair loop
   (`_repair_feedback`) distinguishing validation failures from execution
   failures — this is above the bar of "hope the model guesses the JSON
   shape."
5. **Sovereignty enforcement is defense-in-depth and fails closed**
   (`SovereigntyError` raised, not a silent fallback) at three independent
   layers (F-12).
6. **DAG execution is layer-parallel, dependency-gated, and cascades
   failure correctly** without any shared-state corruption risk between
   concurrent nodes (each node writes only to its own keyed slot).
7. **Audit trail redacts secrets and captures a genuinely useful event
   sequence** (plan created, node started/completed/degraded/blocked,
   tool called/failed/repaired, verification completed, task status
   determined) — closer to "can answer why" than a plain log.

---

## 8. Recommended Target Architecture

Ownership note: per this project's established split, `tools/`,
`retrieval/`, and the knowledge-base implementations are owned by other
contributors; the changes below that touch those directories are flagged
so implementation ownership can be routed correctly. Orchestration/planning
changes are this audit's actionable core.

### MUST FIX
- **M1 — Fix the requirement→step linkage (F-01).** Either (a) have the
  planner emit `requirements` as objects with an optional `step_id` field
  instead of bare strings, defaulting to `None` only when genuinely
  cross-cutting, and pass that through in `extract_requirements`; or (b)
  add an evidence-grounding fallback in the verifier that checks whether
  *any* recorded `Evidence` (not just final text) plausibly satisfies an
  un-mapped requirement, before falling back to keyword overlap. (a) is
  cleaner; (b) is a smaller diff. Either removes the false-failure mode
  that caused the P-101 result.
- **M2 — Give downstream agents structured upstream evidence, not just
  prose (F-02, F-09).** Extend `outputs_for`/`_render_task` (or add a
  parallel `evidence_for(node.depends_on)` call already present on
  `PipelineState`, currently unused by the executor's agent-context path)
  so an agent's prompt includes a compact rendering of upstream
  `Evidence` records (tool, arguments, key output fields), not only the
  upstream node's final paraphrase. This is the direct fix for duplicate
  `equipment_lookup` calls, and it is additive: `PipelineState.evidence_for`
  already exists and is already unused for this purpose
  (`state.py:53-55`).

### SHOULD FIX
- **S1 — Add a cheap pre-planning classification step (F-03).** Not a
  removal of the multi-agent path — an additional, fast, deterministic-or-
  small-model step that decides whether a task is a direct lookup
  (skip planning, call the obvious tool, format, optional lightweight
  verification) versus needing the full Planner→DAG→Verifier path. Keep
  the full path as the default/fallback when classification is unsure —
  do not make misclassification silently drop rigor from a task that
  needed it.
- **S2 — Make tool grants reflect intended role boundaries (F-04).**
  Reduce `equipment_lookup`-style overlap between researcher/analyst/
  executor, or — better, since analysts/executors legitimately sometimes
  need a fact the researcher didn't fetch — keep the grant but change the
  prompt guidance from "look it up yourself" to "look it up only if it is
  not already present in the evidence provided," now that M2 makes
  evidence actually visible.
- **S3 — Formal tool taxonomy (F-13).** Add a `category` field to
  `BaseTool` (retrieval / computation / transformation / document /
  action / verification). Low effort, and lets S1/S2's routing logic be
  written against categories instead of tool names. (Touches `tools/`,
  route through the owning contributor.)
- **S4 — Wire the existing `memory/` package or remove it.** It's fully
  built (`memory_manager.py`, `in_memory.py`) but never imported by the
  orchestrator/executor — decide deliberately rather than leaving working
  code silently unused.

### NICE TO HAVE
- **N1 — Deduplicate identical tool calls within one execution** (F-10):
  low-value for local SQLite lookups, higher value once network/vision
  tools are in the hot path.
- **N2 — Per-claim evidence citation in the summarizer's output** (a
  stronger hallucination defense than requirement/evidence-presence
  checks alone).
- **N3 — Remove or clearly mark `AgentRouter` (F-07) and
  `InspectionAgent`/`BaseAgent.run` (F-08) as legacy/dead, to stop future
  readers from mistaking them for the live request path.

---

## 9. Implementation Plan

### M1 — Requirement → step linkage
- **Files to modify**: `src/agents/planner.py` (extend `_SYSTEM` prompt +
  `Plan.requirements` schema to allow `{"text": str, "step_id": str|null}`
  objects, backward-compatible with plain strings), `src/pipeline/
  requirements.py` (`extract_requirements`: read `step_id` when present).
- **New interface**: `TaskRequirement.source_step_id` already exists — no
  new field needed, just populate it from the richer planner output.
- **Fallback for un-mapped requirements**: in `src/verification/
  verifier.py`, before falling back to `_requirement_mentioned`, add an
  evidence-grounding check: does any `Evidence.content` for the relevant
  producer tool look consistent with the requirement description (simple
  keyword-against-evidence-content check, not just keyword-against-final-
  text). This is a strictly-better fallback, not a new heuristic axis.
- **Expected behavioural change**: P-101-style tasks stop failing
  verification purely due to paraphrase mismatch; genuinely unmet
  requirements still fail.
- **Risks**: planner prompt change could regress JSON-validity rate;
  mitigate by keeping the plain-string form accepted (`step_id` optional).
- **Tests required**: extend `tests/test_agents/test_planner.py` for the
  object form; extend `tests/test_pipeline/test_builder.py`/
  `tests/test_verification/test_verifier.py` for an
  `extract_requirements(plan_with_step_ids) -> verifier` integration test
  (this exact seam is what let F-01 go undetected).

### M2 — Structured evidence into downstream context
- **Files to modify**: `src/pipeline/executor.py` (`_run_node`: build
  `context` from both `outputs_for` and `ctx.state.evidence_for
  (node.depends_on)`), `src/agents/base_agent.py`
  (`_render_task`/`LLMAgent.execute`: accept and render an `evidence`
  section distinctly from `context` text, e.g. `EVIDENCE ALREADY
  RETRIEVED:` block per Evidence record).
- **New interface**: `Agent.execute(..., evidence: list[Evidence] |
  None = None)` — additive, does not break the existing `context` param
  or any current caller.
- **Expected behavioural change**: analyst/executor nodes stop
  re-calling `equipment_lookup` for data already fetched upstream, because
  the data is now visible as structured fact, not implied by a
  paraphrase.
- **Risks**: prompt length growth for deep/wide plans with many upstream
  evidence records — bound with a per-node evidence-item cap or summarize
  large payloads (e.g. long inspection-history lists) before injection.
- **Tests required**: an executor-level test asserting that a second
  node's rendered prompt contains the first node's `Evidence.content`,
  and a scripted-LLM integration test (following the pattern of
  `_ScriptedIncidentLLM` in `test_end_to_end.py`) asserting a model that
  is instructed "don't re-fetch what's already provided" does not
  duplicate the tool call when evidence is present — this can't fully
  prove the *model* won't re-call, but it proves the *evidence is there
  for it to use*, which is the fixable half of the problem.

### S1 — Pre-planning classification
- **Files to create**: `src/agents/intent_classifier.py` (or extend
  `agents/router.py`'s currently-unused `AgentRouter` into this role,
  reusing the name rather than adding a third "router" concept).
- **Files to modify**: `src/orchestrator.py::_run` (branch: classify →
  if "direct lookup" and confidence high, run a single-tool-call fast
  path with a lightweight verification; else the existing full path
  unchanged).
- **Expected behavioural change**: "tell me about P-101" costs 1-2 LLM
  calls instead of 5; "should P-101 be taken out of service" is
  unaffected.
- **Risks**: misclassification of a genuinely complex request as simple
  under-serves it — mitigate by defaulting to the full path whenever
  classification confidence is low, and by keeping the fast path's own
  lightweight verification able to escalate back to the full path (not
  simply return a worse answer).
- **Tests required**: TEST 1 and TEST 9 from the audit brief (simple
  lookup does not trigger full multi-agent execution; complex task still
  gets full planning).

*(S2-S4 and N1-N3 are smaller, single-file changes; omitted here for
brevity but follow the same pattern — each should ship with the specific
regression test from §16 of the audit brief that it addresses.)*

---

## 10. P-101 Execution Diagnosis

- **Why did multiple agents call `equipment_lookup`?** Because tool grants
  overlap across roles (F-04) and downstream agents only receive the
  previous step's prose summary, not its structured data (F-02) — so the
  analyst and executor each independently conclude, correctly *given their
  prompt and the information they actually have*, that they need to fetch
  the data themselves. This is not agents ignoring their dependencies (the
  dependency graph and text context were correctly delivered), not broken
  context propagation (the wiring works exactly as designed), and not the
  executor mis-constructing context — it is a genuine information-loss gap
  between what the pipeline *tracks* (structured `Evidence`) and what it
  *shows* agents (text only).
- **Was this a bug?** The redundant retrieval is an **architectural
  weakness / inefficiency**, not an implementation bug — every component
  did exactly what its code says it does. The one genuine **bug** in this
  trace is F-01 (`extract_requirements` never linking requirements to
  steps on the common path), which is what actually failed verification.
- **Did the summarizer over-answer?** Yes, but by design-following, not
  malfunction: the plan's `requirements` (generated by the planner LLM)
  evidently asked for more than the user's literal question, and the
  summarizer correctly satisfied the plan it was given. The real gap is
  upstream — the planner has no mechanism to keep requirement scope tied
  to the literal request for a narrow factual question (this is the same
  root cause as F-03: no request-type/intent discipline).
- **Why did the verifier fail?** Almost certainly F-01: with
  `source_step_id=None` on every requirement, the deterministic
  step-outcome check never engages, and the keyword-overlap fallback
  rejected a requirement whose wording didn't sufficiently overlap with
  the final paraphrase — not because information was missing or wrong.
- **Was the final factual response actually incorrect?** Nothing in the
  traced logic suggests factual incorrectness. `equipment_lookup` is a
  deterministic local DB read (`db_tool.py`); three redundant successful
  calls to it produce identical, correct data each time. The verification
  failure is a **process/heuristic** failure, not a **correctness**
  failure — this is an important distinction the incident report's framing
  ("verifier failed... despite... broadly consistent" information)
  correctly anticipated.
- **What should the optimal execution path have looked like?** For "Tell
  me about P-101. What is the equipment?" specifically: one retrieval call
  (`equipment_lookup(P-101, info)`, possibly `limits` if "about" is read
  generously), one formatting pass, and a lightweight completeness check —
  not a 4-node plan with a full LLM-critique verifier. That is exactly the
  gap S1 (pre-planning classification) is meant to close; it is a
  legitimate architectural improvement, not evidence that the multi-agent
  path itself is wrong for the tasks that actually need it.
