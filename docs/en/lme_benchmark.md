# LongMemEval (LME) Benchmark

ReMe ships two ReMe-native steps that compose into a [LongMemEval](https://github.com/xiaowu0162/longmemeval)-style
benchmark loop: a direct **answer from session context** step, and an **LLM-judge** step that scores that answer
against the dataset's golden answer. Together with the rest of ReMe's auto-memory / search primitives, they let you
plug ReMe into the official LME harness the same way you would any other tool-equipped service.

```{note}
The example below assumes a service started with `reme start config=reme/config/jinli_lme.yaml`. The benchmark loop
itself is not part of `jinli_lme.yaml`; it lives in the LME harness repo. This page only documents the two ReMe-side
steps and their input/output contract.
```

## Steps at a glance

| Step name | What it reads | What it writes |
|---|---|---|
| `context_answer_step` | `query`, `session_context`, `current_date` | `context_answer` |
| `answer_judge_step` | `query`, `agent_answer`, `golden_answer`, `question_type` | `answer_judgement` (`yes` / `no`) |

Both steps require an `agent_wrapper` (the LLM that performs the answering / judging). They are stateless wrappers
over the wrapper's `reply` call — ReMe does not perform any extra retrieval, only formatting.

## `context_answer_step`

Answers a question directly from the supplied session context (i.e. the conversation history for one dialogue
session in the LME dataset). This mirrors the "direct" baseline in the LongMemEval paper: the model sees only the
history, not the agent's memory store.

Required inputs (from `RuntimeContext`):

```yaml
query: string                # the question being answered
session_context: string      # concatenated history for one session
current_date: string         # `YYYY-MM-DD` of "today" so relative phrases resolve
```

Produces:

```yaml
context_answer: string       # model's free-form answer
metadata:
  query: ...
  session_context: ...
  current_date: ...
  context_answer: ...
```

The default user message template (`user_message`) asks the model to extract relevant facts first and then reason
over them, which matches the public LME prompt.

## `answer_judge_step`

Scores an `agent_answer` against a `golden_answer` with one of four LLM judges. The judge prompt is selected
automatically from `question_type`:

| `question_type` (after normalization) | Judge prompt key |
|---|---|
| `temporal_reasoning` | `temporal_reasoning_system_prompt` |
| `knowledge_update` | `knowledge_update_system_prompt` |
| `single_session_preference` | `single_session_preference_system_prompt` (uses the *preference* rubric user template) |
| anything else | `other_question_types_system_prompt` |

Normalization lowercases the string and replaces `-` and spaces with `_`, so `Temporal-Reasoning`,
`temporal reasoning`, and `TEMPORAL_REASONING` all hit the same judge. Unknown values fall through to the
generic `other_question_types_system_prompt`, which is the right behavior for LME categories other than the
three handled above.

The raw judge output is normalized with a case-insensitive `^\s*(yes|no)\b` match. If that match succeeds, the
final `answer_judgement` is `yes` or `no`; otherwise the lowercased first token of the raw output is kept as-is
(this lets the benchmark harness treat unexpected output as a miss rather than over-reporting).

Required inputs:

```yaml
query: string
agent_answer: string        # typically `context_answer` from the previous step
golden_answer: string       # from the LME dataset
question_type: string       # one of the LME categories above
```

Produces:

```yaml
answer_judgement: string    # `yes` / `no` (or raw lowercased token if non-conforming)
metadata:
  query: ...
  agent_answer: ...
  golden_answer: ...
  question_type: ...
  answer_judgement: ...
  raw_answer_judgement: ... # original model output, useful for debugging mismatched judgements
```

## Wiring the steps into a job

The shipped `jinli_lme.yaml` defines only the standard jobs (`update_index`, `auto_memory`, `version`, `search`)
plus a Tokenizer / Embedding / LLM / Agent stack. To make the two benchmark steps callable from the LME harness,
add a job that bundles them. This stanza can sit in a separate YAML or be appended to `jinli_lme.yaml`:

```yaml
jobs:
  lme_one_question:
    backend: base
    description: "Run context_answer_step + answer_judge_step for a single LME question."
    parameters:
      type: object
      properties:
        query:           {type: string}
        session_context: {type: string}
        current_date:    {type: string}
        agent_answer:    {type: string}
        golden_answer:   {type: string}
        question_type:   {type: string}
      required: [query, session_context, agent_answer, golden_answer, question_type]
    steps:
      - backend: context_answer_step
      - backend: answer_judge_step
```

Invoking it from the LME harness is then just another POST request:

```bash
reme app config=jinli_lme.yaml lme_one_question \
    query="When did Jon lose his job?" \
    session_context="2023-01-19: ..." \
    current_date="2024-03-01" \
    agent_answer="..." \
    golden_answer="..." \
    question_type="temporal_reasoning"
```

## Notes for benchmark authors

- Both steps raise `ValueError` for missing required context fields — the harness should treat that as a
  per-question failure with `question_id` attached, not as a service-level error.
- `question_type` *must* be normalized on the harness side if the LME dataset emits titles like
  `temporal-reasoning`; the step itself only does the safe fallback normalization.
- `answer_judgement` is the field to score against the LME reference scoring script. `raw_answer_judgement` is
  kept for diagnostics when the judge model hallucinates a non-yes/no token.
- The benchmark loop should also keep the per-question metadata captured by both steps (especially
  `session_context` length and `current_date`) so future time-handling refactors can be re-evaluated against
  the same inputs.
