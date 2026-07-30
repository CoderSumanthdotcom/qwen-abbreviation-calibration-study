# Study methods

## Research question

This study asks whether Qwen3 models distinguish an established abbreviation
expansion from a meaning that is supported only by the surrounding context.

## Matched benchmark

The released dataset contains 300 matched sets spanning medicine,
law/finance/business, sports, general language/slang, and
software/technology. Each set contributes three prompts:

1. `real_low_context`: a real abbreviation appears without enough information
   to select one expansion confidently.
2. `real_high_context`: the real abbreviation appears in context that supports
   the intended established expansion.
3. `synthetic_high_context`: the high-context passage is retained, but the real
   abbreviation is replaced by a screened synthetic letter sequence.

This produces 900 prompts. Each of four Qwen3 models answered every prompt in
an independent request, producing 3,600 responses.

## Prompt wording

Every collected prompt ended with:

> What does [ABBREVIATION] mean here?

The exact released dataset is authoritative. A longer planned instruction was
not present in the collected prompts; `deviations.md` explains the resulting
interpretation boundary.

## Model conditions

The public configurations and run manifests document the exact requested and
observed model routes. The primary settings were temperature 0, top-p 1, and
disabled thinking/reasoning output. Web search, retrieval, code execution, and
external tools were unavailable to the models.

## Grading

All reported results use the original project rubric reproduced verbatim in
`grading_rubric.md`:

- Accuracy (0–2) is scored in all three conditions.
- Clarification (0–2) is scored only for `real_low_context`.
- Hallucination (0–2) is scored only for `synthetic_high_context`.

The public scoring script deterministically maps each response to those
metrics. Forty fixed edge-case corrections, listed directly in the script,
apply the same rubric definitions. No alternate grading rubric is used.

For the poster summary, an ideal response is a response with Accuracy score 2
under the condition-specific rubric definition. A definitive unsupported
assignment is a synthetic-high-context response with Hallucination score 2.

## Analysis

Descriptive results are reported by model and condition. Matched comparisons
use the abbreviation set as the experimental unit. The repository includes
paired contrasts, domain summaries, confidence intervals, figures, exact input
hashes, and deterministic analysis code.
