# Data dictionary and release notes

## Prompts

`prompts/v3_900_prompts.jsonl` contains one record per prompt.

Important fields:

- `prompt_id`: unique prompt identifier.
- `set_id`: matched abbreviation-set identifier.
- `condition`: `real_low_context`, `real_high_context`, or
  `synthetic_high_context`.
- `domain`: analysis domain.
- `prompt`: exact submitted prompt text.
- `real_abbreviation` and `synthetic_abbreviation`: matched strings.
- `target_expansion`: intended interpretation.
- `alternate_meanings`: plausible alternatives for the low-context condition.

## Responses

Each file under `responses/` contains the 900 successful responses for one
model. Provider request IDs and generation IDs were removed from the public
copy. Prompt text, completion text, model/provider identifiers, timestamps,
token counts, timing measurements, prompt hashes, and completion status were
retained.

## Grades

`grades/original_rubric_scores.jsonl` contains 3,600 deterministic scores made
with the unchanged original project rubric. The public scoring script includes
40 fixed edge-case corrections under the same scoring definitions.

## Validation

The files under `validation/` document automated quality checks and row-level
benchmark changes. Final synthetic strings are stored in the prompt dataset.

## Counts

- 300 matched sets
- 900 prompts
- 4 models
- 3,600 responses
- 1,200 responses per condition across models

## License

See `../DATA_LICENSE.md`. Third-party references and model outputs may have
additional terms.
