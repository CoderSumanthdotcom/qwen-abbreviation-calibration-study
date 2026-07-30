# Prompt and synthetic-string quality assurance

The public quality-assurance records preserve the benchmark's construction
history:

- `prompt_standardization_changes.jsonl` records the initial conversion to the
  standardized V3 response request.
- `article_agreement_changes.jsonl` records article-agreement repairs.
- `context_remediation_changes.jsonl` records row-level prompt corrections.
- `validation_summary.json` records final automated counts and checks.

The final synthetic strings are stored directly in every matched prompt record.
The construction standard was context-specific: a string was unsuitable when
an established meaning could plausibly fit the matched passage, when it was a
recognizable word or slang term, or when it created a transparent typo cue.
An unrelated external code or abbreviation was not automatically
disqualifying.

See `study_methods.md` for the inclusion and matching rules used in the
reported experiment.
