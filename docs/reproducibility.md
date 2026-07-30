# Reproducibility

## Published experimental unit

The primary experimental unit is the matched abbreviation set, not the
individual prompt. The benchmark contains 300 sets and three conditions per
set, for 900 prompts per model.

## Environment

- Python 3.11 or newer
- Dependencies declared in `pyproject.toml`
- One fresh request per prompt
- Web search, retrieval, code execution, and external tools disabled
- Temperature 0
- Top-p 1
- Thinking/reasoning output disabled

Exact provider routes, model identifiers, seeds, and run dates are recorded in
`configs/` and `data/responses/manifests/`.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,analysis]"
python -m unittest discover -s tests -v
```

## Validate a collection configuration without network access

```bash
python -m qwen_abbrev_study --config configs/qwen3_4b.json --dry-run
```

Dry run is the default. A live run requires an explicit `--execute` flag and
the credential named by the selected config.

## Regenerate the reported analysis

```bash
node scripts/score_responses_original_rubric.mjs
python3 scripts/analyze_original_rubric_results.py
```

The Node script deterministically regenerates scores using the unchanged
original rubric in `docs/grading_rubric.md`. Its 40 fixed edge-case corrections
apply those same definitions. The Python script validates row counts and
matched-set completeness before writing the tables and figures under
`results/`. Its deterministic bootstrap seeds are defined in the script.

## Integrity

The analysis summary records SHA-256 hashes for the prompt dataset and grading
input. Run manifests also record the dataset and configuration hashes used for
collection.

## Scoring authority

`docs/grading_rubric.md` is the sole grading definition for this release.
Accuracy is scored for every condition, Clarification only for
`real_low_context`, and Hallucination only for `synthetic_high_context`.
