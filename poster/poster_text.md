# Can Large Language Models Tell Knowledge from Context?

## Testing established abbreviation knowledge and contextual inference in Qwen3 models

**Sumanth Kaja • [Department or Program] • [Institution] • RISE Symposium 2026**

## Key finding

Qwen3 models interpreted real abbreviations accurately when context was
informative, but almost never spontaneously recognized that plausible
expansions of synthetic letter sequences were inferences rather than
established meanings.

## Background

Large language models can use surrounding information to determine what an
abbreviation probably means. However, generating a plausible interpretation
is not the same as retrieving an abbreviation's established expansion. A model
may infer the intended meaning from context while incorrectly claiming that a
synthetic letter sequence conventionally stands for that meaning.

This distinction is a form of epistemic calibration: can a model accurately
communicate whether its answer comes from established knowledge or from
contextual inference? This matters because confident but unsupported
explanations can mislead users, especially in medicine, law, business, and
technology.

## Research question

When context strongly suggests a particular meaning, can a large language
model distinguish a real abbreviation from a synthetic abbreviation-like
string whose apparent meaning exists only in that context?

We predicted that models would correctly expand real abbreviations when given
informative context but would often assign the same expansion to synthetic
strings without clearly identifying it as an inference.

## Methods

The benchmark contained 300 matched abbreviation sets drawn from five domains:
medicine, law/finance/business, sports, general language/slang, and
software/technology. Each set produced three prompts:

1. **Real abbreviation, low context:** A genuine abbreviation with insufficient
   information to select one meaning confidently.
2. **Real abbreviation, high context:** The same abbreviation with information
   supporting its intended expansion.
3. **Synthetic string, high context:** The real abbreviation was replaced with
   a screened, abbreviation-like sequence while the rest of the passage
   remained unchanged.

Every prompt ended with the same question: "What does [abbreviation] mean
here?" The models were not explicitly told that some strings were synthetic or
instructed to label an answer as established knowledge versus contextual
inference. The experiment therefore measured whether they made that distinction
spontaneously.

Four Qwen3 models—4B, 8B, 14B, and 32B—each answered all 900 prompts, producing
3,600 responses. Prompts were submitted independently without web search or
external tools. Responses received rubric scores measuring interpretation
accuracy and whether the model correctly described the source of its answer.

## Results

Across the four models, ideal-response rates were 39.4% for real abbreviations
with low context, 93.2% for real abbreviations with high context, and 0.5% for
synthetic strings with high context.

Informative context greatly improved performance on genuine abbreviations.
Depending on model size, ideal-response rates increased by approximately 48–58
percentage points between the real low-context and real high-context
conditions.

Performance collapsed when the same informative passages contained synthetic
strings. All four models showed a difference of more than 91 percentage points
between the real-high and synthetic-high conditions. Approximately 51.7% of
synthetic-condition responses presented a context-derived expansion
definitively, while only 6 of 1,200 responses fully avoided the unsupported
assumption.

Increasing model size did not eliminate this failure. The 32B model performed
strongly on real high-context abbreviations but achieved an ideal response on
only 0.3% of synthetic high-context prompts.

## Interpretation

The models were highly capable of using context to generate a fitting
interpretation, but they rarely distinguished that inference from established
abbreviation knowledge. Fluent, contextually appropriate answers may therefore
conceal an important calibration error.

## Limitations and future work

The present analysis is limited to one model family. Responses were scored
with the original project rubric: Accuracy for all conditions, Clarification
for real-low-context prompts, and Hallucination for synthetic-high-context
prompts. The collected prompts omitted a planned instruction asking models to
label their knowledge source, so the results measure spontaneous calibration.
Future work should compare multiple model families and directly compare
prompted versus unprompted calibration.

## Conclusion

Strong contextual reasoning does not necessarily imply reliable knowledge
attribution. In this benchmark, Qwen3 models usually interpreted informative
passages correctly but frequently converted contextual clues into unsupported
claims about what synthetic strings "stand for."

## Layout recommendation

Use three columns. Put the background, research question, and matched example
on the left; methods and the main bar chart in the center; and results,
interpretation, limitations, and conclusion on the right. Use gray for real
low context, blue for real high context, and orange for synthetic high context.
Place `github_repo_qr.png` in the lower-right corner with the label "Data,
responses, code, and results."
