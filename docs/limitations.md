# Limitations and interpretation boundary

1. **One model family.** The four evaluated systems are Qwen3 parameter sizes,
   not independent model families. Results support claims about these runs,
   not all large language models.
2. **Rule-based scoring.** Scores come from a deterministic operationalization
   of the original project rubric. Pattern-based classification may not capture
   every nuance of free-form model responses.
3. **Context-specific synthetic strings.** Synthetic strings were screened as
   nonce strings for their matched passages. The study does not claim that a
   letter sequence has never been used anywhere.
4. **Three-condition design.** There is no synthetic low-context condition.
   The study estimates the context effect for real abbreviations and compares
   real versus synthetic strings under high context; it does not estimate a
   complete two-by-two interaction.
5. **Provider dependence.** Actual model implementations and provider routing
   may change. The manifests document the observed provider and model IDs at
   collection time.
6. **Prompt construction.** Although matched high-context prompts differ only
   in the target string, item selection and screening decisions may still
   affect generalizability.
7. **Rubric operationalization.** "Ideal response" is condition-specific and
   combines interpretation accuracy with calibration behavior. It should not
   be interpreted as generic model accuracy.
8. **No user study.** The experiment evaluates model outputs against a rubric;
   it does not measure how people perceive, trust, or act on those outputs.
9. **Protocol deviation.** The exact collected prompts omitted the planned
   sentence asking models to state whether an answer was an established
   expansion or contextual inference. The calibration results therefore
   measure spontaneous distinction, not compliance with an explicit request.
