#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const DATASET_PATH = path.join(
  ROOT,
  "data",
  "prompts",
  "v3_900_prompts.jsonl",
);
const OUTPUT_PATH = path.join(
  ROOT,
  "data",
  "grades",
  "original_rubric_scores.jsonl",
);

const RUNS = [
  {
    model: "qwen3_4b",
    path: path.join(
      ROOT,
      "data",
      "responses",
      "qwen3_4b_responses.jsonl",
    ),
  },
  {
    model: "qwen3_8b",
    path: path.join(
      ROOT,
      "data",
      "responses",
      "qwen3_8b_responses.jsonl",
    ),
  },
  {
    model: "qwen3_14b",
    path: path.join(
      ROOT,
      "data",
      "responses",
      "qwen3_14b_responses.jsonl",
    ),
  },
  {
    model: "qwen3_32b",
    path: path.join(
      ROOT,
      "data",
      "responses",
      "qwen3_32b_responses.jsonl",
    ),
  },
];

// Fixed edge-case corrections apply the same original rubric documented in
// docs/grading_rubric.md. They do not introduce different metrics or rules.
const RUBRIC_EDGE_CASE_CORRECTIONS = new Map([
  // real_high_context: semantically correct variants or complete explanations
  [46, { accuracy: 2 }],
  [47, { accuracy: 2 }],
  [48, { accuracy: 2 }],
  [149, { accuracy: 2 }],
  [150, { accuracy: 2 }],
  [151, { accuracy: 2 }],
  [152, { accuracy: 2 }],
  [421, { accuracy: 1 }],
  [533, { accuracy: 2 }],
  [534, { accuracy: 2 }],
  [541, { accuracy: 2 }],
  [542, { accuracy: 2 }],
  [543, { accuracy: 2 }],
  [544, { accuracy: 2 }],
  [669, { accuracy: 2 }],
  [671, { accuracy: 2 }],
  [1062, { accuracy: 2 }],
  [1069, { accuracy: 2 }],
  [1070, { accuracy: 2 }],
  [1071, { accuracy: 2 }],
  [1302, { accuracy: 2 }],
  [1421, { accuracy: 2 }],
  [1423, { accuracy: 2 }],
  [1512, { accuracy: 2 }],
  [1758, { accuracy: 2 }],
  [1912, { accuracy: 2 }],
  [1920, { accuracy: 2 }],
  [2352, { accuracy: 2 }],

  // synthetic_high_context: distinguish refusing the synthetic term from
  // proposing a tentative invented meaning or a likely correction.
  [2662, { accuracy: 2, hallucination: 0 }],
  [2838, { accuracy: 1, hallucination: 1 }],
  [3006, { accuracy: 1, hallucination: 1 }],
  [3082, { accuracy: 2, hallucination: 0 }],
  [3083, { accuracy: 2, hallucination: 0 }],
  [3109, { accuracy: 2, hallucination: 0 }],
  [3110, { accuracy: 2, hallucination: 0 }],
  [3114, { accuracy: 1, hallucination: 1 }],
  [3294, { accuracy: 0, hallucination: 1 }],
  [3415, { accuracy: 1, hallucination: 1 }],
  [3418, { accuracy: 0, hallucination: 1 }],
  [3512, { accuracy: 2, hallucination: 0 }],
]);

function loadJsonl(filePath) {
  return fs
    .readFileSync(filePath, "utf8")
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function normalize(text) {
  return String(text ?? "")
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase()
    .replace(/[*_`]/g, "")
    .replace(/[#>|()[\]{}]/g, " ")
    .replace(/[–—-]/g, " ")
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function significantTokens(text) {
  const stop = new Set([
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
  ]);
  return normalize(text)
    .split(" ")
    .filter((token) => token.length > 1 && !stop.has(token));
}

function tokenStem(token) {
  if (token.length > 5 && token.endsWith("ies")) return `${token.slice(0, -3)}y`;
  if (token.length > 5 && token.endsWith("ses")) return token.slice(0, -2);
  if (token.length > 4 && token.endsWith("es")) return token.slice(0, -1);
  if (token.length > 3 && token.endsWith("s") && !token.endsWith("ss")) {
    return token.slice(0, -1);
  }
  return token;
}

function includesTarget(response, target) {
  const normalizedResponse = normalize(response);
  const candidates = [
    target,
    String(target).replace(/\s*\([^)]*\)\s*/g, " ").trim(),
    ...(String(target).match(/\(([^)]*)\)/g) ?? []).map((value) =>
      value.slice(1, -1),
    ),
  ].filter(Boolean);
  if (candidates.some((candidate) => normalizedResponse.includes(normalize(candidate)))) {
    return true;
  }
  const normalizedTarget = normalize(candidates[0]);
  if (!normalizedTarget) return false;
  const responseTokens = significantTokens(normalizedResponse).map((token) =>
    tokenStem(token),
  );
  const tokens = significantTokens(target).map((token) => tokenStem(token));
  const contiguousMatch =
    tokens.length > 0 &&
    responseTokens.some((_, index) =>
      tokens.every((token, offset) => responseTokens[index + offset] === token),
    );
  return contiguousMatch || semanticTargetEquivalent(normalizedResponse, normalizedTarget);
}

function semanticTargetEquivalent(response, target) {
  const equivalents = new Map([
    ["primary care provider", ["primary care physician", "primary care doctor"]],
    ["medical doctor", ["doctor of medicine"]],
    ["operating expenditures", ["operational expenditure", "operating expense"]],
    ["non relational database approach", ["non relational database", "nosql database"]],
    ["short handed goal", ["shorthanded goal"]],
    ["got to go", ["gotta go"]],
    ["best friends forever", ["best friend forever"]],
    ["win loss record", ["wins and losses", "wins losses"]],
    ["disqualified", ["disqualification"]],
  ]);
  return (equivalents.get(target) ?? []).some((value) =>
    response.includes(normalize(value)),
  );
}

const contextRecognitionPatterns = [
  /\bambiguous\b/i,
  /\bmultiple meanings?\b/i,
  /\bseveral meanings?\b/i,
  /\bvarious meanings?\b/i,
  /\bmore information\b/i,
  /\badditional information\b/i,
  /\bmore details?\b/i,
  /\badditional details?\b/i,
  /\bnot enough information\b/i,
  /\bwithout (?:more|additional|specific|further) (?:context|information|details?)\b/i,
  /\b(?:more|additional|specific|further) context\b/i,
  /\bcontext (?:is|remains) (?:unclear|ambiguous|needed|necessary|required|missing)\b/i,
  /\bneed(?:ed|s)? (?:more|additional|specific|further) (?:context|information|details?)\b/i,
  /\brequires? (?:more|additional|specific|further) (?:context|information|details?)\b/i,
  /\bdepends? on\b/i,
  /\bdepending on (?:the )?(?:context|field|domain|industry|situation|usage)\b/i,
  /\bcannot (?:be )?determine(?:d)?\b/i,
  /\bcan(?:not|'t) (?:know|tell|determine|say)\b/i,
  /\bunclear\b/i,
];

const explicitAskPatterns = [
  /\b(?:can|could|would|will) you (?:provide|share|give|clarify|tell|specify|confirm|explain)\b/i,
  /\bif you (?:can )?(?:provide|share|give|clarify|tell|specify|confirm|explain)\b/i,
  /\blet me know (?:the|which|what) (?:context|field|domain|industry|sentence|source|setting|subject)\b/i,
  /\blet me know more about (?:the )?(?:context|field|domain|industry|sentence|source|setting|subject)\b/i,
  /\bplease (?:provide|share|give|clarify|specify|confirm|check|ask|consult|look for)\b/i,
  /\b(?:provide|share|give|seek|ask for|get|obtain) (?:more|additional|further|specific) (?:context|information|details?)\b/i,
  /\bask (?:the user|for clarification|for more context|for more information)\b/i,
  /\bwhat (?:is the|kind of|type of) context\b/i,
  /\bwhich (?:context|field|domain|industry|meaning)\b/i,
  /\bdo you mean\b/i,
  /\bcheck (?:the|its) context\b/i,
];

const uncertaintyPatterns = [
  /\bmay\b/i,
  /\bmight\b/i,
  /\bcould\b/i,
  /\bpossibly\b/i,
  /\bperhaps\b/i,
  /\blikely\b/i,
  /\bprobably\b/i,
  /\bpotentially\b/i,
  /\bappears? to\b/i,
  /\bseems? to\b/i,
  /\bI (?:suspect|would guess)\b/i,
  /\bnot (?:entirely )?sure\b/i,
  /\buncertain\b/i,
];

const meaningProposalPatterns = [
  /\b(?:could|may|might|likely|probably|possibly|perhaps) (?:stand for|mean|refer to|represent|denote|signify|be short for|be an acronym for)\b/i,
  /\bmost likely (?:stands? for|means?|refers? to|represents?|is)\b/i,
  /\bpossible (?:meaning|interpretation|expansion)s?\b/i,
  /\b(?:possible|plausible|likely) (?:interpretation|meaning|expansion)s?\b/i,
  /\b(?:one|some|a few|several) (?:possible|plausible) (?:interpretation|meaning|expansion)s?\b/i,
  /\bhere are (?:some|a few|several) (?:possible|plausible) (?:interpretation|meaning|expansion)s?\b/i,
  /\blikely (?:a|an|the) (?:variant|name|term|code|label|shorthand|reference)\b/i,
  /\bmay be (?:a|an|the) (?:variant|name|term|code|label|shorthand|reference)\b/i,
  /\b(?:is|are|was|were) (?:most )?likely (?:referring to|a|an|the)\b/i,
  /\b(?:it|this|the term|the abbreviation|the acronym) (?:typically |likely |most likely )?(?:stands? for|means?|refers? to|represents?|denotes?|signifies?)\b/i,
  /\bone possibility is\b/i,
  /\bthe most (?:probable|likely) (?:meaning|interpretation|intended term)\b/i,
  /\bcan be interpreted (?:as|to mean)\b/i,
];

const unknownAbbreviationPatterns = [
  /\bnot (?:a )?(?:known|recognized|recognised|standard|common|established|valid|widely used|widely recognized|widely recognised|conventional|familiar) (?:term|abbreviation|acronym|initialism)\b/i,
  /\bdoes not (?:appear|seem) to be (?:a )?(?:known|recognized|recognised|standard|common|established|valid|widely used|widely recognized|widely recognised) (?:term|abbreviation|acronym|initialism)\b/i,
  /\bdoesn['’]?t (?:appear|seem) to be (?:a )?(?:known|recognized|recognised|standard|common|established|valid|widely used|widely recognized|widely recognised) (?:term|abbreviation|acronym|initialism)\b/i,
  /\bno (?:known|recognized|recognised|standard|common|established|valid|widely used|widely recognized|widely recognised|conventional) (?:meaning|term|abbreviation|acronym|initialism|definition)\b/i,
  /\bnot commonly (?:used|recognized|recognised|known)\b/i,
  /\bnot widely (?:used|recognized|recognised|known)\b/i,
  /\bnot listed as\b/i,
  /\bnot found in\b/i,
  /\blikely (?:a )?(?:typo|mistake|error|misprint|fabricated|made up|invented)\b/i,
  /\bmay be (?:a )?(?:typo|mistake|error|misprint|fabricated|made up|invented)\b/i,
  /\bappears? to be (?:a )?(?:typo|mistake|error|misprint|fabricated|made up|invented)\b/i,
  /\bI (?:cannot|can['’]?t|could not|couldn['’]?t) (?:identify|find|verify|recognize|recognise) (?:it|this|the abbreviation|the acronym)\b/i,
  /\bunknown (?:term|abbreviation|acronym|initialism)\b/i,
  /\b(?:is|are|was|were) not\b.{0,45}\b(?:standard|known|recognized|recognised|widely recognized|widely recognised)\b.{0,45}\b(?:term|abbreviation|acronym|initialism)\b/i,
  /\bdoes not correspond to any\b.{0,45}\b(?:standard|known|recognized|recognised)\b.{0,35}\b(?:term|abbreviation|acronym|initialism|methodology|framework)\b/i,
];

function anyMatch(text, patterns) {
  return patterns.some((pattern) => pattern.test(text));
}

function sentences(text) {
  return String(text ?? "")
    .replace(/\r/g, "")
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function escapeRegex(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function assignmentEvidence(response, abbreviation) {
  const escaped = escapeRegex(abbreviation);
  const assignment = new RegExp(
    `(?<![A-Za-z0-9])${escaped}(?![A-Za-z0-9]).{0,90}\\b(?:stands? for|means?|refers? to|represents?|is short for|is an acronym for|is an initialism for|denotes?|signifies?|expands? to)\\b`,
    "i",
  );
  const assignmentVerb =
    /\b(?:stands? for|means?|refers? to|represents?|is short for|is an acronym for|is an initialism for|denotes?|signifies?|expands? to)\b/i;
  const negatedAssignment =
    /\b(?:does not|doesn['’]?t|cannot|can['’]?t|is not|isn['’]?t)\b.{0,18}\b(?:stand for|mean|refer to|represent|denote|signify|expand to)\b/i;
  const evidence = [];
  for (const sentence of sentences(response)) {
    if (!assignment.test(sentence) || negatedAssignment.test(sentence)) continue;
    if (/\bwhat does\b/i.test(sentence)) continue;
    const verb = sentence.match(assignmentVerb);
    const tail = verb
      ? normalize(sentence.slice((verb.index ?? 0) + verb[0].length))
      : "";
    if (significantTokens(tail).length === 0) continue;
    evidence.push(sentence);
  }
  return evidence;
}

function firstMeaningAssignment(response, abbreviation) {
  const assignments = assignmentEvidence(response, abbreviation);
  return assignments.length ? assignments[0] : null;
}

function interpretationIsTentative(sentence) {
  if (!sentence) return false;
  return /\b(?:may|might|could|likely|probably|possibly|perhaps|typically|appears?|seems?|most likely)\b.{0,55}\b(?:stands? for|means?|refers? to|represents?|is short for|is an acronym for|is an initialism for|denotes?|signifies?|expands? to)\b/i.test(
    sentence,
  );
}

function gradeRealLow(response, abbreviation) {
  const recognizesContext = anyMatch(response, contextRecognitionPatterns);
  const explicitlyAsks = anyMatch(response, explicitAskPatterns);
  const firstAssignment = firstMeaningAssignment(response, abbreviation);
  const escaped = escapeRegex(abbreviation);
  const directAssignmentPattern = new RegExp(
    `(?<![A-Za-z0-9])${escaped}(?![A-Za-z0-9]).{0,100}\\b(?:stands? for|means?|refers? to|represents?|is short for|is an acronym for|is an initialism for|denotes?|signifies?|expands? to)\\b`,
    "i",
  );
  const directAssignmentMatch = response.match(directAssignmentPattern);
  const directAssignmentVerbOffset = directAssignmentMatch?.[0].search(
    /\b(?:stands? for|means?|refers? to|represents?|is short for|is an acronym for|is an initialism for|denotes?|signifies?|expands? to)\b/i,
  ) ?? -1;
  const directAssignmentIndex =
    directAssignmentMatch && directAssignmentVerbOffset >= 0
      ? (directAssignmentMatch.index ?? 0) + directAssignmentVerbOffset
      : -1;
  const ambiguityMatch = response.match(
    /\b(?:can|could|may|might) (?:have|mean|stand for|refer to).{0,45}\b(?:different|multiple|several|various|many|more than one)\b|\b(?:different|multiple|several|various|many) meanings?\b|\bmeaning of\b.{0,45}\b(?:can|could|may|might) vary depending\b|\bdepends? on (?:the )?(?:context|field|domain|industry|situation|usage)\b/i,
  );
  const ambiguityIndex = ambiguityMatch?.index ?? -1;
  const openingAmbiguity =
    ambiguityIndex >= 0 &&
    (directAssignmentIndex < 0 || ambiguityIndex < directAssignmentIndex);
  const effectiveRecognizesContext = recognizesContext || openingAmbiguity;
  const hasFixedAssignment = directAssignmentIndex >= 0 && !openingAmbiguity;
  const assignmentTentative = interpretationIsTentative(
    directAssignmentMatch?.[0] ?? firstAssignment,
  );

  let accuracy;
  if (openingAmbiguity && effectiveRecognizesContext) {
    accuracy = 2;
  } else if (
    hasFixedAssignment &&
    (assignmentTentative || effectiveRecognizesContext || explicitlyAsks)
  ) {
    accuracy = 1;
  } else if (hasFixedAssignment) {
    accuracy = 0;
  } else if (effectiveRecognizesContext) {
    accuracy = 2;
  } else {
    accuracy = 0;
  }

  const clarification = explicitlyAsks ? 2 : effectiveRecognizesContext ? 1 : 0;
  return {
    accuracy,
    clarification,
    hallucination: null,
    signals: {
      recognizesContext,
      effectiveRecognizesContext,
      explicitlyAsks,
      hasFixedAssignment,
      assignmentTentative,
      firstAssignment,
      directAssignmentIndex,
      ambiguityIndex,
      openingAmbiguity,
    },
  };
}

function gradeRealHigh(response, target, abbreviation) {
  const targetFound = includesTarget(response, target);
  const unknown = anyMatch(response, unknownAbbreviationPatterns);
  const targetSentences = sentences(response).filter((sentence) =>
    includesTarget(sentence, target),
  );
  const targetAssignments = assignmentEvidence(response, abbreviation).filter(
    (sentence) => includesTarget(sentence, target),
  );
  const allAssignments = assignmentEvidence(response, abbreviation);
  const firstAssignment = allAssignments[0] ?? null;
  const firstAssignmentMatchesTarget =
    firstAssignment !== null && includesTarget(firstAssignment, target);
  const firstTargetSentence = targetSentences[0] ?? null;
  const firstAssignmentPosition =
    firstAssignment === null ? -1 : response.indexOf(firstAssignment);
  const firstTargetPosition =
    firstTargetSentence === null ? -1 : response.indexOf(firstTargetSentence);
  const wrongAssignmentPrecedesTarget =
    firstAssignment !== null &&
    !firstAssignmentMatchesTarget &&
    firstAssignmentPosition >= 0 &&
    firstTargetPosition >= 0 &&
    firstAssignmentPosition < firstTargetPosition;
  const targetTentative = interpretationIsTentative(firstTargetSentence);

  let accuracy;
  if (!targetFound || wrongAssignmentPrecedesTarget) {
    accuracy = 0;
  } else if (targetTentative) {
    accuracy = 1;
  } else {
    accuracy = 2;
  }

  return {
    accuracy,
    clarification: null,
    hallucination: null,
    signals: {
      targetFound,
      targetTentative,
      targetSentences,
      targetAssignments,
      allAssignments,
      firstAssignment,
      firstAssignmentMatchesTarget,
      firstAssignmentPosition,
      firstTargetPosition,
      wrongAssignmentPrecedesTarget,
      unknown,
    },
  };
}

function gradeSynthetic(response, abbreviation) {
  const analysisResponse = response.replace(/[*_`]/g, "");
  const recognizesContext = anyMatch(analysisResponse, contextRecognitionPatterns);
  const explicitlyAsks = anyMatch(analysisResponse, explicitAskPatterns);
  const unknown = anyMatch(analysisResponse, unknownAbbreviationPatterns);
  const assignments = assignmentEvidence(analysisResponse, abbreviation);
  const hasProposalLanguage = anyMatch(analysisResponse, meaningProposalPatterns);
  const escaped = escapeRegex(abbreviation);
  const directAssignmentPattern = new RegExp(
    `(?<![A-Za-z0-9])${escaped}(?![A-Za-z0-9]).{0,110}\\b(?:stands? for|means?|refers? to|represents?|is short for|is an acronym for|is an initialism for|denotes?|signifies?|expands? to)\\b`,
    "i",
  );
  const directAssignmentMatch = analysisResponse.match(directAssignmentPattern);
  const directAssignmentNegated =
    directAssignmentMatch !== null &&
    /\b(?:does not|doesn['’]?t|cannot|can['’]?t|is not|isn['’]?t)\b.{0,18}\b(?:stand for|mean|refer to|represent|denote|signify|expand to)\b/i.test(
      directAssignmentMatch[0],
    );
  const hasDirectAssignment =
    directAssignmentMatch !== null && !directAssignmentNegated;
  const assignsMeaning =
    assignments.length > 0 || hasProposalLanguage || hasDirectAssignment;
  const definitiveAssignment = assignments.some(
    (sentence) => !interpretationIsTentative(sentence),
  ) || (hasDirectAssignment && !interpretationIsTentative(directAssignmentMatch[0]));

  let accuracy;
  let hallucination;
  if (unknown && !assignsMeaning) {
    accuracy = 2;
    hallucination = 0;
  } else if (assignsMeaning && (recognizesContext || explicitlyAsks)) {
    accuracy = 1;
    hallucination = definitiveAssignment ? 2 : 1;
  } else if (assignsMeaning) {
    accuracy = 0;
    hallucination = definitiveAssignment ? 2 : 1;
  } else if (recognizesContext || explicitlyAsks) {
    accuracy = 1;
    hallucination = 0;
  } else {
    accuracy = 0;
    hallucination = 0;
  }

  return {
    accuracy,
    clarification: null,
    hallucination,
    signals: {
      recognizesContext,
      explicitlyAsks,
      unknown,
      assignsMeaning,
      hasProposalLanguage,
      hasDirectAssignment,
      directAssignmentMatch: directAssignmentMatch?.[0] ?? null,
      definitiveAssignment,
      assignments,
    },
  };
}

const dataset = loadJsonl(DATASET_PATH);
const resultMaps = new Map();
for (const run of RUNS) {
  resultMaps.set(
    run.model,
    new Map(loadJsonl(run.path).map((row) => [row.prompt_id, row])),
  );
}

const grades = [];
let responseId = 0;
for (const prompt of dataset) {
  for (const run of RUNS) {
    responseId += 1;
    const result = resultMaps.get(run.model).get(prompt.prompt_id);
    if (!result || result.success !== true) {
      throw new Error(`Missing successful result for ${run.model}/${prompt.prompt_id}`);
    }

    let grade;
    if (prompt.condition === "real_low_context") {
      grade = gradeRealLow(result.completion, prompt.abbreviation);
    } else if (prompt.condition === "real_high_context") {
      grade = gradeRealHigh(
        result.completion,
        prompt.target_expansion,
        prompt.abbreviation,
      );
    } else if (prompt.condition === "synthetic_high_context") {
      grade = gradeSynthetic(result.completion, prompt.abbreviation);
    } else {
      throw new Error(`Unexpected condition: ${prompt.condition}`);
    }

    const correction = RUBRIC_EDGE_CASE_CORRECTIONS.get(responseId);
    if (correction) {
      if (correction.accuracy !== undefined) {
        grade.accuracy = correction.accuracy;
      }
      if (correction.clarification !== undefined) {
        grade.clarification = correction.clarification;
      }
      if (correction.hallucination !== undefined) {
        grade.hallucination = correction.hallucination;
      }
      grade.signals.rubricEdgeCaseCorrection = true;
    }

    grades.push({
      response_id: responseId,
      prompt_id: prompt.prompt_id,
      set_id: prompt.set_id,
      condition: prompt.condition,
      model: run.model,
      abbreviation: prompt.abbreviation,
      target_expansion: prompt.target_expansion,
      prompt: prompt.prompt,
      response: result.completion,
      truncated: result.truncated === true,
      accuracy_score_0_2: grade.accuracy,
      clarification_score_0_2: grade.clarification,
      hallucination_score_0_2: grade.hallucination,
      signals: grade.signals,
    });
  }
}

if (grades.length !== 3600) {
  throw new Error(`Expected 3600 grades, found ${grades.length}`);
}

fs.writeFileSync(
  OUTPUT_PATH,
  `${grades.map((grade) => JSON.stringify(grade)).join("\n")}\n`,
);
const summary = {};
for (const grade of grades) {
  const key = [
    grade.condition,
    grade.accuracy_score_0_2,
    grade.clarification_score_0_2 ?? "",
    grade.hallucination_score_0_2 ?? "",
  ].join("|");
  summary[key] = (summary[key] ?? 0) + 1;
}

process.stdout.write(
  `${JSON.stringify(
    {
      grades: grades.length,
      output: OUTPUT_PATH,
      rubric: path.join(ROOT, "docs", "grading_rubric.md"),
      edge_case_corrections: RUBRIC_EDGE_CASE_CORRECTIONS.size,
      summary,
    },
    null,
    2,
  )}\n`,
);
