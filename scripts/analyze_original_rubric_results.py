#!/usr/bin/env python3
"""Reproducible analysis of scores produced with the original project rubric."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "grades" / "original_rubric_scores.jsonl"
DATASET = ROOT / "data" / "prompts" / "v3_900_prompts.jsonl"
OUTPUT_DIR = ROOT / "results"

MODELS = ("qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b")
MODEL_LABELS = {
    "qwen3_4b": "Qwen3 4B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_14b": "Qwen3 14B",
    "qwen3_32b": "Qwen3 32B",
}
CONDITIONS = (
    "real_low_context",
    "real_high_context",
    "synthetic_high_context",
)
CONDITION_LABELS = {
    "real_low_context": "Real / low context",
    "real_high_context": "Real / high context",
    "synthetic_high_context": "Synthetic / high context",
}
COLORS = {
    "real_low_context": "#9CB6CF",
    "real_high_context": "#2F6B9A",
    "synthetic_high_context": "#D79B4A",
    "score_0": "#2F6B9A",
    "score_1": "#D9B26F",
    "score_2": "#B04A4A",
    "ink": "#1F2933",
    "muted": "#68737D",
    "grid": "#D8DEE5",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denom
    )
    return center - half, center + half


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_difference(
    pairs: list[tuple[int, int]], seed: int, repetitions: int = 10_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(pairs)
    differences: list[float] = []
    for _ in range(repetitions):
        difference = 0
        for _ in range(count):
            left, right = pairs[rng.randrange(count)]
            difference += right - left
        differences.append(difference / count)
    return percentile(differences, 0.025), percentile(differences, 0.975)


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 3600:
        raise ValueError(f"Expected 3,600 grades, found {len(rows)}")
    if len({int(row["response_id"]) for row in rows}) != 3600:
        raise ValueError("Response IDs are not unique")

    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if len(model_rows) != 900:
            raise ValueError(f"{model}: expected 900 rows, found {len(model_rows)}")
        for condition in CONDITIONS:
            subset = [row for row in model_rows if row["condition"] == condition]
            if len(subset) != 300:
                raise ValueError(
                    f"{model}/{condition}: expected 300 rows, found {len(subset)}"
                )
            if len({row["set_id"] for row in subset}) != 300:
                raise ValueError(f"{model}/{condition}: set IDs are not unique")

    for row in rows:
        accuracy = row["accuracy_score_0_2"]
        clarification = row["clarification_score_0_2"]
        hallucination = row["hallucination_score_0_2"]
        if accuracy not in (0, 1, 2):
            raise ValueError(f"Invalid accuracy score: {row['response_id']}")
        if row["condition"] == "real_low_context":
            if clarification not in (0, 1, 2) or hallucination is not None:
                raise ValueError(f"Invalid low-context scores: {row['response_id']}")
        elif row["condition"] == "real_high_context":
            if clarification is not None or hallucination is not None:
                raise ValueError(f"Invalid real-high scores: {row['response_id']}")
        elif row["condition"] == "synthetic_high_context":
            if clarification is not None or hallucination not in (0, 1, 2):
                raise ValueError(f"Invalid synthetic scores: {row['response_id']}")


def descriptive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for condition in CONDITIONS:
            subset = [
                row
                for row in rows
                if row["model"] == model and row["condition"] == condition
            ]
            accuracy = Counter(row["accuracy_score_0_2"] for row in subset)
            ideal = accuracy[2]
            low, high = wilson_interval(ideal, len(subset))
            record: dict[str, Any] = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "n": len(subset),
                "accuracy_0_n": accuracy[0],
                "accuracy_1_n": accuracy[1],
                "accuracy_2_n": accuracy[2],
                "accuracy_mean": sum(
                    row["accuracy_score_0_2"] for row in subset
                )
                / len(subset),
                "ideal_response_rate": ideal / len(subset),
                "ideal_response_ci_low": low,
                "ideal_response_ci_high": high,
            }
            if condition == "real_low_context":
                clarification = Counter(
                    row["clarification_score_0_2"] for row in subset
                )
                record.update(
                    {
                        "clarification_0_n": clarification[0],
                        "clarification_1_n": clarification[1],
                        "clarification_2_n": clarification[2],
                        "explicit_clarification_rate": clarification[2]
                        / len(subset),
                    }
                )
            if condition == "synthetic_high_context":
                hallucination = Counter(
                    row["hallucination_score_0_2"] for row in subset
                )
                record.update(
                    {
                        "hallucination_0_n": hallucination[0],
                        "hallucination_1_n": hallucination[1],
                        "hallucination_2_n": hallucination[2],
                        "definitive_hallucination_rate": hallucination[2]
                        / len(subset),
                        "avoided_assumption_rate": hallucination[0] / len(subset),
                    }
                )
            output.append(record)
    return output


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["set_id"], row["condition"]): row for row in rows
    }
    contrasts = (
        (
            "context_effect_for_real_abbreviations",
            "real_low_context",
            "real_high_context",
        ),
        (
            "real_vs_synthetic_under_high_context",
            "real_high_context",
            "synthetic_high_context",
        ),
    )
    output: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODELS):
        set_ids = sorted(
            {
                row["set_id"]
                for row in rows
                if row["model"] == model
            }
        )
        for contrast_index, (name, left_condition, right_condition) in enumerate(
            contrasts
        ):
            pairs: list[tuple[int, int]] = []
            for set_id in set_ids:
                left = int(
                    by_key[(model, set_id, left_condition)][
                        "accuracy_score_0_2"
                    ]
                    == 2
                )
                right = int(
                    by_key[(model, set_id, right_condition)][
                        "accuracy_score_0_2"
                    ]
                    == 2
                )
                pairs.append((left, right))
            left_only = sum(left == 1 and right == 0 for left, right in pairs)
            right_only = sum(left == 0 and right == 1 for left, right in pairs)
            both = sum(left == 1 and right == 1 for left, right in pairs)
            neither = sum(left == 0 and right == 0 for left, right in pairs)
            left_rate = sum(left for left, _ in pairs) / len(pairs)
            right_rate = sum(right for _, right in pairs) / len(pairs)
            ci_low, ci_high = bootstrap_difference(
                pairs,
                seed=20260726 + model_index * 100 + contrast_index,
            )
            output.append(
                {
                    "contrast": name,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "left_condition": left_condition,
                    "right_condition": right_condition,
                    "n_sets": len(pairs),
                    "left_ideal_rate": left_rate,
                    "right_ideal_rate": right_rate,
                    "difference_right_minus_left": right_rate - left_rate,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "both_ideal": both,
                    "left_only_ideal": left_only,
                    "right_only_ideal": right_only,
                    "neither_ideal": neither,
                    "mcnemar_exact_p": exact_mcnemar(left_only, right_only),
                }
            )
    return output


def domain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["condition"])].append(row)
    for (domain, condition), subset in sorted(grouped.items()):
        ideal = sum(row["accuracy_score_0_2"] == 2 for row in subset)
        low, high = wilson_interval(ideal, len(subset))
        output.append(
            {
                "domain": domain,
                "condition": condition,
                "n": len(subset),
                "ideal_n": ideal,
                "ideal_rate": ideal / len(subset),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_main_figure(
    descriptive: list[dict[str, Any]], path: Path
) -> None:
    width, height = 1800, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(42, bold=True)
    axis_font = font(25)
    label_font = font(26, bold=True)
    small_font = font(22)
    draw.text(
        (90, 42),
        "Ideal-response rates by model and condition",
        font=title_font,
        fill=COLORS["ink"],
    )
    draw.text(
        (90, 98),
        "Ideal response = Accuracy score 2 under the original project rubric",
        font=small_font,
        fill=COLORS["muted"],
    )

    left, top, right, bottom = 150, 190, 1720, 820
    plot_height = bottom - top
    for tick in range(0, 101, 20):
        y = bottom - int(plot_height * tick / 100)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=2)
        text_value = f"{tick}%"
        box = draw.textbbox((0, 0), text_value, font=axis_font)
        draw.text(
            (left - 20 - (box[2] - box[0]), y - 14),
            text_value,
            font=axis_font,
            fill=COLORS["muted"],
        )
    draw.line((left, top, left, bottom), fill=COLORS["ink"], width=3)
    draw.line((left, bottom, right, bottom), fill=COLORS["ink"], width=3)

    group_width = (right - left) / len(MODELS)
    bar_width = 74
    gap = 18
    lookup = {
        (row["model"], row["condition"]): row for row in descriptive
    }
    for model_index, model in enumerate(MODELS):
        group_center = left + group_width * (model_index + 0.5)
        total_bars = len(CONDITIONS) * bar_width + (len(CONDITIONS) - 1) * gap
        start = group_center - total_bars / 2
        for condition_index, condition in enumerate(CONDITIONS):
            record = lookup[(model, condition)]
            rate = float(record["ideal_response_rate"])
            x0 = int(start + condition_index * (bar_width + gap))
            x1 = x0 + bar_width
            y0 = bottom - int(plot_height * rate)
            draw.rectangle(
                (x0, y0, x1, bottom),
                fill=COLORS[condition],
                outline=COLORS["ink"],
                width=1,
            )
            percentage = f"{rate * 100:.1f}"
            box = draw.textbbox((0, 0), percentage, font=small_font)
            draw.text(
                (x0 + (bar_width - (box[2] - box[0])) / 2, y0 - 30),
                percentage,
                font=small_font,
                fill=COLORS["ink"],
            )
        model_text = MODEL_LABELS[model]
        box = draw.textbbox((0, 0), model_text, font=label_font)
        draw.text(
            (group_center - (box[2] - box[0]) / 2, bottom + 24),
            model_text,
            font=label_font,
            fill=COLORS["ink"],
        )

    legend_y = 910
    x = 280
    for condition in CONDITIONS:
        draw.rectangle(
            (x, legend_y, x + 30, legend_y + 30),
            fill=COLORS[condition],
            outline=COLORS["ink"],
        )
        label = CONDITION_LABELS[condition]
        draw.text(
            (x + 42, legend_y - 1),
            label,
            font=axis_font,
            fill=COLORS["ink"],
        )
        x += 470
    image.save(path, dpi=(200, 200))


def draw_hallucination_figure(
    descriptive: list[dict[str, Any]], path: Path
) -> None:
    width, height = 1800, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(42, bold=True)
    label_font = font(27, bold=True)
    small_font = font(23)
    draw.text(
        (90, 42),
        "Synthetic high-context hallucination scores",
        font=title_font,
        fill=COLORS["ink"],
    )
    draw.text(
        (90, 100),
        "0 = avoided assigning a meaning; 1 = tentative assumption; 2 = definitive assumption",
        font=small_font,
        fill=COLORS["muted"],
    )
    records = {
        row["model"]: row
        for row in descriptive
        if row["condition"] == "synthetic_high_context"
    }
    bar_left, bar_right = 410, 1660
    start_y = 215
    bar_height = 92
    row_gap = 42
    for index, model in enumerate(MODELS):
        y0 = start_y + index * (bar_height + row_gap)
        y1 = y0 + bar_height
        draw.text(
            (95, y0 + 25),
            MODEL_LABELS[model],
            font=label_font,
            fill=COLORS["ink"],
        )
        record = records[model]
        counts = [
            int(record["hallucination_0_n"]),
            int(record["hallucination_1_n"]),
            int(record["hallucination_2_n"]),
        ]
        x = bar_left
        for score, count in enumerate(counts):
            segment = (bar_right - bar_left) * count / 300
            color = COLORS[f"score_{score}"]
            draw.rectangle((x, y0, x + segment, y1), fill=color)
            if segment > 80:
                text_value = f"{count / 3:.1f}%"
                box = draw.textbbox((0, 0), text_value, font=small_font)
                draw.text(
                    (
                        x + segment / 2 - (box[2] - box[0]) / 2,
                        y0 + 29,
                    ),
                    text_value,
                    font=small_font,
                    fill="white" if score in (0, 2) else COLORS["ink"],
                )
            x += segment
        draw.rectangle(
            (bar_left, y0, bar_right, y1),
            outline=COLORS["ink"],
            width=2,
        )
    legend_y = 760
    x = 360
    legend = (
        ("score_0", "0 — avoided assumption"),
        ("score_1", "1 — tentative assumption"),
        ("score_2", "2 — definitive assumption"),
    )
    for key, label in legend:
        draw.rectangle(
            (x, legend_y, x + 28, legend_y + 28),
            fill=COLORS[key],
            outline=COLORS["ink"],
        )
        draw.text(
            (x + 40, legend_y - 2),
            label,
            font=small_font,
            fill=COLORS["ink"],
        )
        x += 440
    image.save(path, dpi=(200, 200))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(INPUT)
    prompt_metadata = {
        row["prompt_id"]: row for row in load_jsonl(DATASET)
    }
    if len(prompt_metadata) != 900:
        raise ValueError("Expected 900 unique prompt records in the dataset")
    for row in rows:
        metadata = prompt_metadata.get(row["prompt_id"])
        if metadata is None:
            raise ValueError(f"Missing dataset prompt: {row['prompt_id']}")
        row["domain"] = metadata["domain"]
    validate(rows)
    descriptive = descriptive_rows(rows)
    paired = paired_rows(rows)
    domains = domain_rows(rows)

    write_csv(OUTPUT_DIR / "descriptive_results.csv", descriptive)
    write_csv(OUTPUT_DIR / "paired_contrasts.csv", paired)
    write_csv(OUTPUT_DIR / "domain_results.csv", domains)
    draw_main_figure(
        descriptive, OUTPUT_DIR / "figures" / "ideal_response_rates.png"
    )
    draw_hallucination_figure(
        descriptive,
        OUTPUT_DIR / "figures" / "synthetic_hallucination_scores.png",
    )

    summary = {
        "status": "SCORED WITH THE ORIGINAL PROJECT RUBRIC",
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256(DATASET),
        "rows": len(rows),
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "unique_sets": len({row["set_id"] for row in rows}),
        "descriptive": descriptive,
        "paired_contrasts": paired,
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
