import argparse
import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
import time

import numpy as np
import pandas as pd
from faker import Faker
try:
    import polars as pl
except Exception:
    pl = None


fake = Faker()


def make_logger(enabled: bool):
    def _log(message: str):
        if enabled:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] {message}", flush=True)

    return _log


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def weighted_choice(items, weights, size):
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    idx = np.random.choice(len(items), size=size, p=probs)
    return [items[i] for i in idx]


def build_mask(column_meta: dict, n_rows: int, data_start_row_excel: int) -> np.ndarray:
    target_nulls = int(column_meta.get("null_count", round(column_meta.get("null_ratio", 0) * n_rows)))
    if target_nulls <= 0:
        return np.ones(n_rows, dtype=bool)

    mask = np.ones(n_rows, dtype=bool)
    runs = column_meta.get("gap_profile", {}).get("gap_runs", [])

    run_lengths = []
    for run in runs:
        start = run["from_source_row"] - data_start_row_excel
        end = run["to_source_row"] - data_start_row_excel
        if end < 0 or start >= n_rows:
            continue
        start = max(0, start)
        end = min(n_rows - 1, end)
        if start <= end:
            mask[start : end + 1] = False
            run_lengths.append(end - start + 1)

    current_nulls = int((~mask).sum())
    remaining = target_nulls - current_nulls

    if remaining > 0:
        if run_lengths:
            avg_len = max(1, int(np.median(run_lengths)))
        else:
            avg_len = max(1, int(target_nulls / max(1, column_meta.get("gap_profile", {}).get("gap_count", 1))))
            if avg_len == 1:
                avg_len = 5

        attempts = 0
        while remaining > 0 and attempts < n_rows * 10:
            attempts += 1
            run_len = int(max(1, np.random.poisson(avg_len)))
            run_len = min(run_len, remaining)
            start = random.randint(0, n_rows - 1)
            end = min(n_rows - 1, start + run_len - 1)
            newly_null = int(mask[start : end + 1].sum())
            mask[start : end + 1] = False
            remaining -= newly_null

        if remaining > 0:
            available = np.where(mask)[0]
            if len(available) > 0:
                pick = np.random.choice(available, size=min(remaining, len(available)), replace=False)
                mask[pick] = False

    elif remaining < 0:
        # Preserve provided run blocks as much as possible; reopen random cells if runs exceed target.
        null_rows = np.where(~mask)[0]
        reopen = min(len(null_rows), -remaining)
        if reopen > 0:
            pick = np.random.choice(null_rows, size=reopen, replace=False)
            mask[pick] = True

    return mask


def pick_length(length_profile: dict) -> int:
    common = length_profile.get("common_lengths", [])
    if common:
        lens = [c["length"] for c in common]
        weights = [c["count"] for c in common]
        return int(weighted_choice(lens, weights, 1)[0])
    mn = int(length_profile.get("min_length", 5))
    mx = int(length_profile.get("max_length", max(mn, 20)))
    return random.randint(mn, mx)


def text_to_target_length(target_len: int, mode: str) -> str:
    if mode == "paragraph":
        if target_len > 500:
            chunk = " ".join(fake.paragraph(nb_sentences=random.randint(4, 8)) for _ in range(random.randint(2, 5)))
        elif target_len > 180:
            chunk = " ".join(fake.paragraph(nb_sentences=random.randint(3, 5)) for _ in range(random.randint(1, 3)))
        elif target_len > 60:
            chunk = fake.paragraph(nb_sentences=random.randint(2, 4))
        else:
            chunk = fake.sentence(nb_words=random.randint(4, 12))
    elif mode == "sentence":
        if target_len < 20:
            chunk = " ".join(fake.words(nb=max(1, target_len // 5)))
        elif target_len < 80:
            chunk = fake.sentence(nb_words=max(4, target_len // 7))
        else:
            chunk = fake.paragraph(nb_sentences=max(2, min(5, target_len // 40)))
    else:
        chunk = " ".join(fake.words(nb=max(1, min(6, target_len // 6))))

    chunk = re.sub(r"\s+", " ", chunk).strip()
    if len(chunk) > target_len:
        return chunk[:target_len].rstrip(" ,.;:")
    if len(chunk) < target_len:
        extra = " " + " ".join(fake.words(nb=20))
        chunk = (chunk + extra)[:target_len].rstrip(" ,.;:")
    return chunk


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", normalize_spaces(text))
    return [p.strip() for p in parts if p.strip()]


def infer_bio_style(example_bios):
    if not example_bios:
        return {
            "avg_sentences": 8,
            "avg_words_per_sentence": 16,
            "bullet_bias": 0.2,
        }

    sample = example_bios[: min(600, len(example_bios))]
    sent_counts = []
    word_counts = []
    bullet_hits = 0
    for bio in sample:
        sents = split_sentences(bio)
        if sents:
            sent_counts.append(len(sents))
            for s in sents:
                word_counts.append(len(s.split()))
        if "•" in bio or "\n-" in bio or "\n*" in bio:
            bullet_hits += 1

    avg_sentences = int(round(np.mean(sent_counts))) if sent_counts else 8
    avg_words = int(round(np.mean(word_counts))) if word_counts else 16
    bullet_bias = bullet_hits / max(1, len(sample))
    return {
        "avg_sentences": int(np.clip(avg_sentences, 5, 16)),
        "avg_words_per_sentence": int(np.clip(avg_words, 10, 24)),
        "bullet_bias": float(np.clip(bullet_bias, 0.0, 0.7)),
    }


def load_dummy_profile_style(path: str, log=None):
    style = {
        "summary_avg_words": 22,
        "summary_sentence_weights": [0.62, 0.26, 0.08, 0.03, 0.01],
        "description_avg_words": 20,
        "description_bullet_bias": 0.38,
        "description_sentence_weights": [0.46, 0.34, 0.14, 0.05, 0.01],
    }

    workbook = Path(path)
    if not workbook.exists():
        if log:
            log(f"Local text-style workbook not found: {path}")
        return style

    try:
        df = pd.read_excel(workbook, sheet_name="Dummy Profiles")
    except Exception as e:
        if log:
            log(f"Could not read local text-style workbook: {type(e).__name__}: {e}")
        return style

    if "Summary/Bio" in df.columns:
        vals = [normalize_spaces(str(v)) for v in df["Summary/Bio"].dropna().tolist()]
        if vals:
            word_counts = [len(v.split()) for v in vals]
            sent_counts = [min(8, max(1, len(split_sentences(v)))) for v in vals]
            style["summary_avg_words"] = int(round(np.mean(word_counts)))
            summary_hist = np.bincount(sent_counts, minlength=9)[1:6].astype(float)
            if summary_hist.sum() > 0:
                style["summary_sentence_weights"] = (summary_hist / summary_hist.sum()).tolist()

    if "Description" in df.columns:
        vals = [str(v) for v in df["Description"].dropna().tolist()]
        cleaned = [v for v in vals if normalize_spaces(v)]
        if cleaned:
            word_counts = [len(normalize_spaces(v).replace("- ", "").split()) for v in cleaned]
            sent_counts = [min(8, max(1, len(split_sentences(v)))) for v in cleaned]
            bullet_hits = [1 if ("\n-" in v or "\n*" in v or "•" in v) else 0 for v in cleaned]
            style["description_avg_words"] = int(round(np.mean(word_counts)))
            style["description_bullet_bias"] = float(np.clip(np.mean(bullet_hits), 0.38, 0.42))
            desc_hist = np.bincount(sent_counts, minlength=9)[1:6].astype(float)
            if desc_hist.sum() > 0:
                style["description_sentence_weights"] = (desc_hist / desc_hist.sum()).tolist()

    if log:
        log(
            "Loaded local text style reference: "
            f"summary_avg_words={style['summary_avg_words']}, "
            f"description_avg_words={style['description_avg_words']}, "
            f"description_bullet_bias={style['description_bullet_bias']:.2f}"
        )
    return style


def load_dummy_profile_examples(path: str, log=None):
    workbook = Path(path)
    if not workbook.exists():
        if log:
            log(f"Dummy profile example workbook not found: {path}")
        return {"summaries": [], "descriptions": []}

    try:
        df = pd.read_excel(workbook, sheet_name="Dummy Profile Examples")
    except Exception:
        try:
            df = pd.read_excel(workbook, sheet_name="Paste Ready")
        except Exception as e:
            if log:
                log(f"Could not read dummy profile examples: {type(e).__name__}: {e}")
            return {"summaries": [], "descriptions": []}

    summary_col = next((c for c in df.columns if str(c).strip().lower() == "summary/bio"), None)
    description_col = next((c for c in df.columns if str(c).strip().lower() == "description"), None)

    summaries = []
    descriptions = []
    if summary_col:
        summaries = [str(v).strip() for v in df[summary_col].dropna().tolist() if str(v).strip()]
    if description_col:
        descriptions = [str(v).strip() for v in df[description_col].dropna().tolist() if str(v).strip()]

    if log:
        log(f"Loaded dummy profile examples: summaries={len(summaries)}, descriptions={len(descriptions)}")
    return {"summaries": summaries, "descriptions": descriptions}


def load_bio_examples(uri: str, column: str = "Moreinfo", max_examples: int = 1000, log=None):
    if pl is None:
        if log:
            log("Polars not available; skipping external bio style examples.")
        return []
    try:
        df = pl.read_parquet(uri)
        if column not in df.columns:
            if log:
                log(f"Column '{column}' not found in bio source; skipping.")
            return []
        bios = (
            df[column]
            .drop_nulls()
            .cast(pl.Utf8, strict=False)
            .to_list()
        )
        cleaned = []
        for b in bios:
            if not b:
                continue
            t = normalize_spaces(str(b))
            if len(t) >= 80:
                cleaned.append(t)
            if len(cleaned) >= max_examples:
                break
        if log:
            log(f"Loaded {len(cleaned)} external bio examples for style inference.")
        return cleaned
    except Exception as e:
        if log:
            log(f"Could not load external bio examples: {type(e).__name__}: {e}")
        return []


def limit_text_shape(text: str, max_sentences: int = 8, max_paragraphs: int = 2) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraphs = paragraphs[:max_paragraphs]
    kept = []
    remaining = max_sentences
    for para in paragraphs:
        if remaining <= 0:
            break
        if "\n-" in para:
            lines = [ln for ln in para.splitlines() if ln.strip()]
            head = lines[:1]
            bullets = lines[1 : 1 + max(0, remaining - 1)]
            para_out = "\n".join(head + bullets).strip()
            used = max(1, len(bullets))
        else:
            sents = split_sentences(para)[:remaining]
            para_out = " ".join(sents).strip()
            used = len(sents)
        if para_out:
            kept.append(para_out)
            remaining -= used
    return "\n\n".join(kept).strip()


def trim_to_length(text: str, target_len: int) -> str:
    text = text.strip()
    if len(text) <= target_len:
        return text
    cut = text[:target_len].rstrip()
    if "\n-" in cut:
        return cut.rstrip(" ,.;:")
    for sep in [". ", "; ", ", "]:
        idx = cut.rfind(sep)
        if idx > target_len * 0.6:
            return cut[: idx + (1 if sep == ". " else 0)].rstrip(" ,.;:")
    return cut.rstrip(" ,.;:")


def personalize_example_text(text: str, resource_name: str) -> str:
    text = str(text).strip()
    resource_name = str(resource_name).strip()
    if not text or not resource_name:
        return text

    name_patterns = []
    full_name_match = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(\s+(?:is|has|was)\b)", text)
    if full_name_match:
        full_name = full_name_match.group(1)
        first_name = full_name.split()[0]
        name_patterns.append((re.escape(full_name), resource_name))
        name_patterns.append((re.escape(first_name) + r"'s", resource_name.split()[0] + "'s"))
        name_patterns.append((r"\b" + re.escape(first_name) + r"\b", resource_name.split()[0]))

    pronoun_replacements = {
        r"\bHe\b": "They",
        r"\bShe\b": "They",
        r"\bhe\b": "they",
        r"\bshe\b": "they",
        r"\bHis\b": "Their",
        r"\bHer\b": "Their",
        r"\bhis\b": "their",
        r"\bher\b": "their",
        r"\bhim\b": "them",
    }

    for pattern, replacement in name_patterns:
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in pronoun_replacements.items():
        text = re.sub(pattern, replacement, text)
    return text


def normalize_summary_intro(text: str) -> str:
    words = text.split()
    if len(words) >= 4 and words[2].lower() == "is" and words[3].lower() == "an":
        return "They are an " + " ".join(words[4:])
    return text


def align_summary_subject(text: str, resource_name: str) -> str:
    words = text.split()
    if words and words[0].lower() == "they":
        replacement = resource_name.split()
        if len(words) >= 3 and words[1].lower() == "are":
            words = replacement + ["is"] + words[2:]
        else:
            words = replacement + words[1:]
        return " ".join(words)
    return text


def generate_summary_bio(target_len: int, style: dict) -> str:
    role_pool = [
        "business analyst",
        "technology consultant",
        "risk specialist",
        "financial analyst",
        "data consultant",
        "reporting specialist",
        "transformation consultant",
        "governance analyst",
    ]
    industry_pool = [
        "capital markets",
        "financial services",
        "banking",
        "insurance",
        "public sector",
        "technology",
        "consumer markets",
    ]
    specialty_pool = [
        "data governance",
        "reporting optimisation",
        "ETL pipelines",
        "regulatory reporting",
        "stakeholder management",
        "data transformation",
        "process improvement",
        "requirements analysis",
    ]
    outcomes_pool = [
        "support business and regulatory objectives",
        "improve reporting accuracy",
        "deliver high-quality insights",
        "support complex transformation initiatives",
        "improve data quality and controls",
        "enable scalable delivery outcomes",
    ]

    sentence_options = [1, 2, 3, 4, 5]
    sentence_count = int(weighted_choice(sentence_options, style["summary_sentence_weights"], 1)[0])
    sentence_count = min(sentence_count, 8)
    role = random.choice(role_pool)
    industry = random.choice(industry_pool)
    specialty = random.choice(specialty_pool)
    outcome = random.choice(outcomes_pool)
    years = random.randint(3, 14)

    sentences = [
        f"{random.choice(['Experienced', 'Results-driven', 'Skilled', 'Versatile'])} {role} with experience in {industry}, specialising in {specialty}.",
        f"Proven ability to work across cross-functional teams to {outcome}.",
        f"Strong background in delivery support, analysis, and stakeholder engagement across fast-paced programmes.",
        f"Brings a practical approach to improving controls, reporting quality, and process efficiency.",
        f"Experienced in translating business needs into clear requirements and actionable delivery plans.",
    ]
    text = " ".join(sentences[:sentence_count])
    text = limit_text_shape(text, max_sentences=8, max_paragraphs=2)
    return trim_to_length(text, target_len)


def generate_role_description(target_len: int, style: dict) -> str:
    action_pool = [
        "Supports delivery of data and reporting solutions.",
        "Works with business and technology teams to analyse requirements and improve data processes.",
        "Helps deliver end-to-end transformation initiatives across reporting, controls, and governance.",
        "Collaborates with stakeholders to improve data quality, streamline reporting, and support project delivery.",
        "Contributes to implementation, testing, and process improvement activities across complex programmes.",
    ]
    bullet_pool = [
        "Analysed business requirements and documented delivery needs.",
        "Supported testing, issue management, and implementation planning.",
        "Worked with stakeholders to improve data quality and reporting controls.",
        "Helped coordinate delivery across business and technology teams.",
        "Contributed to process improvement and governance activities.",
    ]

    sentence_options = [1, 2, 3, 4, 5]
    sentence_count = int(weighted_choice(sentence_options, style["description_sentence_weights"], 1)[0])
    sentence_count = min(sentence_count, 8)

    if random.random() < style["description_bullet_bias"]:
        intro = random.choice(action_pool)
        bullet_count = max(2, min(4, sentence_count))
        bullets = random.sample(bullet_pool, k=min(bullet_count, len(bullet_pool)))
        text = intro + "\n\n" + "\n".join(f"- {b}" for b in bullets)
    else:
        sentences = random.sample(action_pool, k=min(sentence_count, len(action_pool)))
        text = " ".join(sentences)

    text = limit_text_shape(text, max_sentences=8, max_paragraphs=2)
    return trim_to_length(text, target_len)


def choose_summary_bio(target_len: int, style: dict, example_pool, resource_name: str):
    if example_pool:
        text = personalize_example_text(random.choice(example_pool), resource_name)
        text = normalize_summary_intro(text)
        text = align_summary_subject(text, resource_name)
        return trim_to_length(text, target_len)
    text = normalize_summary_intro(generate_summary_bio(target_len, style))
    text = align_summary_subject(text, resource_name)
    return trim_to_length(text, target_len)


def choose_role_description(target_len: int, style: dict, example_pool, resource_name: str):
    if example_pool:
        return trim_to_length(personalize_example_text(random.choice(example_pool), resource_name), target_len)
    return generate_role_description(target_len, style)


def format_legacy_date(dt: datetime) -> str:
    # Matches patterns like 'Sep  4 2023 12:00AM'
    return f"{dt.strftime('%b')} {dt.day:2d} {dt.year} 12:00AM"


def format_iso_midnight(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT00:00:00")


def generate_employee_counts(employee_count: int, total_rows: int) -> np.ndarray:
    # Use a smoother distribution than Zipf to avoid unrealistic extreme concentration.
    raw = np.random.gamma(shape=2.5, scale=1.0, size=employee_count).astype(float)
    scaled = raw / raw.sum() * total_rows
    counts = np.floor(scaled).astype(int)

    min_rows = 15
    max_rows = 140
    counts[counts < min_rows] = min_rows
    counts[counts > max_rows] = max_rows

    diff = total_rows - counts.sum()

    if diff > 0:
        order = np.argsort(-raw)
        oi = 0
        while diff > 0:
            i = int(order[oi % len(order)])
            if counts[i] < max_rows:
                counts[i] += 1
                diff -= 1
            oi += 1
    elif diff < 0:
        order = np.argsort(raw)
        oi = 0
        while diff < 0:
            i = int(order[oi % len(order)])
            if counts[i] > min_rows:
                counts[i] -= 1
                diff += 1
            oi += 1

    return counts


def compute_target_counts(total_rows: int, pct_map: dict) -> dict:
    total_pct = float(sum(pct_map.values()))
    raw = {k: total_rows * (v / total_pct) for k, v in pct_map.items()}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total_rows - sum(base.values())
    order = sorted(raw.keys(), key=lambda k: (raw[k] - base[k]), reverse=True)
    for k in order[:remainder]:
        base[k] += 1
    return base


def synthesize_category_pool(column_name: str, unique_count: int):
    key = column_name.lower()

    if "boolean" in key or column_name in {"Interest", "In Process", "Travel Interest"}:
        return ["No", "Yes"]
    if "management level" in key:
        return [f"M{i}" for i in range(1, max(2, unique_count + 1))]
    if "job level" in key:
        return [f"J{i}" for i in range(1, max(2, unique_count + 1))]
    if "resource status" in key:
        return ["Enabled"]
    if "resource type" in key:
        return ["Employee", "Associate", "Contract"][: max(1, unique_count)]
    if "office" in key or "location" in key:
        cities = [
            "North Hub",
            "South Hub",
            "East Hub",
            "West Hub",
            "Metro Hub",
            "Coastal Hub",
            "Central Hub",
            "Lake Hub",
            "Valley Hub",
            "River Hub",
            "Plains Hub",
        ]
        return cities[: max(1, unique_count)]
    if "territory" in key:
        return ["Region United"]
    if "cost centre" in key:
        return [f"CC-{1000 + i}" for i in range(max(1, unique_count))]
    if "network" in key:
        return [f"Network {chr(65 + i)}" for i in range(max(1, unique_count))]
    if "competency" in key:
        return [f"Competency {chr(65 + i)}" for i in range(max(1, unique_count))]
    if "sub" in key or "los" in key:
        return [f"Practice {i+1}" for i in range(max(1, unique_count))]
    if "market" in key or "region" in key:
        return [f"Market {i+1}" for i in range(max(1, unique_count))]
    if "relationship leader" in key or "coach" in key or "contact" in key:
        return [f"Leader {i+1}" for i in range(max(1, unique_count))]
    if "skill level" in key:
        return ["Core", "Applied", "Advanced", "Expert", "Advisor", "Specialist"][: max(1, unique_count)]
    if "proficiency" in key:
        return ["Beginner", "Elementary", "Intermediate", "Advanced", "Upper Advanced", "Expert"][: max(1, unique_count)]

    return [f"{slugify(column_name)}_{i+1}" for i in range(max(1, unique_count))]


def generate_name() -> str:
    return fake.name()


def derive_email(name: str, domain: str = "exampleconsulting.com") -> str:
    parts = re.sub(r"[^a-zA-Z ]", "", name).lower().split()
    if len(parts) >= 2:
        local = f"{parts[0]}.{parts[-1]}"
    else:
        local = parts[0]
    suffix = str(random.randint(1, 999)).zfill(3)
    return f"{local}{suffix}@{domain}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="schema_profile.json")
    parser.add_argument("--output", default="synthetic_staff_talentlink_v3.xlsx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument(
        "--bio-style-uri",
        default="hf://datasets/lang-uk/recruitment-dataset-candidate-profiles-english/data/train-00000-of-00001.parquet",
    )
    parser.add_argument("--bio-style-column", default="Moreinfo")
    parser.add_argument("--local-style-workbook", default="/Users/danza/Downloads/Dummy_Profile_Examples.xlsx")
    args = parser.parse_args()

    log = make_logger(args.verbose)
    t0 = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    Faker.seed(args.seed)
    log("Random seeds initialized.")

    schema = json.loads(Path(args.schema).read_text())
    sheet = schema["sheets"][0]
    log("Schema loaded.")

    sheet_name = sheet["sheet_name"]
    header_row_zero_based = sheet["detected_header_row_zero_based"]
    header_row_excel = sheet["detected_header_row_excel"]
    data_start_row_excel = header_row_excel + 1

    columns = sheet["columns_in_order"]
    col_meta = {c["column_name"]: c for c in sheet["columns"]}

    n_rows = int(sheet["data_row_count"])
    log(f"Target rows: {n_rows}")

    # Optional external style inference; never copied directly into output.
    bio_examples = load_bio_examples(args.bio_style_uri, args.bio_style_column, log=log)
    bio_style = infer_bio_style(bio_examples)
    local_text_style = load_dummy_profile_style(args.local_style_workbook, log=log)
    local_example_pool = load_dummy_profile_examples(args.local_style_workbook, log=log)
    log(
        f"Bio style profile: avg_sentences={bio_style['avg_sentences']}, "
        f"avg_words_per_sentence={bio_style['avg_words_per_sentence']}, "
        f"bullet_bias={bio_style['bullet_bias']:.2f}"
    )

    id_meta = col_meta["Local employee ID"]
    employee_count = int(id_meta["repetition_profile"]["unique_count"])
    employee_row_counts = generate_employee_counts(employee_count, n_rows)

    # Efficient unique ID sampling without allocating a huge integer array.
    local_ids = random.sample(range(200000000, 1_000_000_000), k=employee_count)
    local_ids = [str(x) for x in local_ids]

    tl_lengths = [
        c["length"]
        for c in col_meta["TalentLink ID"]["length_profile"].get("common_lengths", [{"length": 7, "count": 1}])
    ]
    tl_weights = [
        c["count"]
        for c in col_meta["TalentLink ID"]["length_profile"].get("common_lengths", [{"length": 7, "count": 1}])
    ]

    talentlink_ids = []
    for _ in range(employee_count):
        ln = int(weighted_choice(tl_lengths, tl_weights, 1)[0])
        start = 10 ** (ln - 1)
        end = (10**ln) - 1
        talentlink_ids.append(str(random.randint(start, end)))

    employees = []
    for i in range(employee_count):
        name = generate_name()
        employees.append(
            {
                "Local employee ID": local_ids[i],
                "TalentLink ID": talentlink_ids[i],
                "Resource Name": name,
                "PwC email": derive_email(name),
                "Unnamed_BC": f"r{local_ids[i]}",
            }
        )
    log(f"Generated base employee table for {employee_count} unique employees.")

    # Build row-to-employee mapping (contiguous blocks to imitate grouped profile rows).
    employee_for_row = []
    for idx, ct in enumerate(employee_row_counts):
        employee_for_row.extend([idx] * int(ct))
    employee_for_row = employee_for_row[:n_rows]
    log("Built employee-to-row expansion mapping.")

    data = {c: [None] * n_rows for c in columns}

    base_categorical_columns = [
        "Resource Status",
        "Resource Type",
        "Management level",
        "Job level",
        "Global LoS",
        "Global Network",
        "Global Net Competency",
        "Go to Market",
        "PwC office",
        "PwC Territory",
        "Relationship Leader",
        "Career Coach",
        "Resource Management contact",
        "HR contact",
        "Local LoS",
        "Cost Centre",
        "Global Network__2",
        "Global Net Competency__2",
        "Go to Market__2",
        "Local SubLoS2",
        "Local SubLoS3",
        "Local SubLoS4",
        "Strategic Region",
        "Strategic Market",
    ]

    emp_profile = [{} for _ in range(employee_count)]

    for c in base_categorical_columns:
        meta = col_meta[c]
        u = int(meta.get("repetition_profile", {}).get("unique_count", 1) or 1)
        pool = synthesize_category_pool(c, u)
        if c in {"Interest", "In Process", "Travel Interest"}:
            pool = ["No", "Yes"]

        weights = np.arange(len(pool), 0, -1, dtype=float)
        if len(pool) == 1:
            weights = np.array([1.0])
        picks = weighted_choice(pool, weights, employee_count)

        for i in range(employee_count):
            emp_profile[i][c] = picks[i]
    log("Generated employee-level categorical profiles.")

    # Numeric/constant employee-level fields.
    grade_low = 5
    for i in range(employee_count):
        emp_profile[i]["Grade Code"] = str(grade_low + (i % 28))
        emp_profile[i]["30 day availability forecast"] = int(max(0, np.random.normal(35, 18)))
        # Keep near-constant hidden metadata-like fields synthetic.
        emp_profile[i]["Unnamed_BD"] = 48932.125
        emp_profile[i]["Unnamed_BE"] = "|Synthetic profile filter in 'Enabled'| |Synthetic domain in 'Practice 1'|"

        # Longer person-level summary text with variable lengths.
        summary_len = pick_length(col_meta["Summary/bio"]["length_profile"])
        emp_profile[i]["Summary/bio"] = choose_summary_bio(
            summary_len, local_text_style, local_example_pool["summaries"], employees[i]["Resource Name"]
        )
    log("Generated employee-level numeric, summary, and constant fields.")

    # Fill core person-consistent columns row by row (column-backed lists for speed).
    for r in range(n_rows):
        eidx = employee_for_row[r]
        emp = employees[eidx]
        profile = emp_profile[eidx]

        data["Local employee ID"][r] = emp["Local employee ID"]
        data["TalentLink ID"][r] = emp["TalentLink ID"]
        data["Resource Name"][r] = emp["Resource Name"]
        data["PwC email"][r] = emp["PwC email"]
        data["Unnamed_BC"][r] = emp["Unnamed_BC"]

        for c in base_categorical_columns:
            data[c][r] = profile[c]

        data["Grade Code"][r] = profile["Grade Code"]
        data["30 day availability forecast"][r] = profile["30 day availability forecast"]
        data["Summary/bio"][r] = profile["Summary/bio"]
        data["Unnamed_BD"][r] = profile["Unnamed_BD"]
        data["Unnamed_BE"][r] = profile["Unnamed_BE"]
        if (r + 1) % args.log_interval == 0:
            log(f"Core identity fill progress: {r + 1}/{n_rows} rows")

    # Build masks from gap metadata and target null counts.
    nonnull_masks = {
        c: build_mask(col_meta[c], n_rows, data_start_row_excel)
        for c in columns
    }
    log("Built per-column non-null masks.")

    # Empty columns remain empty by definition.
    for c in columns:
        if col_meta[c].get("semantic_type") == "empty":
            nonnull_masks[c][:] = False

    # Dependency repair: expiration date only when issued date exists.
    cert_issue_col = "Certification Issued Date"
    cert_exp_col = "Certification Expiration Date"
    bad = np.where(nonnull_masks[cert_exp_col] & ~nonnull_masks[cert_issue_col])[0]
    if len(bad) > 0:
        nonnull_masks[cert_exp_col][bad] = False
        candidates = np.where(nonnull_masks[cert_issue_col] & ~nonnull_masks[cert_exp_col])[0]
        if len(candidates) > 0:
            take = min(len(bad), len(candidates))
            chosen = np.random.choice(candidates, size=take, replace=False)
            nonnull_masks[cert_exp_col][chosen] = True

    # Dependency repair: job end date only when start date exists.
    job_start_col = "Job History Start Date"
    job_end_col = "Job History End Date"
    bad = np.where(nonnull_masks[job_end_col] & ~nonnull_masks[job_start_col])[0]
    if len(bad) > 0:
        nonnull_masks[job_end_col][bad] = False
        candidates = np.where(nonnull_masks[job_start_col] & ~nonnull_masks[job_end_col])[0]
        if len(candidates) > 0:
            take = min(len(bad), len(candidates))
            chosen = np.random.choice(candidates, size=take, replace=False)
            nonnull_masks[job_end_col][chosen] = True
    log("Applied dependent-field mask repairs for date consistency.")

    # Custom generation rules requested for profile-detail columns.
    custom_rule_columns = [
        "Skill Level 1",
        "Capability Name",
        "Proficiency",
        "Interest",
        "Certification Issued Date",
        "Certification Expiration Date",
        "Job History Start Date",
        "Job History End Date",
        "Position",
        "Company",
        "Location",
        "Description",
        "Education title/type of degree",
        "Area of Study",
        "Academic Institution",
        "In Process",
        "Travel Interest",
        "Project Type Interest",
    ]
    for c in custom_rule_columns:
        nonnull_masks[c][:] = True

    # Target row distribution using largest-remainder method.
    target_pct = {
        "Skills": 75.2,
        "Job History": 14.2,
        "Certifications/Credentials": 5.3,
        "Language": 2.9,
        "Education": 1.7,
        "Travel": 0.8,
    }
    target_counts = compute_target_counts(n_rows, target_pct)
    log(f"Target Skill Level 1 counts: {target_counts}")

    # Build row indices by employee (employee rows are contiguous blocks).
    row_indices_by_emp = [[] for _ in range(employee_count)]
    for r, eidx in enumerate(employee_for_row):
        row_indices_by_emp[eidx].append(r)

    # Reserve at least one Skills row per person, then allocate other types from remaining rows.
    reserved_skill_row = {}
    available_rows_by_person = {}
    for eidx, rows in enumerate(row_indices_by_emp):
        keep = random.choice(rows)
        reserved_skill_row[eidx] = keep
        available_rows_by_person[eidx] = set(rows)
        available_rows_by_person[eidx].remove(keep)

    selected_rows_by_type = {
        "Language": set(),
        "Certifications/Credentials": set(),
        "Job History": set(),
        "Education": set(),
        "Travel": set(),
    }
    per_person_counts = {
        "Language": np.zeros(employee_count, dtype=int),
        "Certifications/Credentials": np.zeros(employee_count, dtype=int),
        "Job History": np.zeros(employee_count, dtype=int),
        "Education": np.zeros(employee_count, dtype=int),
        "Travel": np.zeros(employee_count, dtype=int),
    }

    def pick_row_for_person(eidx: int):
        if not available_rows_by_person[eidx]:
            return None
        row = random.choice(tuple(available_rows_by_person[eidx]))
        available_rows_by_person[eidx].remove(row)
        return row

    def allocate_rows(kind: str, total: int, max_per_person: int, min_people: int = 0):
        eligible = [i for i in range(employee_count) if len(available_rows_by_person[i]) > 0]
        random.shuffle(eligible)
        base_people = min(min_people, total, len(eligible))
        assigned_people = 0
        for eidx in eligible:
            if assigned_people >= base_people:
                break
            if per_person_counts[kind][eidx] >= max_per_person:
                continue
            row = pick_row_for_person(eidx)
            if row is None:
                continue
            selected_rows_by_type[kind].add(row)
            per_person_counts[kind][eidx] += 1
            total -= 1
            assigned_people += 1
            if total == 0:
                break

        while total > 0:
            idx = [
                i
                for i in range(employee_count)
                if len(available_rows_by_person[i]) > 0 and per_person_counts[kind][i] < max_per_person
            ]
            if not idx:
                # Fallback: if caps deadlock while rows are still available, continue with available rows.
                idx = [i for i in range(employee_count) if len(available_rows_by_person[i]) > 0]
                if not idx:
                    raise RuntimeError(f"Cannot allocate remaining {kind} rows.")
            rem = np.array([len(available_rows_by_person[i]) for i in idx], dtype=float)
            cur = np.array([per_person_counts[kind][i] for i in idx], dtype=float)
            weights = rem / np.power(cur + 1.0, 1.2)
            weights = weights / weights.sum()
            eidx = int(np.random.choice(idx, p=weights))
            row = pick_row_for_person(eidx)
            if row is None:
                continue
            selected_rows_by_type[kind].add(row)
            per_person_counts[kind][eidx] += 1
            total -= 1

    # Allocate exact totals with realistic per-person caps and coverage.
    allocate_rows("Travel", target_counts["Travel"], max_per_person=1, min_people=target_counts["Travel"])
    allocate_rows("Education", target_counts["Education"], max_per_person=3, min_people=int(employee_count * 0.75))
    allocate_rows("Language", target_counts["Language"], max_per_person=5, min_people=int(employee_count * 0.6))
    allocate_rows("Certifications/Credentials", target_counts["Certifications/Credentials"], max_per_person=8, min_people=int(employee_count * 0.55))
    allocate_rows("Job History", target_counts["Job History"], max_per_person=80, min_people=int(employee_count * 0.9))

    # Final row type map.
    row_type_by_row = ["Skills"] * n_rows
    for kind, rows in selected_rows_by_type.items():
        for rr in rows:
            row_type_by_row[rr] = kind

    # Rebalance per-person distributions while preserving global totals.
    person_row_sets = [set(rows) for rows in row_indices_by_emp]

    def rows_of_type(eidx: int, rtype: str):
        return [r for r in person_row_sets[eidx] if row_type_by_row[r] == rtype]

    def rows_of_skills(eidx: int):
        return [r for r in person_row_sets[eidx] if row_type_by_row[r] == "Skills"]

    def swap_type_between_people(from_emp: int, to_emp: int, rtype: str):
        donor_rows = rows_of_type(from_emp, rtype)
        receiver_skill_rows = rows_of_skills(to_emp)
        if not donor_rows or len(receiver_skill_rows) <= 1:
            return False
        donor_row = random.choice(donor_rows)
        receiver_row = random.choice(receiver_skill_rows)
        row_type_by_row[donor_row] = "Skills"
        row_type_by_row[receiver_row] = rtype
        return True

    # Job History: cap at 25 rows/person and ensure broad coverage.
    max_job_per_person = 25
    changed = True
    while changed:
        changed = False
        donors = [i for i in range(employee_count) if len(rows_of_type(i, "Job History")) > max_job_per_person]
        receivers = [i for i in range(employee_count) if len(rows_of_type(i, "Job History")) < max_job_per_person]
        if donors and receivers:
            for d in donors:
                while len(rows_of_type(d, "Job History")) > max_job_per_person:
                    candidates = sorted(receivers, key=lambda x: len(rows_of_type(x, "Job History")))
                    moved = False
                    for rcv in candidates:
                        if rcv == d:
                            continue
                        if swap_type_between_people(d, rcv, "Job History"):
                            moved = True
                            changed = True
                            break
                    if not moved:
                        break

    # Language: most people 0-1 rows, fewer 2, very few >2.
    lang_max = 5
    changed = True
    while changed:
        changed = False
        donors = [i for i in range(employee_count) if len(rows_of_type(i, "Language")) > lang_max]
        receivers = [i for i in range(employee_count) if len(rows_of_type(i, "Language")) == 0]
        if donors and receivers:
            for d in donors:
                while len(rows_of_type(d, "Language")) > lang_max and receivers:
                    rcv = receivers.pop(0)
                    if swap_type_between_people(d, rcv, "Language"):
                        changed = True

    # Additional hard cap to avoid extreme language outliers.
    changed = True
    while changed:
        changed = False
        donors = [i for i in range(employee_count) if len(rows_of_type(i, "Language")) > 5]
        if not donors:
            break
        receivers = [i for i in range(employee_count) if len(rows_of_type(i, "Language")) <= 1]
        if not receivers:
            break
        for d in donors:
            while len(rows_of_type(d, "Language")) > 5 and receivers:
                rcv = random.choice(receivers)
                if swap_type_between_people(d, rcv, "Language"):
                    changed = True

    # Limit number of people with >2 language rows to a very small group.
    def people_with_lang_gt2():
        return [i for i in range(employee_count) if len(rows_of_type(i, "Language")) > 2]

    changed = True
    while changed:
        changed = False
        gt2 = people_with_lang_gt2()
        if len(gt2) <= 3:
            break
        receivers = [i for i in range(employee_count) if len(rows_of_type(i, "Language")) <= 1]
        if not receivers:
            break
        for d in gt2[3:]:
            if len(rows_of_type(d, "Language")) <= 2:
                continue
            for rcv in receivers:
                if swap_type_between_people(d, rcv, "Language"):
                    changed = True
                    break

    # Hard-enforce language max=5 while preserving global language row total.
    lang_deficit = 0
    for eidx in range(employee_count):
        lrows = rows_of_type(eidx, "Language")
        while len(lrows) > 5:
            rr = random.choice(lrows)
            row_type_by_row[rr] = "Skills"
            lang_deficit += 1
            lrows = rows_of_type(eidx, "Language")
    if lang_deficit > 0:
        for _ in range(lang_deficit):
            candidates = [
                i for i in range(employee_count)
                if len(rows_of_type(i, "Language")) <= 1 and len(rows_of_skills(i)) > 1
            ]
            if not candidates:
                candidates = [i for i in range(employee_count) if len(rows_of_skills(i)) > 1]
            if not candidates:
                break
            eidx = random.choice(candidates)
            rr = random.choice(rows_of_skills(eidx))
            row_type_by_row[rr] = "Language"

    # Generate row-level one-to-many profile details.
    company_pool = [fake.company() for _ in range(250)]
    position_pool = [
        "Analyst",
        "Senior Analyst",
        "Consultant",
        "Senior Consultant",
        "Manager",
        "Senior Manager",
        "Director",
        "Project Lead",
        "Data Engineer",
        "Risk Advisor",
        "Finance Associate",
        "Audit Associate",
        "Product Manager",
    ]
    location_pool = [
        "London, United Kingdom",
        "Birmingham, United Kingdom",
        "Manchester, United Kingdom",
        "Leeds, United Kingdom",
        "Edinburgh, United Kingdom",
        "Glasgow, United Kingdom",
        "Dublin, Ireland",
        "Remote",
        "Hybrid",
        "Client Site",
    ]
    skills_pool = [
        "Data Analysis",
        "Financial Modeling",
        "Regulatory Reporting",
        "Risk Assessment",
        "Internal Controls",
        "Cloud Architecture",
        "Cybersecurity Governance",
        "Python Development",
        "SQL Optimization",
        "Stakeholder Management",
        "Process Improvement",
        "Business Analysis",
        "Project Delivery",
        "Audit Planning",
        "Machine Learning",
        "Data Governance",
        "Tax Compliance",
        "Treasury Operations",
        "Strategic Planning",
        "Change Management",
    ]
    language_pool = [
        "English",
        "Spanish",
        "French",
        "German",
        "Arabic",
        "Mandarin",
        "Portuguese",
        "Hindi",
        "Italian",
        "Dutch",
        "Polish",
        "Urdu",
    ]
    proficiency_values = [
        "Advanced",
        "Beginner",
        "Elementary",
        "Intermediate",
        "Upper Advanced",
        "Upper Intermediate",
    ]
    cert_name_pool = [
        "AWS Certified Solutions Architect - Associate",
        "Microsoft Certified: Azure Fundamentals",
        "Google Professional Data Engineer",
        "Certified ScrumMaster (CSM)",
        "PRINCE2 Practitioner",
        "PMP - Project Management Professional",
        "CISA - Certified Information Systems Auditor",
        "CISM - Certified Information Security Manager",
        "CFA Level I",
        "ACCA Qualification",
        "Lean Six Sigma Green Belt",
        "ITIL 4 Foundation",
    ]
    degree_titles = [
        "Bachelor's degree",
        "Master's degree",
        "PhD",
        "Certificate",
        "Secondary education",
    ]
    degree_weights = [56, 28, 4, 6, 6]
    area_of_study_pool = [
        "Computer Science",
        "Accounting",
        "Economics",
        "Business Administration",
        "Law",
        "Data Science",
        "Information Systems",
        "Finance",
        "Engineering",
        "Other",
    ]
    area_weights = [14, 12, 10, 16, 8, 12, 10, 12, 4, 2]
    universities = [
        "University of Oxford",
        "University of Cambridge",
        "Imperial College London",
        "London School of Economics and Political Science",
        "King's College London",
        "University of Manchester",
        "University of Birmingham",
        "University of Leeds",
        "University of Edinburgh",
        "University of Bristol",
        "University College London",
        "University of Warwick",
    ]
    colleges = [
        "Open University",
        "Birkbeck, University of London",
        "City, University of London",
    ]
    schools = [
        "St Mary's Secondary School",
        "Greenfield Secondary School",
        "Westbrook High School",
    ]
    base_profile_date = datetime(2024, 1, 1)
    # Keep in-progress education very rare.
    education_rows = [r for r in range(n_rows) if row_type_by_row[r] == "Education"]
    in_progress_rows = set(random.sample(education_rows, k=min(3, len(education_rows))))

    # Person-level language sets to preserve consistency across repeated rows.
    person_languages = {}
    for eidx in range(employee_count):
        k = int(per_person_counts["Language"][eidx])
        if k <= 0:
            person_languages[eidx] = []
            continue
        langs = ["English"]
        if k > 1:
            others = [x for x in language_pool if x != "English"]
            langs.extend(random.sample(others, k=min(k - 1, len(others))))
        person_languages[eidx] = langs[:k]

    # Build per-person job chronology templates.
    person_job_timelines = {}
    for eidx in range(employee_count):
        k = int(per_person_counts["Job History"][eidx])
        if k <= 0:
            person_job_timelines[eidx] = []
            continue
        base = datetime(random.randint(2004, 2015), random.randint(1, 12), random.randint(1, 28))
        timeline = []
        prev_end = base
        for j in range(k):
            start = prev_end + timedelta(days=random.randint(20, 260))
            # Current role only optionally on latest role.
            if j == (k - 1) and random.random() < 0.18:
                end = None
                prev_end = start
            else:
                end = start + timedelta(days=random.randint(240, 1500))
                prev_end = end
            timeline.append((start, end))
        person_job_timelines[eidx] = timeline

    for r in range(n_rows):
        eidx = employee_for_row[r]
        row_type = row_type_by_row[r]

        # Skill Level 1 constrained categories (exact labels).
        data["Skill Level 1"][r] = row_type

        if row_type == "Skills":
            data["Capability Name"][r] = random.choice(skills_pool)
            if random.random() < 0.18:
                data["Interest"][r] = "Yes" if random.random() < 0.35 else "No"
        elif row_type == "Language":
            # Keep person-level language consistency.
            lang_rows = [idx for idx in row_indices_by_emp[eidx] if row_type_by_row[idx] == "Language"]
            lang_pos = lang_rows.index(r)
            lang = person_languages[eidx][lang_pos] if lang_pos < len(person_languages[eidx]) else "English"
            data["Capability Name"][r] = lang
            if lang == "English":
                data["Proficiency"][r] = "Upper Advanced" if random.random() < 0.92 else "Advanced"
            else:
                data["Proficiency"][r] = weighted_choice(proficiency_values, [16, 10, 14, 24, 22, 14], 1)[0]
            if random.random() < 0.10:
                data["Interest"][r] = "Yes" if random.random() < 0.4 else "No"
        elif row_type == "Certifications/Credentials":
            if random.random() < 0.97:
                data["Capability Name"][r] = random.choice(cert_name_pool)
            if random.random() < 0.12:
                data["Interest"][r] = "Yes" if random.random() < 0.35 else "No"

        # Certification rows.
        cert_issue = None
        if row_type == "Certifications/Credentials" and random.random() < 0.35:
            cert_issue = datetime(2017, 1, 1) + timedelta(days=random.randint(0, 3650))
            data["Certification Issued Date"][r] = format_legacy_date(cert_issue)

        if row_type == "Certifications/Credentials" and cert_issue is not None and random.random() < 0.22:
            if cert_issue is None:
                cert_issue = datetime(2017, 1, 1) + timedelta(days=random.randint(0, 3650))
            cert_exp = cert_issue + timedelta(days=random.randint(180, 1825))
            data["Certification Expiration Date"][r] = format_legacy_date(cert_exp)

        # Job history one-to-many rows with chronological ordering.
        if row_type == "Job History":
            j_rows = [idx for idx in row_indices_by_emp[eidx] if row_type_by_row[idx] == "Job History"]
            j_pos = j_rows.index(r)
            start, end = person_job_timelines[eidx][j_pos]
            data["Job History Start Date"][r] = format_legacy_date(start)
            ln = pick_length(col_meta["Position"]["length_profile"])
            data["Position"][r] = random.choice(position_pool)[:ln]
            ln = pick_length(col_meta["Company"]["length_profile"])
            data["Company"][r] = random.choice(company_pool)[:ln]
            if random.random() < 0.55:
                ln = pick_length(col_meta["Location"]["length_profile"])
                data["Location"][r] = random.choice(location_pool)[:ln]
            if random.random() < 0.85:
                ln = pick_length(col_meta["Description"]["length_profile"])
                data["Description"][r] = choose_role_description(
                    ln, local_text_style, local_example_pool["descriptions"], employees[eidx]["Resource Name"]
                )
            if end is not None:
                data["Job History End Date"][r] = format_legacy_date(end)

        # Education rows.
        if row_type == "Education":
            title = None
            if random.random() < 0.90:
                title = weighted_choice(degree_titles, degree_weights, 1)[0]
                ln = pick_length(col_meta["Education title/type of degree"]["length_profile"])
                data["Education title/type of degree"][r] = title[:ln]

            if random.random() < 0.88:
                area = weighted_choice(area_of_study_pool, area_weights, 1)[0]
                ln = pick_length(col_meta["Area of Study"]["length_profile"])
                data["Area of Study"][r] = area[:ln]

            # Academic institution always filled for education rows.
            if title and ("Bachelor" in title or "Master" in title or "PhD" in title):
                inst = random.choice(universities)
            elif title and ("Certificate" in title):
                inst = random.choice(colleges)
            elif title and ("Secondary" in title):
                inst = random.choice(schools)
            else:
                inst = random.choice(universities)
            ln = pick_length(col_meta["Academic Institution"]["length_profile"])
            data["Academic Institution"][r] = inst[:ln]

            in_progress = "Yes" if r in in_progress_rows else "No"
            if in_progress == "Yes" and title and ("PhD" in title or "Secondary" in title):
                data["Education title/type of degree"][r] = "Master's degree"
            data["In Process"][r] = in_progress

        # Travel rows (sparse fields, Travel Interest if populated is always Yes).
        if row_type == "Travel":
            if random.random() < 0.35:
                ln = pick_length(col_meta["Project Type Interest"]["length_profile"])
                data["Project Type Interest"][r] = text_to_target_length(ln, "sentence")
            if random.random() < 0.65:
                data["Travel Interest"][r] = "Yes"

        if nonnull_masks["Additional Information"][r]:
            ln = pick_length(col_meta["Additional Information"]["length_profile"])
            data["Additional Information"][r] = text_to_target_length(ln, "sentence")

        # Last edited date present in all rows.
        if nonnull_masks["Profile Last Edited (YYYY/MM/DD)"][r]:
            dt = base_profile_date + timedelta(days=random.randint(0, 730))
            data["Profile Last Edited (YYYY/MM/DD)"][r] = format_iso_midnight(dt)
        if (r + 1) % args.log_interval == 0:
            log(f"Profile-detail generation progress: {r + 1}/{n_rows} rows")

    # Enforce strict conditional blanking for controlled columns.
    allowed_by_type = {
        "Skills": {"Capability Name", "Interest"},
        "Language": {"Capability Name", "Proficiency", "Interest"},
        "Certifications/Credentials": {
            "Capability Name",
            "Interest",
            "Certification Issued Date",
            "Certification Expiration Date",
        },
        "Job History": {
            "Job History Start Date",
            "Job History End Date",
            "Position",
            "Company",
            "Location",
            "Description",
        },
        "Education": {
            "Education title/type of degree",
            "Area of Study",
            "Academic Institution",
            "In Process",
        },
        "Travel": {"Project Type Interest", "Travel Interest"},
    }
    for r in range(n_rows):
        rtype = data["Skill Level 1"][r]
        allowed = allowed_by_type.get(rtype, set())
        for col in custom_rule_columns:
            if col == "Skill Level 1":
                continue
            if col not in allowed:
                data[col][r] = None

    # Apply null masks strictly at the end to preserve required sparsity patterns.
    for c in columns:
        mask = nonnull_masks[c]
        col_arr = np.array(data[c], dtype=object)
        col_arr[~mask] = None
        data[c] = col_arr.tolist()
    log("Applied final null masks.")

    # Enforce derived-field consistency after final masking.
    for r in range(n_rows):
        if data["Local employee ID"][r] is not None:
            data["Unnamed_BC"][r] = f"r{data['Local employee ID'][r]}"
    log("Enforced derived ID consistency.")

    df = pd.DataFrame(data, columns=columns)

    # Excel output with header row at the detected position (blank rows above).
    output_path = Path(args.output)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            startrow=header_row_zero_based,
        )
    log(f"Workbook written: {output_path}")
    log(f"Completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
