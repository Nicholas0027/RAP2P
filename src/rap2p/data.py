"""SocioBench ingestion, canonical demographics, and the three leakage-safe splits
(ID-respondent, OOD-intersection, OOD-item) built from a single partition.

Raw file layout expected under `paths.sociobench_repo` (matches
https://github.com/JiaWANG-TJ/SocioBench):

    Dataset_all/q&a/issp_qa_{domain}.json
        -> list of {"question_id", "question", "answer": {code: text, ...}, ...}
    Dataset_all/A_GroundTruth_sampling500/issp_answer_{domain}.json
        -> list of {"person_id", "attributes": {<40+ ISSP codebook label -> value>},
                    "questions_answer": {question_id (lowercase): answer_code, ...}}

Only question ids present in the domain's q&a whitelist are kept; anything else
(e.g. country-specific derived fields) is dropped before any demographic or
leakage logic runs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import stable_int, write_json

# ---------------------------------------------------------------------------
# Canonical demographics
# ---------------------------------------------------------------------------

ATTRIBUTE_PATTERNS: dict[str, list[str]] = {
    "country": [r"country.*iso\s*3166.*alphanumeric", r"^country prefix iso\s*3166"],
    "sex": [r"^sex of respondent$", r"^sex$"],
    "age": [r"^age of respondent$"],
    "education": [
        r"isced.*simplified.*highest completed degree",
        r"isced.*highest completed degree",
        r"highest completed (degree of )?education.*international comparison",
        r"comparative.*highest completed degree of education",
    ],
    "employment": [r"^currently, formerly, or never in paid work$", r"^main status$"],
    "marital_status": [r"^living in steady partnership$", r"^legal partnership status$"],
    "urbanicity": [r"place of living.*urban.*rural", r"urban.*rural"],
    "subjective_income": [r"^top-bottom self-placement", r"subjective.*(income|class|social position)"],
}

MISSING_PATTERNS = (
    "nap",
    "not applicable",
    "no answer",
    "don't know",
    "do not know",
    "not available",
    "refused",
)

CANONICAL_DEMOGRAPHIC_COLUMNS = [
    "country",
    "sex",
    "age_bin",
    "education",
    "income_quintile",
    "employment",
    "marital_status",
    "urbanicity",
]

INTERSECTION_DEFAULT_ATTRIBUTES = ["age_bin", "education", "income_quintile", "urbanicity"]


def _clean_value(value: Any) -> str:
    if value is None:
        return "Missing"
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or any(pattern in text.lower() for pattern in MISSING_PATTERNS):
        return "Missing"
    return text


def _find_attribute(attributes: Mapping[str, Any], patterns: Iterable[str]) -> str:
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for key, value in attributes.items():
            if regex.search(key):
                cleaned = _clean_value(value)
                if cleaned != "Missing":
                    return cleaned
    return "Missing"


def _parse_age(value: str) -> float | None:
    match = re.search(r"\d{1,3}", value)
    if not match:
        return None
    age = int(match.group())
    return float(age) if 14 <= age <= 110 else None


def age_bin(value: str) -> str:
    age = _parse_age(value)
    if age is None:
        return "Missing"
    if age < 30:
        # ISSP samples adults (18+); _parse_age tolerates 14-17 for robustness,
        # and any such respondent lands in this youngest bucket despite its label.
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 65:
        return "45-64"
    return "65+"


def income_quintile(value: str) -> str:
    """Bin a self-placement top-bottom scale (assumed 1=top .. 10=bottom, the
    common ISSP encoding) into quintiles Q5 (top) .. Q1 (bottom). Falls back to
    'Missing' whenever the raw value cannot be parsed as a number in [1, 10];
    verify this assumption against the actual codebook entry for each domain
    before trusting income_quintile for anything beyond coarse stratification.
    """
    match = re.search(r"\d{1,2}", value)
    if not match:
        return "Missing"
    scale = int(match.group())
    if not 1 <= scale <= 10:
        return "Missing"
    # scale=1 is "top" -> highest quintile (Q5); scale=10 is "bottom" -> Q1.
    inverted = 11 - scale
    quintile = min(5, max(1, math.ceil(inverted / 2)))
    return f"Q{quintile}"


def canonicalize_attributes(attributes: Mapping[str, Any]) -> dict[str, str]:
    canonical = {name: _find_attribute(attributes, patterns) for name, patterns in ATTRIBUTE_PATTERNS.items()}
    canonical["age_bin"] = age_bin(canonical.pop("age"))
    canonical["income_quintile"] = income_quintile(canonical.pop("subjective_income"))
    return {key: canonical[key] for key in CANONICAL_DEMOGRAPHIC_COLUMNS}


def demographic_record_fields(row: Any) -> dict[str, Any]:
    """Every prediction record (baselines and inference.py alike) should carry
    these so eval/metrics.py's group_js_metrics/worst_group_accuracy and
    eval/heterogeneity.py's matching can operate on *any* method's output, not
    just the ones produced by inference.py.
    """
    return {column: getattr(row, column) for column in CANONICAL_DEMOGRAPHIC_COLUMNS}


def format_demographics(canonical: Mapping[str, str]) -> str:
    labels = {
        "country": "Country",
        "sex": "Gender",
        "age_bin": "Age group",
        "education": "Education",
        "income_quintile": "Income quintile (self-reported)",
        "employment": "Employment",
        "marital_status": "Marital/partnership status",
        "urbanicity": "Urbanicity",
    }
    return "; ".join(f"{labels[key]}: {canonical.get(key, 'Missing')}" for key in labels)


def extract_weight(attributes: Mapping[str, Any]) -> float:
    preferred = [r"^combination of all weights$", r"^weighting factor$", r"^weight$", r"survey weight"]
    for pattern in preferred:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for key, value in attributes.items():
            if regex.search(key):
                try:
                    number = float(str(value).replace(",", "."))
                    if math.isfinite(number) and number > 0:
                        return number
                except (TypeError, ValueError):
                    continue
    return 1.0


def _option_sort_key(code: str) -> tuple[int, float | str]:
    try:
        return (0, float(code))
    except ValueError:
        return (1, code)


def infer_ordinal(question: str, options: list[str]) -> bool:
    cues = (
        "agree", "satisfied", "important", "often", "likely", "good or bad",
        "increase or decrease", "too high", "strongly", "always", "never",
        "scale of", "very happy", "extent",
    )
    text = (question + " " + " ".join(options)).lower()
    return any(cue in text for cue in cues)


def load_domain(repo: str | Path, domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repo = Path(repo)
    qa_path = repo / "Dataset_all" / "q&a" / f"issp_qa_{domain}.json"
    answer_path = repo / "Dataset_all" / "A_GroundTruth_sampling500" / f"issp_answer_{domain}.json"
    if not qa_path.exists() or not answer_path.exists():
        raise FileNotFoundError(f"Missing official SocioBench files for {domain}: {qa_path}, {answer_path}")
    return json.loads(qa_path.read_text(encoding="utf-8")), json.loads(answer_path.read_text(encoding="utf-8"))


def parse_sociobench(repo: str | Path, domains: Iterable[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse raw SocioBench files into one row per (respondent, valid answered item).

    No item/country curation and no split assignment happens here -- see
    `curate_items_and_countries` and `assign_splits` for that. Keeping parsing
    separate from curation means the audit trail always shows exactly how much
    was dropped and why.
    """
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"domains": {}, "canonical_attribute_missing": Counter()}

    for domain in domains:
        questions, respondents = load_domain(repo, domain)
        qmap = {str(item["question_id"]).lower(): item for item in questions}
        domain_audit: Counter = Counter(respondents=len(respondents), whitelist_questions=len(qmap))

        for respondent in respondents:
            person_id = str(respondent["person_id"])
            panel_id = f"{domain}::{person_id}"
            attributes = respondent.get("attributes") or {}
            canonical = canonicalize_attributes(attributes)
            for key, value in canonical.items():
                if value == "Missing":
                    audit["canonical_attribute_missing"][key] += 1
            demographic_text = format_demographics(canonical)
            weight = extract_weight(attributes)

            for raw_qid, raw_answer in (respondent.get("questions_answer") or {}).items():
                domain_audit["raw_response_fields"] += 1
                qid = str(raw_qid).lower()
                question = qmap.get(qid)
                if question is None:
                    domain_audit["excluded_nonwhitelist_fields"] += 1
                    continue
                answer_map = {str(k): str(v) for k, v in question["answer"].items()}
                answer_code = str(raw_answer)
                if answer_code not in answer_map:
                    domain_audit["excluded_invalid_answers"] += 1
                    continue
                option_codes = sorted(answer_map, key=_option_sort_key)
                options = [answer_map[code] for code in option_codes]
                answer_index = option_codes.index(answer_code)
                is_ordinal = infer_ordinal(str(question["question"]), options)
                if not is_ordinal:
                    domain_audit["excluded_non_ordinal"] += 1
                    continue
                record = {
                    "row_id": f"{panel_id}::{qid}",
                    "panel_id": panel_id,
                    "person_id": person_id,
                    "domain": domain,
                    "question_id": qid,
                    "question_key": f"{domain}::{qid}",
                    "question": str(question["question"]).strip(),
                    "option_codes_json": json.dumps(option_codes),
                    "options_json": json.dumps(options, ensure_ascii=False),
                    "n_options": len(options),
                    "answer_code": answer_code,
                    "answer_index": answer_index,
                    "answer_text": answer_map[answer_code],
                    "is_ordinal": True,
                    "survey_weight": weight,
                    "demographic_text": demographic_text,
                    **canonical,
                }
                rows.append(record)
                domain_audit["valid_responses"] += 1

        audit["domains"][domain] = dict(domain_audit)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No valid ordinal SocioBench rows were parsed")
    if frame["row_id"].duplicated().any():
        raise ValueError("Duplicate panel-question rows found during parsing")
    audit["canonical_attribute_missing"] = dict(audit["canonical_attribute_missing"])
    audit["total_valid_rows"] = int(len(frame))
    audit["total_panels"] = int(frame["panel_id"].nunique())
    return frame, audit


# ---------------------------------------------------------------------------
# Curation: item selection, country selection, minimum-answered filter
# ---------------------------------------------------------------------------


def curate_items_and_countries(
    frame: pd.DataFrame,
    max_items_per_domain: int,
    countries_per_domain: int,
    max_respondents_per_country: int,
    min_answered_items_per_respondent: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Frozen-before-modeling dataset curation: highest-coverage ordinal items,
    largest-sample countries, deterministic per-country capping, and a
    minimum-answered-items respondent filter. None of these choices depend on
    any model's predictions.
    """
    audit: dict[str, Any] = {}
    kept_items: dict[str, list[str]] = {}
    for domain, group in frame.groupby("domain"):
        coverage = group.groupby("question_key")["panel_id"].nunique()
        # (-count, key) tuple sort: exact coverage ties at the cutoff boundary
        # break deterministically by key, not by pandas' non-stable quicksort.
        ranked = sorted(coverage.items(), key=lambda pair: (-pair[1], pair[0]))
        kept_items[domain] = [key for key, _ in ranked[:max_items_per_domain]]
    audit["kept_items_per_domain"] = {domain: len(items) for domain, items in kept_items.items()}

    frame = frame[frame.apply(lambda row: row["question_key"] in kept_items[row["domain"]], axis=1)].copy()

    kept_countries: dict[str, list[str]] = {}
    for domain, group in frame.groupby("domain"):
        sizes = group.drop_duplicates("panel_id").groupby("country")["panel_id"].nunique()
        sizes = sizes.drop(index="Missing", errors="ignore")
        ranked = sorted(sizes.items(), key=lambda pair: (-pair[1], pair[0]))
        kept_countries[domain] = [key for key, _ in ranked[:countries_per_domain]]
    audit["kept_countries_per_domain"] = kept_countries

    frame = frame[frame.apply(lambda row: row["country"] in kept_countries[row["domain"]], axis=1)].copy()

    # Deterministic per-(domain, country) respondent cap.
    panels = frame[["panel_id", "domain", "country"]].drop_duplicates("panel_id")
    keep_panel_ids: set[str] = set()
    for (domain, country), group in panels.groupby(["domain", "country"]):
        ranked = sorted(group["panel_id"], key=lambda pid: stable_int(seed, "country_cap", pid))
        keep_panel_ids.update(ranked[:max_respondents_per_country])
    frame = frame[frame["panel_id"].isin(keep_panel_ids)].copy()

    # Minimum-answered-items filter (counts only kept items, post country cap).
    answered_counts = frame.groupby("panel_id")["question_key"].nunique()
    eligible_panels = answered_counts[answered_counts >= min_answered_items_per_respondent].index
    dropped_sparse = int((~frame["panel_id"].isin(eligible_panels)).sum())
    frame = frame[frame["panel_id"].isin(eligible_panels)].copy()

    audit["dropped_rows_sparse_respondents"] = dropped_sparse
    audit["final_rows"] = int(len(frame))
    audit["final_panels"] = int(frame["panel_id"].nunique())
    audit["final_panels_per_domain"] = frame.groupby("domain")["panel_id"].nunique().to_dict()
    return frame, audit


# ---------------------------------------------------------------------------
# Item holdout (OOD-Item axis)
# ---------------------------------------------------------------------------


def assign_item_holdout(frame: pd.DataFrame, unseen_fraction: float, seed: int) -> pd.DataFrame:
    """Mark a deterministic, frozen-before-modeling subset of items per domain
    as `is_unseen_item`. Unseen items are removed from `calibration_orders`
    entirely (see below) so they can never leak into any respondent's history.
    """
    output = frame.copy()
    output["is_unseen_item"] = False
    for domain, group in output.groupby("domain"):
        items = sorted(group["question_key"].unique())
        ranked = sorted(items, key=lambda qk: stable_int(seed, "item_holdout", qk))
        n_unseen = max(1, round(len(items) * unseen_fraction))
        unseen = set(ranked[:n_unseen])
        output.loc[output["question_key"].isin(unseen) & (output["domain"] == domain), "is_unseen_item"] = True
    return output


# ---------------------------------------------------------------------------
# Splits: ID-respondent (stratified) + OOD-intersection carve-out
# ---------------------------------------------------------------------------


def _collapse_strata(panels: pd.DataFrame, minimum: int = 5) -> pd.Series:
    candidates = [
        ["domain", "country", "sex", "age_bin"],
        ["domain", "country", "sex"],
        ["domain", "country"],
        ["domain"],
    ]
    strata = pd.Series(index=panels.index, dtype="object")
    unresolved = pd.Series(True, index=panels.index)
    for columns in candidates:
        labels = panels[columns].fillna("Missing").astype(str).agg("|".join, axis=1)
        counts = labels.loc[unresolved].value_counts()
        usable = labels.map(counts).ge(minimum) & unresolved
        strata.loc[usable] = labels.loc[usable]
        unresolved &= ~usable
    strata.loc[unresolved] = panels.loc[unresolved, "domain"].astype(str)
    return strata


def assign_intersection_holdout(
    frame: pd.DataFrame,
    attributes: list[str],
    holdout_fraction: float,
    min_cell_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hold out entire demographic-intersection cells: every respondent in a
    held-out cell is removed from the train/validation/test pool and placed in
    a dedicated `ood_intersection` split. Training still sees every individual
    attribute value, just never that exact combination.
    """
    panels = frame[["panel_id", "domain", *attributes]].drop_duplicates("panel_id").copy()
    panels["cell"] = panels[attributes].astype(str).agg("|".join, axis=1)
    cell_sizes = panels.groupby(["domain", "cell"])["panel_id"].nunique()
    valid_cells = cell_sizes[cell_sizes >= min_cell_size]

    held_out_cells: set[tuple[str, str]] = set()
    for domain in valid_cells.index.get_level_values("domain").unique():
        domain_cells = sorted(valid_cells.loc[domain].index.tolist())
        ranked = sorted(domain_cells, key=lambda cell: stable_int(seed, "intersection_holdout", domain, cell))
        n_holdout = max(1, round(len(domain_cells) * holdout_fraction))
        held_out_cells.update((domain, cell) for cell in ranked[:n_holdout])

    panels["is_ood_intersection"] = panels.apply(
        lambda row: (row["domain"], row["cell"]) in held_out_cells, axis=1
    )
    holdout_panel_ids = set(panels.loc[panels["is_ood_intersection"], "panel_id"])
    audit = {
        "n_valid_cells_per_domain": valid_cells.groupby("domain").size().to_dict() if len(valid_cells) else {},
        "n_held_out_cells": len(held_out_cells),
        "n_held_out_panels": len(holdout_panel_ids),
    }
    output = frame.copy()
    output["is_ood_intersection"] = output["panel_id"].isin(holdout_panel_ids)
    return output, audit


def assign_splits(
    frame: pd.DataFrame,
    seed: int,
    ratios: tuple[float, float, float],
    intersection_attributes: list[str],
    intersection_holdout_fraction: float,
    min_cell_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_ratio, val_ratio, test_ratio = ratios
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to one")

    frame, intersection_audit = assign_intersection_holdout(
        frame, intersection_attributes, intersection_holdout_fraction, min_cell_size, seed
    )

    panel_columns = ["panel_id", "domain", "country", "sex", "age_bin", "is_ood_intersection"]
    panels = frame[panel_columns].drop_duplicates("panel_id").reset_index(drop=True)
    id_pool = panels[~panels["is_ood_intersection"]].reset_index(drop=True)
    strata = _collapse_strata(id_pool)

    train_idx, hold_idx = train_test_split(
        id_pool.index, test_size=val_ratio + test_ratio, random_state=seed, stratify=strata
    )
    hold = id_pool.loc[hold_idx]
    hold_strata = _collapse_strata(hold, minimum=5)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        hold.index, test_size=relative_test, random_state=seed + 1, stratify=hold_strata
    )

    split_map: dict[str, str] = {panel_id: "ood_intersection" for panel_id in panels.loc[panels["is_ood_intersection"], "panel_id"]}
    split_map.update({panel_id: "train" for panel_id in id_pool.loc[train_idx, "panel_id"]})
    split_map.update({panel_id: "validation" for panel_id in id_pool.loc[val_idx, "panel_id"]})
    split_map.update({panel_id: "test" for panel_id in id_pool.loc[test_idx, "panel_id"]})

    output = frame.copy()
    output["split"] = output["panel_id"].map(split_map)
    if output["split"].isna().any():
        raise RuntimeError("Some panels were not assigned to a split")
    return output, intersection_audit


def add_question_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    # Idempotent: recompute from scratch if stat columns already exist
    # (e.g. re-running on an already-processed frame).
    output = output.drop(columns=[c for c in ("question_mean", "question_std") if c in output.columns])
    denominator = np.maximum(output["n_options"].to_numpy() - 1, 1)
    output["normalized_answer"] = output["answer_index"].to_numpy() / denominator
    # Train respondents AND seen items only: per-item stats for an unseen item
    # would be derived from real train answers to that item -- currently these
    # columns are informational only (no loss reads them), but the exclusion
    # keeps them safe to wire into a loss later. Unseen items get neutral
    # defaults (midpoint mean, wide std) instead of leaked estimates.
    seen_train = output[output["split"].eq("train") & ~output["is_unseen_item"]]
    stats = (
        seen_train
        .groupby("question_key")["normalized_answer"]
        .agg(question_mean="mean", question_std="std")
    )
    stats["question_std"] = stats["question_std"].fillna(0.25).clip(lower=0.05)
    output = output.join(stats, on="question_key")
    output["question_mean"] = output["question_mean"].fillna(0.5)
    output["question_std"] = output["question_std"].fillna(0.25)
    return output


# ---------------------------------------------------------------------------
# Calibration orders (history/target eligibility per K, leakage-safe for
# unseen items: they are never inserted into an order, so they can never be
# sampled as history for anyone, and calibration draws never reach into them.)
# ---------------------------------------------------------------------------


def calibration_orders(frame: pd.DataFrame, seeds: Iterable[int]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen_only = frame[~frame["is_unseen_item"]]
    for panel_id, group in seen_only.groupby("panel_id", sort=True):
        qids = group["question_id"].astype(str).tolist()
        for seed in seeds:
            ordered = sorted(qids, key=lambda qid: stable_int(seed, panel_id, qid))
            records.append({"panel_id": panel_id, "seed": int(seed), "ordered_question_ids_json": json.dumps(ordered)})
    return pd.DataFrame(records)


def validate_no_leakage(frame: pd.DataFrame) -> dict[str, Any]:
    split_counts = frame.groupby("panel_id")["split"].nunique()
    row_unique = not frame["row_id"].duplicated().any()
    unseen_in_ood_intersection_train = frame[frame["split"].eq("train") & frame["is_unseen_item"]]
    report = {
        "panel_in_one_split": bool(split_counts.max() == 1),
        "row_ids_unique": bool(row_unique),
        "panels_by_split": frame.groupby("split")["panel_id"].nunique().to_dict(),
        "rows_by_split": frame["split"].value_counts().to_dict(),
        "unseen_item_rows_in_train": int(len(unseen_in_ood_intersection_train)),
    }
    if not report["panel_in_one_split"] or not report["row_ids_unique"]:
        raise AssertionError(f"Leakage audit failed (split/row_id): {report}")
    return report


def validate_item_holdout_leakage(frame: pd.DataFrame, orders: pd.DataFrame) -> dict[str, Any]:
    """Confirm unseen items never appear inside any calibration-order sequence
    (i.e. can never be drawn as K-shot history for any respondent, train or
    test) -- the guard the OOD-Item axis depends on.
    """
    unseen_keys = set(frame.loc[frame["is_unseen_item"], "question_id"].astype(str))
    offenders = 0
    for row in orders.itertuples(index=False):
        ordered = set(json.loads(row.ordered_question_ids_json))
        offenders += len(ordered & unseen_keys)
    report = {"unseen_items": len(unseen_keys), "unseen_items_found_in_calibration_orders": offenders}
    if offenders:
        raise AssertionError(f"Item-holdout leakage detected: {report}")
    return report


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def prepare_sociobench(config: Mapping[str, Any]) -> dict[str, Any]:
    repo = Path(config["paths"]["sociobench_repo"])
    output_dir = Path(config["paths"]["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = config["data"]
    seed = int(config["seed"])

    frame, parse_audit = parse_sociobench(repo, data_config["domains"])
    frame, curation_audit = curate_items_and_countries(
        frame,
        max_items_per_domain=int(data_config["max_items_per_domain"]),
        countries_per_domain=int(data_config["countries_per_domain"]),
        max_respondents_per_country=int(data_config["max_respondents_per_country"]),
        min_answered_items_per_respondent=int(data_config["min_answered_items_per_respondent"]),
        seed=seed,
    )
    frame = assign_item_holdout(frame, float(data_config["unseen_item_fraction"]), seed)
    frame, intersection_audit = assign_splits(
        frame,
        seed,
        tuple(float(x) for x in data_config["respondent_split"]),
        list(data_config["intersection_attributes"]),
        float(data_config["intersection_holdout_fraction"]),
        int(data_config["min_cell_size"]),
    )
    frame = add_question_statistics(frame)
    all_seeds = [int(data_config["calibration_seed"]), *[int(s) for s in data_config["calibration_seeds_secondary"]]]
    orders = calibration_orders(frame, all_seeds)
    leakage = validate_no_leakage(frame)
    item_leakage = validate_item_holdout_leakage(frame, orders)

    frame.to_parquet(output_dir / "responses.parquet", index=False)
    orders.to_parquet(output_dir / "calibration_orders.parquet", index=False)
    panel_columns = ["panel_id", "person_id", "domain", "split", "demographic_text", *CANONICAL_DEMOGRAPHIC_COLUMNS, "survey_weight"]
    frame[panel_columns].drop_duplicates("panel_id").to_parquet(output_dir / "panels.parquet", index=False)
    item_columns = ["question_key", "domain", "question_id", "question", "is_unseen_item"]
    frame[item_columns].drop_duplicates("question_key").to_parquet(output_dir / "items.parquet", index=False)

    audit = {
        "parse": parse_audit,
        "curation": curation_audit,
        "intersection_holdout": intersection_audit,
        "leakage": leakage,
        "item_holdout_leakage": item_leakage,
    }
    write_json(output_dir / "audit.json", audit)
    return audit


class PanelStore:
    """In-memory access to item rows, deterministic K-shot histories, and the
    three evaluation splits (id-respondent test / ood_intersection / unseen items).
    """

    def __init__(self, responses: pd.DataFrame, calibration_orders_frame: pd.DataFrame):
        self.responses = responses.reset_index(drop=True)
        self.by_panel = {
            panel_id: group.sort_values("question_id").reset_index(drop=True)
            for panel_id, group in self.responses.groupby("panel_id", sort=False)
        }
        self.order_lookup: dict[tuple[str, int], list[str]] = {}
        for row in calibration_orders_frame.itertuples(index=False):
            self.order_lookup[(row.panel_id, int(row.seed))] = json.loads(row.ordered_question_ids_json)

    @classmethod
    def from_dir(cls, processed_dir: str | Path) -> "PanelStore":
        processed_dir = Path(processed_dir)
        return cls(
            pd.read_parquet(processed_dir / "responses.parquet"),
            pd.read_parquet(processed_dir / "calibration_orders.parquet"),
        )

    def history_rows(self, panel_id: str, target_question_id: str, k: int, seed: int) -> pd.DataFrame:
        if k <= 0:
            return self.by_panel[panel_id].iloc[0:0].copy()
        ordered = [qid for qid in self.order_lookup[(panel_id, seed)] if qid != target_question_id]
        selected = set(ordered[:k])
        panel = self.by_panel[panel_id]
        order_rank = {qid: rank for rank, qid in enumerate(ordered[:k])}
        history = panel[panel["question_id"].isin(selected)].copy()
        history["_rank"] = history["question_id"].map(order_rank)
        return history.sort_values("_rank").drop(columns="_rank")

    def target_rows(
        self,
        split: str,
        k: int,
        seed: int,
        domains: Iterable[str] | None = None,
        item_pool: str = "seen",
    ) -> pd.DataFrame:
        """`split` accepts "train" / "validation" / "test" / "ood_intersection".
        `item_pool` selects which items are eligible as targets:
        "seen" (default, ID-respondent + OOD-intersection axes),
        "unseen" (OOD-Item axis), or "all".
        """
        frame = self.responses[self.responses["split"].eq(split)]
        if domains is not None:
            frame = frame[frame["domain"].isin(list(domains))]
        if item_pool == "seen":
            frame = frame[~frame["is_unseen_item"]]
        elif item_pool == "unseen":
            frame = frame[frame["is_unseen_item"]]
        elif item_pool != "all":
            raise ValueError(f"Unknown item_pool: {item_pool}")
        if k <= 0:
            return frame.copy()
        keep: list[bool] = []
        for row in frame.itertuples(index=False):
            calibration = set(self.order_lookup[(row.panel_id, int(seed))][:k])
            keep.append(row.question_id not in calibration)
        return frame.loc[np.asarray(keep)].copy()


# ---------------------------------------------------------------------------
# Synthetic data for the zero-GPU smoke path (RAP2P_SMOKE=1). Never used to
# produce a reported number -- see README "Running it".
# ---------------------------------------------------------------------------


def make_synthetic_panels(
    n_panels: int = 40,
    n_items_per_domain: int = 10,
    n_unseen_items: int = 2,
    domains: tuple[str, ...] = ("Environment", "Role of Government"),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sexes = ["Male", "Female"]
    age_bins = ["18-29", "30-44", "45-64", "65+"]
    educations = ["Primary", "Secondary", "Tertiary"]
    quintiles = [f"Q{i}" for i in range(1, 6)]
    employments = ["Employed", "Unemployed", "Retired"]
    maritals = ["Married", "Single", "Divorced"]
    urbanicities = ["Urban", "Rural"]
    countries = ["A", "B", "C", "D"]

    rows: list[dict[str, Any]] = []
    for domain in domains:
        item_ids = [f"v{i}" for i in range(n_items_per_domain)]
        unseen = set(item_ids[:n_unseen_items])
        for panel_index in range(n_panels):
            person_id = f"{domain}_{panel_index}"
            panel_id = f"{domain}::{person_id}"
            canonical = {
                "country": countries[panel_index % len(countries)],
                "sex": sexes[panel_index % 2],
                "age_bin": age_bins[panel_index % 4],
                "education": educations[panel_index % 3],
                "income_quintile": quintiles[panel_index % 5],
                "employment": employments[panel_index % 3],
                "marital_status": maritals[panel_index % 3],
                "urbanicity": urbanicities[panel_index % 2],
            }
            demographic_text = format_demographics(canonical)
            latent = rng.normal()
            for item_id in item_ids:
                n_options = 5
                score = int(np.clip(round(2 + latent + rng.normal(scale=0.5)), 0, n_options - 1))
                rows.append(
                    {
                        "row_id": f"{panel_id}::{item_id}",
                        "panel_id": panel_id,
                        "person_id": person_id,
                        "domain": domain,
                        "question_id": item_id,
                        "question_key": f"{domain}::{item_id}",
                        "question": f"Synthetic ordinal question {item_id} for {domain}",
                        "option_codes_json": json.dumps([str(i) for i in range(n_options)]),
                        "options_json": json.dumps([f"Option {i}" for i in range(n_options)]),
                        "n_options": n_options,
                        "answer_code": str(score),
                        "answer_index": score,
                        "answer_text": f"Option {score}",
                        "is_ordinal": True,
                        "survey_weight": 1.0,
                        "demographic_text": demographic_text,
                        "is_unseen_item": item_id in unseen,
                        **canonical,
                    }
                )
    frame = pd.DataFrame(rows)
    frame, _ = assign_splits(frame, seed, (0.7, 0.1, 0.2), INTERSECTION_DEFAULT_ATTRIBUTES, 0.15, min_cell_size=2)
    frame = add_question_statistics(frame)
    orders = calibration_orders(frame, [seed])
    return frame, orders
