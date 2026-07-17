from __future__ import annotations

import json

from rap2p.data import (
    PanelStore,
    add_question_statistics,
    age_bin,
    assign_item_holdout,
    assign_splits,
    calibration_orders,
    canonicalize_attributes,
    curate_items_and_countries,
    income_quintile,
    parse_sociobench,
    validate_item_holdout_leakage,
    validate_no_leakage,
)


def _write_toy_repo(tmp_path, n_respondents: int = 200, n_items: int = 12):
    repo = tmp_path / "SocioBench"
    qa_dir = repo / "Dataset_all" / "q&a"
    answer_dir = repo / "Dataset_all" / "A_GroundTruth_sampling500"
    qa_dir.mkdir(parents=True)
    answer_dir.mkdir(parents=True)

    ordinal_stub = "How much do you agree? Please answer on a scale of strongly agree to strongly disagree."
    questions = [
        {"question_id": f"V{i}", "question": ordinal_stub, "answer": {"1": "Agree", "2": "Neither", "3": "Disagree"}}
        for i in range(n_items)
    ]
    countries = ["A", "B", "C", "D", "E"]
    respondents = []
    for index in range(n_respondents):
        respondents.append(
            {
                "person_id": index,
                "attributes": {
                    "Country Prefix ISO 3166 Code - alphanumeric": countries[index % len(countries)],
                    "Sex of Respondent": "Female" if index % 2 else "Male",
                    "Age of respondent": str(20 + index % 60),
                    "Highest completed education level: Categories for international comparison": "Secondary",
                    "Top-bottom self-placement": str(1 + index % 10),
                    "Currently, formerly, or never in paid work": "Employed",
                    "Living in steady partnership": "Married",
                    "Place of living: urban - rural": "Urban" if index % 3 else "Rural",
                },
                "questions_answer": {
                    f"v{i}": 1 + (index + i) % 3 for i in range(n_items)
                },
            }
        )
    (qa_dir / "issp_qa_toy.json").write_text(json.dumps(questions))
    (answer_dir / "issp_answer_toy.json").write_text(json.dumps(respondents))
    return repo


def test_canonicalize_attributes_and_bins():
    canonical = canonicalize_attributes(
        {
            "Sex of Respondent": "Female",
            "Age of respondent": "37",
            "Top-bottom self-placement": "2",
            "Highest completed education level: Categories for international comparison": "Secondary",
        }
    )
    assert canonical["sex"] == "Female"
    assert canonical["age_bin"] == "30-44"
    assert canonical["income_quintile"] == "Q5"  # scale=2 -> inverted=9 -> ceil(9/2)=5
    assert age_bin("not a number") == "Missing"
    assert income_quintile("not a number") == "Missing"


def test_parse_curate_split_and_leakage(tmp_path):
    repo = _write_toy_repo(tmp_path)
    frame, parse_audit = parse_sociobench(repo, ["toy"])
    assert parse_audit["total_valid_rows"] > 0
    assert frame["is_ordinal"].all()

    frame, curation_audit = curate_items_and_countries(
        frame, max_items_per_domain=8, countries_per_domain=3,
        max_respondents_per_country=1000, min_answered_items_per_respondent=5, seed=7,
    )
    assert curation_audit["final_rows"] > 0
    assert frame["country"].nunique() <= 3
    assert frame.groupby("question_key")["panel_id"].nunique().shape[0] <= 8

    frame = assign_item_holdout(frame, unseen_fraction=0.25, seed=7)
    assert frame["is_unseen_item"].any()
    n_items = frame["question_key"].nunique()
    n_unseen = frame.loc[frame["is_unseen_item"], "question_key"].nunique()
    assert n_unseen == max(1, round(n_items * 0.25))

    frame, intersection_audit = assign_splits(
        frame, seed=7, ratios=(0.7, 0.1, 0.2),
        intersection_attributes=["age_bin", "education", "income_quintile", "urbanicity"],
        intersection_holdout_fraction=0.2, min_cell_size=2,
    )
    assert set(frame["split"].unique()) <= {"train", "validation", "test", "ood_intersection"}

    frame = add_question_statistics(frame)
    orders = calibration_orders(frame, [11])

    leakage = validate_no_leakage(frame)
    assert leakage["panel_in_one_split"]
    assert leakage["row_ids_unique"]

    item_leakage = validate_item_holdout_leakage(frame, orders)
    assert item_leakage["unseen_items_found_in_calibration_orders"] == 0


def test_intersection_holdout_removes_whole_cells(tmp_path):
    repo = _write_toy_repo(tmp_path, n_respondents=300)
    frame, _ = parse_sociobench(repo, ["toy"])
    frame, _ = curate_items_and_countries(
        frame, max_items_per_domain=8, countries_per_domain=5,
        max_respondents_per_country=1000, min_answered_items_per_respondent=5, seed=1,
    )
    frame = assign_item_holdout(frame, 0.2, seed=1)
    frame, audit = assign_splits(
        frame, seed=1, ratios=(0.7, 0.1, 0.2),
        intersection_attributes=["age_bin", "education", "income_quintile", "urbanicity"],
        intersection_holdout_fraction=0.3, min_cell_size=5,
    )
    ood = frame[frame["split"].eq("ood_intersection")]
    non_ood = frame[~frame["split"].eq("ood_intersection")]
    ood_cells = set(zip(ood["age_bin"], ood["education"], ood["income_quintile"], ood["urbanicity"]))
    non_ood_cells = set(zip(non_ood["age_bin"], non_ood["education"], non_ood["income_quintile"], non_ood["urbanicity"]))
    assert not (ood_cells & non_ood_cells), "a held-out intersection cell leaked into the training pool"


def test_panel_store_history_never_includes_target_or_unseen(tmp_path):
    repo = _write_toy_repo(tmp_path)
    frame, _ = parse_sociobench(repo, ["toy"])
    frame, _ = curate_items_and_countries(
        frame, max_items_per_domain=10, countries_per_domain=3,
        max_respondents_per_country=1000, min_answered_items_per_respondent=5, seed=3,
    )
    frame = assign_item_holdout(frame, 0.2, seed=3)
    frame, _ = assign_splits(
        frame, seed=3, ratios=(0.7, 0.1, 0.2),
        intersection_attributes=["age_bin", "education", "income_quintile", "urbanicity"],
        intersection_holdout_fraction=0.2, min_cell_size=2,
    )
    frame = add_question_statistics(frame)
    orders = calibration_orders(frame, [5])
    store = PanelStore(frame, orders)

    row = frame[~frame["is_unseen_item"]].iloc[0]
    history = store.history_rows(row.panel_id, row.question_id, 3, 5)
    assert row.question_id not in set(history["question_id"])
    assert not history["is_unseen_item"].any()

    unseen_targets = store.target_rows("test", 0, 5, item_pool="unseen")
    assert unseen_targets["is_unseen_item"].all()
    seen_targets = store.target_rows("test", 0, 5, item_pool="seen")
    assert not seen_targets["is_unseen_item"].any()


def test_curate_items_and_countries_respects_caps(tmp_path):
    repo = _write_toy_repo(tmp_path, n_respondents=500)
    frame, _ = parse_sociobench(repo, ["toy"])
    frame, audit = curate_items_and_countries(
        frame, max_items_per_domain=6, countries_per_domain=2,
        max_respondents_per_country=20, min_answered_items_per_respondent=3, seed=9,
    )
    counts = frame.drop_duplicates("panel_id").groupby("country").size()
    assert (counts <= 20).all()
    assert frame["country"].nunique() <= 2
