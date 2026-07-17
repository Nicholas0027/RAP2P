"""Demographic-prior MIRT baseline: the classical (non-LLM) prior-to-posterior
comparison point for RQ1. A demographic-only linear prior over a low-dimensional
latent trait, refined at test time by a handful of MAP gradient steps on the K
known answers. If this alone matches RAP2P, the core hypothesis (LLM item
semantics + target-aware routing add something beyond a classical latent-trait
update) has not been supported.

This is a *categorical* (per-item, per-option) discrimination model, not a
strict graded-response IRT model with ordered thresholds -- documented
simplification, see README. It cannot generalize to OOD-Item (no discrimination
parameters exist for an item never seen in training), so `predict_mirt` refuses
`item_pool="unseen"` rather than silently returning meaningless numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from ..data import CANONICAL_DEMOGRAPHIC_COLUMNS, demographic_record_fields
from ..utils import write_json

DEMOGRAPHIC_COLUMNS = CANONICAL_DEMOGRAPHIC_COLUMNS


class DemographicMIRT(nn.Module):
    def __init__(self, n_features: int, n_questions: int, max_options: int, dimensions: int = 4):
        super().__init__()
        self.demographic_prior = nn.Linear(n_features, dimensions)
        self.discrimination = nn.Parameter(torch.randn(n_questions, max_options, dimensions) * 0.05)
        self.intercept = nn.Parameter(torch.zeros(n_questions, max_options))
        self.max_options = max_options

    def item_logits(self, theta: torch.Tensor, question_index: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
        discrimination = self.discrimination[question_index]
        intercept = self.intercept[question_index]
        logits = torch.einsum("bd,bcd->bc", theta, discrimination) + intercept
        mask = torch.arange(self.max_options, device=logits.device).unsqueeze(0) < n_options.unsqueeze(1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def forward(self, question_index: torch.Tensor, features: torch.Tensor, n_options: torch.Tensor) -> torch.Tensor:
        theta = self.demographic_prior(features)
        return self.item_logits(theta, question_index, n_options)


def _panel_features(frame: pd.DataFrame, encoder: OneHotEncoder | None = None):
    panels = frame[["panel_id", *DEMOGRAPHIC_COLUMNS]].drop_duplicates("panel_id").sort_values("panel_id")
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        features = encoder.fit_transform(panels[DEMOGRAPHIC_COLUMNS].fillna("Missing"))
    else:
        features = encoder.transform(panels[DEMOGRAPHIC_COLUMNS].fillna("Missing"))
    return panels, features.astype(np.float32), encoder


def fit_mirt(
    responses: pd.DataFrame,
    output_dir: str | Path,
    dimensions: int = 4,
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 0.02,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = responses[responses["split"].eq("train") & ~responses["is_unseen_item"]].copy()
    _, features, encoder = _panel_features(train)
    question_keys = sorted(train["question_key"].unique())
    question_lookup = {value: index for index, value in enumerate(question_keys)}
    panels = train[["panel_id", *DEMOGRAPHIC_COLUMNS]].drop_duplicates("panel_id").sort_values("panel_id")
    panel_lookup = {value: index for index, value in enumerate(panels["panel_id"])}

    dataset = TensorDataset(
        torch.tensor([panel_lookup[value] for value in train["panel_id"]], dtype=torch.long),
        torch.tensor([question_lookup[value] for value in train["question_key"]], dtype=torch.long),
        torch.tensor(train["n_options"].to_numpy(), dtype=torch.long),
        torch.tensor(train["answer_index"].to_numpy(), dtype=torch.long),
        torch.tensor(train["survey_weight"].to_numpy(), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DemographicMIRT(features.shape[1], len(question_keys), int(train["n_options"].max()), dimensions).to(device)
    panel_features = torch.from_numpy(features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        losses = []
        for panel_index, question_index, n_options, answer, weight in loader:
            panel_index, question_index, n_options, answer, weight = (
                v.to(device) for v in (panel_index, question_index, n_options, answer, weight)
            )
            logits = model(question_index, panel_features[panel_index], n_options)
            loss = (F.cross_entropy(logits, answer, reduction="none") * weight.clamp(0.1, 10)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    torch.save(model.state_dict(), output_dir / "mirt_state.pt")
    import joblib

    joblib.dump(encoder, output_dir / "demographic_encoder.joblib")
    metadata = {
        "dimensions": dimensions, "n_features": features.shape[1], "max_options": int(train["n_options"].max()),
        "question_keys": question_keys, "history": history,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def load_mirt(output_dir: str | Path, device: str | torch.device = "cpu"):
    import joblib

    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    encoder = joblib.load(output_dir / "demographic_encoder.joblib")
    model = DemographicMIRT(metadata["n_features"], len(metadata["question_keys"]), metadata["max_options"], metadata["dimensions"])
    model.load_state_dict(torch.load(output_dir / "mirt_state.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, encoder, metadata


def predict_mirt(
    responses: pd.DataFrame,
    history_lookup,  # callable(panel_id, target_question_id, k, seed) -> DataFrame, see data.PanelStore.history_rows
    target_rows_fn,  # callable(split, k, seed, item_pool) -> DataFrame, see data.PanelStore.target_rows
    model: DemographicMIRT,
    encoder: OneHotEncoder,
    metadata: dict[str, Any],
    k_values: Iterable[int],
    calibration_seed: int,
    output_path: str | Path,
    posterior_steps: int = 50,
    posterior_lr: float = 0.08,
    prior_precision: float = 1.0,
    device: str | torch.device = "cpu",
    split: str = "test",
    item_pool: str = "seen",
) -> pd.DataFrame:
    if item_pool == "unseen":
        raise ValueError(
            "Standard MIRT has no discrimination/intercept parameters for unseen items -- "
            "report this cell as N/A in Table 3 rather than calling predict_mirt(item_pool='unseen')."
        )
    device = torch.device(device)
    evaluation = responses[responses["split"].eq(split)]
    panels, features, _ = _panel_features(evaluation, encoder)
    panel_lookup = {value: index for index, value in enumerate(panels["panel_id"])}
    question_lookup = {value: index for index, value in enumerate(metadata["question_keys"])}
    feature_tensor = torch.from_numpy(features).to(device)
    with torch.no_grad():
        prior = model.demographic_prior(feature_tensor)

    outputs: list[dict[str, Any]] = []
    for k in k_values:
        theta = nn.Parameter(prior.detach().clone())
        history_rows = []
        if int(k) > 0:
            for panel_id in panels["panel_id"]:
                history_rows.extend(history_lookup(panel_id, "__none__", int(k), calibration_seed).to_dict("records"))
        if history_rows:
            history = pd.DataFrame(history_rows)
            pidx = torch.tensor([panel_lookup[v] for v in history["panel_id"]], device=device)
            qidx = torch.tensor([question_lookup[v] for v in history["question_key"]], device=device)
            nopt = torch.tensor(history["n_options"].to_numpy(), device=device)
            answer = torch.tensor(history["answer_index"].to_numpy(), device=device)
            optimizer = torch.optim.Adam([theta], lr=posterior_lr)
            for _ in range(posterior_steps):
                logits = model.item_logits(theta[pidx], qidx, nopt)
                loss = F.cross_entropy(logits, answer) + 0.5 * prior_precision * (theta - prior).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        targets = target_rows_fn(split, int(k), calibration_seed, item_pool=item_pool)
        for start in range(0, len(targets), 8192):
            batch = targets.iloc[start : start + 8192]
            pidx = torch.tensor([panel_lookup[v] for v in batch["panel_id"]], device=device)
            qidx = torch.tensor([question_lookup[v] for v in batch["question_key"]], device=device)
            nopt = torch.tensor(batch["n_options"].to_numpy(), device=device)
            with torch.no_grad():
                probabilities = F.softmax(model.item_logits(theta[pidx], qidx, nopt), dim=-1).cpu().numpy()
            for row, probability in zip(batch.itertuples(index=False), probabilities):
                probability = probability[: int(row.n_options)]
                probability = probability / probability.sum()
                target = int(row.answer_index)
                predicted = int(probability.argmax())
                outputs.append(
                    {
                        "method": "mirt", "row_id": row.row_id, "panel_id": row.panel_id, "domain": row.domain,
                        "question_id": row.question_id, "question_key": row.question_key, "k": int(k), "split": split,
                        "item_pool": item_pool, "calibration_seed": int(calibration_seed), "option_seed": 0,
                        "answer_index": target, "n_options": int(row.n_options), "survey_weight": float(row.survey_weight),
                        "probabilities_json": json.dumps(probability.tolist()), "predicted_index": predicted,
                        "nll": float(-np.log(max(probability[target], 1e-12))),
                        "brier": float(np.square(probability - np.eye(len(probability))[target]).sum()),
                        "normalized_ordinal_error": float(abs(predicted - target) / max(1, len(probability) - 1)),
                        "correct": int(predicted == target),
                        **demographic_record_fields(row),
                    }
                )
    output = pd.DataFrame(outputs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output
