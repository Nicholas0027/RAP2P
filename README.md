# RAP2P: Response-Anchored Profile-to-PEFT

Plain Python project (no notebooks) implementing the RAP2P design for few-response
survey-respondent personalization: a shared, rank-block-gated LoRA basis on an 8B backbone,
routed per (respondent, target question) by a demographic prior and a **target-aware**,
correlation-graph-biased summary of a handful of the respondent's real answers.

This is a standalone project. It is not a fork of `../track1` (SocioHyperLoRA), although the
data-cleaning and orchestration conventions were informed by it. See `paper/draft.md` for the
full write-up this code is meant to produce evidence for.

## Compute budget (validated, not aspirational)

Single 24GB-class GPU (RTX 4090 / L4 / A10), bf16 compute, 4-bit NF4 backbone, seq_len 256,
gradient checkpointing on, effective batch 64:

| Run | Steps | Seeds | Hours/seed | Subtotal |
|---|---:|---:|---:|---:|
| Global QLoRA | 1200–1800 | 2 | 5–7 | 10–14 |
| Context QLoRA | 1200–1800 | 2 | 5–7 | 10–14 |
| P2P-Static router | 800–1200 | 2 | 3–4.5 | 6–9 |
| RAP2P router | 800–1200 | 3 | 3–4.5 | 9–13.5 |
| RAP2P-noGraph (retrained ablation) | 800–1200 | 1 | 3–4.5 | 3–4.5 |
| RAP2P-noHistory (retrained ablation, cross-check only) | 800–1200 | 1 | 3–4.5 | 3–4.5 |
| **Training subtotal** | | | | **41–59 GPU-hours** |
| Inference (all K, all splits, permutations, matched-pair panels) | | | | 8–15 GPU-hours |
| Engineering slack / reruns (~1.5x) | | | | +25–35 GPU-hours |
| **Total** | | | | **~75–110 GPU-hours ≈ 3–5 days on one GPU** |

This fits inside the AAAI-27 window with 5–7 days left for data engineering, debugging, and
writing. It does **not** include the country-generalization axis, the full 4-domain SocioBench
sweep, or an exhaustive loss-term ablation — those were deliberately cut; see `paper/draft.md`
§ Limitations. If you have more than one GPU, parallelize across runs (they are independent),
not across steps within a run.

Router-only runs (P2P-Static, RAP2P, ablations) still require a **full forward pass** through
the 8B backbone every step because the loss depends on backbone logits — they are not cheap in
the way "1–2M trainable parameters" might suggest. The savings versus Global/Context QLoRA come
from fewer optimizer states and slightly fewer steps, not from a smaller forward pass.

## Day-by-day plan (12-day AAAI window)

| Day | Work | Go/no-go gate |
|---|---|---|
| 1 | Clone SocioBench, build canonical demographics + splits, precompute embeddings + item graph. Run `scripts/verify_no_leakage.py`. Run `scripts/check_matched_pairs.py` to confirm enough matched pairs exist for the heterogeneity metric (§ below) | Splits leakage-free; matched-pair count ≥ 200 per domain, else relax `matched_pair` thresholds in config **before** any GPU spend |
| 2 | Train Global QLoRA + Context QLoRA (2 seeds each, can run concurrently on 2 GPUs or sequentially) | Both converge (validation NLL decreasing); Context QLoRA baseline numbers in hand |
| 3 | Train P2P-Static (2 seeds), RAP2P (first seed only) | **Checkpoint**: does RAP2P beat Context QLoRA on the dev slice at K=3? If not by a clear margin, see `paper/draft.md` § Fallback framing — do not silently keep grinding on the same claim |
| 4 | Finish remaining RAP2P seeds, RAP2P-noGraph, RAP2P-noHistory-retrained | All checkpoints ready for full evaluation |
| 5 | Full inference sweep: all K, all 3 splits, permutation robustness, matched-pair heterogeneity, API baseline calls, bootstrap + Holm correction | Tables 1–3 + heterogeneity table populated with real numbers |
| 6–7 | Buffer for reruns, figures, writing | — |

## Directory layout

```
configs/mvp.yaml           single config; everything reads from here
src/rap2p/
  config.py                YAML loading + path resolution
  data.py                  SocioBench ingestion, canonical demographics, item/country curation,
                           item holdout, and all three splits (assign_splits/assign_intersection_holdout)
                           built from one partition -- there is no separate "splits" module
  item_graph.py            leakage-safe shrinkage Spearman item-item correlation C_jk
  embeddings.py             frozen demographic/item sentence-embedding cache
  prompting.py              prompt construction + option-label permutation
  batching.py              episodic K-shot sampler, modality dropout, collation
  models/
    common.py              backbone loading, label-token restricted logits, losses, checkpoint I/O
    gating.py               DynamicRankBlockLoRALinear — the RAP2P adapter layer
    demographic_prior.py     demographic encoder -> prior router contribution
    response_anchoring.py    target-aware attention history encoder + correlation-graph bias
    rap2p_model.py            full RAP2P wrapper (prior-residual gate, K-evidence weighting)
    p2p_static.py              static-profile baseline, same capacity, no target conditioning
  training.py               shared training loop (Global/Context QLoRA, P2P-Static, RAP2P, ablations)
  inference.py              prediction pass + the "free" ablation flags (see below)
  workflows.py              high-level orchestration used by scripts/
  baselines/                majority, demographic-frequency, MIRT, local + API prompting
  eval/                     metrics (expected-score based, no Monte Carlo panel sampling),
                           bootstrap+Holm, permutation robustness, matched-pair heterogeneity
scripts/                    CLI entry points, one per pipeline stage (see below)
tests/                      logic tests runnable without torch/transformers installed
paper/draft.md              paper draft with placeholders for every number this code produces
```

## Running it

```bash
pip install -e .[llm,dev]          # add [api] if running the strong-API baseline

python scripts/prepare_sociobench.py    --config configs/mvp.yaml   # parses, curates, splits, in one pass
python scripts/verify_no_leakage.py     --config configs/mvp.yaml
python scripts/check_matched_pairs.py   --config configs/mvp.yaml
python scripts/precompute_embeddings.py --config configs/mvp.yaml
python scripts/precompute_item_graph.py --config configs/mvp.yaml

python scripts/train.py --config configs/mvp.yaml --run global_qlora   --seed 1701
python scripts/train.py --config configs/mvp.yaml --run context_qlora --seed 1701
python scripts/train.py --config configs/mvp.yaml --run p2p_static    --seed 1701
python scripts/train.py --config configs/mvp.yaml --run rap2p         --seed 1701
python scripts/train.py --config configs/mvp.yaml --run rap2p_no_graph --seed 1701
python scripts/train.py --config configs/mvp.yaml --run rap2p_no_history_retrained --seed 1701
# repeat with the remaining seeds listed under `runs.<name>.seeds` in the config

python scripts/evaluate_all.py --config configs/mvp.yaml   # Tables 1-3, heterogeneity, permutation, bootstrap
python scripts/run_api_baseline.py --config configs/mvp.yaml --provider anthropic
python scripts/make_figures.py --config configs/mvp.yaml
```

For a zero-GPU plumbing check (never a reported result):

```bash
RAP2P_SMOKE=1 python scripts/train.py --config configs/mvp.yaml --run rap2p --seed 0 --smoke
```

Smoke mode uses `model.smoke_backbone` (a tiny CPU-sized model), a synthetic in-memory dataset
generated by `src/rap2p/data.py:make_synthetic_panels`, and 20 optimizer steps. It exists to
catch shape/plumbing bugs before spending real GPU time, not to validate the science.

## Baseline naming honesty

- **P2P-Static** is a matched-capacity, target-independent-routing control built on the same
  rank-block LoRA architecture, router hidden size, warm start, and basis learning rate as
  RAP2P. It is **not** a reproduction of the official P2P hypernetwork stack (different
  training data format, different profile encoder). Report it as "P2P-style static-profile
  control," not as "P2P."
- **Global QLoRA** is a standard contiguous rank-16 PEFT LoRA — matched to RAP2P's basis in
  total rank/alpha/target modules, but not block-gated. Its trained checkpoint **warm-starts**
  every rank-block-basis run (`runs.*.init_basis_from: global_qlora` in the config;
  `workflows.initialize_basis_from_peft_checkpoint` splits the rank-16 A/B into 4 blocks and
  scales lora_B by 4 so the initial ΔW under a uniform gate reproduces the Stage-1 adapter
  exactly). **Train `global_qlora` before any p2p_static/rap2p run**, or those runs will fail
  fast with a missing-checkpoint error.
- **Context QLoRA** is the same PEFT architecture as Global QLoRA, retrained from scratch with
  demographics + history text appended to the prompt.
- **Demographic MIRT** is a categorical (per-item, per-option discrimination) model, not a
  strict graded-response IRT model with ordered thresholds — a documented simplification; see
  `baselines/mirt.py` and the paper's Limitations.
- The **API baseline** is evaluated on a fixed 500-respondent stratified subsample (see
  `evaluation.api` in the config) for cost/rate-limit reasons; it is a reference point, not part
  of the primary statistical comparison. Unparseable responses are recorded with
  `parse_failed=True` and the raw text, never silently coerced into an answer.

## What this project deliberately does not attempt

See `paper/draft.md` § Limitations for the full list and rationale. In short: no
unseen-country generalization axis, no exhaustive per-loss-term ablation (only two training
losses are used), no 4-domain sweep, no 7B/8B backbone comparison. These are the same cuts
discussed and justified in the design review that produced this repository.
