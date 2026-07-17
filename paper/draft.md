# RAP2P: Response-Anchored Profile-to-PEFT with Psychometric Graph Routing for Few-Response Survey Personalization

> Draft status: method, experimental design, and every table/figure are final;
> all numeric results are placeholders (`[--]`) pending the runs in
> `../README.md` "Compute budget". Column names in every table match exactly
> what `scripts/evaluate_all.py` writes to `artifacts/metrics/*.csv`, so filling
> this draft in is a copy-paste job, not a rewrite.

## Abstract

Large language models simulating survey respondents are typically conditioned
either on static demographic profiles or on nothing at all, and achieve only
30–40% individual-level accuracy on large sociological benchmarks such as
SocioBench. We ask whether a handful of a respondent's real answers — as few
as one to eight — can be turned into an instant, parameter-level adaptation
that a static demographic prior cannot provide. We introduce RAP2P
(Response-Anchored Profile-to-PEFT), which routes a shared, rank-block-gated
LoRA basis using (i) a demographic prior conditioned on the target question and
(ii) a target-aware, attention-weighted summary of the respondent's known
answers, biased by a leakage-safe item-item correlation graph estimated from
training respondents only. A prior-residual gate combines both signals with an
evidence weight that grows with the number of known answers. We compare RAP2P
against demographic and classical-IRT baselines, a matched-capacity static
P2P-style profile control, and a Context-QLoRA baseline that receives the
identical information as plain prompt tokens, on two SocioBench domains
(Environment, Role of Government) across three generalization axes built from
a single data partition (unseen respondents, unseen demographic
intersections, unseen items). We additionally introduce Matched-Profile
Heterogeneity, a test of whether the model distinguishes two demographically
identical respondents whose true answers diverge. [Results summary: TBD.]

## 1. Introduction

Demographic personas describe population-level tendencies, not individuals.
Two respondents with identical age, education, income, and country can hold
opposite views, and a model conditioned only on demographics cannot tell them
apart at $K=0$ known answers. SocioBench aggregates over 480,000 real ISSP
respondents across 30+ countries and 10 domains and reports that current LLMs
reach only 30–40% individual-level accuracy, with systematic gaps across
demographic subgroups (Wang et al., SocioBench, EMNLP 2025). Separately, P2P
(Tan et al., ACL 2026) shows that a hypernetwork can map a user profile
directly to a full set of PEFT parameters and generalize to unseen users
without per-user optimization. Between these two results sits an open
question this paper answers: **given a demographic prior and a small number of
a respondent's real answers, can a lightweight, target-aware routing mechanism
turn that sparse evidence into an individual-level parameter adaptation that
a static profile-to-parameter mapping, or an equivalent amount of prompt
context, cannot achieve?**

We make the following contributions.

1. **Response-Anchored Profile-to-PEFT (RAP2P)**: a shared rank-block-gated
   LoRA basis routed by a prior-residual gate that combines a
   question-conditioned demographic prior with a *target-aware* attention
   summary of the respondent's $K$ known answers, biased by a leakage-safe
   item-item correlation graph (Section 3).
2. **A controlled experimental design built from one data partition**: unseen
   respondent, unseen demographic-intersection, and unseen-item splits that
   share a single train/validation/test assignment, plus a capacity-matched
   P2P-style static-profile control that isolates exactly what target-aware
   routing adds over a static profile-to-parameter mapping (Section 4).
3. **Matched-Profile Heterogeneity**, a new evaluation that directly tests
   whether a model differentiates demographically identical respondents whose
   true answers diverge — the sharpest test of "not a stereotype machine"
   available short of full psychometric panel modeling (Section 5.4).

We deliberately scope the study to two SocioBench domains and drop several
axes (cross-country generalization, an exhaustive per-loss-term ablation) that
a fuller study would include; Section 7 states these cuts explicitly rather
than silently.

## 2. Related Work

**Personalization via hypernetworks.** P2P (Tan et al., 2026) trains a
hypernetwork to map an encoded user profile directly to a full LoRA adapter,
generalizing to unseen users without per-user optimization and running
roughly 33x faster than OPPU-style per-user LoRA training at inference time.
RAP2P shares P2P's "instant, no per-user gradient step" goal but differs in
what the routing signal is conditioned on: P2P's profile encoding is static
per user; RAP2P recomputes its response-anchoring summary for every target
question, using an item-item correlation prior to decide which of the K known
answers are relevant to the current question. Our P2P-Static baseline
(Section 4.3) isolates exactly this difference under matched architecture
capacity.

**Group-level and progressive adaptation.** PROPER (ACL 2025) bridges
population- and user-level LoRA through an intermediate group level, with a
user-aware router assigning respondents to group experts and a further
individual LoRA stage. RAP2P does not introduce a discrete group level;
instead, the demographic prior and response-anchoring branches jointly and
continuously determine the rank-block mixture, and the group structure (if
any) is left for the model to discover through the shared basis rather than
imposed via hard clustering.

**LoRA mixtures and routing.** Our rank-block gating is architecturally a
LoRA mixture-of-experts: a shared rank-$R$ basis is decomposed into $B$
rank-$r$ blocks ($R = B \times r$) mixed by a per-example softmax gate,
related to LoRA-MoE / routed-adapter approaches such as LoraHub, AdaLoRA, and
X-LoRA. The contribution here is not the mixing mechanism itself but what
drives the gate: a demographic prior and a target-aware, psychometrically
biased summary of a respondent's own answers, rather than a task embedding or
a data-driven cluster assignment.

**Representativeness beyond marginals.** Beyond Marginal Distributions
(ACL Findings 2026) shows that matching single-item marginal distributions
does not imply matching the joint (item-item) correlation structure of a
population. We evaluate this directly (Table 2) using expected-score item-item
correlations rather than committing to a training-time structural loss —
Section 3.4 explains why structural fidelity is measured, not optimized for.

**Robustness to option framing.** Questioning the Survey Responses of LLMs
(NeurIPS 2024) shows LLM survey answers can be sensitive to option order and
label scheme. We adopt randomized label mapping during training and averaged,
fixed-permutation evaluation at test time (Section 4.5).

## 3. Method

### 3.1 Problem statement

For an unseen respondent $i$ with demographics $d_i$, $K$ known
question-answer pairs $H_i^K = \{(q_k, y_{ik})\}_{k=1}^K$, and a target
question $q_j$, we predict

$$p(y_{ij} \mid d_i, H_i^K, q_j)$$

without any per-respondent gradient update. $K \in \{0, 1, 3, 5, 8\}$ is
sampled per training batch (variable-$K$ training, one checkpoint covers every
$K$) and fixed per evaluation pass.

### 3.2 Shared rank-block-gated LoRA basis

On the backbone's last $L$ decoder layers, for each of the $M$ target modules
(`q_proj`, `v_proj`), we maintain one population-shared LoRA of rank
$R = B \times r$ ($B{=}4$ blocks of rank $r{=}4$, so $R{=}16$), split along the
rank dimension into $B$ blocks $(A_b, B_b)_{b=1}^{B}$:

$$\Delta W^{(l,m)} = \sum_{b=1}^{B} g_b^{(l,m)} \, B_b^{(l,m)} A_b^{(l,m)}, \qquad g^{(l,m)} = \mathrm{softmax}(\cdot) \in \Delta^{B-1}$$

Implemented without materializing a per-example full-rank matrix: $A$ is
applied once (concatenated across blocks), the gate scales the low-rank
intermediate per block, and $B$ is applied once (see
`src/rap2p/models/gating.py:DynamicRankBlockLoRALinear`). Every respondent
uses the *same* basis atoms $(A_b, B_b)$; only the $B$-dimensional mixture
weight is personalized, giving a per-(respondent, question) parameter state of
$L \times M \times B$ scalars — with $L{=}8$, $M{=}2$, $B{=}4$, that is **64
values**.

### 3.3 Prior-residual gate

$$g_{ij}^{(l,m)} = \mathrm{softmax}\!\left[\, b^{(l,m)} \;+\; r_{ij}^{d,(l,m)} \;+\; \rho_{K_i} \, r_{ij}^{H,(l,m)} \,\right], \qquad \rho_{K_i} = \frac{K_i}{K_i + \tau}, \ \tau{=}2$$

- $b^{(l,m)}$: a learned, respondent-independent bias — the population
  default, dominant at $K{=}0$ with uninformative demographics.
- $r_{ij}^{d}$: the **demographic prior router** (Section 3.4).
- $r_{ij}^{H}$: the **target-aware response-anchoring** summary
  (Section 3.5), damped by $\rho_{K_i}$ so it contributes nothing when no
  answers are available and grows smoothly as more answers accumulate.
- $K_i$ is each respondent's **actual** number of known answers in the batch
  (from the history mask, not the batch's nominal $K$): a respondent with
  fewer answered items than the sampled $K$ gets a correspondingly smaller
  evidence weight.

### 3.4 Demographic prior router

$$r_{ij}^{d} = R_d\big([\,h_i^d\,;\,\pi(e_j)\,]\big), \qquad h_i^d = \mathrm{MLP}\big(\mathrm{Embed}_{\text{frozen}}(\text{demographic text}_i)\big)$$

$h_i^d$ is a small trainable MLP over a *frozen* sentence embedding of the
respondent's formatted demographic string (country, sex, age bin, education,
income quintile, employment, marital status, urbanicity — see
`data.py:format_demographics`); $e_j$ is the frozen embedding of the target
question, passed through a small learned projection $\pi$ before
concatenation. Conditioning on the target item lets the same demographic
profile push the gate differently depending on which question is asked
(income matters more for a taxation item than a national-pride item).

### 3.5 Target-aware response anchoring (the central mechanism)

Each known answer is represented as

$$r_{ik} = \mathrm{MLP}\big([\, e_k \,;\, E_y(y_{ik}) \,]\big)$$

where $e_k$ is the (frozen) embedding of known question $k$ and $E_y$ is a
small *trainable* answer-position embedding (not baked into a frozen text
encoder — the model learns how much an answer's position should shift the
representation). The $K$ known answers are weighted by an attention score that
mixes learned semantic similarity with a precomputed, leakage-safe item-item
correlation prior $C_{jk}$:

$$\alpha_{ijk} = \mathrm{softmax}_k\!\left[\, \frac{(W_q e_j)^\top (W_k e_k)}{\sqrt{d}} \;+\; \gamma\, C_{jk} \,\right], \qquad h_{i \to j}^{H} = \sum_{k=1}^K \alpha_{ijk}\, r_{ik}$$

$C_{jk}$ is a shrinkage-corrected Spearman correlation between items $j$ and
$k$, estimated **only on training-split respondents**:

$$C_{jk} = \frac{n_{jk}}{n_{jk} + \lambda}\, \rho_{\text{Spearman}}(q_j, q_k), \qquad \lambda{=}50$$

with $C_{jk}$ zeroed when the co-answering count $n_{jk}$ falls below a
minimum threshold (`data.correlation_min_n_jk`). This is the paper's central
inductive bias: **the same $K$ known answers are weighted differently
depending on what is being predicted**, using the survey's own item structure
as a prior on relevance — in contrast to a static profile encoder that
summarizes the $K$ answers once, identically, regardless of the target
question (our P2P-Static control, Section 4.3).

### 3.6 Training objective

Only two loss terms:

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + 0.1\, \mathcal{L}_{\mathrm{ord}}, \qquad \mathcal{L}_{\mathrm{ord}} = \frac{1}{n_{\text{options}}-1}\sum_{c} p(c)\,|c - y_{ij}|$$

reading only the option-label token logits (`A`.."J"), with the label-to-option
mapping re-randomized every epoch during training and averaged over fixed
permutations at test time (Section 4.5). A router-collapse guard adds
$0.01 \sum_b (\bar g_b - 1/B)^2$ **only if** triggered (any block's mean gate
share exceeds `router_collapse_threshold`), so it does not appear in the loss
for a healthy run. We deliberately do **not** add a KL, marginal-distribution,
or structural-correlation loss term: Table 2's structural fidelity (Section
5.3) is therefore an emergent property of accurate individual-level modeling,
not a directly optimized target — a stronger and more falsifiable claim than
training the correlation structure in and then reporting it back out.

### 3.7 Modality dropout: training-time "free" ablations

During RAP2P's main training run, demographics, history, and the correlation
bias are independently dropped per example with probability 0.20, 0.20, 0.15
respectively (`data.modality_dropout`). Because the model is trained to
function with any subset of these signals missing, most of Table 4's
ablations (no-demographics, no-history, no-correlation-graph, uniform-gate)
can be run at **inference time** on the single trained checkpoint by hard-zeroing
the corresponding keep-mask, rather than requiring a separately retrained
model for each. This is a compute-saving methodological choice, not a free
lunch: Section 4.4 explains the one ablation (RAP2P-noGraph) that *is*
retrained from scratch, and why, plus the cross-check (RAP2P-noHistory-retrained)
used to confirm the "free" ablations track what a true retrain would show.

## 4. Experimental Setup

### 4.1 Data

Two SocioBench (ISSP) domains: **Environment** and **Role of Government**.
Per domain: the 20 highest-coverage ordinal items retained; the 4 largest-sample
countries kept, capped at 3,000 respondents/country; respondents with fewer
than 12 answered (kept) items dropped. See `data.py:curate_items_and_countries`
for the exact, frozen-before-modeling curation rule and
`configs/mvp.yaml:data` for every threshold.

| | Environment | Role of Government |
|---|---:|---:|
| Countries kept | [--] | [--] |
| Items kept (of which unseen: 15%) | [--] | [--] |
| Respondents (train / val / test / ood-intersection) | [--] / [--] / [--] / [--] | [--] / [--] / [--] / [--] |
| Total (respondent, item) rows | [--] | [--] |

### 4.2 Three generalization axes from one partition

All three come from a single `assign_splits` call (`data.py`), not three
separate re-splits:

- **ID — Unseen Respondent**: standard 70/10/20 respondent-level split,
  stratified by domain/country/sex/age-bin.
- **OOD — Unseen Intersection**: demographic cells (age-bin × education ×
  income-quintile × urbanicity) with $\geq$40 respondents are, with
  probability matching `intersection_holdout_fraction` (15%), held out
  *entirely* — every respondent in a held-out cell moves to a dedicated
  `ood_intersection` split, never appearing in train/validation/test. Training
  still sees every individual attribute value, never that exact combination.
- **OOD — Unseen Item**: 15% of kept items per domain are excluded from *every*
  respondent's calibration history and from training targets; they appear only
  as test-time targets. `data.py:validate_item_holdout_leakage` asserts this
  holds before any GPU time is spent.

### 4.3 Baselines

| ID | Method | Input | Training cost |
|---|---|---|---:|
| B0 | Majority | $q$ | CPU, minutes |
| B1 | Demographic Frequency | $d, q$ | CPU, minutes |
| B2 | Question Frequency (no demographics) | $q$ | CPU, minutes |
| B3 | Demographic MIRT | $d, H_i^K, q$ | CPU, [--] |
| B4 | Local-8B Persona | $d, q$ | inference only |
| B5 | Local-8B Sparse-ICL | $d, H_i^K, q$ | inference only |
| B6 | Strong-API Sparse-ICL (500-respondent subsample) | $d, H_i^K, q$ | inference only |
| B7 | **Global QLoRA** | $d, q$ | GPU |
| B8 | **Context QLoRA** | $d, H_i^K, q$ (in prompt) | GPU |
| B9 | **P2P-Static** | $d, H_i^K$ (static profile, not target-conditioned) | GPU |
| **Ours** | **RAP2P** | $d, H_i^K, q$ (target-conditioned routing) | GPU |

B3 is the classical-latent-trait comparison point: if demographic MIRT alone
matches RAP2P, the gain is coming from a classical prior-posterior update, not
from LLM item semantics or target-aware routing (Section 6, Claim 2). Note
that the implemented MIRT is a *categorical* (per-item, per-option
discrimination) model, not a strict graded-response IRT model with ordered
thresholds — a documented simplification, restated in Limitations. B7/B8 use
a standard contiguous LoRA of the same total rank (16), alpha, and target
modules as RAP2P's basis — matched capacity, not the same block-gated
architecture; B9 (**P2P-Static**) shares RAP2P's exact rank-block
architecture, router hidden size, and warm-started basis, isolating precisely
the target-conditioned-routing difference. B9 is *not* a reproduction of the
official P2P hypernetwork stack — different training data format, different
profile encoder, mixes weights over a shared basis rather than generating
full adapter weights — report it as "a P2P-style static-profile control,"
never as "P2P" (see `README.md` "Baseline naming honesty").

Global/Context QLoRA and RAP2P/P2P-Static/ablations receive **the same text
prompt** (demographics + target question, no history text) except Context
QLoRA, whose entire purpose is to test in-context personalization
(`prompting.py` module docstring). This equalizes the information available
through the text channel across every method except the one baseline built to
test the text channel itself.

### 4.4 Training matrix and seeds

| Run | Seeds | Purpose |
|---|---|---|
| Global QLoRA | 1701, 7 | population survey adapter; its checkpoint **warm-starts** every rank-block basis below (`init_basis_from: global_qlora` in the config, implemented in `workflows.initialize_basis_from_peft_checkpoint`: the contiguous rank-16 LoRA is split into the 4 blocks, with lora_B scaled by $B$ so the initial $\Delta W$ under a uniform gate reproduces the Stage-1 adapter exactly) |
| Context QLoRA | 1701, 7 | RQ2 reference: same information, delivered as tokens |
| P2P-Static | 1701, 7 | RQ2 reference: same information, static (non-target-conditioned) routing; same warm start, same basis LR |
| RAP2P | 1701, 7, 42 | main method |
| RAP2P-noGraph | 1701 | correlation-graph term disabled entirely at construction time, $\gamma$ non-trainable (a true retrain, not a dropout-based ablation) |
| RAP2P-noHistory-retrained | 1701 | history branch never observed during training (cross-checks the "free" no-history ablation on the main RAP2P checkpoint, Section 3.7) |

All rank-block-basis runs share the same two-tier learning rate (router
5e-4, basis 2e-5) and the same warm start, so no ablation gap can be
attributed to an LR or initialization mismatch. All comparisons use
**matched seed counts** (2–3 per method) so no reported gap is a single-seed
artifact; see `README.md` "Compute budget" for the resulting ~75–110
GPU-hour estimate.

### 4.5 Robustness and statistics

- **Option-order robustness**: 5 `option_seed` values, each threaded into the
  collator's per-row deterministic label permutation and re-run over the same
  deterministic 500-respondent subsample of the ID test split
  (`evaluation.permutation` in the config); report semantic-answer
  **permutation consistency** (fraction of (respondent, item) pairs whose
  predicted answer is stable across all 5 permutations). Covered methods:
  RAP2P, Context QLoRA, and Local-8B Sparse-ICL.
- **Respondent-level paired bootstrap**: `evaluation.bootstrap_resamples`
  (2,000) resamples, domain-then-panel grouped, 95% CI
  (`eval/bootstrap.py:paired_bootstrap`).
- **Holm correction** at `evaluation.holm_alpha` (0.05) across the family of
  method-vs-Context-QLoRA / method-vs-P2P-Static / method-vs-MIRT comparisons
  at every $K$ (`eval/bootstrap.py:holm_correction`).

## 5. Results

*(All values below are placeholders; column names match
`artifacts/metrics/*.csv` exactly.)*

### 5.1 Table 1 — Few-response personalization (ID test split, `item_pool=seen`)

Primary metric: `accuracy` (mean of `correct`). Secondary: `mae`
(`normalized_ordinal_error`), `nll`. $\Delta_K$ = accuracy(K) − accuracy(0);
AUAC = *unweighted mean* accuracy over the (unevenly spaced) grid
$K \in \{0,1,3,5,8\}$ — not a trapezoidal integral. This table is assembled
by `eval/metrics.py:build_main_table` from the seed-aggregated macro table
(`table1_main.csv`): per-seed methods (e.g. `rap2p_seed1701/7/42`) are
collapsed to one row per method, reported as mean over seeds.

| Method | K=0 | K=1 | K=3 | K=5 | K=8 | AUAC ↑ | MAE(K=5) ↓ | NLL(K=5) ↓ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Majority | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Demographic Frequency | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Question Frequency | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Demographic MIRT | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Local-8B Persona | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Local-8B Sparse-ICL | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Strong-API Sparse-ICL$^\dagger$ | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Global QLoRA | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| Context QLoRA | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| P2P-Static | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |
| **RAP2P** | [--] | [--] | [--] | [--] | [--] | [--] | [--] | [--] |

$^\dagger$ Evaluated on a fixed 500-respondent subsample; NLL/Brier are a hard-label
approximation (no logprob access), report accuracy/MAE as primary for this row.
Responses that contain no valid standalone option letter are recorded with
`parse_failed=True` (plus the raw response text) rather than silently coerced;
report the parse-failure rate alongside this row.

**Go/no-go reading of this table** (see `README.md` day-3 checkpoint): if
RAP2P does not clear Context QLoRA by a clear, bootstrap-significant margin at
$K{\in}\{3,5\}$, the paper's central claim is not supported by Table 1 alone —
see Section 6 "Claim strength ladder" for how the framing degrades gracefully
using Table 2/3 instead of collapsing entirely.

### 5.2 Bootstrap comparisons (`bootstrap_accuracy.csv`)

| method | reference | k | difference | ci_low | ci_high | p_two_sided | holm_reject_null |
|---|---|--:|--:|--:|--:|--:|---|
| rap2p_seed1701 | context_qlora_seed1701 | 0 | [--] | [--] | [--] | [--] | [--] |
| rap2p_seed1701 | context_qlora_seed1701 | 1 | [--] | [--] | [--] | [--] | [--] |
| rap2p_seed1701 | context_qlora_seed1701 | 3 | [--] | [--] | [--] | [--] | [--] |
| rap2p_seed1701 | context_qlora_seed1701 | 5 | [--] | [--] | [--] | [--] | [--] |
| rap2p_seed1701 | context_qlora_seed1701 | 8 | [--] | [--] | [--] | [--] | [--] |
| rap2p_seed1701 | p2p_static_seed1701 | 0–8 | [--] | [--] | [--] | [--] | [--] |
| rap2p_seed1701 | mirt | 0–8 | [--] | [--] | [--] | [--] | [--] |

### 5.3 Table 2 — Population structure (macro domain, ID test split)

Group JS: the `js` column of `group_js.csv` (survey-weighted, marginal
Jensen-Shannon); `corr_err` / `corr_sim` (expected-score item-item
correlation RMSE / correlation-of-correlations against human data, in
`correlation_structure.csv`); `worst_group_accuracy` (min over demographic
cells with $n \geq 30$).

| Method | K | Group JS ↓ | CorrErr ↓ | CorrSim ↑ | Worst-group Acc ↑ |
|---|--:|--:|--:|--:|--:|
| Local Sparse-ICL | 5 | [--] | [--] | [--] | [--] |
| Context QLoRA | 5 | [--] | [--] | [--] | [--] |
| P2P-Static | 5 | [--] | [--] | [--] | [--] |
| RAP2P-noGraph | 5 | [--] | [--] | [--] | [--] |
| **RAP2P** | 5 | [--] | [--] | [--] | [--] |

Recall (Section 3.6) that no training loss directly optimizes Group JS or
CorrErr — this table is an emergent-generalization test, not a loss-matching
exercise.

### 5.4 Matched-Profile Heterogeneity (`heterogeneity.csv`)

Pairs must satisfy **both** halves of the definition: (i) demographics matched
on age-bin/education/income-quintile/urbanicity (exact or at most one
mismatch), and (ii) *known answers that actually diverge* — mean normalized
answer distance $\geq$ `min_answer_divergence` (0.35) over a canonical set of
`selection_items` (5) commonly-answered items
(`eval/heterogeneity.py:filter_pairs_by_answer_divergence`). The selection
items are recorded per pair and **excluded from the target comparison**, so
the variable used to select pairs never contaminates the outcome being
measured. `check_matched_pairs.py`'s day-1 gate reports the
divergence-filtered pair count per domain (not just raw demographic
matches). True vs. predicted divergence is then compared on the remaining
shared held-out target items.

| Method | K | n_pairs | ρ (Spearman, pred vs. true divergence) ↑ | DRR (→1) |
|---|--:|--:|--:|--:|
| Context QLoRA | 5 | [--] | [--] | [--] |
| P2P-Static | 5 | [--] | [--] | [--] |
| **RAP2P** | 5 | [--] | [--] | [--] |

DRR $\approx 0$ means the model collapses every demographically-matched pair
onto the same answer (a stereotype machine); DRR $\approx 1$ means predicted
divergence tracks true divergence in magnitude.

### 5.5 Table 3 — Generalization and ablations

**Generalization** (accuracy, K=5, macro domain):

| Method | ID — Unseen Respondent | OOD — Unseen Intersection | OOD — Unseen Item |
|---|--:|--:|--:|
| Demographic MIRT | [--] | [--] | N/A (no item parameters) |
| Context QLoRA | [--] | [--] | [--] |
| P2P-Static | [--] | [--] | [--] |
| **RAP2P** | [--] | [--] | [--] |

**Ablations** (accuracy, K=5, ID test; ✝ = "free" modality-dropout ablation on
the main RAP2P checkpoint, all others independently retrained):

| Variant | Acc | Δ vs. full RAP2P | Mechanism removed |
|---|--:|--:|---|
| RAP2P (full) | [--] | — | — |
| No demographics ✝ | [--] | [--] | prior branch |
| No history ✝ | [--] | [--] | response-anchoring branch |
| No history (retrained) | [--] | [--] | cross-check vs. the ✝ row above |
| No correlation graph ✝ | [--] | [--] | $\gamma C_{jk}$ term |
| RAP2P-noGraph (retrained, $\gamma{=}0$ fixed) | [--] | [--] | cross-check vs. the ✝ row above |
| Uniform gate ✝ | [--] | [--] | the router itself (tests whether learned mixing matters at all) |

**Consistency check**: the "✝ free" and "retrained" rows for no-history and
no-correlation-graph should be close; a large gap between them means the
modality-dropout training scheme is not a faithful stand-in for a true
ablation, and the paper should report retrained numbers only.

### 5.6 Option-order robustness (`permutation_consistency.csv`)

| Method | K | n_permutations | Permutation consistency ↑ |
|---|--:|--:|--:|
| Local Sparse-ICL | 5 | 5 | [--] |
| Context QLoRA | 5 | 5 | [--] |
| **RAP2P** | 5 | 5 | [--] |

## 6. Discussion

### Claim strength ladder

The paper's claim degrades gracefully depending on what Table 1/2/3 actually
show — decided by the numbers, not chosen in advance:

- **Strongest**: RAP2P beats Context QLoRA and P2P-Static on accuracy/MAE/NLL
  at $K{\in}\{3,5\}$ *and* on Group JS/CorrErr *and* on Matched-Profile
  Heterogeneity. Full story: target-aware parameter routing outperforms both
  in-context personalization and static profile-to-parameter mapping, and the
  gain is not bought by sacrificing population structure.
- **Acceptable**: RAP2P ties Context QLoRA on raw accuracy but wins clearly on
  OOD-intersection generalization and Matched-Profile Heterogeneity. Framing:
  *parameter-space personalization preserves within-group heterogeneity more
  effectively than context-space personalization*, at lower inference-time
  token cost and fixed 64-scalar per-user state.
- **Weak / not supported**: Context QLoRA matches or beats RAP2P on every axis.
  The core hypothesis (target-aware routing adds value beyond in-context
  personalization) is not supported; report this plainly rather than
  re-running ablations until something looks favorable.

### What Demographic MIRT tells us

If Demographic MIRT (B3) is statistically indistinguishable from RAP2P at
every $K$, the gain in this paper is coming from a classical prior-posterior
latent-trait update, not from LLM item semantics or target-aware routing — a
legitimate, narrower finding ("classical latent-trait models remain a strong
baseline for this task") that the draft should state directly rather than
bury.

## 7. Limitations

Stated up front rather than discovered by a reviewer:

- **No cross-country generalization axis.** SocioBench spans 30+ countries;
  this study evaluates unseen respondents, unseen demographic intersections,
  and unseen items, but not unseen countries. Left for future work.
- **Two domains, not ten.** Environment and Role of Government were chosen
  before looking at any model result (data-curation freeze, `data.py`); the
  findings should not be extrapolated to all ten SocioBench domains without
  further evidence.
- **The income-quintile binning assumes a specific ISSP scale direction**
  (1=top .. 10=bottom self-placement; `data.py:income_quintile`). This is
  load-bearing for the OOD-Intersection axis and the Matched-Profile
  Heterogeneity pairing, and must be verified against the actual ISSP
  codebook per domain before trusting those axes; the code flags it, and we
  restate it here.
- **Ordinal-item selection is an English-keyword heuristic**
  (`data.py:infer_ordinal`): it silently determines the item universe before
  curation. A hand-audited item whitelist would be stronger; the heuristic's
  false-negative rate has not been measured.
- **Stratified splitting backs off for small cells.** The 70/10/20 split's
  domain/country/sex/age-bin stratification degrades to coarser strata when a
  cell is too small (`data.py:_collapse_strata`), so the stated stratification
  is approximate for sparse combinations.
- **The MIRT baseline is a categorical simplification**, not a
  graded-response model with ordered thresholds (Section 4.3); conclusions
  about "classical IRT" strictly apply to this variant.
- **Only two training losses.** $\mathcal{L}_{\mathrm{CE}} + 0.1\,\mathcal{L}_{\mathrm{ord}}$,
  no marginal/structural/KL loss term — a deliberate simplification (Section
  3.6) that makes Table 2 a cleaner emergent-generalization test but forgoes
  the option to directly optimize for structural fidelity if the emergent
  result is weak.
- **Modality-dropout ablations are not fully independent of "free."** Four of
  seven ablation rows in Table 3 come from one jointly trained checkpoint with
  inference-time masking rather than independent retraining; the two
  retrained cross-check rows (RAP2P-noGraph, RAP2P-noHistory-retrained) exist
  specifically to validate this choice, not as free-standing scientific
  claims on their own.
- **The API baseline's probabilities are a hard-label approximation.** No
  logprob access; NLL/Brier for that row are not a calibration measurement,
  and unparseable responses are flagged (`parse_failed`) rather than scored.
- **This produces probabilistic *synthetic* survey responses, not a
  replacement for real participants.** Every claim in this paper is about
  conditional answer distributions under the modeling assumptions above, not
  about "digital twins" of real people.

## 8. Reproducibility

All configuration lives in `configs/mvp.yaml`; every number in this draft
traces to a file under `artifacts/metrics/`. Splits, calibration histories,
option-label permutation seeds, and item-graph coefficients are all
deterministic given the config's `seed` and are written to disk alongside
`artifacts/processed/audit.json` for inspection. See `README.md` for the exact
command sequence used to produce every table above.

## Figures

*(Generated by `scripts/make_figures.py`; Fig. 1 is a schematic, not
data-driven, and is not auto-generated.)*

- **Fig. 1** (hand-made): demographic prior → sparse-response posterior →
  rank-block gate → shared LoRA basis → prediction, with the item-correlation
  graph feeding the response-anchoring attention score.
- **Fig. 2** (`fig2_adaptation_curve.pdf`): accuracy / MAE / NLL vs. $K$,
  one line per method, ID test split, macro domain.
- **Fig. 3** (`fig3_correlation_heatmaps.pdf`): human vs. predicted item-item
  correlation matrices (expected-score based) for Context QLoRA, P2P-Static,
  and RAP2P at $K{=}5$.
- **Fig. 6** (`fig6_bootstrap_forest.pdf`): bootstrap accuracy differences
  (95% CI) for every method-vs-reference comparison in Table row 5.2, one row
  per (method, reference, K).
