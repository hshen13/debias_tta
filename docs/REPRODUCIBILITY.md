# Reproducibility: Paper Table Provenance

This page maps every table in the [ACL 2026 paper](https://aclanthology.org/2026.acl-long.919/) to the notebook, cell heading, function, configuration, or dictionary that generates or assembles it. Search a notebook for the quoted cell heading to jump to the source.

A full audit of these mappings against the paper and the released code — numeric checks, recomputation of the human-evaluation statistics, and code-to-paper differences in implementation details and hyperparameters — is in [`CAP_TTA_provenance_and_data_audit_EN.md`](../CAP_TTA_provenance_and_data_audit_EN.md). This page is the short map; the audit is the reference for any discrepancy.

Notebook names below refer to files in the repository root. `CAPTTA_qwen3_tta_main_updated.ipynb` (NoTrig row of Table 4) is not included in this release.

## Shared pipeline for Tables 1, 3, and 4

- **Prompt sets:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Cell: `## 3) Prompts: SAFE vs TOXIC`
  - Functions: `build_toxic_prompts_rtp`, `build_safe_prompts_writingprompts`
- **Final BiasBench and quality scoring:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Cell: `# Final Bias Benchmark Scoring (NOT trigger benchmarks) + Quality (PPL, rep4, lengths)`
  - Functions: `bias_bench_score`, `_prob_positive`, `_detexd_risk`, `ppl_on_text`, `fluency_from_ppl`
- **Online trigger scoring:** `CAPTTA_qwen3_tta_main.ipynb`
  - Functions/classes: `toxic_prob_committee_mean`, `BiasScorer4Type`

## Table 1 — ID/OOD Bias and Speed

- **Generation for all five model rows:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Cell: `## 1) Models (baseline generation only)`
  - Object: `MODELS`
  - Cell: `## 6) Phase 1: Generate ONLY`
  - Functions: `run_split_generate_only`, `generate_one_segment`
- **Table assembly:** same notebook
  - Cell: `# Recompute Table~\ref{tab:main_results} from SCORED CSVs (HARD-CODED)`
  - Dictionary: `PATHS`

## Table 2 — OOD Detection

All rows are in `OOD.ipynb`:

1. `# 1) Load Qwen3 + Background LM (for LLR)`
2. `# 2) Load datasets & build ID/OOD splits`
3. `# 3) Scoring functions` — `score_llr`
4. `# 4) Embedding-based detectors (Mahalanobis / kNN)` — `extract_reps`, `fit_knn`, `fit_mahalanobis`
5. `# 5) Metrics` — `eval_detector`
6. `# 6) Significance tests` — `bootstrap_ci_metric`
7. `# 7) Run all detectors + benchmark comparison table`

## Table 3 — Main Results

### Static/debiased model rows

File: `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`

- `Qwen-3-4B`: `MODELS` entry with `key="qwen3_4b"` + `## 6) Phase 1: Generate ONLY`
- `DeepSeek-8B`: `MODELS` entry with `key="deepseek_r1_8b"` + Phase 1
- `Mistral`: `MODELS` entry with `key="mistral_7b_instruct"` + Phase 1
- `Self-correction`: `MODELS` entry with `key="qwen4b_self_correct"` + Phase 1
- `DS-8B-debiased`: `MODELS` entry with `key="deepseek_r1_8b_debiased"` + Phase 1

### TTA rows

File: `CAPTTA_qwen3_tta_main.ipynb`

- `Qwen-SGD`
  - Cell: `# 1) method compare`
  - Configuration: `kind="sgd10"`, `update_kind="sgd_10"`, `epsilon=0.3`
  - Function: `tta_step_sgd_10`
- `Qwen-ADAMW`
  - Cell: `# 1) method compare`
  - Configuration: `kind="adamw10"`, `update_kind="adamw_10"`, `epsilon=0.3`
  - Function: `tta_step_adamw_10`
- `Qwen-Prec-trig`
  - Cell: `# 1) method compare`
  - Configuration: `kind="precond"`, `epsilon=0.3`
  - Functions: `run_main_experiments`, `tta_lora_update_precond`, `estimate_preconditioner_diag`
- `Qwen-Prec-notrig`
  - Cell: `# 2) epsilon sweep (precond)`
  - Entry: `epsilon=0`
- `Qwen-Prec-trig-2`
  - Cell/configuration: `BASE_PRECOND_THEORETICAL`

### Table assembly

File: `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`

- Cell: `# FINAL PAPER TABLE GENERATOR (RECOMPUTE MEAN ± STD)`
- Dictionary/function: `MAIN_ROWS`, `metrics_from_csv`
- StereoSet/StereoDet/Delicate columns: cell `# MAIN TABLE (EXPANDED BIAS)`

## Table 4 — Ablations

- `Qwen Baseline`
  - File: `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Location: `MODELS` entry `qwen3_4b` + `## 6) Phase 1: Generate ONLY`
- `Epsilon: eps0.2 / eps0.25`
  - File: `CAPTTA_qwen3_tta_main.ipynb`
  - Cell: `# 2) epsilon sweep (precond)`
- `Segments: nseg2 / nseg8`
  - File: `CAPTTA_qwen3_tta_main.ipynb`
  - Cell: `# 5) length ablation`
  - Loop: `for nseg in [2, 4, 8]`
- `SegTokens: tok64 / tok256`
  - File: `CAPTTA_qwen3_tta_main.ipynb`
  - Cell: `# 5) length ablation`
  - Loop: `for seg_tokens in [64, 128, 256]`
- `NoTrig: eps0,nseg16,tok128`
  - File: `CAPTTA_qwen3_tta_main_updated.ipynb`
  - Cell: `=== 2) Qwen3 新增配置：epsilon=0，128 token/段，16 段 ===`
  - Configuration: `BASE_QWEN3_EPS0_16SEG`
- `MultiTrigger: multi0 / multi1`
  - File: `CAPTTA_qwen3_tta_main.ipynb`
  - Cell: `# 4) multi-trigger vs single-trigger`
  - Loop: `for multi in [True, False]`
- `DeepSeek Baseline`
  - File: `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Location: `MODELS` entry `deepseek_r1_8b` + Phase 1
- `DeepSeek Precond eps0.3`
  - File: `CAPTTA_qwen3_tta_main.ipynb`
  - Cell: `# ========= DeepSeek 接入（复用你现有 run_main_experiments…）=========`

**Table assembly:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`

- Cell: `# FINAL PAPER TABLE GENERATOR (RECOMPUTE MEAN ± STD)`
- Dictionary: `ABL_ROWS`
- NoTrig selected-output cell: `# SELECTED-ONLY (4 files) -> new CSV + paper-style LaTeX`

## Table 5 — Human Evaluation Bias Rate

- **Source data:** `human_evaluation_with_models.csv`
- **Aggregation/table-generation cell:** not included in the repository notebooks.

## Table 6 — Generated Example, prompt_id=117

- **Display/scoring code:** `CAPTTA_qwen3_tta_main.ipynb`
  - Section: `# 3) 评分器：committee bias_score（tox_mean）`
  - Functions: `score_committee`, `show_record_with_scoring`
- **Source outputs:** TTA output from `run_main_experiments` and Qwen baseline output from `## 6) Phase 1: Generate ONLY` in `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`.

## Table 7 — Generated Example, prompt_id=3

- **Display/scoring code:** `CAPTTA_qwen3_tta_main.ipynb`
  - Section: `# 3) 评分器：committee bias_score（tox_mean）`
  - Functions: `score_committee`, `show_record_with_scoring`
- **Source outputs:** TTA output from `run_main_experiments` and Qwen baseline output from `## 6) Phase 1: Generate ONLY` in `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`.

## Table 8 — Router Examples

File: `CAPTTA_qwen3_tta_main.ipynb`

- Function/cell: `pretty_print_scores`
- Demo cell: `# ---- demo examples ----`
- Scoring implementation: `BiasScorer4Type`, `toxic_prob_committee_mean`

## Table 9 — IRB Screening Responses

- **Source document:** `2026-002_FA_Shen2.pdf`
- **Code-generated:** no.

## Table 10 — Fleiss' Kappa

- **Source data:** `human_evaluation_with_models.csv`
- **Aggregation/table-generation cell:** `compute_fleiss_kappa.ipynb`.

## Table 11 — Reference Benchmark Values

- Literature values copied from the original benchmark papers.
- No repository computation or table-generation code.
