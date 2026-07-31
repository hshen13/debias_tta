# CAP-TTA: Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation

This repository contains the official implementation for the paper **"Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation"** (Shen et al., 2026).

📄 **Paper:** [arXiv:2603.13683](https://arxiv.org/abs/2603.13683)


### Content Warning: While this public dataset has been filtered for safety, some prompts may still contain distressing or uncomfortable material. All datasets are sourced from other public datasets for research purposes.


## Overview



Although debiased LLMs perform well on known bias patterns, they often fail to generalize to unfamiliar bias prompts, producing toxic outputs. This work first validates that such high-bias prompts constitute a **distribution shift** via OOD detection, then shows that static debiased models degrade under this shift.

To adapt on-the-fly, we propose **CAP-TTA** (Preconditioned Context-Aware Test-Time Adaptation), a test-time adaptation framework that:

- **Monitors** bias/toxicity risk online during generation.
- **Triggers** lightweight updates only when the bias-risk score crosses a threshold, avoiding unnecessary parameter drift.
- **Updates** only a small LoRA adapter module using a precomputed **diagonal preconditioner** for fast, stable few-step updates.

Across toxic-prompt settings and benchmarks, CAP-TTA reduces bias (confirmed by human evaluation) while achieving much lower update latency than AdamW/SGD. It also mitigates catastrophic forgetting, significantly improving narrative fluency over SOTA debiasing baselines while maintaining comparable debiasing effectiveness.

## Repository Structure

| File | Description |
| --- | --- |
| `CAPTTA_qwen3_tta_main.ipynb` | Main pipeline: CAP-TTA test-time adaptation on top of a Qwen3 backbone with LoRA updates, trigger logic, and preconditioned optimizer. |
| `OOD.ipynb` | Out-of-distribution detection experiments validating that high-bias prompts constitute a distribution shift relative to the debiasing training set. |
| `bias_clean_dedup (2).ipynb` | Data preparation: cleaning and deduplication of bias/toxicity prompt corpora used for training and evaluation. |

## Method at a Glance

CAP-TTA reframes debiasing as a **continual adaptation problem under distribution shift**:

1. **Trigger** — A bias-risk score is computed per generation step. Updates fire only when the score exceeds a threshold, keeping inference overhead low on safe prompts.
2. **Context-aware LoRA update** — Only LoRA adapter weights are updated; the base model stays frozen.
3. **Diagonal preconditioner** — A precomputed diagonal preconditioner rescales gradients for faster and more stable few-step adaptation than vanilla SGD/AdamW.

## Getting Started

### Requirements

The notebooks use the Hugging Face stack with a Qwen3 backbone and PEFT/LoRA. Typical dependencies:

```bash
pip install torch transformers peft accelerate datasets bitsandbytes
pip install scikit-learn numpy pandas tqdm jupyter
```

A CUDA-capable GPU with sufficient VRAM to run Qwen3 + LoRA is recommended.

### Running the Notebooks

Run in the following order for a full reproduction:

1. `bias_clean_dedup (2).ipynb` — Prepare and deduplicate the bias prompt data.
2. `OOD.ipynb` — Reproduce the OOD detection analysis showing distribution shift on high-bias prompts.
3. `CAPTTA_qwen3_tta_main.ipynb` — Run CAP-TTA test-time adaptation and evaluate against baselines.

Adjust dataset paths, model checkpoints, and the bias-risk threshold at the top of each notebook as needed.

## Citation

If you find this work useful, please cite the paper:

```bibtex
@article{shen2026captta,
  title   = {Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation},
  author  = {Shen, Hanwen and Ying, Ting and Lu, Jiajie and Wang, Shanshan},
  journal = {arXiv preprint arXiv:2603.13683},
  year    = {2026}
}
```

## License

Unless stated otherwise, the code in this repository is released for research use. Please refer to the licenses of the underlying base models (e.g., Qwen3) and datasets when using or redistributing.

## Contact

For questions about the paper or code, please open an issue or contact the authors listed on the arXiv page.

## Paper Table Provenance

The entries below identify the notebook/file and the searchable cell heading, function, configuration, or dictionary used to generate or assemble each table in the paper.

### Shared pipeline for Tables 1, 3, and 4

- **Prompt sets:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Cell: `## 3) Prompts: SAFE vs TOXIC`
  - Functions: `build_toxic_prompts_rtp`, `build_safe_prompts_writingprompts`
- **Final BiasBench and quality scoring:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Cell: `# Final Bias Benchmark Scoring (NOT trigger benchmarks) + Quality (PPL, rep4, lengths)`
  - Functions: `bias_bench_score`, `_prob_positive`, `_detexd_risk`, `ppl_on_text`, `fluency_from_ppl`
- **Online trigger scoring:** `CAPTTA_qwen3_tta_main.ipynb`
  - Functions/classes: `toxic_prob_committee_mean`, `BiasScorer4Type`

### Table 1 — ID/OOD Bias and Speed

- **Generation for all five model rows:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`
  - Cell: `## 1) Models (baseline generation only)`
  - Object: `MODELS`
  - Cell: `## 6) Phase 1: Generate ONLY`
  - Functions: `run_split_generate_only`, `generate_one_segment`
- **Table assembly:** same notebook
  - Cell: `# Recompute Table~\ref{tab:main_results} from SCORED CSVs (HARD-CODED)`
  - Dictionary: `PATHS`

### Table 2 — OOD Detection

All rows are in `OOD.ipynb`:

1. `# 1) Load Qwen3 + Background LM (for LLR)`
2. `# 2) Load datasets & build ID/OOD splits`
3. `# 3) Scoring functions` — `score_llr`
4. `# 4) Embedding-based detectors (Mahalanobis / kNN)` — `extract_reps`, `fit_knn`, `fit_mahalanobis`
5. `# 5) Metrics` — `eval_detector`
6. `# 6) Significance tests` — `bootstrap_ci_metric`
7. `# 7) Run all detectors + benchmark comparison table`

### Table 3 — Main Results

#### Static/debiased model rows

File: `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`

- `Qwen-3-4B`: `MODELS` entry with `key="qwen3_4b"` + `## 6) Phase 1: Generate ONLY`
- `DeepSeek-8B`: `MODELS` entry with `key="deepseek_r1_8b"` + Phase 1
- `Mistral`: `MODELS` entry with `key="mistral_7b_instruct"` + Phase 1
- `Self-correction`: `MODELS` entry with `key="qwen4b_self_correct"` + Phase 1
- `DS-8B-debiased`: `MODELS` entry with `key="deepseek_r1_8b_debiased"` + Phase 1

#### TTA rows

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

#### Table assembly

File: `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`

- Cell: `# FINAL PAPER TABLE GENERATOR (RECOMPUTE MEAN ± STD)`
- Dictionary/function: `MAIN_ROWS`, `metrics_from_csv`
- StereoSet/StereoDet/Delicate columns: cell `# MAIN TABLE (EXPANDED BIAS)`

### Table 4 — Ablations

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

### Table 5 — Human Evaluation Bias Rate

- **Source data:** `human_evaluation_with_models.csv`
- **Aggregation/table-generation cell:** not included in the repository notebooks.

### Table 6 — Generated Example, prompt_id=117

- **Display/scoring code:** `CAPTTA_qwen3_tta_main.ipynb`
  - Section: `# 3) 评分器：committee bias_score（tox_mean）`
  - Functions: `score_committee`, `show_record_with_scoring`
- **Source outputs:** TTA output from `run_main_experiments` and Qwen baseline output from `## 6) Phase 1: Generate ONLY` in `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`.

### Table 7 — Generated Example, prompt_id=3

- **Display/scoring code:** `CAPTTA_qwen3_tta_main.ipynb`
  - Section: `# 3) 评分器：committee bias_score（tox_mean）`
  - Functions: `score_committee`, `show_record_with_scoring`
- **Source outputs:** TTA output from `run_main_experiments` and Qwen baseline output from `## 6) Phase 1: Generate ONLY` in `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`.

### Table 8 — Router Examples

File: `CAPTTA_qwen3_tta_main.ipynb`

- Function/cell: `pretty_print_scores`
- Demo cell: `# ---- demo examples ----`
- Scoring implementation: `BiasScorer4Type`, `toxic_prob_committee_mean`

### Table 9 — IRB Screening Responses

- **Source document:** `2026-002_FA_Shen2.pdf`
- **Code-generated:** no.

### Table 10 — Fleiss' Kappa

- **Source data:** `human_evaluation_with_models.csv`
- **Aggregation/table-generation cell:** `compute_fleiss_kappa.ipynb`.

### Table 11 — Reference Benchmark Values

- Literature values copied from the original benchmark papers.
- No repository computation or table-generation code.

## Model Checkpoints

### Generation and Backbone Models

- [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) — main Qwen3 backbone for CAP-TTA and the Qwen baselines.
- [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base) — Qwen3 base checkpoint used in the OOD-detection experiments.
- [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) — DeepSeek-R1-8B baseline and CAP-TTA backbone.
- [`hirundo-io/DeepSeek-R1-Distill-Llama-8B-Debiased`](https://huggingface.co/hirundo-io/DeepSeek-R1-Distill-Llama-8B-Debiased) — offline debiased DeepSeek baseline.
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) — Mistral baseline.
- `fenffef/Qwen-4B-Instruct-2505-Self-correct` — self-correction/Sherlock checkpoint used in the experiment code. The currently available repository is [`fenffef/Qwen-4B-Instruct-2507-Self-correct`](https://huggingface.co/fenffef/Qwen-4B-Instruct-2507-Self-correct).

### Online Trigger Committee

- [`s-nlp/roberta_toxicity_classifier`](https://huggingface.co/s-nlp/roberta_toxicity_classifier)
- [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert)
- [`unitary/unbiased-toxic-roberta`](https://huggingface.co/unitary/unbiased-toxic-roberta)

### Offline Bias Reporting Models

- [`grammarly/detexd-roberta-base`](https://huggingface.co/grammarly/detexd-roberta-base)
- [`henryscheible/stereoset_trainer_roberta-base_finetuned`](https://huggingface.co/henryscheible/stereoset_trainer_roberta-base_finetuned)
- [`Narrativa/distilroberta-finetuned-stereotype-detection`](https://huggingface.co/Narrativa/distilroberta-finetuned-stereotype-detection)

### Quality and OOD Evaluation

- [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) — perplexity/fluency evaluation and the likelihood-ratio background language model.
