# CAP-TTA Paper Table Provenance and Data Audit

**English translation, corrected against the released repository and the paper**  
**Audit date:** 30 July 2026  
**Repository snapshot audited:** `hshen13/debias_tta`, commit `5e371ca3f570569a1a08cbca6afaf2da7f92d489`

## 1. Scope and audit standard

This document translates the uploaded Chinese provenance map into English and then audits it against:

1. the uploaded 34-page paper, *Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation*;
2. the public GitHub repository at the commit above;
3. active code, commented-out code, saved notebook outputs, and hard-coded table mappings; and
4. the released `human_evaluation_with_models.csv`, whose counts and Fleiss’ κ values were independently recomputed.

The following status labels are used:

- **RECOMPUTED** — independently recomputed from released row-level data.
- **MATCHES STORED OUTPUT** — the paper number matches an output already saved in a notebook, but the underlying model inference was not rerun.
- **SOURCE VERIFIED** — the row-to-file/cell mapping is verified, but no released raw result file permits independent numeric recomputation.
- **MISMATCH** — the paper, active code, commented code, or table generator disagree.
- **NOT REPRODUCIBLE FROM THE RELEASE** — the selection rule or necessary raw artifact is absent.

This is a source-and-output audit, not a new GPU reproduction. Therefore, “matches stored output” does **not** mean that a result remains valid after fixing implementation errors.

---

## 2. Executive verdict

Most printed values in Tables 1, 2, 3, and 4 can be matched to saved notebook outputs or hard-coded table paths. However, the release contains several material implementation and reproducibility problems. The most important are:

1. **The paper calls the main model “Qwen3-4B Base,” but the generation and CAP-TTA code loads `Qwen/Qwen3-4B`, not `Qwen/Qwen3-4B-Base`.** The OOD notebook uses the Base checkpoint, so Table 2 and the generation experiments do not use the same Qwen checkpoint.
2. **Two of the three online trigger models are multi-label classifiers but are decoded with softmax instead of sigmoid.** This changes every committee trigger score, trigger/no-trigger decision, routed type, update count, trigger rate, update time, generated continuation after an update, and therefore all TTA rows and trigger-based figures.
3. **The released preconditioned method performs one online update per trigger, not ten.** `precond_steps=10` controls the number of offline gradient-collection steps used to estimate the preconditioner; `tta_lora_update_precond` itself explicitly performs one update and is called once per routed type. This directly invalidates the paper’s “ten preconditioned steps” description and makes the latency comparison with ten-step SGD/AdamW non-like-for-like.
4. **The online safety loss is not the conditional loss described in the paper.** The code concatenates `history + safe_text`, then sets `labels=input_ids`, so it trains on the history as well as the safe continuation and does not mask padding. The intended objective is to optimize the safe suffix conditional on the history.
5. **The released LoRA scope differs from the paper.** Active code updates only `q_proj`, `k_proj`, `v_proj`, and `o_proj`; the paper says it also updates `gate_proj`, `up_proj`, and `down_proj`.
6. **Several published hyperparameters differ from active code:** generic safe corpus 200 vs. 300 texts; standard preconditioned learning rate `1e-3` vs. `3e-4`; maximum delta norm `0.25` vs. `0.5`; update length 384 vs. the active 256; and Trigger-2 epsilon `0.22` vs. active `0.18`.
7. **Table 3/4 means and standard deviations are computed over CSV rows, which are segment rows, not prompt-level narrative aggregates.** The code’s `mean_std` does not group by `prompt_id`. This also creates a risk of pseudo-replication when segment rows are used as if independent prompt observations.
8. **The 20-prompt human-evaluation subset cannot be uniquely recovered from the released CSV.** The CSV contains 21 prompt IDs with exactly five ratings for all three models. The paper’s 20-prompt κ values can be obtained by removing one of several candidate prompt IDs, but the release does not say which one was removed or provide the separate fluency/attention-check responses needed to reproduce the filter.

Consequently, the published table values are best described as **matching the authors’ stored outputs under the released implementation**, not as fully verified experimental results. Corrected results require a clean rerun after fixing the checkpoint, trigger activation, online-step count, conditional loss masking, LoRA scope, and hyperparameter specification.

---

# Part I. Corrected English Provenance Map

## 3. Repository-name normalization

The uploaded map uses filenames from an earlier private/upload bundle. At the audited public commit, use these names:

| Earlier name in the Chinese map | Current public-repository name/status |
|---|---|
| `CAPTTA_qwen3_tta_main__1_.ipynb` | `CAPTTA_qwen3_tta_main.ipynb` |
| `colab_baselines_generate_then_score_cl_exp2__1___6_.ipynb` | `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb` |
| `CAPTTA_qwen3_tta_main_updated.ipynb` | Not present at current HEAD; historical branch copy |
| `CAPTTA_qwen3_tta_main_mistral7b.ipynb` | Not present at current HEAD; historical branch copy |
| `CAPTTA_qwen3_tta_main_multigpu_groups_qwen32b_v3.ipynb` | Not present at current HEAD; historical branch copy |
| `bias_clean_merged_shift_baselines_tta_ablations.ipynb` | Present in repository history/search; not the principal paper-table generator |

The historical copy relationship in the original map should therefore be retained only as a **historical note**, not as a description of files currently available to a reproducer.

## 4. Shared code for Tables 1, 3, and 4

Each row is intended to be traced to a full code filename and a Ctrl+F-searchable cell marker, section title, function name, or configuration string.

### Prompt sets

**File:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`  
**Cell:** `## 3) Prompts: SAFE vs TOXIC`  
**Functions:** `build_toxic_prompts_rtp`, `build_safe_prompts_writingprompts`

The generation benchmark uses 300 WritingPrompts prompts and 300 RTP prompts filtered at prompt toxicity `>= 0.4`. This is distinct from the second OOD pipeline in `OOD.ipynb`, which samples RTP prompts with non-null toxicity but does not apply the `>= 0.4` filter.

### Final reporting scores

**File:** `colab_baselines_generate_then_score_cl_exp2_(1)_(6).ipynb`  
**Cell:** `# Final Bias Benchmark Scoring (NOT trigger benchmarks) + Quality (PPL, rep4, lengths)`  
**Functions:** `bias_bench_score`, `_prob_positive`, `_detexd_risk`, `ppl_on_text`, `fluency_from_ppl`

### Online trigger score

**File:** `CAPTTA_qwen3_tta_main.ipynb`  
**Functions/classes:** `toxic_prob_committee_mean`, `ToxicityModel`, `BiasScorer4Type`

**Audit warning:** the active `ToxicityModel.p_toxic` uses softmax for every model with more than one logit. This is correct for the single-label s-nlp model but incorrect for both Unitary multi-label models.

---

## 5. Table 1 — ID/OOD safety and speed

### Provenance translation

| Paper row | Generation source | Table-construction source | Audit status |
|---|---|---|---|
| Qwen3-4B | `MODELS` entry `qwen3_4b`; Phase 1 generation | `# Recompute Table~\ref{tab:main_results} from SCORED CSVs (HARD-CODED)` and its `PATHS` map | Values match stored table output; model is `Qwen/Qwen3-4B`, not Base |
| DeepSeek-R1-8B | `MODELS` entry `deepseek_r1_8b`; Phase 1 generation | Same hard-coded table cell | Values match stored output |
| DeepSeek-R1-8B-Debiased | `MODELS` entry `deepseek_r1_8b_debiased` | Same | Values match stored output |
| Mistral-7B-Instruct | `MODELS` entry `mistral_7b_instruct`, HF checkpoint `mistralai/Mistral-7B-Instruct-v0.3` | Same | Numeric values match; “Offline Detox” is not supported by the checkpoint identity |
| Qwen self-correction / “Sherlock” | `MODELS` entry `qwen4b_self_correct`, HF checkpoint `fenffef/Qwen-4B-Instruct-2505-Self-correct` | Same | Numeric values match; “Sherlock” and “Yes (CoT)” do not match the released metadata |

### Metadata discrepancies

The current table cell labels Mistral as **`SOTA Alignment`**, not `Offline Detox`, and labels the self-correction row as **`Yes (Prompt)`**, not `Yes (CoT)`. The “Sherlock” name is also not the name of the loaded HF checkpoint. These are paper-label changes, not values generated by the code.

### Printed values

The paper reports:

| Model | ID | OOD | Speed |
|---|---:|---:|---:|
| Qwen3-4B | 0.289 | 0.452 | 19.4 |
| DeepSeek-R1-8B | 0.395 | 0.454 | 26.0 |
| DeepSeek-R1-8B-Debiased | 0.389 | 0.471 | 21.9 |
| Mistral-7B-Instruct | 0.449 | 0.525 | 25.3 |
| Self-correction | 0.395 | 0.437 | 18.8 |

These values match the stored table output to the printed precision, but no released result CSV bundle permits a clean external recomputation.

---

## 6. Table 2 — OOD detection

All three rows come from the **second** pipeline in `OOD.ipynb`:

- `# 1) Load Qwen3 + Background LM (for LLR)` — `Qwen/Qwen3-4B-Base` and GPT-2.
- `# 2) Load datasets & build ID/OOD splits` — `N_ID_TRAIN=800`, `N_ID_TEST=800`, `N_OOD=2000`.
- `# 3) Scoring functions` — `score_llr`.
- `# 4) Embedding-based detectors (Mahalanobis / kNN)` — `extract_reps`, `fit_knn(k=10)`, `fit_mahalanobis`.
- `# 5) Metrics` and `# 6) Significance tests` — `eval_detector`, `bootstrap_ci_metric(B=300)`.
- `# 7) Run all detectors + benchmark comparison table` — execution and saved output.

### Numerical audit

| Detector | Stored AUROC | Paper AUROC | Stored AUPR | Paper AUPR | Stored 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|
| kNN | 0.992205 | 99.22% | 0.996157 | 99.62% | [0.9888, 0.9951] | MATCHES STORED OUTPUT |
| Mahalanobis | 0.988055 | 98.81% | 0.994573 | 99.46% | [0.9839, 0.9917] | MATCHES STORED OUTPUT |
| LLR | 0.707419 | 70.74% | 0.866736 | 86.67% | [0.6891, 0.7260] | MATCHES STORED OUTPUT |

### Interpretation and calibration problems

1. The second pipeline admits any RTP prompt whose toxicity is non-null. It therefore establishes a **dataset/style-source shift between WritingPrompts and RTP**, not specifically a shift caused by the paper’s `toxicity >= 0.4` high-bias subset.
2. The kNN threshold is fitted using neighbors of the training representations against the same training set, so each point’s zero-distance self-neighbor is included.
3. The notebook’s held-out ID false-positive rates are approximately 32.1% for kNN and 99.25% for Mahalanobis, despite a nominal 95th-percentile training threshold. This does not change rank-based AUROC/AUPR, but it makes threshold-based OOD-rate interpretation unreliable.

---

## 7. Table 3 — Main results

### Row provenance

| Paper row | Generation source | Table-source status |
|---|---|---|
| Qwen-3-4B | Baseline notebook: `MODELS` entry `qwen3_4b` + Phase 1 | Included in final generator |
| DeepSeek-8B | `deepseek_r1_8b` + Phase 1 | Included |
| Mistral | `mistral_7b_instruct`, v0.3 + Phase 1 | Included |
| Self-correction | `qwen4b_self_correct` + Phase 1 | Included |
| DS-8B-debiased | `deepseek_r1_8b_debiased` + Phase 1 | Included |
| Qwen-SGD | Main notebook, active `kind="sgd10", eps=0.3`; function `tta_step_sgd_10` | Included |
| Qwen-ADAMW | Active `kind="adamw10", eps=0.3`; function `tta_step_adamw_10` | Included |
| Qwen-Prec-trig | Active `kind="precond", eps=0.3`; `tta_lora_update_precond`; `estimate_preconditioner_diag` | Included |
| Qwen-Prec-notrig | Epsilon sweep, `eps=0` | Included, but the name is semantically wrong: it is ungated/always-trigger, not “no update” |
| Qwen-Prec-trig-2 | `BASE_PRECOND_THEORETICAL` | Active code uses epsilon 0.18; paper appendix says 0.22; row is absent from the released final `MAIN_ROWS` table generator |

### Numeric audit

The final paper-table output reproduces the first nine rows to the paper’s printed precision. The Trigger-2 row is not selected by the released final table generator, so its exact source file cannot be uniquely proven from that cell.

| Row | PPL | Fluency | BB Bias | Trigger rate | Update (s) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Qwen-3-4B | 13.491 | 0.298 | 0.452 | — | — | MATCHES STORED OUTPUT |
| DeepSeek-8B | 21.361 | 0.255 | 0.454 | — | — | MATCHES STORED OUTPUT |
| Mistral | 212.5 | 0.275 | 0.525 | — | — | MATCHES; extreme PPL dispersion needs outlier disclosure |
| Self-correction | 22.092 | 0.262 | 0.437 | — | — | MATCHES STORED OUTPUT |
| DS-8B-debiased | 22.894 | 0.256 | 0.471 | — | — | MATCHES STORED OUTPUT |
| Qwen-SGD | 13.498 | 0.298 | 0.460 | 0.262 | 5.720 | MATCHES STORED OUTPUT |
| Qwen-ADAMW | 22.749 | 0.304 | 0.468 | 0.290 | 5.276 | MATCHES STORED OUTPUT |
| Qwen-Prec-trig | 13.119 | 0.307 | 0.443 | 0.256 | 0.839 | MATCHES STORED OUTPUT, but implementation differs materially from paper |
| Qwen-Prec-notrig | 13.460 | 0.303 | 0.456 | 1.000 | 0.991 | MATCHES; rename to “ungated/always-update” |
| Qwen-Prec-trig-2 | 13.877 | 0.293 | 0.437 | 0.778 | 0.403 | Number printed in paper; exact 0.18-vs-0.22 artifact selection unresolved |

The arithmetic claims based on the printed numbers are correct: `13.877` is approximately 37.2% below `22.092`, and `0.293` is approximately 11.8% above `0.262`. The scientific comparison is nevertheless affected by the implementation discrepancies described in Part II.

### Statistical-symbol inconsistency

The Table 3 caption says `†/††` are paired t-tests against the baseline, while the body explains the no-trigger `†` using a difference-in-differences test with `p=0.069`. Those are not the same test. The caption must identify the statistic actually used for each symbol.

---

## 8. Table 4 — Ablations

### Provenance translation

| Paper row | Source |
|---|---|
| Qwen baseline | Baseline notebook, same static Qwen row as Table 3 |
| Epsilon `eps0.2`, `eps0.25` | Main notebook: `# 2) epsilon sweep (precond)` |
| Segments `nseg2`, `nseg8` | Main notebook: `# 5) length ablation`, `for nseg in [2,4,8]` |
| Segment tokens `tok64`, `tok256` | Same cell, `for seg_tokens in [64,128,256]` |
| `eps0,nseg16,tok128` | Historical `CAPTTA_qwen3_tta_main_updated.ipynb`, `BASE_QWEN3_EPS0_16SEG`; this generating cell is absent from current HEAD |
| Multi-trigger `multi0`, `multi1` | Main notebook: `# 4) multi-trigger vs single-trigger`, including commented/active run history |
| DeepSeek baseline | Baseline notebook, `deepseek_r1_8b` |
| DeepSeek preconditioned | DeepSeek integration block in main notebook; the entire released block is commented out |

### Numeric audit

The paper’s Table 4 values match the final stored paper-table output to the printed precision. Important qualifications:

- `PPL=19.427` is the final reported value for the 64-token setting, not a label or another metric.
- `PPL=10.628` is consistently used for the 256-token setting in both the main table and the appendix of the uploaded PDF.
- Older output cells in the notebook contain stale, different intermediate values. The paper follows the later final generator, not every visible notebook output.
- `NoTrig` is not “no trigger”: epsilon zero yields trigger rate 1.000.
- The 16-segment and DeepSeek generating code is not executable as released without restoring historical or commented code.

---

## 9. Tables 5, 9, and 10 — Human evaluation and IRB

The original Chinese map said these did not appear in the uploaded code bundle. That statement is no longer accurate for the current repository: it now contains `human_evaluation_with_models.csv` and an IRB PDF. However, it still does not contain a complete, explicit script that reproduces the paper’s 20-prompt filter.

### Independent recomputation from the released CSV

The released CSV contains:

- 441 ratings;
- 31 reviewers;
- 30 prompt IDs;
- 90 prompt×model items;
- 81 items with exactly five ratings;
- 9 items with four ratings.

For all 81 items with exactly five ratings, the paper’s first half of Table 10 is reproduced exactly after rounding:

| Subset | κ recomputed | Paper κ | Items |
|---|---:|---:|---:|
| Overall | 0.294047 | 0.294 | 81 |
| Qwen base | 0.415323 | 0.415 | 29 |
| Self-correction | 0.178333 | 0.178 | 29 |
| CAP-TTA | 0.303947 | 0.304 | 23 |

### Common-prompt discrepancy

The CSV contains **21**, not 20, prompt IDs for which all three models have exactly five ratings. Using all 21 yields:

| Model | κ | Majority-biased items | Total | Percentage |
|---|---:|---:|---:|---:|
| Overall | 0.305019 | — | 63 items | — |
| Qwen base | 0.438503 | 3 | 21 | 14.286% |
| Self-correction | 0.166176 | 3 | 21 | 14.286% |
| CAP-TTA | 0.325275 | 2 | 21 | 9.524% |

The paper instead reports a 20-prompt subset, 15% / 15% / 10%, and κ values 0.297 / 0.433 / 0.156 / 0.318. Those values are internally consistent for a 20-prompt subset, but the released bias-only CSV does not uniquely determine which prompt was excluded. The paper states that a separate fluency question was used as an attention check/filter; those responses and the deterministic exclusion rule must be released to reproduce the 20-prompt subset.

A standalone recomputation script and its output files accompany this report.

---

## 10. Tables 6 and 7 — Generation examples

**Display/scoring helper:** `CAPTTA_qwen3_tta_main.ipynb`, commented display area containing `score_committee` and `show_record_with_scoring`; a related helper appears in `bias_clean_merged_shift_baselines_tta_ablations.ipynb`.

The example text and scores are printed in the paper, but the exact source result CSV/JSONL rows and deterministic selection code for prompt IDs 117 and 3 are not released in a way that permits independent reconstruction. The examples are therefore **paper-visible but not end-to-end reproducible from the repository**.

The displayed trigger scores are also affected by the softmax/sigmoid error.

---

## 11. Table 8 — Router examples

**File:** `CAPTTA_qwen3_tta_main.ipynb`  
**Cell:** `pretty_print_scores`, followed by `# ---- demo examples ----`  
**Inputs:** four hard-coded sentences, epsilon 0.3.

The numbers in Table 8 match the notebook’s saved output. They should not be treated as valid classifier probabilities because two committee members are decoded with the wrong activation. The clearest symptom is that `unitary/toxic-bert` assigns approximately 0.535–0.540 to benign sentences such as “Black holes are fascinating objects in astrophysics” and “The priest gave a sermon about compassion and forgiveness.” Under the model’s intended multi-label sigmoid decoding, these values need not resemble the released softmax values.

---

## 12. Table 11 — Reference benchmark numbers

These are literature-transcribed reference values, not reruns by the CAP-TTA code. The table should say explicitly “quoted from original benchmark papers” and should not be visually grouped with the authors’ reproduced measurements without that distinction.

---

## 13. Figures and textual statistics

| Item | Corrected provenance/status |
|---|---|
| Figures 1, 8, 9 | Schematic / LoRA diagram / Prolific screenshot; not data plots generated by the released analysis cells |
| Figures 2 and 3 | Based on the standard `kind="precond", eps=0.3` run; the release contains the rendered figures/output, but a clean standalone plotting cell and raw CSV are not available for independent regeneration |
| Figure 4 | Baseline table notebook, cell headed `# 3) One figure: color=model, split + method fixed (scatter + line)` |
| Figures 5, 6, 7 | Baseline/table notebook, cell beginning `# FIX: include epsilon=0 precond ... in ALL plots`, plus `# Top-tier Analysis: Plain vs TTA` |
| DiD `p=0.069` | Cell containing `# MODE: ALL segments (anchored or adjacent)`, anchored segment-0 analysis |
| Table 3 significance | `# SIGNIFICANCE TESTS FOR MAIN + ABLATION EXPERIMENTS`, `paired_stats`; paper caption and body disagree on which test produces the dagger |
| Catastrophic-forgetting plots | Cell explicitly labeled `# HARD-CODED ...`; not a data-derived reproducibility path suitable for a paper claim |

All trigger-score figures must be regenerated after correcting multi-label activation. All method-comparison figures must be regenerated after correcting the main checkpoint, online-step count, and conditional-loss implementation.

---

# Part II. Material Code-to-Paper Discrepancies

## 14. Main checkpoint mismatch

### Released code

- Main CAP-TTA notebook: `QWEN3_ID = "Qwen/Qwen3-4B"`.
- Baseline notebook: `qwen3_4b` also points to `Qwen/Qwen3-4B`.
- OOD notebook: the second pipeline points to `Qwen/Qwen3-4B-Base`.

### Paper

The paper repeatedly describes the main system and Table 1 row as Qwen3-4B **Base / Pretrained**.

### Effect

This is not a naming-only difference: Base and post-trained/instruction checkpoints are different models. The paper currently attributes generation, fluency, bias, and TTA results from one checkpoint to another, while the OOD analysis uses the Base checkpoint. The checkpoint name must be corrected in the paper, or all generation/TTA experiments must be rerun on `Qwen3-4B-Base`.

## 15. Trigger committee uses the wrong activation

### Released code

For every model with more than one logit, `ToxicityModel.p_toxic` computes:

```python
probs = torch.softmax(logits, dim=-1)[0]
```

It then averages the chosen value from:

- `s-nlp/roberta_toxicity_classifier`;
- `unitary/unbiased-toxic-roberta`;
- `unitary/toxic-bert`.

### Correct semantics

The s-nlp model is a two-class, single-label classifier, so softmax is appropriate. Both Unitary models are multi-label classifiers whose labels are independent and require sigmoid. Their official configs identify `problem_type="multi_label_classification"`; the unbiased model also explicitly sets `function_to_apply="sigmoid"`.

### Effect on published data

This is upstream of every online decision. It changes:

- `bias_score_trigger`;
- whether a segment exceeds epsilon;
- routed type(s) and selected SafeBank data;
- number and position of updates;
- trigger rate and update-time totals;
- all post-update generated segments;
- Table 3 TTA rows;
- Table 4 TTA ablations;
- Tables 6–8 trigger annotations;
- Figures 2, 3, and 7, and any figure filtered by trigger behavior.

The published TTA numbers cannot be repaired by changing a label in the manuscript; the experiments must be rerun.

## 16. “Ten preconditioned steps” are not implemented online

### Released code

`precond_steps` is passed only to `estimate_preconditioner_diag(... steps=precond_steps ...)`. The online function `tta_lora_update_precond` documents itself as **“One preconditioned update step”** and performs one forward/backward/update. The main loop calls it once per routed type.

### Paper

The paper states that CAP-TTA uses ten preconditioned steps online.

### Effect

- The algorithm executed is different from the method described.
- The reported 0.839-second CAP-TTA update time is compared with ten-step SGD/AdamW, while the released preconditioned method takes one online step.
- The apparent latency advantage is therefore not a controlled optimizer comparison.
- `Qwen-Prec-trig-2` is likewise described as four online steps, but its `precond_steps=4` controls preconditioner estimation in this code path.

A corrected implementation needs separate parameters, for example `precond_estimation_steps` and `online_update_steps`, and must rerun every optimizer comparison.

## 17. Conditional safe-suffix loss is not implemented

### Released code

The main loop creates:

```python
texts = [f"{history}\n\n{s}" for s in safe_samples]
```

The loss function then sets:

```python
labels = input_ids
```

for the entire concatenated text and does not replace padding labels with `-100`.

### Paper objective

The paper and Algorithm 1 describe fitting the safe sample conditional on the history, i.e. optimizing the safe continuation while treating the history as context.

### Effect

The gradient includes tokens from the prompt and all previous generated segments, not only the safe suffix. Padding can also contribute to the causal-LM loss. With truncation, the safe suffix may be partially removed when history is long. The actual update is therefore not the stated `-log p(safe | history)` objective.

The fix is to construct per-example labels, mask every history token and every padding token with `-100`, and verify that at least one safe-suffix token survives truncation.

## 18. LoRA target mismatch

### Released active code

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Paper appendix

The paper additionally lists `gate_proj`, `up_proj`, and `down_proj`.

### Effect

The trainable parameter set, Fisher/preconditioner dimension, update magnitude, GPU cost, and model behavior differ. The paper must describe the four attention targets actually used, or the experiments must be rerun with the seven-module scope.

## 19. Hyperparameter mismatch table

| Quantity | Paper | Active released code | Status |
|---|---:|---:|---|
| Main Qwen checkpoint | Qwen3-4B Base / pretrained | `Qwen/Qwen3-4B` | MISMATCH |
| Generic WikiText safe corpus | 200 texts | loads `generic_safe_wikitext_300.json`; saved output says 300 | MISMATCH |
| Standard preconditioned LR | `1e-3` | `3e-4` | MISMATCH |
| Standard max delta norm | `0.25` | `0.5` | MISMATCH |
| CAP-TTA update max length | 384 in method description | active `max_len_update=256` passed into update | MISMATCH / paper internally inconsistent |
| Main online preconditioned steps | 10 | 1 | MISMATCH |
| Trigger-2 epsilon | 0.22 | active 0.18; stale output names include 0.22 | MISMATCH |
| Trigger-2 online steps | 4 | 1; value 4 is used for preconditioner estimation | MISMATCH |
| LoRA target modules | attention + MLP projections | four attention projections only | MISMATCH |

## 20. Aggregation unit and significance

The final table helper converts the selected metric column to numeric and calls `.mean()` / `.std()` directly. TTA CSVs contain one row per segment; for the default setting this is 1,200 rows from 300 prompts × 4 segments. No `groupby("prompt_id")` is applied in the table-metric function.

Therefore:

- the published means are segment-row means;
- the published SDs are segment-row SDs;
- captions suggesting prompt-level variability are inaccurate;
- paired tests must aggregate to one value per prompt or use a hierarchical/repeated-measures model;
- varying the number of segments changes the number and weighting of observations, complicating direct ablation comparisons.

The numerical means may still equal the stored row-level averages, but the reporting unit must be corrected and statistical tests rerun at the prompt level.

## 21. OOD claim is broader than the implemented test

The paper links high bias to OOD shift. The Table 2 pipeline actually distinguishes WritingPrompts from an unfiltered RTP sample. This supports a source/style shift, but it does not isolate high-toxicity prompts or demonstrate that toxicity level causes the shift. A suitable test would use matched RTP subsets stratified by toxicity, or compare high- vs low-toxicity prompts within RTP while controlling for source/style.

## 22. Human-evaluation selection must be made deterministic

To make Tables 5 and 10 reproducible, release:

1. the fluency/attention-check responses;
2. the exact exclusion rule;
3. the final 20 prompt IDs;
4. a script that asserts five bias ratings per model per retained prompt; and
5. the aggregation and Fleiss’ κ computation.

Without those, both the 20-prompt denominator and the common-subset κ values depend on an unstated removal.

---

# Part III. Recommended Corrections and Rerun Order

## 23. Minimum code corrections

1. Select and document one Qwen checkpoint consistently.
2. Decode classifiers according to `config.problem_type`:
   - sigmoid for multi-label;
   - softmax for single-label multiclass;
   - sigmoid for a single logit.
3. Separate offline preconditioner-estimation steps from online update steps.
4. Mask history and padding in the causal-LM labels.
5. Set the LoRA target modules to the intended, documented list.
6. Freeze a single configuration manifest per paper row and write it beside each output CSV.
7. Aggregate metrics once per prompt/narrative before means, SDs, and paired tests.
8. Save and release prompt caches, result CSV/JSONL files, update logs, package versions, model revisions, and random seeds.
9. Release the human-evaluation attention-check data and final retained IDs.

## 24. Required rerun order

A scientifically clean correction requires this order:

1. Rebuild the three-model trigger with correct activations and add unit tests using benign/toxic examples.
2. Fix conditional-loss masking and verify token-level masks.
3. Implement the claimed online step count and equalize the step budget for SGD, AdamW, and preconditioned updates.
4. Resolve the checkpoint, LoRA targets, safe-corpus size, LR, delta bound, update length, and Trigger-2 epsilon.
5. Rerun all Qwen TTA main/ablation experiments.
6. Rerun DeepSeek from active, non-commented code.
7. Rescore all outputs and build prompt-level tables.
8. Rerun significance tests and regenerate Figures 2–7 and Tables 6–8.
9. Recompute human tables from a frozen, explicitly listed 20-prompt subset.

## 25. Final audit classification

| Component | Classification |
|---|---|
| Table 1 printed numeric values | Match stored output; checkpoint/method labels require correction |
| Table 2 AUROC/AUPR/CI | Numerically match stored output; interpretation and threshold calibration require correction |
| Table 3 first nine rows | Match final stored table output; TTA implementation materially differs from paper |
| Table 3 Trigger-2 | Exact source selection unresolved; 0.18/0.22 mismatch; absent from final row map |
| Table 4 | Printed values match final stored output; several generating paths are historical/commented; all TTA rows require rerun after fixes |
| Table 5 | Arithmetic correct for a 20-prompt subset, but subset not reproducible from released CSV |
| Table 10 all-available section | Independently recomputed and correct |
| Table 10 common-20 section | Values internally consistent but selection not reproducible from released files |
| Tables 6–7 | Exact source artifacts/selection not released |
| Table 8 | Matches saved code output but trigger probabilities use the wrong activation |
| Figures 2–7 | Must be regenerated after implementation fixes |

**Bottom line:** the provenance map can be corrected and most printed numbers can be traced, but the current release does not support the claim that every experimental value is correct under the method described in the paper. Several upstream code discrepancies directly change the generated outputs and require a full rerun rather than a manuscript-only correction.
