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
