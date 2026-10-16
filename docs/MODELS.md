# Model Checkpoints

All models used in the CAP-TTA experiments, grouped by role. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for where each is loaded.

## Generation and Backbone Models

- [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) — main Qwen3 backbone for CAP-TTA and the Qwen baselines.
- [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base) — Qwen3 base checkpoint used in the OOD-detection experiments.
- [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) — DeepSeek-R1-8B baseline and CAP-TTA backbone.
- [`hirundo-io/DeepSeek-R1-Distill-Llama-8B-Debiased`](https://huggingface.co/hirundo-io/DeepSeek-R1-Distill-Llama-8B-Debiased) — offline debiased DeepSeek baseline.
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) — Mistral baseline.
- `fenffef/Qwen-4B-Instruct-2505-Self-correct` — self-correction/Sherlock checkpoint used in the experiment code. The currently available repository is [`fenffef/Qwen-4B-Instruct-2507-Self-correct`](https://huggingface.co/fenffef/Qwen-4B-Instruct-2507-Self-correct).

## Online Trigger Committee

- [`s-nlp/roberta_toxicity_classifier`](https://huggingface.co/s-nlp/roberta_toxicity_classifier)
- [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert)
- [`unitary/unbiased-toxic-roberta`](https://huggingface.co/unitary/unbiased-toxic-roberta)

## Offline Bias Reporting Models

- [`grammarly/detexd-roberta-base`](https://huggingface.co/grammarly/detexd-roberta-base)
- [`henryscheible/stereoset_trainer_roberta-base_finetuned`](https://huggingface.co/henryscheible/stereoset_trainer_roberta-base_finetuned)
- [`Narrativa/distilroberta-finetuned-stereotype-detection`](https://huggingface.co/Narrativa/distilroberta-finetuned-stereotype-detection)

## Quality and OOD Evaluation

- [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) — perplexity/fluency evaluation and the likelihood-ratio background language model.
