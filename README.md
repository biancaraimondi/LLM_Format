<h1 align="center"> From Reasoning to Code: GRPO Optimization for
Underrepresented Languages </h1>
<p align="center">
    <strong>Federico Pennino</strong><sup>1</sup>,
    <strong>Bianca Raimondi</strong><sup>1</sup>,
  <strong>Massimo Rondelli</strong><sup>1</sup>,
  <strong>Andrea Gurioli</strong><sup>1</sup>,
  <strong>Maurizio Gabbrielli</strong><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>University of Bologna, Bologna, Italy<br>
</p>
<p align="center">
   <a href="https://arxiv.org/abs/2506.11027"><img src="https://img.shields.io/badge/arXiv-2506.11027-b31b1b.svg?style=flat-square" alt="Paper"></a>
   <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg?style=flat-square" alt="License"></a>
</p>

# Abstract
Generating accurate and executable code using Large Language Models (LLMs) remains a significant challenge for underrepresented programming languages, such as Prolog and Lisp, due to the scarcity of public training data compared to high-resource languages like Python. This paper introduces a generalizable Reinforcement Learning (RL) approach that combines small-scale versions of the Qwen2.5-Coder model with Group Relative Policy Optimization (GRPO) to enable effective code generation through reasoning. To address the limitations of sparse datasets, we integrate execution-driven feedback directly into the RL loop, utilizing a reward system that exploits both logical correctness and structural formatting. Experimental results on GSM8K dataset demonstrate significant improvements in reasoning quality and code accuracy across underrepresented languages. These findings underscore the potential of our approach to benefit a wide range of programming languages lacking extensive training resources by leveraging symbolic reasoning and interpreter-based feedback.

---

# How It Works

The model is trained on GSM8K math word problems and must solve each one by generating valid **Prolog** code. Every response must follow this structured format:

```
<reasoning>
...chain-of-thought reasoning...
</reasoning>
<code>
...prolog facts and rules...
</code>
<query>
...prolog query to extract the answer...
</query>
```

Correctness is verified at training time by executing the generated Prolog knowledge base against the expected numerical answer using a live SWI-Prolog interpreter (via `pyswip`), with a 5-second timeout per execution.

---

# Setup
We strongly suggest to create a conda environment in which installing the needed dependencies. You can do that like this:
### Install Python dependencies
```bash
conda create -n <env_name>
conda activate <env_name>
pip install -r requirements.txt
```

### Weights & Biases

Training logs to W&B. Set your entity in `train.py` before running:

```python
os.environ["WANDB_ENTITY"] = "your-wandb-username"
```

---

# Training

### Quick start

Edit `train.sh` to select the model size(s) and prompting mode(s), then run:

```bash
./train.sh
```

Or invoke directly:

```bash
# Zero-shot, 7B model
python train.py --model_B 7 --one_shot 0

# One-shot, 3B model
python train.py --model_B 3 --one_shot 1

# Five-shot, 1.5B model
python train.py --model_B 1.5 --one_shot 5
```

Available model sizes (`--model_B`): `0.5`, `1.5`, `3`, `7`  
Available prompting modes (`--one_shot`): `0` (zero-shot), `1` (one-shot), `5` (five-shot)

### Prompting strategies

Each mode prepends a different system prompt to every GSM8K training question:

| Mode | `--one_shot` | Description |
|------|-------------|-------------|
| Zero-shot | `0` | Instructions only — no Prolog examples |
| One-shot | `1` | Instructions + 1 worked Prolog example |
| Five-shot | `5` | Instructions + 5 worked Prolog examples |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base models | `Qwen2.5-Coder-{0.5,1.5,3,7}B-Instruct` |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Quantization | 4-bit |
| Learning rate | 5e-6 |
| LR scheduler | cosine |
| Optimizer | paged AdamW 8-bit |
| Warmup ratio | 0.1 |
| Weight decay | 0.1 |
| Batch size | 1 (grad accumulation = 4) |
| Generations per step | 8 |
| Max prompt length | 512 tokens |
| Max completion length | 1024 tokens |
| Max steps | 1500 |
| Checkpoint every | 500 steps |
| Max gradient norm | 0.1 |

Checkpoints are saved under `Qwen-{SIZE}B[_one_shot|_five_shot][_length]/checkpoint-{step}/`.

### Reward functions

Training uses a composite reward combining six signals:

| Reward | Max value | What it measures |
|--------|-----------|-----------------|
| `correctness_reward_func` | +2 | Executes the Prolog KB + query via SWI-Prolog; returns `+2` if the numerical answer matches ground truth, `-1` otherwise |
| `xmlcount_reward_func` | +0.75 | Assigns partial credit (+0.125 each) for the presence of correct XML tags (`<reasoning>`, `</reasoning>`, `<code>`, `</code>`, `<query>`, `</query>`); penalises `-0.5` if the query leaks into `<code>` |
| `soft_format_reward_func` | +0.5 | Regex check that all three sections appear in the right order |
| `count_reasoning` | unbounded | +0.0001 per word in the reasoning section — encourages the model to think |
| `count_code` | unbounded | +0.0001 per word in the code section — encourages completeness |
| `length_correctness_reasoning_reward_func` | +1 | Bonus for correct solutions whose reasoning length falls within a target range (90–130 words) |

---

# Evaluation

### Quick start

Edit `test.sh` to select models, checkpoints, datasets, and prompting modes, then run:

```bash
./test.sh
```

Or invoke directly:

```bash
# Base model (no fine-tuning), zero-shot, GSM8K test set
python test.py --model_B 7 --checkpoint '' --one_shot 0 --length '' --dataset gsm8k

# Fine-tuned checkpoint 1500, one-shot, Rosetta Code benchmark
python test.py --model_B 7 --checkpoint 1500 --one_shot 1 --length '_length' --dataset rosetta2

# Fine-tuned checkpoint 1000, one-shot, GSM-Symbolic (partition p1)
python test.py --model_B 7 --checkpoint 1000 --one_shot 1 --length '_length' --dataset GSM-Symbolic --p p1
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--model_B` | Model size: `0.5`, `1.5`, `3`, `7` |
| `--checkpoint` | Checkpoint step (`500`, `1000`, `1500`) or `''` for the base model |
| `--one_shot` | Prompting mode: `0`, `1`, or `5` |
| `--length` | Suffix for the checkpoint directory: `''`, `'_'`, or `'_length'` |
| `--dataset` | `gsm8k`, `rosetta2`, or `GSM-Symbolic` |
| `--p` | GSM-Symbolic partition (e.g. `p1`, `p2`); only used when `--dataset GSM-Symbolic` |

### Datasets

| Dataset | Split | Source |
|---------|-------|--------|
| GSM8K | test (1319 problems) | `openai/gsm8k` via HuggingFace |
| GSM-Symbolic | test (1300 problems) | `apple/GSM-Symbolic` via HuggingFace |
| Rosetta Code (Prolog) | — | `data/prolog_tasks_ground_truth.csv` |

### Metrics

Accuracy is computed as **pass@1**: a question is considered solved if at least one of the 4 generated samples produces the correct answer when executed by SWI-Prolog.

Each evaluation run saves a CSV to `results/` with per-question correctness, raw generated code for all 4 samples, and summary statistics.

---

# Citation
```bibtex
@misc{pennino2026reasoningcodegrpooptimization,
      title={From Reasoning to Code: GRPO Optimization for Underrepresented Languages}, 
      author={Federico Pennino and Bianca Raimondi and Massimo Rondelli and Andrea Gurioli and Maurizio Gabbrielli},
      year={2026},
      eprint={2506.11027},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.11027}, 
}
```
