# Catching the Seams

**Stanford CS221 (Autumn 2026) Final Project (Classroom Track)**  
**Authors:** Deepti Gupta (`dgupta13`), Lingyin Wu (`lingyin`), Rohith Pamulasai (`r1021989`), Aditya Anand Bangalore (`adianand`)  
**Proposal:** [`docs/proposal/Proposal_final_submitted.pdf`](docs/proposal/Proposal_final_submitted.pdf)

## Overview

Mixed Human-AI text rarely sits purely at 0% or 100% synthetic prose: writers often draft with an LLM and revise selected sentences, or paste generated lines into human drafts. Document-level detectors average signal across whole articles, while single-sentence classifiers lose accuracy on short 10-to-25-word lines.

Reading a passage left to right one sentence at a time lets the detector inspect three signals together at step $t$: 18 clues from line $t$ ($\phi(c_t)$), 5 seam jumps between line $t-1$ and line $t$ ($\psi(c_{t-1}, c_t)$), and the prior predicted label $\tilde{y}_{t-1}$. Linear weights $w \in \mathbb{R}^{2\times 32}$ produce an exact additive decision margin $\Delta Q_t = Q_w(s_t, \text{AI}) - Q_w(s_t, \text{Human})$ showing which features drove each line label.

## Benchmark Construction

Our M3 benchmark builds leak-free episodes from pre-generated static articles in the **Defactify Text Dataset** ([Roy et al., arXiv:2510.22874](https://arxiv.org/abs/2510.22874); [`Rajarshi-Roy-research/Defactify_Text_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset)), pairing New York Times human reporting with static articles from Gemma-2-9B, Mistral-7B, Qwen-2-72B, Llama-8B, Yi-Large, and GPT-4o (zero LLM inference is run during benchmark creation or training).

- **Split isolation:** Training groups sharing sentences with validation or `test.parquet` are removed (`1,236` blocked groups), and remaining clean groups partition strictly by `story_group` into a `253`-article human bigram LM corpus, `Train` (`630` passages, `6,774` lines), and `Clean Held-Out Test` (`133` passages, `1,423` lines), alongside official `Validation` (`133` passages, `1,454` lines) and noisy `OOD Test` (`133` passages, `1,215` lines).
- **Formatting hygiene:** Strips markdown headers, section tags (`Climate|Headline`), tracker chrome, and repairs fused scrape punctuation (`book.Learn -> book. Learn`).
- **Passage lengths:** All episodes run `6` to `14` sentences (`150` to `392` words), with `0` cross-split sentence overlap.

## Repository Layout

```text
catching-the-seams/
├── configs/default.yaml                  # Shared dataset, feature, baseline, and RL hyperparameters
├── docs/proposal/Proposal_final_submitted.pdf
├── scripts/
│   ├── build_benchmark.py                # Step 1: Clean Defactify Parquet splits into Slice H / E(k)
│   ├── extract_features.py               # Step 2: Fit ReferenceBigramLM and cache standardized features
│   ├── run_baselines.py                  # Step 3: Fit Baseline B1 and Baseline B3
│   ├── train_rl.py                       # Step 4: Train Linear Q-learning, SARSA, and E2 ablations
│   └── evaluate.py                       # Step 5: Compute E1, E2, E3, OOD metrics, and walkthrough
├── src/seam_detector/
│   ├── schema.py                         # Label, LineRecord, Passage, LinePrediction dataclasses
│   ├── config_util.py                    # YAML config loader
│   ├── data/                             # Defactify Parquet cleaner and Slice H / E(k) builder
│   ├── features/                         # phi(c_t), psi(c_{t-1}, c_t), and 32-dim state featurizer
│   ├── baselines/                        # B1 smoothness proxy, B3 SGD logistic regression, and oracle bounds
│   ├── rl/                               # Sequential MDP environment, Q-learning, and SARSA agents
│   └── eval/                             # Macro-F1, Slice H FPR, AI-share MAE, and faithfulness flip checks
└── tests/test_smoke.py                   # Unit tests across schema, cleaning, features, MDP, RL, and metrics
```

## Running the Pipeline and Tests

Run the unit tests with Python's standard library runner:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run all five CPU stages end-to-end (`configs/default.yaml` sets default paths and hyperparameters):

```bash
PYTHONPATH=src python3 scripts/build_benchmark.py --config configs/default.yaml
PYTHONPATH=src python3 scripts/extract_features.py --config configs/default.yaml
PYTHONPATH=src python3 scripts/run_baselines.py --config configs/default.yaml
PYTHONPATH=src python3 scripts/train_rl.py --config configs/default.yaml
PYTHONPATH=src python3 scripts/evaluate.py --config configs/default.yaml
```

Override `--parquet-dir`, `--data-dir`, `--ckpt-dir`, or `--out-dir` explicitly when pointing at custom folders.

## CS221 Classroom Track AI-Use Disclosure

Per the [CS221 Classroom Track AI Policy](https://stanford-cs221.github.io/autumn2026/project-classroom.html), students must review and own all core code before milestone submission:

| Area | Role and Student Verification |
| :--- | :--- |
| Packaging & CLI starter files | Repository layout, YAML config loader, `pyproject.toml`, and CLI argument parsing generated during setup and reviewed by the team. |
| Benchmark hygiene & visualization | Parquet error checks, cleaning regexes, and result plotting scripts audited against processed JSON summaries on CPU. |
| Core detection & RL algorithms | Feature extraction (`phi`, `psi`), logistic baseline, MDP rewards, Q-learning/SARSA updates, and E1/E3 metrics reviewed line by line and tested by the four authors before submission. |
| Milestone M4 roadmap | Gemini zero-shot baseline (`B2` in `baselines/gemini_zeroshot.py`), frozen Gemma 3 1B features, and gold human-edit evaluation remain next milestones. |
