# Catching the Seams: Explainable Line-Level AI Text Detection with Q-Learning

**Stanford CS221 (Autumn 2026) Final Project — Classroom Track**  
**Authors:** Deepti Gupta (`dgupta13`), Lingyin Wu (`lingyin`), Rohith Pamulasai (`r1021989`), Aditya Anand Bangalore (`adianand`)  
**Proposal:** [`docs/proposal/Proposal_final_submitted.pdf`](docs/proposal/Proposal_final_submitted.pdf)

---

## Overview

Real-world AI writing is frequently hybrid: an author drafts with an LLM and rewrites selected sentences, or inserts AI-polished sentences into human prose. Document-level detectors pool signal across the entire passage (letting human edits dilute remaining AI lines), while independent single-sentence classifiers struggle on short 10–25 token sentences.

**Key idea:** Human–AI editing leaves *seams*—abrupt sentence-to-sentence shifts in log-probability curvature, length, syntax depth, and style. We formulate line-by-line provenance tagging as a sequential Markov Decision Process (MDP) solved with linear function approximation (`Q-learning` and on-policy `SARSA`). Because $Q_w(s_t, a) = w_a^\top f(s_t)$ is linear, every line verdict $\hat{y}_t \in \{\text{Human}, \text{AI}\}$ decomposes exactly into additive feature contributions toward the margin $\Delta Q_t = Q_w(s_t, \text{AI}) - Q_w(s_t, \text{Human})$.

```text
Passage s = (c_1, ..., c_n)
   │
   ▼  left-to-right at step t
State s_t = (phi(c_t), psi(c_{t-1}, c_t), y_hat_{t-1}, t/n)
   │
   ▼  Linear Q-agent: Q_w(s_t, a) = w^T Phi(s_t, a),  a in {Human, AI}
Verdict y_hat_t + exact top feature contributions to Delta Q_t
```

---

## Source Dataset & Benchmark Construction

We ground our benchmark in the **Defactify Text Dataset** ([Roy et al., arXiv:2510.22874](https://arxiv.org/abs/2510.22874); [`Rajarshi-Roy-research/Defactify_Text_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset)):

| Split | Total Articles | `Human_story` (NYT) | Per-LLM Count (`Gemma-2-9B`, `Mistral-7B`, `Qwen-2-72B`, `LLaMA-8B`, `Yi-Large`, `GPT-4o`) |
| :--- | ---: | ---: | ---: |
| `train` | 51,247 | 7,321 | 7,321 each |
| `validation` | 10,983 | 1,569 | 1,569 each |
| `test` | 10,963 | 1,558 | 1,565–1,569 each |
| **Total** | **73,193** | **10,448** | **10,455–10,459 each** |

Each record provides `Text`, `Label_A` (`0` = Human, `1` = AI), and `Label_B` (`Human_story` or source LLM). Because raw Defactify rows are whole articles, our benchmark pipeline constructs line-level provenance episodes of $n \in [6, 20]$ sentences (150–400 words) while preserving split boundaries:

1. **`Slice H` (Pure Human Passages, ~200 passages):** Sampled from `Human_story` (`Label_A = 0`) to measure false-positive rates on unedited human writing.
2. **`Slice E(k)` (Hybrid Seam Passages, ~400 passages across $k \in \{0, 25, 50, 75\}\%$):** Sampled from `Label_A = 1` LLM articles where $k\%$ of sentences are rewritten (by team members for a ~60-passage gold subset in `data/gold_human_edits/` and by controlled voice-rewrite passes for the remaining benchmark), recording exact sentence-level edit logs $y^*_{1..n}$.
3. **Held-Out Case Study (`data/case_study/`):** Qualitative line-by-line evaluation on contested literary prose (*The Serpent in the Grove*).

---

## Repository Layout & Team Leads

The repository mirrors the four-lead division in Proposal Appendix A.3:

```text
catching-the-seams/
├── configs/default.yaml                     # Dataset, feature, MDP, and RL hyperparameters
├── docs/proposal/Proposal_final_submitted.pdf
├── scripts/
│   ├── build_benchmark.py                   # Step 1: Download Defactify & build Slice H / Slice E(k)
│   ├── extract_features.py                  # Step 2: Offline Gemma 3 1B curvature + phi / psi cache
│   ├── run_baselines.py                     # Step 3: B1 (Fast-DetectGPT), B2 (Gemini), B3 (LogReg), Oracle
│   ├── train_rl.py                          # Step 4: Linear Q-learning & SARSA training on CPU
│   └── evaluate.py                          # Step 5: E1 robustness, E2 state ablation, E3 faithfulness
├── src/seam_detector/
│   ├── schema.py                            # Passage, LineRecord, and LinePrediction dataclasses
│   ├── data/                                # Lead 1 (Benchmark): Defactify loader & Slice H / E(k) builder
│   ├── features/                            # Lead 2 (Features): phi(c_t), psi(c_{t-1}, c_t), Phi(s_t, a)
│   ├── baselines/                           # Lead 2 (Baselines): B1, B2, B3, and Edit-Log Oracle
│   ├── rl/                                  # Lead 3 (MDP & RL): Sequential env, Q-learning, SARSA
│   └── eval/                                # Lead 4 (Eval & XAI): Macro-F1, FPR, faithfulness, explanations
└── tests/
    └── test_smoke.py                        # Package import and schema smoke tests
```

---

## Setup & How to Run

### 1. Environment Setup

Using `uv` (recommended) or `pip` with Python 3.10+:

```bash
git clone https://github.com/dg2k11/catching-the-seams.git
cd catching-the-seams

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install optional local NLP / reference-LM dependencies when running offline feature extraction:

```bash
pip install -e ".[nlp]"
python3 -m spacy download en_core_web_sm
```

### 2. Pipeline Commands

Once each module's implementation is completed, run the end-to-end workflow with:

```bash
# 1. Stream/download Defactify splits and construct Slice H + Slice E(k)
python3 scripts/build_benchmark.py --config configs/default.yaml

# 2. Compute offline per-line features phi(c_t) and seam shifts psi(c_{t-1}, c_t)
python3 scripts/extract_features.py --config configs/default.yaml

# 3. Evaluate B1 (Fast-DetectGPT), B2 (Gemini zero-shot), B3 (per-line LogReg), and Oracle
python3 scripts/run_baselines.py --config configs/default.yaml

# 4. Train linear Q-learning and SARSA agents in NumPy on CPU
python3 scripts/train_rl.py --config configs/default.yaml --algorithm q_learning
python3 scripts/train_rl.py --config configs/default.yaml --algorithm sarsa

# 5. Run E1 (edit-rate robustness), E2 (state ablation), and E3 (explanation faithfulness)
python3 scripts/evaluate.py --config configs/default.yaml
```

Run tests with:

```bash
pytest
```

---

## Code Authorship & AI-Use Provenance (CS221 Classroom Track)

Per the [CS221 Classroom Track AI Policy](https://stanford-cs221.github.io/autumn2026/project-classroom.html) and Project Guidelines, all core algorithmic, dataset-construction, feature-extraction, baseline, RL, and evaluation logic should be written by the four student authors. The table below tracks file-level provenance across the repository and should be updated at each milestone (`progress.pdf` and `final.pdf`):

| Path / Module | Provenance Status | Notes |
| :--- | :--- | :--- |
| `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `LICENSE`, `configs/default.yaml` | AI-scaffolded boilerplate (non-core) | Initial project structure, config template, and documentation; reviewed and maintained by student authors |
| `src/seam_detector/schema.py`, `tests/test_smoke.py`, `scripts/*.py` (CLI arg parsing & stubs) | AI-scaffolded boilerplate (non-core) | Data structures and empty CLI entrypoints (`NotImplementedError` placeholders; zero core algorithm code) |
| `src/seam_detector/data/*` | **Human-written (core — in progress)** | Defactify passage segmentation, `Slice H` / `Slice E(k)` construction, and gold human-edit pipeline |
| `src/seam_detector/features/*` | **Human-written (core — in progress)** | Offline `Gemma 3 1B` curvature/log-prob pass, linguistic/rhetorical features $\phi(c_t)$, seam shifts $\psi(c_{t-1}, c_t)$ |
| `src/seam_detector/baselines/*` | **Human-written (core — in progress)** | `B1` Fast-DetectGPT threshold, `B2` zero-shot LLM baseline, `B3` SGD logistic regression, and Oracle bounds |
| `src/seam_detector/rl/*` | **Human-written (core — in progress)** | Sequential provenance MDP, $\epsilon$-greedy linear Q-learning, on-policy SARSA, and heuristic policy evaluation |
| `src/seam_detector/eval/*` | **Human-written (core — in progress)** | Pooled line-level Macro-F1, `Slice H` FPR, passage AI-share error, `E1`–`E3` ablations, and linear explanation renderer |
| `docs/proposal/Proposal_final_submitted.pdf` | **Human-written** | Authored by the four students (LaTeX/TikZ formatting disclosed in Proposal Table 5) |

---

## Citation

If referencing the underlying source dataset, cite:

```bibtex
@misc{roy2026comprehensivedatasethumanvs,
  title         = {A Comprehensive Dataset for Human vs. AI Generated Text Detection},
  author        = {Rajarshi Roy and Gurpreet Singh and Ashhar Aziz and Shashwat Bajpai and Nasrin Imanpour and Shwetangshu Biswas and Kapil Wanaskar and Parth Patwa and Subhankar Ghosh and Shreyas Dixit and Nilesh Ranjan Pal and Vipula Rawte and Ritvik Garimella and Gaytri Jena and Amitava Das and Amit Sheth and Vasu Sharma and Aishwarya Naresh Reganti and Vinija Jain and Aman Chadha},
  year          = {2026},
  eprint        = {2510.22874},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2510.22874}
}
```
