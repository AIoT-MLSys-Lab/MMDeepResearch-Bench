# MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Benchmark: MMDR-Bench](https://img.shields.io/badge/Benchmark-MMDR--Bench-red.svg)](#citation)

This repository maintains the codebase of the end-to-end evaluation framework of MMDeepResearch-Bench (MMDR benchmark).

---

## Update (10 August 2026)

Following a reproducibility audit of commit `9f24d03`, we updated the released
evaluator to make future runs and leaderboard exports unambiguous:

- Task split and difficulty are now handled separately. Canonical task IDs
  `Q0-Q99` are Research and `Q100-Q139` are Daily; difficulty remains an
  independent easy, medium, hard, or complex label.
- New records store `task_split`, `task_difficulty`, and
  `evaluator_semantics="canonical-v2"`.
- `detail/leaderboard_export.py` exports task-level and aggregate scores,
  records source hashes and metric denominators, and can reject incomplete
  140-task leaderboard rows.
- The official aggregation is
  `TRACE = 0.6*EVI + 0.4*VEF` and
  `Overall = 0.2*GEN + 0.3*EVI + 0.2*VEF + 0.3*MM`.
- Historical leaderboard evaluation was staged: GEN, EVI, and MM were scored
  first, VEF was evaluated separately, and the retained components were fused
  offline. This maintenance update does not silently rewrite frozen paper
  scores.
- Missing or unusable task-level MOSAIC output contributes `MM = 0` to the
  all-task aggregate, while Sem., Acc., and VQA use conditional denominators
  over applicable, successfully routed outputs. `mm_na_penalty.py` was not used
  for the published leaderboard.

---

## ✨ Key Features

### 🔬 Innovative Metrics for Grounded Research Quality
- **FLAE (Formula-LLM Adaptive Evaluation):** Measures report quality (readability, insightfulness, structure).
- **TRACE (Trustworthy Retrieval-Aligned Citation Evaluation):** Verifies citation support and claim-URL alignment.
  - **VEF (Visual Evidence Fidelity):** A strict gatekeeper enforcing alignment between textual claims and visual evidence (PASS/FAIL).
- **MOSAIC (Multimodal Support-Aligned Integrity Check):** Validates consistency between generated text and visual artifacts (Charts, Diagrams, Photos).

### 🛠️ Engineering & Usability
- **Smart Resume:** Skips already-completed tasks to reduce time and API cost.
- **Graceful Stop:** Safe shutdown via CLI (`stop`, `exit`) or `Ctrl+C`, ensuring partial results are flushed.
- **Precision Debugging:** Run a single case with `--quiz_first` or `--quiz_index`.
- **Multi-Provider Support:** Google Gemini, Azure OpenAI, OpenRouter.

---

## 📦 Installation

### 1) Clone
```bash
git clone https://github.com/AIoT-MLSys-Lab/MMDeepResearch-Bench.git
cd MMDeepResearch-Bench
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### 1) Create `.env`

```bash
cp env.txt .env
```

### 2) Edit `.env`

Example (adjust to your providers/models):

```ini
# --- Roles ---
MMDR_REPORT_PROVIDER=gemini       # gemini | azure | openrouter
MMDR_JUDGE_PROVIDER=gemini

# --- Models ---
MMDR_REPORT_MODEL=gemini-1.5-pro
MMDR_JUDGE_MODEL=gemini-2.5-pro
MMDR_JUDGE_TEMPERATURE=0.2
MMDR_WEIGHTS=2,3,3,2

# --- API Keys / Endpoints ---
GEMINI_API_KEY=AIza...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
OPENROUTER_API_KEY=...
```

---

## 🚀 Usage

### 1) Quick verification (recommended first run)

Run the **first question only** to confirm API + paths:

```bash
python run_pipeline.py --quiz_first
```

### 2) Full batch run

Process all tasks in `quiz.jsonl`:

```bash
python run_pipeline.py --run_id experiment_v1
```

### 3) Targeted debugging

Re-run a single item by 1-based index:

```bash
python run_pipeline.py --quiz_index 5 --run_id debug_q5
```

### 4) Parallel mode

```bash
python run_pipeline.py --max_workers 4
```

---

## 🎮 Runtime Controls

| Command | Action |
|---------|--------|
| `stop` + Enter | Safely stop after current tasks finish; saves outputs |
| `Ctrl+C` | Triggers the same graceful shutdown behavior |

---

## 📂 Output Structure

Outputs are written to `reports_runs/<RUN_ID>/`:

```text
reports_runs/experiment_v1/
├── reports/                  # Markdown research reports
│   ├── Q1.md
│   └── ...
├── results/
│   └── experiment_v1.jsonl   # detailed logs (scores/errors/timings)
├── summary/
│   ├── experiment_v1.json    # machine-readable aggregated metrics
│   ├── experiment_v1.txt     # human-readable summary
│   ├── experiment_v1.leaderboard.csv
│   ├── experiment_v1.task_scores.csv
│   └── experiment_v1.leaderboard_manifest.json
└── mm/                       # multimodal intermediate artifacts
```

The pipeline writes the leaderboard and task-level CSV files automatically. To
export an existing run, or to fail fast when any of the 140 task-level component
scores is missing, run:

```bash
python export_leaderboard.py \
  --results reports_runs/experiment_v1/results/experiment_v1.jsonl \
  --model_name "Your model name" \
  --require_complete
```

`--require_complete` is recommended for every leaderboard submission. The
manifest records the source SHA256, task/component counts, formulas, and whether
the row is submission-ready.

---

## 📊 Metrics Explanation

The pipeline outputs three aggregate scores and one final combined score:

| Aggregate | Full Name | Sub-metrics (Leaderboard) |
|-----------|-----------|--------------------------|
| **FLAE / GEN** | General report quality | **Read.** = `general.R`, **Insh.** = `general.I`, **Stru.** = `general.S` |
| **TRACE** | Citation and visual-evidence fidelity | **Vef.** = thresholded `scores.VEF`, **Con.** = `evidence.E_con`, **Cov.** = `evidence.E_cov`, **Fid.** = `evidence.E_fid` |
| **MOSAIC / MM** | Multimodal evidence integrity | **Sem.** = `semantic`, **Acc.** = `data_accuracy`, **VQA** = `vqa_score` from the final routed MOSAIC summary |
| **FINAL_MMDR** | Official aggregate | `0.2*FLAE + 0.5*TRACE + 0.3*MOSAIC` |

All sub-metrics are available in the output JSON file under `aggregates.{research|all}.submetrics`:

```text
submetrics.general   ->  general.R, general.I, general.S, general.C, ...
submetrics.evidence  ->  evidence.E_con, evidence.E_cov, evidence.E_fid, evidence.E_div, ...
submetrics.mm        ->  mm.avg_metric_by_dim.semantic, .faithful, .data_accuracy, .vqa_score, ...
```

The fixed internal TRACE coefficient is `lambda_VEF=0.4`, so
`TRACE = 0.6*EVI + 0.4*VEF`. Equivalently, the implementation-level formula is
`FINAL_MMDR = 0.2*GEN + 0.3*EVI + 0.2*VEF + 0.3*MM`, which is why the CLI weight
tuple is `2,3,3,2` in `(GEN,EVI,MM,VEF)` order.

`VEF_Raw` is an audit field on the judge's pre-threshold scale and must not be
used as the leaderboard `Vef.` column. The canonical task split is fixed by QID:
`Q0-Q99` are Research and `Q100-Q139` are Daily. MOSAIC dimensions are routed by
visual type, so their manifest counts can be below 140 even for a complete run;
each displayed submetric is averaged over tasks where that dimension applies.

**Compatibility note for commit `9f24d03`:** the earlier text summary exposed
`VEF_Raw_Judge`, and the historical table assembly used that audit value for the
displayed `Vef.` breakdown. Retained caches contain both 0-10 and 0-100 raw
values, so that display is not a valid leaderboard component. This does not
change the published `Overall`, which was computed from thresholded `scores.VEF`.
The exporter above is the canonical submission path and will be used to correct
the affected breakdown column.

A complete example generated from a retained current-schema run is provided in
`reference_outputs/gemini31flash/`.

For detailed computation logic, see:
- `scoring_general.py` -- GEN (FLAE)
- `scoring_evidence.py` -- EVI (TRACE)
- `mm_router5_aggregate.py` -- MM (MOSAIC)
- `accuracy.py` -- VEF (verification gating)

---

## 🧾 Citation

If you find this codebase or the MMDR-Bench dataset useful in your research, please cite:

```bibtex
@misc{huang2026mmdeepresearchbenchbenchmarkmultimodaldeep,
      title={MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents}, 
      author={Peizhou Huang and Zixuan Zhong and Zhongwei Wan and Donghao Zhou and Samiul Alam and Xin Wang and Zexin Li and Zhihao Dou and Li Zhu and Jing Xiong and Chaofan Tao and Yan Xu and Dimitrios Dimitriadis and Tuo Zhang and Mi Zhang},
      year={2026},
      eprint={2601.12346},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.12346}, 
}
```

---

## 📬 Contact and Community Results

If you run MMDR-Bench and obtain interesting results, please submit them through our Google Form:

[Submit results and feedback via Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfOUzuaLJorHAJ3P7g5-vM9mMYB-P3Fuep_6Ln1rhI8hocq-w/viewform?usp=header)

We welcome reports on:
- new model results
- reproduction logs
- implementation issues
- suggestions for future benchmark extensions

---

## 📜 License

This project is released under the **Apache-2.0 License**. See [LICENSE](LICENSE).
