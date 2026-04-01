# MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Benchmark: MMDR-Bench](https://img.shields.io/badge/Benchmark-MMDR--Bench-red.svg)](#citation)

This repository maintains the codebase of the end-to-end evaluation framework of MMDeepResearch-Bench (MMDR benchmark).

---

## ✨ Key Features

### 🔬 Innovative Metrics for Grounded Research Quality
- **FLAE (Formula-LLM Adaptive Evaluation):** Measures report quality (readability, insightfulness, structure).
- **TRACE (Trustworthy Retrieval-Aligned Citation Evaluation):** Verifies citation support and claim–URL alignment.
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
git clone [https://github.com/YourUsername/MMDR.git](https://github.com/YourUsername/MMDR.git)
cd MMDR

```

### 2) Install dependencies

```bash
pip install -r requirements.txt

```

---

## ⚙️ Configuration

### 1) Create `.env`

```bash
cp env.txt

```

### 2) Edit `.env`

Example (adjust to your providers/models):

```ini
# --- Roles ---
MMDR_REPORT_PROVIDER=gemini       # gemini | azure | openrouter
MMDR_JUDGE_PROVIDER=azure         # recommended: strong reasoning model

# --- Models ---
MMDR_REPORT_MODEL=gemini-1.5-pro
MMDR_JUDGE_MODEL=gpt-4o

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
| --- | --- |
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
│   └── experiment_v1.txt     # aggregated stats (pass rate/avg scores)
└── mm/                       # multimodal intermediate artifacts

```

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
## Contact and Community Results

If you run MMDR-Bench and obtain interesting results, please submit them through our Google Form:

[Submit results and feedback via Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfOUzuaLJorHAJ3P7g5-vM9mMYB-P3Fuep_6Ln1rhI8hocq-w/viewform?usp=header)

We welcome reports on:
- new model results
- reproduction logs
- implementation issues
- suggestions for future benchmark extensions

---


## 📜 License

This project is released under the **Apache-2.0 License**. See [LICENSE](https://www.google.com/search?q=LICENSE).
