# BGEbench

Bug Growth Elasticity Benchmark — a unified experiment framework for assessing
LLM-generated code quality through defect-growth modeling, reliability
assessment, and verification feedback.

## Modules

| Module | Focus |
|--------|-------|
| `growth/` | Measure Bug Growth Elasticity (BGE) from generated code defects |
| `reliability/` | Reliability risk scoring via severity-weighted defects and RRS |
| `feedback/` | Verification feedback loop reducing defects vs direct generation |

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Set your API key before running:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

Or edit `config/settings.yaml` directly. Proxy settings are also in the config file.

## Usage

```bash
# Full pipeline (growth -> reliability -> feedback)
python scripts/run_all.py

# Individual experiments
python scripts/run_growth.py        # BGE measurement
python scripts/run_reliability.py   # Reliability assessment
python scripts/run_feedback.py      # Verification feedback
```

## Data layout

```
data/
├── tasks/       # Cached benchmark copies
├── generated/   # Raw LLM outputs
├── repaired/    # Repair artifacts
├── logs/        # Tool raw outputs
└── results/     # Final CSV/JSONL tables
```
