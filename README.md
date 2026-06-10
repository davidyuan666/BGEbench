# BGEbench

Bug Growth Elasticity Benchmark — a unified experiment framework for assessing
LLM-generated code quality through defect-growth modeling, reliability
assessment, and verification feedback.

## Three experiments

| Module | Paper | Focus |
|--------|-------|-------|
| `wsse/` | WSSE 2026 | Measure Bug Growth Elasticity (BGE) from generated code defects |
| `icre/` | ICRE 2026 | Reliability assessment via severity-weighted defects and RRS |
| `eiecc/` | EIECC 2026 | Verification feedback loop reducing defects vs direct generation |

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Copy `config/settings.yaml` and set your API key:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

Or edit `config/settings.yaml` directly.

## Usage

```bash
# Full pipeline (WSSE → ICRE → EIECC)
python scripts/run_all.py

# Individual experiments
python scripts/run_wsse.py    # BGE measurement
python scripts/run_icre.py    # Reliability assessment
python scripts/run_eiecc.py   # Verification feedback
```

## Data layout

```
data/
├── tasks/       # Cached benchmark copies
├── generated/   # Raw LLM outputs
├── repaired/    # EIECC repair artifacts
├── logs/        # Tool raw outputs
└── results/     # Final CSV/JSONL tables
```
