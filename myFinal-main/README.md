# L2 speech-style speaker verification

Evaluate x-vector and ECAPA-TDNN on L2-ARCTIC and Svarah under read/spontaneous trial conditions.

Paper draft: [`paper/paper_edit.tex`](paper/paper_edit.tex) (Overleaf notes in [`paper/README.md`](paper/README.md)).

## Setup

```bash
bash setup_env.sh          # creates conda env l2-spkverif
conda activate l2-spkverif
export HF_CACHE_ROOT="$(pwd)/hf_cache"

hf auth login              # required for gated HuggingFace datasets
python download.py         # downloads audio, writes indices/*.json
python run_eval.py --device auto
```

Enrollment ablation (Svarah RR, 1 vs 2 enrollment utterances):

```bash
python run_ablation_enroll.py --device auto
```

## Layout

| Path | Purpose |
|------|---------|
| `config.py` | Paths, models, trial settings |
| `core.py` | Indices, trials, embeddings, EER |
| `download.py` | HF download + index build |
| `run_eval.py` | Full evaluation → `results/summary.json` |
| `run_ablation_enroll.py` | Enrollment-length ablation |
| `indices/` | Generated utterance indices (rebuilt by `download.py`) |
| `data/` | Downloaded audio (gitignored) |
| `results/` | JSON outputs (`summary.json` tracked) |

## Requirements

- Linux, conda, HuggingFace account with access to KoelLabs/L2Arctic and ai4bharat/Svarah
- GPU optional (`--device cuda`); CPU works but is slow
