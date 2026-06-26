"""Paths and experiment settings for L2 speech-style speaker verification."""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
INDEX_DIR = os.path.join(PROJECT_ROOT, "indices")

# Defaults match download.py output.
L2_ARCTIC_ROOT = os.path.join(PROJECT_ROOT, "data", "L2Arctic")
SVARAH_ROOT = os.path.join(PROJECT_ROOT, "data", "Svarah")

DATASETS = {
    "l2_arctic": L2_ARCTIC_ROOT,
    "svarah": SVARAH_ROOT,
}

MODELS = {
    "xvector": "speechbrain/spkrec-xvect-voxceleb",
    "ecapa": "speechbrain/spkrec-ecapa-voxceleb",
}

# Matched and mismatched enrollment/test style pairs.
CONDITIONS = {
    "read_read": ("read", "read"),
    "spontaneous_spontaneous": ("spontaneous", "spontaneous"),
    "read_spontaneous": ("read", "spontaneous"),
}

STYLE_ALIASES = {
    "read": ("read", "arctic", "prompted", "scripted"),
    "spontaneous": (
        "spontaneous",
        "spont",
        "conversational",
        "conv",
        "conversation",
        "suitcase",
    ),
}

SAMPLE_RATE = 16000
MIN_UTTS_PER_STYLE = 2
ENROLL_UTTS = 1
IMPOSTORS_PER_TRIAL = 5
RANDOM_SEED = 42

HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_ROOT", os.path.join(PROJECT_ROOT, "hf_cache")
)
