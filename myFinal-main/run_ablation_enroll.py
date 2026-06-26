#!/usr/bin/env python3
"""Ablation: enrollment utterance count on Svarah Read--Read trials."""

from __future__ import annotations

import argparse
import json
import os

import torch

from config import MODELS, RESULTS_DIR
from core import (
    build_trials,
    cosine_score,
    embed_paths,
    load_index,
    summarize_by_condition,
)
from run_eval import _pick_device


def run_ablation(model_key: str, enroll_utts: int, device: str) -> dict:
    utterances = load_index("svarah")
    conditions = {"read_read": ("read", "read")}
    trials = build_trials(utterances, "svarah", enroll_utts=enroll_utts, conditions=conditions)
    if not trials:
        raise RuntimeError(f"No trials for enroll_utts={enroll_utts}")

    rows = []
    for trial in trials:
        enroll_emb = embed_paths(model_key, list(trial.enroll_paths), device=device)
        test_emb = embed_paths(model_key, [trial.test_path], device=device)
        rows.append(
            {
                "condition": trial.condition,
                "label": trial.label,
                "score": cosine_score(enroll_emb, test_emb),
            }
        )

    return {
        "dataset": "svarah",
        "model": model_key,
        "condition": "read_read",
        "enroll_utts": enroll_utts,
        "num_trials": len(trials),
        "by_condition": summarize_by_condition(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrollment-length ablation on Svarah RR.")
    parser.add_argument("--model", choices=sorted(MODELS), action="append")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    models = args.model or list(MODELS)
    device = _pick_device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for enroll_utts in (1, 2):
        for model_key in models:
            print(f"Svarah RR | {model_key} | enroll_utts={enroll_utts} | {device}")
            result = run_ablation(model_key, enroll_utts, device)
            out_path = os.path.join(
                RESULTS_DIR, f"ablation_svarah_rr_enroll{enroll_utts}_{model_key}.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            eer = result["by_condition"]["read_read"]["eer_percent"]
            print(f"  EER={eer:.2f}% -> {out_path}")
            results.append(result)

    summary_path = os.path.join(RESULTS_DIR, "ablation_enroll_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
