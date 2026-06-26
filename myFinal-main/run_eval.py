#!/usr/bin/env python3
"""Run style-conditioned speaker verification and report EER."""

from __future__ import annotations

import argparse
import json
import os

import torch

from config import DATASETS, MODELS, RESULTS_DIR
from core import (
    build_trials,
    cosine_score,
    embed_paths,
    load_index,
    summarize_by_condition,
    summarize_index,
    trials_to_json,
)


def _pick_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def evaluate_dataset(dataset: str, model_key: str, device: str) -> dict:
    utterances = load_index(dataset)
    trials = build_trials(utterances, dataset)
    if not trials:
        raise RuntimeError(f"No trials built for {dataset}; check index coverage.")

    rows = []
    for trial in trials:
        enroll_emb = embed_paths(model_key, list(trial.enroll_paths), device=device)
        test_emb = embed_paths(model_key, [trial.test_path], device=device)
        rows.append(
            {
                "dataset": trial.dataset,
                "condition": trial.condition,
                "speaker_id": trial.speaker_id,
                "label": trial.label,
                "score": cosine_score(enroll_emb, test_emb),
                "test_path": trial.test_path,
                "impostor_id": trial.impostor_id,
            }
        )

    return {
        "dataset": dataset,
        "model": model_key,
        "index_summary": summarize_index(utterances),
        "num_trials": len(trials),
        "by_condition": summarize_by_condition(rows),
        "trials": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate x-vector / ECAPA on L2 datasets.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    parser.add_argument("--model", choices=sorted(MODELS), action="append")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-trials", action="store_true")
    args = parser.parse_args()

    datasets = args.dataset or list(DATASETS)
    models = args.model or list(MODELS)
    device = _pick_device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summaries = []
    for dataset in datasets:
        trials = build_trials(load_index(dataset), dataset)
        if args.save_trials:
            trial_path = os.path.join(RESULTS_DIR, f"{dataset}_trials.json")
            with open(trial_path, "w", encoding="utf-8") as f:
                json.dump(trials_to_json(trials), f, indent=2)
            print(f"Saved trials to {trial_path}")

        for model_key in models:
            print(f"Evaluating {dataset} with {model_key} on {device} ...")
            result = evaluate_dataset(dataset, model_key, device)
            out_path = os.path.join(RESULTS_DIR, f"{dataset}_{model_key}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"  wrote {out_path}")
            for condition, metrics in result["by_condition"].items():
                print(
                    f"  {condition}: EER={metrics['eer_percent']:.2f}% "
                    f"(targets={metrics['num_targets']}, impostors={metrics['num_impostors']})"
                )
            all_summaries.append(
                {"dataset": dataset, "model": model_key, "by_condition": result["by_condition"]}
            )

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
