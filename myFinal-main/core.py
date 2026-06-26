"""Shared logic: indices, trials, embeddings, and EER."""

from __future__ import annotations

import io
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, Dataset, DatasetDict

from config import (
    CONDITIONS,
    ENROLL_UTTS,
    HF_CACHE_DIR,
    IMPOSTORS_PER_TRIAL,
    INDEX_DIR,
    MIN_UTTS_PER_STYLE,
    MODELS,
    RANDOM_SEED,
    SAMPLE_RATE,
    STYLE_ALIASES,
)


# --- data model ---


@dataclass(frozen=True)
class Utterance:
    dataset: str
    speaker_id: str
    style: str
    path: str
    utt_id: str


@dataclass(frozen=True)
class Trial:
    dataset: str
    condition: str
    speaker_id: str
    label: int
    enroll_paths: tuple[str, ...]
    test_path: str
    impostor_id: str | None = None


# --- index ---


def _style_from_text(text: str) -> str | None:
    lowered = text.lower()
    for canonical, aliases in STYLE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return canonical
    return None


def save_index(utterances: list[Utterance], dataset: str) -> str:
    os.makedirs(INDEX_DIR, exist_ok=True)
    out_path = os.path.join(INDEX_DIR, f"{dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(u) for u in utterances], f, indent=2)
    return out_path


def load_index(dataset: str) -> list[Utterance]:
    path = os.path.join(INDEX_DIR, f"{dataset}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing index: {path}\nRun: python download.py --dataset {dataset}")
    with open(path, encoding="utf-8") as f:
        return [Utterance(**row) for row in json.load(f)]


def summarize_index(utterances: list[Utterance]) -> dict:
    by_style: dict[str, set[str]] = {"read": set(), "spontaneous": set()}
    counts = {"read": 0, "spontaneous": 0}
    for utt in utterances:
        by_style[utt.style].add(utt.speaker_id)
        counts[utt.style] += 1
    return {
        "num_utterances": len(utterances),
        "num_speakers": len({u.speaker_id for u in utterances}),
        "read_utterances": counts["read"],
        "spontaneous_utterances": counts["spontaneous"],
        "speakers_with_read": len(by_style["read"]),
        "speakers_with_spontaneous": len(by_style["spontaneous"]),
    }


# --- HF audio decode (no torchcodec) ---


def disable_audio_decode(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
    if isinstance(dataset, DatasetDict):
        return DatasetDict({name: disable_audio_decode(split) for name, split in dataset.items()})
    for column, feature in dataset.features.items():
        if isinstance(feature, Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    return dataset


def audio_column(dataset: Dataset) -> str:
    for column, feature in dataset.features.items():
        if isinstance(feature, Audio):
            return column
    raise KeyError("No Audio column found in dataset.")


def decode_audio_field(audio: dict) -> tuple[np.ndarray, int]:
    if audio.get("array") is not None:
        return np.asarray(audio["array"], dtype=np.float32).reshape(-1), int(audio["sampling_rate"])
    if audio.get("bytes"):
        data, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sample_rate)
    if audio.get("path"):
        data, sample_rate = sf.read(audio["path"], dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sample_rate)
    raise ValueError("Audio field has no array, bytes, or path.")


def write_wav(path: str, array, sample_rate: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    waveform = torch.tensor(array, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform, sample_rate)


# --- trials ---


def _group_by_speaker(utterances: list[Utterance]) -> dict[str, dict[str, list[Utterance]]]:
    grouped: dict[str, dict[str, list[Utterance]]] = defaultdict(
        lambda: {"read": [], "spontaneous": []}
    )
    for utt in utterances:
        grouped[utt.speaker_id][utt.style].append(utt)
    return grouped


def _eligible_speakers(
    grouped,
    enroll_style: str,
    test_style: str,
    enroll_utts: int,
) -> list[str]:
    eligible = []
    for speaker_id, styles in grouped.items():
        enroll_pool = styles[enroll_style]
        test_pool = styles[test_style]
        if enroll_style == test_style:
            if len(enroll_pool) < enroll_utts + 1:
                continue
        elif len(enroll_pool) < enroll_utts or len(test_pool) < 1:
            continue
        eligible.append(speaker_id)
    return sorted(eligible)


def build_trials(
    utterances: list[Utterance],
    dataset: str,
    enroll_utts: int | None = None,
    conditions: dict[str, tuple[str, str]] | None = None,
) -> list[Trial]:
    enroll_n = ENROLL_UTTS if enroll_utts is None else enroll_utts
    trial_conditions = CONDITIONS if conditions is None else conditions
    rng = random.Random(RANDOM_SEED)
    grouped = _group_by_speaker(utterances)
    trials: list[Trial] = []

    for condition, (enroll_style, test_style) in trial_conditions.items():
        speakers = _eligible_speakers(grouped, enroll_style, test_style, enroll_n)
        if len(speakers) < 2:
            continue

        for speaker_id in speakers:
            enroll_pool = list(grouped[speaker_id][enroll_style])
            test_pool = list(grouped[speaker_id][test_style])
            rng.shuffle(enroll_pool)
            rng.shuffle(test_pool)

            enroll = enroll_pool[:enroll_n]
            if enroll_style == test_style:
                test_candidates = [
                    u for u in enroll_pool[enroll_n:] + test_pool
                    if u.path not in {e.path for e in enroll}
                ]
            else:
                test_candidates = [u for u in test_pool if u.path not in {e.path for e in enroll}]

            if not test_candidates:
                continue

            test_utt = test_candidates[0]
            trials.append(
                Trial(
                    dataset=dataset,
                    condition=condition,
                    speaker_id=speaker_id,
                    label=1,
                    enroll_paths=tuple(u.path for u in enroll),
                    test_path=test_utt.path,
                )
            )

            impostors = [s for s in speakers if s != speaker_id]
            rng.shuffle(impostors)
            for impostor_id in impostors[:IMPOSTORS_PER_TRIAL]:
                impostor_enroll = grouped[impostor_id][enroll_style][:enroll_n]
                if len(impostor_enroll) < enroll_n:
                    continue
                trials.append(
                    Trial(
                        dataset=dataset,
                        condition=condition,
                        speaker_id=speaker_id,
                        label=0,
                        enroll_paths=tuple(u.path for u in impostor_enroll),
                        test_path=test_utt.path,
                        impostor_id=impostor_id,
                    )
                )
    return trials


def trials_to_json(trials: list[Trial]) -> list[dict]:
    return [asdict(t) for t in trials]


# --- embeddings ---


def _configure_hf_cache() -> None:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))


@lru_cache(maxsize=2)
def load_model(model_key: str, device: str) -> object:
    _configure_hf_cache()
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source=MODELS[model_key],
        savedir=os.path.join(HF_CACHE_DIR, model_key),
        run_opts={"device": device},
    )


def _load_audio(path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform


def embed_paths(model_key: str, paths: list[str], device: str = "cpu") -> torch.Tensor:
    classifier = load_model(model_key, device)
    embeddings = []
    for path in paths:
        signal = _load_audio(path).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(signal)
        embeddings.append(emb.squeeze().cpu())
    return torch.stack(embeddings).mean(dim=0)


def cosine_score(enroll_emb: torch.Tensor, test_emb: torch.Tensor) -> float:
    enroll = enroll_emb / enroll_emb.norm(p=2)
    test = test_emb / test_emb.norm(p=2)
    return float(torch.dot(enroll, test).item())


# --- EER ---


def compute_eer(scores: list[float], labels: list[int]) -> dict:
    if not scores:
        raise ValueError("No scores provided.")
    if len(set(labels)) < 2:
        raise ValueError("Need both target and impostor trials to compute EER.")

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores_arr)
    scores_sorted = scores_arr[order]
    labels_sorted = labels_arr[order]

    targets = int(labels_arr.sum())
    impostors = int(len(labels_arr) - targets)
    tp = fp = 0
    prev_score = None
    best_eer = 1.0
    best_threshold = scores_sorted[-1]

    for score, label in zip(scores_sorted, labels_sorted):
        if prev_score is not None and score != prev_score:
            eer = 0.5 * ((1.0 - tp / targets) + (fp / impostors))
            if eer < best_eer:
                best_eer, best_threshold = eer, prev_score
        tp += label == 1
        fp += label == 0
        prev_score = score

    eer = 0.5 * ((1.0 - tp / targets) + (fp / impostors))
    if eer < best_eer:
        best_eer = eer
        best_threshold = prev_score if prev_score is not None else best_threshold

    return {
        "eer": float(best_eer),
        "eer_percent": float(best_eer * 100.0),
        "threshold": float(best_threshold),
        "num_trials": len(scores),
        "num_targets": targets,
        "num_impostors": impostors,
    }


def summarize_by_condition(results: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict[str, list]] = {}
    for row in results:
        grouped.setdefault(row["condition"], {"scores": [], "labels": []})
        grouped[row["condition"]]["scores"].append(row["score"])
        grouped[row["condition"]]["labels"].append(row["label"])
    return {cond: compute_eer(p["scores"], p["labels"]) for cond, p in grouped.items()}
