#!/usr/bin/env python3
"""Download L2-ARCTIC and Svarah from Hugging Face."""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from datasets import load_dataset

from config import DATASETS, HF_CACHE_DIR, PROJECT_ROOT
from core import (
    Utterance,
    audio_column,
    decode_audio_field,
    disable_audio_decode,
    save_index,
    summarize_index,
    write_wav,
)

L2_ARCTIC_HF_ID = "KoelLabs/L2Arctic"
SVARAH_HF_ID = "ai4bharat/Svarah"
SPONTANEOUS_CHUNK_SEC = 10.0
SPONTANEOUS_MIN_CHUNK_SEC = 3.0


def _chunk_waveform(array, sample_rate: int) -> list[tuple[np.ndarray, int]]:
    waveform = np.asarray(array, dtype=np.float32).reshape(-1)
    chunk_len = int(SPONTANEOUS_CHUNK_SEC * sample_rate)
    min_len = int(SPONTANEOUS_MIN_CHUNK_SEC * sample_rate)
    if len(waveform) <= chunk_len:
        return [(waveform, sample_rate)]

    chunks: list[tuple[np.ndarray, int]] = []
    start = 0
    while start < len(waveform):
        end = min(start + chunk_len, len(waveform))
        piece = waveform[start:end]
        if len(piece) < min_len:
            if chunks:
                prev, sr = chunks[-1]
                chunks[-1] = (np.concatenate([prev, piece]), sr)
            else:
                chunks.append((piece, sample_rate))
            break
        chunks.append((piece, sample_rate))
        start = end
    return chunks


def _export_segments(
    utterances: list[Utterance],
    out_dir: str,
    style: str,
    speaker: str,
    utt_id: str,
    array,
    sample_rate: int,
    dataset: str,
    chunk_spontaneous: bool,
) -> None:
    segments = (
        _chunk_waveform(array, sample_rate)
        if chunk_spontaneous
        else [(np.asarray(array, dtype=np.float32).reshape(-1), sample_rate)]
    )
    for chunk_idx, (segment, sr) in enumerate(segments):
        chunk_id = f"{utt_id}_chunk_{chunk_idx}" if len(segments) > 1 else utt_id
        wav_path = os.path.join(out_dir, style, speaker, f"{chunk_id}.wav")
        write_wav(wav_path, segment, sr)
        utterances.append(
            Utterance(
                dataset=dataset,
                speaker_id=speaker,
                style=style,
                path=os.path.abspath(wav_path),
                utt_id=chunk_id,
            )
        )


def download_l2_arctic(out_dir: str) -> list[Utterance]:
    print(f"Loading {L2_ARCTIC_HF_ID} ...")
    dataset_dict = disable_audio_decode(load_dataset(L2_ARCTIC_HF_ID, cache_dir=HF_CACHE_DIR))
    utterances: list[Utterance] = []

    for split_name, style in (("scripted", "read"), ("spontaneous", "spontaneous")):
        print(f"Exporting {split_name} ...")
        if style == "spontaneous":
            spontaneous_dir = os.path.join(out_dir, style)
            if os.path.isdir(spontaneous_dir):
                shutil.rmtree(spontaneous_dir)

        ds = dataset_dict[split_name]
        for i, row in enumerate(ds):
            speaker = next(
                str(row[k]).upper()
                for k in ("speaker_code", "speaker_id", "speaker", "spk_id")
                if row.get(k)
            )
            utt_id = next(
                str(row[k])
                for k in ("utterance_id", "utt_id", "file", "filename")
                if row.get(k)
            ) if any(row.get(k) for k in ("utterance_id", "utt_id", "file", "filename")) else f"{split_name}_{i:05d}"
            array, sample_rate = decode_audio_field(row["audio"])
            _export_segments(
                utterances, out_dir, style, speaker, utt_id, array, sample_rate,
                "l2_arctic", chunk_spontaneous=(style == "spontaneous"),
            )
            if (i + 1) % 200 == 0:
                print(f"  processed {i + 1}/{len(ds)} recordings")
        print(f"  done ({sum(1 for u in utterances if u.style == style)} {style} utterances)")

    return utterances


def _svarah_style(row: dict) -> str:
    text = (row.get("text") or "").strip().lower()
    if not text:
        return "read"
    if "?" in text or text.startswith(
        ("can ", "please ", "what ", "how ", "tell ", "list ", "show ", "give ", "i want", "could you")
    ):
        return "spontaneous"
    if any(m in text for m in ("application id", "scholarship", "passport", "pension", "nsp", "aadhaar", "upi")):
        return "spontaneous"
    if len(text.split()) <= 5 and float(row.get("duration") or 0) < 4.0:
        return "spontaneous"
    return "read"


def _svarah_speaker(row: dict) -> str:
    parts = [row.get(k, "") for k in ("gender", "primary_language", "native_place_state", "native_place_district", "age-group")]
    return "_".join(str(p).replace(" ", "") for p in parts if p)


def download_svarah(out_dir: str, split: str = "test") -> list[Utterance]:
    print(f"Loading {SVARAH_HF_ID} (split={split}) ...")
    ds = disable_audio_decode(load_dataset(SVARAH_HF_ID, split=split, cache_dir=HF_CACHE_DIR))
    audio_key = audio_column(ds)
    utterances: list[Utterance] = []

    for i, row in enumerate(ds):
        speaker = _svarah_speaker(row)
        style = _svarah_style(row)
        audio = row[audio_key]
        path = audio.get("path") if isinstance(audio, dict) else None
        utt_id = os.path.splitext(os.path.basename(path))[0] if path else f"utt_{i:05d}"
        array, sample_rate = decode_audio_field(audio)
        _export_segments(
            utterances, out_dir, style, speaker, utt_id, array, sample_rate,
            "svarah", chunk_spontaneous=False,
        )
        if (i + 1) % 500 == 0:
            print(f"  exported {i + 1}/{len(ds)} utterances")

    print(f"  done ({len(utterances)} utterances)")
    return utterances


DOWNLOADERS = {
    "l2_arctic": lambda out: download_l2_arctic(out),
    "svarah": lambda out: download_svarah(out),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help="Dataset to download (repeat for multiple; default: all).",
    )
    args = parser.parse_args()

    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    selected = args.dataset or list(DATASETS)

    for name in selected:
        out_dir = DATASETS[name]
        os.makedirs(out_dir, exist_ok=True)
        utterances = DOWNLOADERS[name](out_dir)
        out_path = save_index(utterances, name)
        stats = summarize_index(utterances)
        print(f"\n[{name}] index -> {out_path}")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
