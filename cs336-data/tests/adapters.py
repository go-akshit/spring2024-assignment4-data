#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

import sys
sys.path.insert(0, './cs336-data/cs336_data')
from cc_filtering import (
    extract_text,
    language_detection,
    mask_email,
    mask_phone_numbers,
    mask_ips,
    classify_nsfw,
    classify_toxic_speech,
    gopher_quality_filter,
    exact_deduplication, 
    minhash_lsh_deduplication
)

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return language_detection(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
     return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    exact_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_lsh_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
