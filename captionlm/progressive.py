"""Transcribe a long file chunk by chunk, handing out the transcript so far.

`BaseParakeet.transcribe` already does exactly this internally -- it walks
overlapping windows and merges each into a running token list -- but it keeps
the running list to itself and returns only at the end, so a UI has nothing to
show for the length of the audio. `chunk_callback` reports progress as two
numbers, not text.

The streaming API next door (`transcribe_stream` / `StreamingParakeet`) is not
a substitute here: `add_audio` calls `model.decode()` directly, bypassing
`generate()`, which is where `BiasedParakeetTDTCTC` runs the CTC word spotter.
Streaming that way would silently drop every bit of term biasing, which is the
entire product.

So this mirrors the library's loop and calls `on_partial` after each merge. The
merging itself is the library's own, not a reimplementation.

It also decodes chunks in batches rather than one at a time, which the
library's loop cannot do and which is where most of the runtime went --
see docs/results/2026-09-05-decode-speed.md.
"""
from typing import Callable, Optional

import mlx.core as mx
from parakeet_mlx.alignment import (
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from parakeet_mlx.audio import get_logmel, load_audio
from parakeet_mlx.parakeet import AlignedResult, DecodingConfig

from captionlm.config import DECODE_BATCH


def transcribe_progressive(
    model,
    path: str,
    *,
    chunk_duration: float = 120.0,
    overlap_duration: float = 15.0,
    decode_batch: int = DECODE_BATCH,
    dtype: mx.Dtype = mx.bfloat16,
    decoding_config: DecodingConfig = DecodingConfig(),
    on_partial: Optional[Callable[[AlignedResult, float, float], None]] = None,
) -> AlignedResult:
    """Transcribe `path`, calling `on_partial(result, done, total)` per chunk.

    `done` and `total` are seconds, so a caller can render progress from the
    same callback it renders text from. The final result is returned as well,
    identical in shape to `model.transcribe`.

    Chunks are decoded `decode_batch` at a time. Every chunk still gets its
    own partial, in order, but they arrive in bursts of `decode_batch`: that
    is the whole cost of the batching, and it buys 2.4x on an hour of audio.
    """
    rate = model.preprocessor_config.sample_rate
    audio = load_audio(path, rate, dtype)
    total = len(audio) / rate

    if total <= chunk_duration:
        result = model.generate(get_logmel(audio, model.preprocessor_config),
                                decoding_config=decoding_config)[0]
        if on_partial:
            on_partial(result, total, total)
        return result

    chunk_samples = int(chunk_duration * rate)
    overlap_samples = int(overlap_duration * rate)
    tokens: list = []
    pending: list[tuple[mx.array, int, int]] = []   # mel, start, end

    def decode_pending() -> None:
        nonlocal tokens
        if not pending:
            return
        chunks = model.generate_batch(
            [mel for mel, _, _ in pending], decoding_config=decoding_config
        )
        for chunk, (_, start, end) in zip(chunks, pending):
            offset = start / rate
            for sentence in chunk.sentences:
                for token in sentence.tokens:
                    token.start += offset
                    token.end = token.start + token.duration

            if tokens:
                try:
                    tokens = merge_longest_contiguous(
                        tokens, chunk.tokens, overlap_duration=overlap_duration
                    )
                except RuntimeError:
                    tokens = merge_longest_common_subsequence(
                        tokens, chunk.tokens, overlap_duration=overlap_duration
                    )
            else:
                tokens = chunk.tokens

            if on_partial:
                on_partial(
                    sentences_to_result(
                        tokens_to_sentences(tokens, decoding_config.sentence)
                    ),
                    end / rate,
                    total,
                )
        pending.clear()

    for start in range(0, len(audio), chunk_samples - overlap_samples):
        end = min(start + chunk_samples, len(audio))
        if end - start < model.preprocessor_config.hop_length:
            break  # prevent zero-length log mel

        pending.append((get_logmel(audio[start:end], model.preprocessor_config), start, end))
        if len(pending) == decode_batch:
            decode_pending()
    decode_pending()

    return sentences_to_result(tokens_to_sentences(tokens, decoding_config.sentence))
