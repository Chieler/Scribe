"""The two gates from the live-captions design, measured rather than assumed.

    venv/bin/python scripts/measure_live.py <audio.wav>

Gate 1, duty cycle: what fraction of realtime does the rolling re-transcribe
actually consume? The design projects ~25% at 110m from batch throughput,
assuming per-call overhead on a 30s generate() is negligible.

Gate 2, acceptance: does the committed live transcript match what the batch
path produces for the same audio? Approach A's whole claim is that rolling
re-transcribe preserves measured quality.
"""
import argparse
import os
import time

import mlx.core as mx
import numpy as np
from parakeet_mlx.audio import load_audio

from captionlm.config import MODEL_ID_110M
from captionlm.eval import _wer
from captionlm.live import LiveSession
from captionlm.progressive import transcribe_progressive
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer
from captionlm.biased_model import load_biased_model


def _feed(model, wav: str, chunk: float = 1.0, **kw):
    """Run a file through a LiveSession as if it arrived in real time."""
    rate = model.preprocessor_config.sample_rate
    audio = np.array(load_audio(wav, rate, mx.float32), copy=False)
    session = LiveSession(model, **kw)
    committed, timings = [], []
    step = int(chunk * rate)
    for i in range(0, len(audio), step):
        started = time.monotonic()
        committed += session.add_audio(audio[i : i + step])
        timings.append(time.monotonic() - started)
    committed += session.finish()
    return session, committed, timings, len(audio) / rate


def measure_duty_cycle(model, wav: str, **kw) -> dict:
    session, _, timings, seconds = _feed(model, wav, **kw)
    passes = [t for t in timings if t > 0.01]      # calls that actually decoded
    return {
        "passes": len(passes),
        "mean_decode_s": float(np.mean(passes)) if passes else 0.0,
        "max_decode_s": float(np.max(passes)) if passes else 0.0,
        "duty_cycle": float(np.sum(timings)) / seconds,
        "interval": session.interval,
        "degraded": session.degraded,
        "audio_s": seconds,
    }


def measure_acceptance(model, wav: str, reference: str, **kw) -> dict:
    """Live transcript versus batch transcript of the same audio.

    Both are scored against the reference so the number is a WER a reader
    already knows how to interpret, rather than a similarity score between two
    hypotheses.
    """
    _, committed, _, _ = _feed(model, wav, **kw)
    live_text = "".join(t.text for t in committed).strip()
    batch_text = transcribe_progressive(model, wav).text.strip()

    # _wer applies eval.py's normaliser to both sides, so this number is
    # directly comparable to every other WER this project has recorded.
    live_wer = _wer([reference], [live_text])
    batch_wer = _wer([reference], [batch_text])
    return {
        "live_wer": live_wer,
        "batch_wer": batch_wer,
        "delta": live_wer - batch_wer,
        "live_text": live_text,
        "batch_text": batch_text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--terms", help="Bias term list (default: <wav>.terms.txt if present)")
    parser.add_argument("--model", default=MODEL_ID_110M)
    args = parser.parse_args()

    model = load_biased_model(args.model)
    terms_path = args.terms or os.path.splitext(args.wav)[0] + ".terms.txt"
    if os.path.isfile(terms_path):
        terms = load_term_list(terms_path)
        model.context_graph = build_context_graph(
            terms, load_tokenizer(args.model), len(model.vocabulary)
        )
        print(f"biased by {len(terms)} terms from {terms_path}")
    else:
        print("no term list; running unbiased")

    d = measure_duty_cycle(model, args.wav)
    print(f"\nGATE 1  duty cycle")
    print(f"  audio            {d['audio_s']:.1f}s")
    print(f"  decode passes    {d['passes']}")
    print(f"  mean per pass    {d['mean_decode_s']:.3f}s")
    print(f"  worst pass       {d['max_decode_s']:.3f}s")
    print(f"  duty cycle       {d['duty_cycle']:.1%}")
    print(f"  final interval   {d['interval']:.1f}s (2.0 = never retuned)")
    print(f"  degraded         {d['degraded']}")
    verdict = "PASS" if d["duty_cycle"] < 0.8 and not d["degraded"] else "FAIL"
    print(f"  -> {verdict}")

    ref_path = os.path.splitext(args.wav)[0] + ".txt"
    if os.path.isfile(ref_path):
        reference = open(ref_path, encoding="utf-8").read().strip()
        a = measure_acceptance(model, args.wav, reference)
        print(f"\nGATE 2  live against batch")
        print(f"  batch WER        {a['batch_wer']:.4f}")
        print(f"  live WER         {a['live_wer']:.4f}")
        print(f"  delta            {a['delta']:+.4f}")
        print(f"  -> {'PASS' if a['delta'] < 0.02 else 'FAIL'}")
    else:
        print(f"\nGATE 2 skipped: no reference at {ref_path}")


if __name__ == "__main__":
    main()
