"""Project-wide configuration constants."""
from dataclasses import dataclass

# Both are hybrid TDT-CTC, which is what the CTC-WS spotter requires: a
# TDT-only or CTC-only checkpoint has no compatible pair of heads. Measured
# on the read-aloud eval set at cb_weight 3.0 -- 110m: 0.1160 unbiased,
# 0.0989 biased, 438 MB, ~47x realtime. 1.1b: 0.1009 unbiased, 0.0778
# biased, 4.0 GB, ~28x realtime. Neither trains anything at use time.
MODEL_ID_110M = "mlx-community/parakeet-tdt_ctc-110m"
MODEL_ID_1_1B = "mlx-community/parakeet-tdt_ctc-1.1b"

# The small model stays the default: 4.0 GB is the user's disk and the
# user's download, and that is their choice to make with --model, not ours
# to make silently on their behalf.
MODEL_ID = MODEL_ID_110M
TERM_EXTRACTION_TOP_N = 50

# How many chunks share one TDT decode loop. The decode step is dispatch-
# bound (0.49 ms at batch 1, 0.71 ms at batch 8), so this is close to free
# speed: measured on a 55-minute Earnings-21 clip with the 110m model,
# 54.7 s at 1, 23.0 s at 4, 19.0 s at 8, with byte-identical transcripts.
# It stops at 4 because every chunk in a batch is decoded before any of its
# text is handed out, and the browser path shows partials as they land: 4
# chunks is about 7 minutes of audio between updates already.
DECODE_BATCH = 4

# SEC EDGAR requires a descriptive contact string in the User-Agent header
# on every request. Fill this in with a real contact before running
# captionlm/eval_dataset.py against the live API.
SEC_CONTACT = "captionlm/1.0 (chielerli@gmail.com)"


@dataclass
class SpotterConfig:
    """CTC-WS hyperparameters. Defaults match NeMo's own
    run_word_spotter defaults; retune cb_weight against your own term
    list via eval.py's --cb-weight sweep before trusting these."""

    beam_threshold: float = 5.0
    cb_weight: float = 3.0
    ctc_ali_token_weight: float = 0.5
    keyword_threshold: float = -5.0
    blank_threshold: float = 0.8
    non_blank_threshold: float = 0.001
    intersection_threshold: float = 30.0


# Earnings-21 per-clip (the shipped per-document configuration) was net -9
# at the stock cb_weight=3.0 on clips of 50-90 minutes; every weight >=2.0
# tested regressed on that same long-form audio. Capping to 1.0 above this
# duration cut the damage to net -2 at zero runtime cost, because duration
# is known before transcription starts. That is one sweep, not a validated
# threshold -- see docs/HANDOFF.md for the validation still required.
# for the follow-up this needs before it's trusted as more than a stopgap.
LONG_FORM_DURATION_S = 300.0
LONG_FORM_CB_WEIGHT = 1.0


def select_cb_weight(duration_s: float, base_weight: float) -> float:
    """Cap cb_weight for long-form audio. base_weight is returned
    unchanged for anything at or under LONG_FORM_DURATION_S."""
    if duration_s > LONG_FORM_DURATION_S:
        return LONG_FORM_CB_WEIGHT
    return base_weight
