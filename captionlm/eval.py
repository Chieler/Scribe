"""Score domain-biased captioning against a labeled clip set: term
F-score and WER, with biasing on vs off. Run captionlm/build_eval_set.py
first to populate clip_dir automatically from Earnings-21, or hand-collect
<clip>.wav/<clip>.txt pairs yourself.

Each clip is biased with its own <clip>.terms.txt when present, matching
what cli.py does: one document biases its own audio. The combined term
list passed on the command line is used for F-score scoring, so the
metric's vocabulary stays constant across clips.

WER is reported three ways: overall, and split into B-WER (reference
words on the term list) and U-WER (the rest). Overall WER cannot settle
whether biasing paid, because listed words are a small share of the
words and a gain on them hides under run-to-run noise -- see bias_wer.

Three term metrics, because one number cannot carry all three questions:
F-score says how well listed terms that were spoken came out; the
injection rate says how often a listed term that was NOT spoken got put
in anyway, which F-score has no denominator for; per_term says which
terms are responsible for either. The headline pair is recall against
injection rate -- precision cannot improve under biasing and is a
guardrail, not a score -- and "yield" reduces that pair to one number.
"""
import argparse
import glob
import json
import os
import re

import jiwer

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID
from captionlm.fusion import WHISPER_MODEL_ID, fuse_text, second_opinion
from captionlm.progressive import transcribe_progressive
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer
from captionlm.vendor.fscore import compute_fscore, keyword_stats


def load_clips(clip_dir: str, fallback_terms: list[str] | None = None) -> list[dict]:
    clips = []
    for wav_path in sorted(glob.glob(os.path.join(clip_dir, "*.wav"))):
        base = os.path.splitext(wav_path)[0]
        txt_path = base + ".txt"
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            reference = f.read().strip()
        terms_path = base + ".terms.txt"
        terms = load_term_list(terms_path) if os.path.isfile(terms_path) else list(fallback_terms or [])
        clips.append({"wav": wav_path, "text": reference, "terms": terms})
    return clips


def transcribe_clips(model, clips: list[dict]) -> list[dict]:
    samples = []
    for clip in clips:
        result = transcribe_progressive(model, clip["wav"])
        samples.append({"text": clip["text"], "pred_text": result.text})
    return samples


def transcribe_baseline(model, clips: list[dict]) -> list[dict]:
    model.context_graph = None
    return transcribe_clips(model, clips)


def transcribe_biased(
    model,
    clips: list[dict],
    tokenizer,
    blank_idx: int,
    cb_weight: float | None = None,
    second_opinion_model: str | None = None,
) -> list[dict]:
    if cb_weight is not None:
        model.spotter_config.cb_weight = cb_weight
    samples = []
    for clip in clips:
        model.context_graph = build_context_graph(clip["terms"], tokenizer, blank_idx)
        sample = transcribe_clips(model, [clip])[0]
        if second_opinion_model:
            sample["pred_text"] = fuse_text(
                sample["pred_text"],
                second_opinion(clip["wav"], second_opinion_model),
                clip["terms"],
            )
        samples.append(sample)
    return samples


# jiwer 4.0.0's default transform does NOT normalize -- wer_default is
# [RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords], so
# jiwer.wer("Net Sales grew.", "net sales grew") is 1.0. The model emits
# cased, punctuated text and the references are rebuilt from the .nlp
# token+punctuation columns, so without this the score conflates
# recognition errors with casing and punctuation mismatch.
#
# jiwer's own wer_standardize is not the fix: it includes RemoveWhiteSpace,
# which glues every word into a single token. Compose it explicitly.
_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _wer(references: list[str], hypotheses: list[str]) -> float:
    # NB: jiwer.wer takes reference_transform, NOT truth_transform.
    return jiwer.wer(
        references,
        hypotheses,
        reference_transform=_NORMALIZE,
        hypothesis_transform=_NORMALIZE,
    )


def _bias_vocabulary(terms: list[str]) -> set[str]:
    """The term list as a set of single normalized words.

    B-WER is a per-word metric and the alignment it reads is per-word, so
    a multi-word entry like "DERIK DE BRUIN" has to enter the vocabulary
    as its three words. That over-counts: "de" is now a biased word
    everywhere it appears. Every published B-WER does this -- the metric
    is defined over a word-level bias vocabulary, not over spans -- and
    the alternative, span alignment, would not be comparable to anyone
    else's number.
    """
    vocab: set[str] = set()
    for term in terms:
        vocab.update(re.findall(r"[a-z0-9]+", term.lower()))
    return vocab


def bias_wer(references: list[str], hypotheses: list[str], terms: list[str]) -> dict:
    """B-WER and U-WER: the same errors, split by whether the reference
    word is on the bias list.

    Plain WER hides the thing biasing is for. Listed terms are a few
    percent of the words, so recovering all of them moves overall WER by
    less than the noise between two runs, while a regression on ordinary
    words that the biasing caused is invisible underneath the gain. The
    split makes both directions legible: B-WER is the score, U-WER is the
    guardrail that says biasing did not damage the rest of the sentence.

    Errors are attributed by REFERENCE word, so a substitution or
    deletion of a listed word is a B-WER error whatever came out instead.
    Insertions have no reference word, so they go by the inserted word:
    hallucinating "Paxos" is a B-WER error, and that is what keeps B-WER
    from being gameable by emitting listed terms everywhere. Each rate's
    denominator is its own share of the reference words, so B-WER and
    U-WER do not average back to WER.
    """
    vocab = _bias_vocabulary(terms)
    out = jiwer.process_words(
        references,
        hypotheses,
        reference_transform=_NORMALIZE,
        hypothesis_transform=_NORMALIZE,
    )
    errors = {True: 0, False: 0}
    words = {True: 0, False: 0}
    for reference, hypothesis, chunks in zip(out.references, out.hypotheses, out.alignments):
        for chunk in chunks:
            if chunk.type == "insert":
                for word in hypothesis[chunk.hyp_start_idx : chunk.hyp_end_idx]:
                    errors[word in vocab] += 1
                continue
            for word in reference[chunk.ref_start_idx : chunk.ref_end_idx]:
                biased = word in vocab
                words[biased] += 1
                if chunk.type != "equal":
                    errors[biased] += 1
    return {
        "b_wer": errors[True] / words[True] if words[True] else None,
        "u_wer": errors[False] / words[False] if words[False] else None,
        "b_words": words[True],
        "u_words": words[False],
        "b_errors": errors[True],
        "u_errors": errors[False],
    }

def _term_words(text: str) -> str:
    """Text reduced to space-delimited words, padded so that a substring
    search for " term " is a whole-word search."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def unspoken_injections(clips: list[dict], samples: list[dict]) -> dict:
    """How often biasing puts a listed term into the transcript that the
    speaker never said.

    The F-score cannot measure this. Its precision is tp/(tp+fp), and fp
    only counts terms that fired -- there is no denominator for the terms
    that were listed, never spoken, and correctly stayed out. That
    population is where biasing carries pure risk, and on real documents
    it is most of the list: measured per clip, 10-38% of an extracted term
    list is actually spoken on Earnings-21, against 100% on the read-aloud
    set, where the document IS the transcript and nothing can be falsely
    accepted at all.

    Counted per (clip, term) pair, so the denominator matches the number
    of independent chances biasing had to inject something. Run it on the
    unbiased predictions too: a term can land in the hypothesis by chance,
    and that rate is the floor this one has to beat.
    """
    unspoken = 0
    injected = []
    for clip, sample in zip(clips, samples):
        reference = _term_words(clip["text"])
        hypothesis = _term_words(sample["pred_text"])
        for term in clip["terms"]:
            key = _term_words(term)
            if key.strip() == "" or key in reference:
                continue
            unspoken += 1
            if key in hypothesis:
                injected.append(term)
    return {
        "unspoken_terms": unspoken,
        "injected": injected,
        "injection_rate": len(injected) / unspoken if unspoken else None,
    }


def per_term_stats(all_terms: list[str], baseline: list[dict], biased: list[dict]) -> list[dict]:
    """One row per term that either occurred or fired, worst first.

    The aggregate F-score cannot say WHICH terms misbehave, and the answer
    changes the fix: sixteen false accepts spread over sixteen terms is a
    threshold problem, sixteen from one term is a bad list entry.
    """
    base = keyword_stats(baseline, all_terms)
    bias = keyword_stats(biased, all_terms)
    rows = []
    for term, (tp, gt, fp) in bias.items():
        b_tp, b_gt, b_fp = base[term]
        if not (gt or fp or b_gt or b_fp):
            continue
        rows.append(
            {
                "term": term,
                "spoken": gt,
                "baseline_correct": b_tp,
                "biased_correct": tp,
                "baseline_false_accepts": b_fp,
                "biased_false_accepts": fp,
            }
        )
    rows.sort(key=lambda r: (-(r["biased_false_accepts"]), -(r["spoken"] - r["biased_correct"]), r["term"]))
    return rows


def biasing_yield(per_term: list[dict], baseline: dict, biased: dict) -> dict:
    """Recall against injection, in the same unit: term occurrences.

    Precision is the wrong scoreboard for biasing and cannot improve.
    The unbiased baseline's false accepts are already near zero -- it
    fails by not emitting the term at all, not by emitting the wrong one
    -- and biasing works by emitting listed terms MORE often, so every
    extra emission is either a true positive or a false accept and
    precision only has a downward path. Measured at 1.1b: 67 term misses
    fixed, 12 false accepts created, precision 0.9853 -> 0.9630.

    What the trade actually is: occurrences recovered, minus occurrences
    hallucinated from the list. Both counts, both attributable, and the
    injections are netted against the unbiased arm because a term can
    land in the transcript by chance -- on Earnings-21, 12 of the 17
    injections at cb_weight 0.5 also happen with biasing switched off.
    """
    recovered = sum(r["biased_correct"] - r["baseline_correct"] for r in per_term)
    injected = len(biased["injected"]) - len(baseline["injected"])
    return {
        "terms_recovered": recovered,
        "injections_added": injected,
        "net_term_gain": recovered - injected,
    }


def score(clips: list[dict], baseline: list[dict], biased: list[dict], all_terms: list[str]) -> dict:
    references = [c["text"] for c in clips]
    baseline_preds = [s["pred_text"] for s in baseline]
    biased_preds = [s["pred_text"] for s in biased]

    baseline_p, baseline_r, baseline_f = compute_fscore(baseline, all_terms)
    biased_p, biased_r, biased_f = compute_fscore(biased, all_terms)

    per_clip = [
        {
            "clip": os.path.basename(clip["wav"]),
            "reference": clip["text"],
            "baseline_pred": base_pred,
            "biased_pred": bias_pred,
            "baseline_wer": _wer([clip["text"]], [base_pred]),
            "biased_wer": _wer([clip["text"]], [bias_pred]),
            "n_terms": len(clip["terms"]),
        }
        for clip, base_pred, bias_pred in zip(clips, baseline_preds, biased_preds)
    ]

    baseline_injections = unspoken_injections(clips, baseline)
    biased_injections = unspoken_injections(clips, biased)
    per_term = per_term_stats(all_terms, baseline, biased)

    return {
        "baseline": {
            "wer": _wer(references, baseline_preds),
            "precision": baseline_p,
            "recall": baseline_r,
            "fscore": baseline_f,
            **bias_wer(references, baseline_preds, all_terms),
            **baseline_injections,
        },
        "biased": {
            "wer": _wer(references, biased_preds),
            "precision": biased_p,
            "recall": biased_r,
            "fscore": biased_f,
            **bias_wer(references, biased_preds, all_terms),
            **biased_injections,
        },
        "yield": biasing_yield(per_term, baseline_injections, biased_injections),
        "per_clip": per_clip,
        "per_term": per_term,
    }


def run_eval(
    clip_dir: str,
    term_list_path: str,
    model_id: str = MODEL_ID,
    cb_weight: float | None = None,
    baseline: list[dict] | None = None,
    second_opinion_model: str | None = None,
) -> dict:
    all_terms = load_term_list(term_list_path)
    clips = load_clips(clip_dir, fallback_terms=all_terms)
    assert clips, f"no .wav/.txt pairs found in {clip_dir}"

    model = load_biased_model(model_id)
    if baseline is None:
        baseline = transcribe_baseline(model, clips)

    tokenizer = load_tokenizer(model_id)
    biased = transcribe_biased(
        model, clips, tokenizer, len(model.vocabulary), cb_weight, second_opinion_model
    )

    stats = score(clips, baseline, biased, all_terms)
    stats["cb_weight"] = cb_weight
    stats["baseline_samples"] = baseline
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate domain-biased captioning")
    parser.add_argument("clip_dir", help="Directory of <clip>.wav/<clip>.txt pairs")
    parser.add_argument("terms", help="Path to the combined domain term list (.txt), used for F-score scoring")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cb-weight", type=float, default=None)
    parser.add_argument(
        "--second-opinion",
        nargs="?",
        const=WHISPER_MODEL_ID,
        default=None,
        metavar="MODEL_ID",
        help="Fuse each biased hypothesis with an independent model's transcript "
        "of the same audio (see captionlm/fusion.py).",
    )
    parser.add_argument("--out-json", help="Write full per-clip results to this path as JSON")
    args = parser.parse_args()

    stats = run_eval(
        args.clip_dir, args.terms, args.model, args.cb_weight,
        second_opinion_model=args.second_opinion,
    )
    for label in ("baseline", "biased"):
        s = stats[label]
        rate = "n/a" if s["injection_rate"] is None else f"{s['injection_rate']:.4f}"
        b = "n/a" if s["b_wer"] is None else f"{s['b_wer']:.3f}"
        u = "n/a" if s["u_wer"] is None else f"{s['u_wer']:.3f}"
        print(
            f"{label}: WER={s['wer']:.3f} B-WER={b} ({s['b_words']} words) U-WER={u} "
            f"P={s['precision']:.3f} R={s['recall']:.3f} "
            f"F={s['fscore']:.3f} inject={rate} ({len(s['injected'])}/{s['unspoken_terms']} unspoken)"
        )

    y = stats["yield"]
    print(
        f"yield: recovered {y['terms_recovered']:+d} term occurrences, "
        f"injected {y['injections_added']:+d} -> net {y['net_term_gain']:+d}"
    )

    worst = [r for r in stats["per_term"] if r["biased_false_accepts"]][:5]
    if worst:
        print("worst false accepts: " + ", ".join(
            f"{r['term']}({r['biased_false_accepts']})" for r in worst
        ))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in stats.items() if k != "baseline_samples"}, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
