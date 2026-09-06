import json

from captionlm.eval import (
    _wer,
    bias_wer,
    load_clips,
    per_term_stats,
    score,
    transcribe_biased,
    unspoken_injections,
)


def test_load_clips_pairs_wav_and_txt(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")
    (tmp_path / "clip2.wav").write_bytes(b"\x00")
    (tmp_path / "clip2.txt").write_text("second clip", encoding="utf-8")
    (tmp_path / "orphan.wav").write_bytes(b"\x00")  # no matching .txt

    clips = load_clips(str(tmp_path))

    assert len(clips) == 2
    texts = {c["text"] for c in clips}
    assert texts == {"hello there", "second clip"}
    assert all(c["wav"].endswith(".wav") for c in clips)


def test_load_clips_reads_per_clip_terms(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")
    (tmp_path / "clip1.terms.txt").write_text("net sales\nfree cash flow\n", encoding="utf-8")

    clips = load_clips(str(tmp_path), fallback_terms=["ignored"])

    assert clips[0]["terms"] == ["net sales", "free cash flow"]


def test_load_clips_falls_back_when_no_per_clip_terms(tmp_path):
    (tmp_path / "clip1.wav").write_bytes(b"\x00")
    (tmp_path / "clip1.txt").write_text("hello there", encoding="utf-8")

    clips = load_clips(str(tmp_path), fallback_terms=["combined term"])

    assert clips[0]["terms"] == ["combined term"]


def test_transcribe_biased_builds_a_graph_per_clip(monkeypatch):
    graphs_seen = []

    class FakeResult:
        text = "hello"

    class FakeModel:
        context_graph = None

    monkeypatch.setattr(
        "captionlm.eval.transcribe_progressive",
        lambda model, wav_path: graphs_seen.append(model.context_graph) or FakeResult(),
    )
    monkeypatch.setattr(
        "captionlm.eval.build_context_graph",
        lambda terms, tokenizer, blank_idx: tuple(terms),
    )

    clips = [
        {"wav": "a.wav", "text": "x", "terms": ["alpha"]},
        {"wav": "b.wav", "text": "y", "terms": ["beta"]},
    ]
    samples = transcribe_biased(FakeModel(), clips, tokenizer=None, blank_idx=0)

    assert graphs_seen == [("alpha",), ("beta",)]
    assert [s["text"] for s in samples] == ["x", "y"]


def test_score_emits_one_record_per_clip():
    clips = [
        {"wav": "/tmp/a.wav", "text": "net sales grew", "terms": ["net sales"]},
        {"wav": "/tmp/b.wav", "text": "free cash flow rose", "terms": ["free cash flow"]},
    ]
    baseline = [
        {"text": "net sales grew", "pred_text": "net sails grew"},
        {"text": "free cash flow rose", "pred_text": "free cash flow rose"},
    ]
    biased = [
        {"text": "net sales grew", "pred_text": "net sales grew"},
        {"text": "free cash flow rose", "pred_text": "free cash flow rose"},
    ]

    stats = score(clips, baseline, biased, ["net sales", "free cash flow"])

    assert [r["clip"] for r in stats["per_clip"]] == ["a.wav", "b.wav"]
    assert stats["per_clip"][0]["baseline_wer"] > 0.0
    assert stats["per_clip"][0]["biased_wer"] == 0.0
    assert stats["biased"]["wer"] < stats["baseline"]["wer"]
    assert stats["per_clip"][0]["n_terms"] == 1


def test_wer_ignores_case_and_punctuation_but_counts_real_errors():
    # Regression: jiwer 4.0.0's default transform does not normalize, and
    # jiwer.wer() takes reference_transform (not truth_transform).
    assert _wer(["Net Sales grew."], ["net sales grew"]) == 0.0
    assert _wer(["net sales grew"], ["net sails grew"]) > 0.0


def test_score_ignores_case_and_punctuation():
    clips = [{"wav": "/tmp/a.wav", "text": "Net Sales grew.", "terms": ["net sales"]}]
    samples = [{"text": "Net Sales grew.", "pred_text": "net sales grew"}]

    stats = score(clips, samples, samples, ["net sales"])

    assert stats["baseline"]["wer"] == 0.0


def test_injection_rate_counts_only_terms_that_were_never_spoken():
    # The population F-score has no denominator for: listed, not said, and
    # therefore pure downside if biasing puts it in anyway.
    clips = [{"wav": "/tmp/a.wav", "text": "net sales grew", "terms": ["net sales", "EBITDA"]}]
    clean = [{"text": "net sales grew", "pred_text": "net sales grew"}]
    hallucinated = [{"text": "net sales grew", "pred_text": "net sales EBITDA grew"}]

    assert unspoken_injections(clips, clean) == {
        "unspoken_terms": 1,
        "injected": [],
        "injection_rate": 0.0,
    }
    assert unspoken_injections(clips, hallucinated)["injection_rate"] == 1.0


def test_injection_rate_is_none_when_every_listed_term_is_spoken():
    # The read-aloud set's shape: the document IS the transcript, so there
    # is nothing to falsely accept and the metric must say so rather than
    # report a flattering zero.
    clips = [{"wav": "/tmp/a.wav", "text": "net sales grew", "terms": ["net sales"]}]
    samples = [{"text": "net sales grew", "pred_text": "net sales grew"}]

    assert unspoken_injections(clips, samples)["injection_rate"] is None


def test_injection_matching_is_whole_word():
    clips = [{"wav": "/tmp/a.wav", "text": "we shipped it", "terms": ["ray"]}]
    samples = [{"text": "we shipped it", "pred_text": "we sprayed it"}]

    assert unspoken_injections(clips, samples)["injected"] == []


def test_per_term_stats_puts_the_terms_doing_damage_first():
    baseline = [{"text": "we deployed alpha today", "pred_text": "we deployed alpha today"}]
    biased = [{"text": "we deployed alpha today", "pred_text": "we deployed beta today"}]

    rows = per_term_stats(["alpha", "beta"], baseline, biased)

    assert rows[0]["term"] == "beta"
    assert rows[0]["spoken"] == 0
    assert rows[0]["biased_false_accepts"] == 1
    alpha = next(r for r in rows if r["term"] == "alpha")
    assert (alpha["baseline_correct"], alpha["biased_correct"]) == (1, 0)


def test_yield_nets_recovered_terms_against_hallucinated_ones():
    # Recall against injection in one unit. Precision cannot improve under
    # biasing, so this is the pair that says whether biasing paid.
    clips = [{"wav": "/tmp/a.wav", "text": "we deployed alpha today", "terms": ["alpha", "beta"]}]
    baseline = [{"text": "we deployed alpha today", "pred_text": "we deployed alfa today"}]
    biased = [{"text": "we deployed alpha today", "pred_text": "we deployed alpha beta today"}]

    stats = score(clips, baseline, biased, ["alpha", "beta"])

    # "alpha" recovered; "beta" was listed, never spoken, and put in anyway.
    assert stats["yield"] == {
        "terms_recovered": 1,
        "injections_added": 1,
        "net_term_gain": 0,
    }


def test_bias_wer_splits_errors_by_reference_word():
    # One substitution of a listed word, one of an unlisted word, and one
    # inserted listed word. The insertion has no reference word, so it is
    # attributed by what came out -- otherwise emitting listed terms
    # everywhere would improve B-WER.
    stats = bias_wer(
        ["paxos runs on every node"],
        ["taxes runs on each node paxos"],
        ["Paxos", "linearizability"],
    )

    assert (stats["b_words"], stats["u_words"]) == (1, 4)
    assert stats["b_errors"] == 2  # the substitution and the insertion
    assert stats["u_errors"] == 1  # "every" -> "each"
    assert stats["b_wer"] == 2.0
    assert stats["u_wer"] == 0.25


def test_bias_wer_expands_multiword_terms_into_words():
    # The alignment is per word, so a multi-word entry enters the
    # vocabulary as its words -- "de" counts as biased wherever it lands.
    stats = bias_wer(
        ["derik de bruin said hello"],
        ["derek de bruin said hello"],
        ["Derik De Bruin"],
    )

    assert (stats["b_words"], stats["u_words"]) == (3, 2)
    assert stats["b_errors"] == 1
    assert stats["u_wer"] == 0.0
