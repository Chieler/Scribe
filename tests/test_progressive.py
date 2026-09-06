"""The chunk loop and its callback contract, with the model stubbed out.

What is worth checking here is not the acoustics -- that is the library's --
but that a caller gets a partial per chunk, that the seconds it is handed line
up with the audio, and that the last partial is the returned result.
"""
import mlx.core as mx
import pytest
from parakeet_mlx.alignment import AlignedToken, sentences_to_result, tokens_to_sentences

from captionlm import progressive


class FakeModel:
    """Emits one token per chunk, named for the chunk it came from."""

    class preprocessor_config:
        sample_rate = 16000
        hop_length = 160

    def __init__(self):
        self.calls = 0
        self.batches = 0

    def generate(self, mel, *, decoding_config=None):
        self.calls += 1
        token = AlignedToken(id=self.calls, text=f" chunk{self.calls}", start=0.0, duration=1.0)
        token.end = 1.0
        return [sentences_to_result(tokens_to_sentences([token]))]

    def generate_batch(self, mels, *, decoding_config=None):
        self.batches += 1
        return [self.generate(mel)[0] for mel in mels]


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(progressive, "get_logmel", lambda audio, cfg: audio)
    return FakeModel()


def _audio(monkeypatch, seconds):
    monkeypatch.setattr(progressive, "load_audio",
                        lambda path, rate, dtype: mx.zeros((int(seconds * rate),)))


def test_audio_shorter_than_one_chunk_yields_a_single_partial(stubbed, monkeypatch):
    _audio(monkeypatch, 30)
    seen = []
    result = progressive.transcribe_progressive(
        stubbed, "x.wav", chunk_duration=120.0, on_partial=lambda r, d, t: seen.append((d, t))
    )
    assert seen == [(30.0, 30.0)]
    assert stubbed.calls == 1
    assert result.text.strip() == "chunk1"


def test_a_long_file_reports_a_partial_per_chunk_up_to_its_length(stubbed, monkeypatch):
    _audio(monkeypatch, 300)
    seen = []
    progressive.transcribe_progressive(
        stubbed, "x.wav", chunk_duration=120.0, overlap_duration=15.0,
        on_partial=lambda r, d, t: seen.append((round(d, 1), t)),
    )
    # 120s windows stepping 105s: partials land at 120, 225, 300, and the last
    # one is the end of the audio rather than past it.
    assert [d for d, _ in seen] == [120.0, 225.0, 300.0]
    assert all(t == 300.0 for _, t in seen)


def test_the_transcript_only_grows(stubbed, monkeypatch):
    # A UI renders every partial. One that shrank would make text disappear
    # under the reader.
    _audio(monkeypatch, 300)
    lengths = []
    progressive.transcribe_progressive(
        stubbed, "x.wav", chunk_duration=120.0,
        on_partial=lambda r, d, t: lengths.append(len(r.text)),
    )
    assert lengths == sorted(lengths) and len(lengths) > 1


def test_the_last_partial_is_what_gets_returned(stubbed, monkeypatch):
    _audio(monkeypatch, 300)
    last = []
    result = progressive.transcribe_progressive(
        stubbed, "x.wav", chunk_duration=120.0, on_partial=lambda r, d, t: last.append(r.text)
    )
    assert result.text == last[-1]


def test_chunks_are_decoded_in_batches(stubbed, monkeypatch):
    # 6 chunks at batch 2 is 3 decode calls, not 6, and the partials still
    # arrive one per chunk in order.
    _audio(monkeypatch, 540)
    seen = []
    progressive.transcribe_progressive(
        stubbed, "x.wav", chunk_duration=120.0, overlap_duration=15.0, decode_batch=2,
        on_partial=lambda r, d, t: seen.append(round(d, 1)),
    )
    assert stubbed.calls == 6
    assert stubbed.batches == 3
    assert seen == sorted(seen) and len(seen) == 6
