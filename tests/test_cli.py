import wave

from parakeet_mlx.alignment import AlignedResult, AlignedSentence, AlignedToken

import captionlm.cli as cli
from captionlm.cli import configure_document_bias, format_timestamp, get_audio_duration, result_to_srt
from captionlm.config import SpotterConfig


def test_get_audio_duration(tmp_path):
    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 8_000)

    duration = get_audio_duration(str(audio_path))
    assert 0.49 < duration < 0.51


def test_configure_document_bias_applies_and_resets_the_duration_cap(monkeypatch):
    """One cached model must not carry a long-recording cap into the next file."""
    class Model:
        spotter_config = SpotterConfig(cb_weight=3.0)

    model = Model()
    monkeypatch.setattr(
        "captionlm.cli.get_audio_duration",
        lambda path: 301.0 if path == "long.wav" else 120.0,
    )

    assert configure_document_bias(model, "long.wav", base_weight=3.0) == 1.0
    assert model.spotter_config.cb_weight == 1.0
    assert configure_document_bias(model, "short.wav", base_weight=3.0) == 3.0
    assert model.spotter_config.cb_weight == 3.0


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(65.25) == "00:01:05,250"
    assert format_timestamp(3661.001) == "01:01:01,001"


def test_result_to_srt():
    token = AlignedToken(id=0, text=" hello world", start=0.0, duration=1.5)
    sentence = AlignedSentence(text=" hello world", tokens=[token])
    result = AlignedResult(text=" hello world", sentences=[sentence])

    srt = result_to_srt(result)

    assert srt.splitlines() == [
        "1",
        "00:00:00,000 --> 00:00:01,500",
        "hello world",
        "",
    ]


def test_result_to_txt_keeps_final_words_without_caption_timing():
    first = AlignedSentence(
        text=" hello hello everyone",
        tokens=[AlignedToken(id=0, text=" hello hello everyone", start=0.0, duration=1.0)],
    )
    second = AlignedSentence(
        text=" welcome back",
        tokens=[AlignedToken(id=1, text=" welcome back", start=1.0, duration=1.0)],
    )
    result = AlignedResult(text=" hello hello everyone welcome back", sentences=[first, second])

    assert cli.result_to_txt(result) == "hello everyone\nwelcome back\n"


def test_caption_file_can_return_plain_text(monkeypatch):
    result = AlignedResult(
        text=" hello world",
        sentences=[
            AlignedSentence(
                text=" hello world",
                tokens=[AlignedToken(id=0, text=" hello world", start=0.0, duration=1.0)],
            )
        ],
    )

    monkeypatch.setattr(cli, "load_biased_model", lambda model_id: object())
    monkeypatch.setattr(cli, "transcribe_progressive", lambda model, path: result)

    assert cli.caption_file("recording.wav", None, output_format="txt") == "hello world\n"
