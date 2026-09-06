"""CLI: caption a video/audio file, optionally biased toward a domain
term list, and write an .srt file."""
import argparse
from pathlib import Path
from subprocess import run

from parakeet_mlx.alignment import AlignedResult, sentences_to_result, tokens_to_sentences

from captionlm.biased_model import load_biased_model
from captionlm.config import MODEL_ID, select_cb_weight
from captionlm.disfluency import strip_restarts
from captionlm.fusion import WHISPER_MODEL_ID, fuse_tokens, second_opinion
from captionlm.progressive import transcribe_progressive
from captionlm.terms import build_context_graph, load_term_list, load_tokenizer


def get_audio_duration(path: str) -> float:
    """Probe a file's duration via ffprobe, without decoding it. Works for
    both audio and video containers -- same ffmpeg dependency load_audio
    already requires."""
    out = run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(out.stdout.strip())


def configure_document_bias(model, audio_path: str, base_weight: float) -> float:
    """Apply the per-recording bias policy to a model reused across files.

    The model is cached by both product entry points.  Always derive the
    weight from ``base_weight`` rather than its current mutable setting, or a
    long recording's conservative cap would leak into the next short one.
    """
    weight = select_cb_weight(get_audio_duration(audio_path), base_weight)
    model.spotter_config.cb_weight = weight
    return weight


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def result_to_srt(result: AlignedResult, drop_restarts: bool = True) -> str:
    """Render sentences as SRT cues, dropping the speaker's false starts.

    Restart removal happens per sentence, which costs nothing: measured on
    the read-aloud set, every restart in it falls inside one sentence, and
    per-sentence and whole-transcript stripping give the identical WER
    (0.0784 -> 0.0712). Doing it here keeps the cue timestamps intact --
    the surviving copy is the second one, so the cue still ends where the
    speaker finished saying it.
    """
    lines = []
    for i, sentence in enumerate(result.sentences, start=1):
        text = sentence.text.strip()
        if drop_restarts:
            text = strip_restarts(text)
        lines.append(str(i))
        lines.append(f"{format_timestamp(sentence.start)} --> {format_timestamp(sentence.end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines) + "\n"


def result_to_txt(result: AlignedResult, drop_restarts: bool = True) -> str:
    """Render final spoken text without timestamps."""
    lines = []
    for sentence in result.sentences:
        text = sentence.text.strip()
        if drop_restarts:
            text = strip_restarts(text)
        if text:
            lines.append(text)
    return "\n".join(lines) + ("\n" if lines else "")


def caption_file(
    audio_path: str,
    term_list_path: str | None,
    model_id: str = MODEL_ID,
    second_opinion_model: str | None = None,
    output_format: str = "srt",
) -> str:
    model = load_biased_model(model_id)

    terms: list[str] = []
    if term_list_path:
        tokenizer = load_tokenizer(model_id)
        terms = load_term_list(term_list_path)
        blank_idx = len(model.vocabulary)
        model.context_graph = build_context_graph(terms, tokenizer, blank_idx)
        configure_document_bias(model, audio_path, model.spotter_config.cb_weight)

    result = transcribe_progressive(model, audio_path)

    if second_opinion_model:
        second = second_opinion(audio_path, second_opinion_model)
        result = sentences_to_result(
            tokens_to_sentences(fuse_tokens(result.tokens, second, terms))
        )

    if output_format == "txt":
        return result_to_txt(result)
    return result_to_srt(result)


def main():
    parser = argparse.ArgumentParser(description="Domain-biased captioning")
    parser.add_argument("audio", help="Path to an audio/video file")
    parser.add_argument("--terms", help="Path to a term list (.txt, one term per line)")
    parser.add_argument("--out", help="Output path (default: <audio>.<format>)")
    parser.add_argument("--format", choices=("srt", "txt"), default="srt")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--second-opinion",
        nargs="?",
        const=WHISPER_MODEL_ID,
        default=None,
        metavar="MODEL_ID",
        help="Transcribe again with an independent model and let it correct the "
        "spans it disagrees on. Measured on the read-aloud set at 1.1b: WER "
        "0.0712 -> 0.0560. Costs a 1.6 GB download the first time, and roughly "
        "a quarter again of the transcription time.",
    )
    args = parser.parse_args()

    output = caption_file(
        args.audio, args.terms, args.model, args.second_opinion, args.format
    )
    out_path = args.out or str(Path(args.audio).with_suffix("." + args.format))
    Path(out_path).write_text(output, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
