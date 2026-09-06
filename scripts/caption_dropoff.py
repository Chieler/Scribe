"""Caption every audio file in a drop-off directory, biased by the
document sitting next to it.

This is the product path end to end, in one command: a user has a
recording and the document it is about, and wants an .srt in which the
domain terms are spelled correctly. Pair them by filename --
`standup.m4a` next to `standup.pdf` -- and this extracts the terms,
compiles them into the context graph, and transcribes.

Nothing is trained. The term list becomes a trie at call time, which is
the whole reason CTC-WS was chosen over fine-tuning: the user waits for
transcription only, never for training.

An audio file with no matching document is still transcribed, unbiased,
with a warning -- that is the honest baseline, not a failure.
"""
import argparse
import glob
import os

from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences

from captionlm.biasable import extract_bias_terms
from captionlm.build_eval_set import convert_to_wav
from captionlm.cli import configure_document_bias, result_to_srt, result_to_txt
from captionlm.config import MODEL_ID, SpotterConfig
from captionlm.doc_import import extract_text
from captionlm.biased_model import load_biased_model
from captionlm.fusion import WHISPER_MODEL_ID, fuse_tokens, second_opinion
from captionlm.progressive import transcribe_progressive
from captionlm.terms import build_context_graph, load_tokenizer

AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".mov", ".aac", ".flac", ".ogg", ".webm", ".mkv"}
DOC_EXTS = {".txt", ".md", ".pdf", ".docx", ".doc", ".csv", ".rtf", ".html"}


SEPARATORS = "-_. "
# What this script writes next to a recording. They sit in the drop-off looking
# exactly like inputs -- `standup.terms.txt` reads as a document for
# `standup.m4a`, `standup.converted.wav` as a recording of its own -- so a run
# would feed its own output back in.
GENERATED = (".terms.txt", ".converted.wav", ".srt", ".transcript.txt")
# The drop-off's own instructions. Harmless while an unmatched document was
# ignored; once they are shared it would bias every recording toward this
# repository's vocabulary.
NOT_A_SOURCE = {"README.md"}


def partition_documents(directory: str) -> tuple[dict[str, list[str]], list[str]]:
    """Documents grouped by the recording they name, and the ones naming none.

    A document belongs to a recording when its basename is the recording's, or
    starts with it followed by a separator, so one recording can be biased by
    several documents at once: `standup.m4a` takes `standup.pdf`,
    `standup-notes.txt` and `standup_slides.md`. A document that could match
    two recordings goes to the longer one, so `standup-part2.txt` biases
    `standup-part2.m4a` and not `standup.m4a`.

    A document matching no recording is shared: `glossary.pdf` was dropped in
    to be used, and requiring it to be renamed after the recording is a
    contract nobody keeps. Naming a document after a recording is what scopes
    it to that one recording alone.
    """
    audio: dict[str, str] = {}
    docs: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        if path.lower().endswith(GENERATED) or os.path.basename(path) in NOT_A_SOURCE:
            continue
        base, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower()
        if ext in AUDIO_EXTS:
            audio[base] = path
        elif ext in DOC_EXTS:
            docs.append((base, path))

    named: dict[str, list[str]] = {b: [] for b in audio}
    shared: list[str] = []
    for base, path in docs:
        owners = [
            a for a in audio
            if base == a or (base.startswith(a) and base[len(a)] in SEPARATORS)
        ]
        if owners:
            named[max(owners, key=len)].append(path)
        else:
            shared.append(path)
    return {audio[b]: sorted(d) for b, d in named.items()}, sorted(shared)


def find_pairs(directory: str) -> list[tuple[str, list[str]]]:
    """(audio, every document that biases it) for every audio file."""
    named, shared = partition_documents(directory)
    return [(audio, sorted(docs + shared)) for audio, docs in named.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="dropoff")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--mode", choices=("terminology", "ner", "combined"), default="combined")
    parser.add_argument("--top-n", type=int, default=None, help="Cap the term list (default: no cap)")
    parser.add_argument("--cb-weight", type=float, default=None,
                        help="Biasing strength. Default is SpotterConfig's 3.0, "
                             "which measured best on the read-aloud eval set.")
    parser.add_argument("--second-opinion", nargs="?", const=WHISPER_MODEL_ID, default=None,
                        metavar="MODEL_ID",
                        help="Transcribe again with an independent model and let it "
                             "correct the spans it disagrees on. Measured at 1.1b: "
                             "read-aloud WER 0.0712 -> 0.0560, Earnings-21 0.1929 -> "
                             "0.1858, with term recall up on both. Costs a 1.6 GB "
                             "download and about a quarter again of the runtime.")
    args = parser.parse_args()

    pairs = find_pairs(args.directory)
    if not pairs:
        raise SystemExit(
            f"No audio found in {args.directory}/. Drop in a recording and, next to it, "
            f"the document it is about, with the SAME basename:\n"
            f"    {args.directory}/standup.m4a\n"
            f"    {args.directory}/standup.pdf"
        )

    print(f"{len(pairs)} recording(s) in {args.directory}/\nloading {args.model} ...", flush=True)
    model = load_biased_model(args.model)
    tokenizer = load_tokenizer(args.model)
    blank_idx = len(model.vocabulary)
    base_cb_weight = args.cb_weight if args.cb_weight is not None else SpotterConfig().cb_weight

    for audio, docs in pairs:
        base = os.path.splitext(audio)[0]
        print(f"\n=== {os.path.basename(audio)} ===")

        terms: list[str] = []
        if docs:
            text = "\n\n".join(extract_text(d) for d in docs)
            terms = extract_bias_terms(text, mode=args.mode, top_n=args.top_n)
            with open(f"{base}.terms.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(terms) + "\n")
            named = ", ".join(os.path.basename(d) for d in docs)
            print(f"  {len(terms)} terms from {named} -> {os.path.basename(base)}.terms.txt")
            print(f"    {', '.join(terms[:10])}{' ...' if len(terms) > 10 else ''}")
            model.context_graph = build_context_graph(terms, tokenizer, blank_idx)
        else:
            print("  no matching document -- transcribing UNBIASED")
            model.context_graph = None

        wav = audio
        if not audio.lower().endswith(".wav"):
            wav = f"{base}.converted.wav"
            convert_to_wav(audio, wav)

        if docs:
            weight = configure_document_bias(model, wav, base_cb_weight)
            print(f"  document bias: cb_weight={weight:g}")

        result = transcribe_progressive(model, wav)

        if args.second_opinion:
            print("  asking for a second opinion ...", flush=True)
            result = sentences_to_result(
                tokens_to_sentences(
                    fuse_tokens(result.tokens, second_opinion(wav, args.second_opinion), terms)
                )
            )

        out = f"{base}.srt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(result_to_srt(result))
        txt_out = f"{base}.transcript.txt"
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(result_to_txt(result))
        print(f"  wrote {out} and {txt_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
